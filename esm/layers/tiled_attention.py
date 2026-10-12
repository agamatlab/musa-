"""
Memory-efficient FP32 attention for MUSA, in pure PyTorch (no custom kernel).

Why: MUSA's torch_musa has no fused FP32 flash/efficient SDPA kernel, so
F.scaled_dot_product_attention falls back to the "math" path and materializes
the full L x L score matrix -> O(L^2) memory (360 MiB @ L=2048 in the probe).

This implements the FlashAttention algorithm (tiling + online softmax) with
ordinary torch ops. It never holds the full L x L scores: it streams over K/V
in blocks of `block_k`, keeping a running max (m) and running denominator (l)
per query row. Peak transient scores = [.. , L, block_k] instead of [.. , L, L]
-> O(L) memory, still FP32 -> no dtype change, no block_34 divergence risk.

Scope: INFERENCE (the gluon forward/logits path). The Python block-loop is
autograd-traceable but would store per-block intermediates for backward, which
defeats the memory goal; for training you'd wrap this in a custom
torch.autograd.Function (recompute in backward) or torch.utils.checkpoint.

Drop-in: in esm/layers/attention.py MultiHeadAttention.forward, replace the
SDPA call(s) with:
    context_BHLD = tiled_attention(query_BHLD, key_BHLD, value_BHLD, mask_BHLL)
(shapes are already [B, H, L, D] there).
"""
import torch


def tiled_attention(q, k, v, attn_mask=None, block_k: int = 512):
    """
    q, k, v : [B, H, Lq, D] and [B, H, Lk, D]  (same dtype/device; FP32 ok)
    attn_mask : [B, H, Lq, Lk] bool, True = attend; or None.
    block_k : K/V block size. Smaller = less memory, more loop overhead.
    returns context : [B, H, Lq, D]
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    scale = D ** -0.5
    neg_inf = torch.finfo(q.dtype).min

    out = torch.zeros_like(q)                                    # accumulator
    m = torch.full((B, H, Lq, 1), neg_inf, device=q.device, dtype=q.dtype)  # running max
    l = torch.zeros((B, H, Lq, 1), device=q.device, dtype=q.dtype)          # running sum

    for start in range(0, Lk, block_k):
        end = min(start + block_k, Lk)
        k_blk = k[:, :, start:end, :]                            # [B,H,bk,D]
        v_blk = v[:, :, start:end, :]

        s = torch.matmul(q, k_blk.transpose(-1, -2)) * scale     # [B,H,Lq,bk]  (never LxL)
        if attn_mask is not None:
            s = s.masked_fill(~attn_mask[:, :, :, start:end], neg_inf)

        m_blk = s.max(dim=-1, keepdim=True).values               # [B,H,Lq,1]
        m_new = torch.maximum(m, m_blk)
        p = torch.exp(s - m_new)                                 # [B,H,Lq,bk]
        alpha = torch.exp(m - m_new)                             # rescale old stats

        l = l * alpha + p.sum(dim=-1, keepdim=True)
        out = out * alpha + torch.matmul(p, v_blk)               # [B,H,Lq,D]
        m = m_new

    # rows that attended to nothing (fully-masked) -> l==0; keep them zero, no nan
    out = out / l.clamp_min(torch.finfo(q.dtype).tiny)
    return out


# ---- quick self-check vs the reference math path (run on any device) ----
if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    B, H, L, D = 1, 20, 1024, 64
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    ref = F.scaled_dot_product_attention(q, k, v)          # math path
    got = tiled_attention(q, k, v, block_k=256)
    print("max_abs_diff vs reference:", (ref - got).abs().max().item())
    # expect ~1e-6 (FP32 numerical noise) -> same math, O(L) memory
