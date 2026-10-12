#!/usr/bin/env python
"""
ESM3-on-MUSA accuracy regression suite. Run after every change that can touch numerics.

Ground truth  CPU FP32, computed once and FROZEN on disk (results/accuracy_ref/), so every
              future run is judged against the same answers (rebuild with --rebuild-ref).
Yardstick     CPU running the upstream BF16 recipe (Biohub/esm: whole model BF16 + autocast),
              also frozen. BF16 variants must be "no worse than normal BF16".

Variants (MUSA) declare a tier:
  exact  byte-identical to a named baseline variant (all 7 outputs, every case)
  fp32 / tf32 / bf16   thresholds below (vs the CPU FP32 reference)
Every variant is also run once through MUSA graph replay, which must equal eager bytes.

Cases: real proteins (76/129/238 aa), random L=64/512/1024, a B=2 batch, and an input WITH
coordinates (exercises block-0 geometric attention). Each is also run with 15% masked
residues for masked-LM cross-entropy.

Metrics per case: per-output relative L2 error + max abs; sequence top-1 agreement and KL;
structure top-1 / top-5 agreement and the FP32 probability gap at flipped positions
(near-tie analysis); embedding cosine (min/mean); masked-LM ΔCE; per-block hidden-state drift.

  tools/wrun.sh python /workspace/agha/tests/accuracy_suite.py                 # all variants
  tools/wrun.sh python /workspace/agha/tests/accuracy_suite.py --variants musa_bf16 musa_bf16_qkln
Exit code 1 if any variant FAILs.
"""
import argparse, contextlib, json, math, os, sys, time
from datetime import datetime
import torch, torch_musa  # noqa: F401
import torch.nn.functional as F
import esm.layers.attention as A
import esm.layers.rotary as RT
import esm.models.esm3 as E
from esm.layers.blocks import UnifiedTransformerBlock
from esm.models.esm3 import ESM3
from esm.utils.bf16_linears import prepare_bf16_linears

OUT = "/workspace/agha/results"
REF_DIR = f"{OUT}/accuracy_ref"
REF_VERSION = 1
FIELDS = ("sequence_logits", "structure_logits", "secondary_structure_logits",
          "sasa_logits", "function_logits", "residue_logits", "embeddings")
DEV = torch.device("musa:0")
KICK = torch.zeros(1, device=DEV)
torch.set_num_threads(min(32, os.cpu_count() or 8))

# ------------------------------------------------------------------ tiers / thresholds
TIERS = {
    #           rel L2 per output         seq top1  emb cos min  |ΔCE|   struct top1 floor
    "fp32": dict(rel=1e-4, rel_yard=None, seq=1.000, cos=0.99999, ce=1e-3, struct=0.99),
    "tf32": dict(rel=1e-2, rel_yard=None, seq=0.995, cos=0.9999,  ce=5e-3, struct=0.95),
    # bf16: per output rel L2 must be <= 1.5x the upstream-BF16 yardstick (or <= 2e-3)
    "bf16": dict(rel=2e-3, rel_yard=1.5,  seq=0.98,  cos=0.999,   ce=2e-2, struct=None),
}
STRUCT_YARD_MARGIN = 0.10      # bf16: structure top-1 may be at most 10 points below yardstick

# ------------------------------------------------------------------ variants
def _flags(fast_rotary=True, qkln=False, skip=True, tiled=False, tf32=False):
    RT.FAST_ROTARY, A.QK_LN_NATIVE_BF16 = fast_rotary, qkln
    E.SKIP_GEOM_WITHOUT_COORDS, A.USE_TILED_ATTENTION = skip, tiled
    with contextlib.suppress(Exception):
        torch.backends.musa.matmul.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

BF16_AC = lambda: torch.autocast("musa", dtype=torch.bfloat16)
NO_AC = contextlib.nullcontext
VARIANTS = {
    # name: (weights, flags, autocast, tier, exact_baseline)
    "musa_fp32":        ("fp32", dict(),                    NO_AC,   "fp32", None),
    "musa_fp32_norot":  ("fp32", dict(fast_rotary=False),   NO_AC,   "exact", "musa_fp32"),
    "musa_fp32_noskip": ("fp32", dict(skip=False),          NO_AC,   "exact", "musa_fp32"),
    "musa_tf32":        ("fp32", dict(tf32=True),           NO_AC,   "tf32", None),
    "musa_bf16":        ("bf16", dict(),                    BF16_AC, "bf16", None),
    "musa_bf16_norot":  ("bf16", dict(fast_rotary=False),   BF16_AC, "exact", "musa_bf16"),
    "musa_bf16_qkln":   ("bf16", dict(qkln=True),           BF16_AC, "bf16", None),
}

# ------------------------------------------------------------------ cases
REAL = {
    "ubiquitin_76": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "lysozyme_129": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPC"
                    "SALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "gfp_238": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQH"
               "DFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKV"
               "NFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
}
TRACE_CASE = "gfp_238"

def helix_coords(L):
    """Idealized alpha-helix backbone (N, CA, C) per residue; BOS/EOS rows are NaN."""
    i = torch.arange(L, dtype=torch.float64)
    th = torch.deg2rad(100.0 * i)
    def at(r, dth, dz):
        return torch.stack([r * torch.cos(th + dth), r * torch.sin(th + dth), 1.5 * i + dz], -1)
    xyz = torch.stack([at(1.6, -0.45, -0.75), at(2.3, 0.0, 0.0), at(1.7, 0.45, 0.8)], 1).float()
    xyz[0] = float("nan"); xyz[-1] = float("nan")
    return xyz.unsqueeze(0)

def build_cases(tok):
    g = torch.Generator().manual_seed(20261012)
    def rand_seq(n):
        return "".join("ACDEFGHIKLMNPQRSTVWY"[j] for j in torch.randint(0, 20, (n,), generator=g))
    seqs = dict(REAL)
    for n in (62, 510, 1022):
        seqs[f"random_{n + 2}"] = rand_seq(n)
    cases = {}
    for name, s in seqs.items():
        cases[name] = {"ids": torch.tensor(tok.encode(s)).unsqueeze(0)}
    b = [torch.tensor(tok.encode(rand_seq(126))) for _ in range(2)]
    cases["batch2_128"] = {"ids": torch.stack(b)}
    ids = torch.tensor(tok.encode(REAL["ubiquitin_76"])).unsqueeze(0)
    cases["coords_helix_78"] = {"ids": ids, "coords": helix_coords(ids.shape[1])}
    for name, c in cases.items():                                   # 15% masked copy
        B, L = c["ids"].shape
        gm = torch.Generator().manual_seed(L)
        pos = torch.randperm(L - 2, generator=gm)[: max(1, int(0.15 * (L - 2)))] + 1
        m = c["ids"].clone(); m[:, pos] = tok.mask_token_id
        c["masked"], c["pos"] = m, pos
    return cases

# ------------------------------------------------------------------ running
def wait(dev):
    if dev.type == "musa":
        _ = KICK.cpu()

def run_case(model, dev, ctx, c, trace=False):
    hidden = []
    hooks = []
    if trace:
        for blk in model.modules():
            if isinstance(blk, UnifiedTransformerBlock):
                hooks.append(blk.register_forward_hook(lambda m, i, o: hidden.append(o[0].detach().float().cpu())))
    kw = {"structure_coords": c["coords"].to(dev)} if "coords" in c else {}
    try:
        with torch.inference_mode(), ctx():
            o = model(sequence_tokens=c["ids"].to(dev), **kw); wait(dev)
            outs = {f: getattr(o, f).detach().float().cpu() for f in FIELDS}
            for h in hooks:
                h.remove()
            hooks = []
            mo = model(sequence_tokens=c["masked"].to(dev), **kw); wait(dev)
            lg = mo.sequence_logits.detach().float().cpu()[:, c["pos"]]
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), c["ids"][:, c["pos"]].reshape(-1)).item()
    finally:
        for h in hooks:
            h.remove()
    return {"outs": outs, "ce": ce, "hidden": hidden or None}

def run_all(model, dev, ctx, cases):
    return {n: run_case(model, dev, ctx, c, trace=(n == TRACE_CASE)) for n, c in cases.items()}

def graph_check(model, ctx, c):
    """Capture one forward (no coords case) and compare replay to eager, byte for byte."""
    x = c["ids"].to(DEV)
    with torch.inference_mode(), ctx():
        e = model(sequence_tokens=x); wait(DEV)
        e = {f: getattr(e, f).detach().clone() for f in FIELDS}
    static = x.clone()
    s = torch.musa.Stream(); s.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(s), torch.inference_mode():
        for _ in range(3):
            with ctx():
                model(sequence_tokens=static)
    torch.musa.current_stream().wait_stream(s); wait(DEV)
    gr = torch.musa.MUSAGraph()
    with torch.musa.stream(s), torch.inference_mode():
        gr.capture_begin()
        try:
            with ctx():
                go = model(sequence_tokens=static)
        finally:
            gr.capture_end()
    torch.musa.current_stream().wait_stream(s); wait(DEV)
    gr.replay(); wait(DEV)
    ok = all(torch.equal(getattr(go, f), e[f]) for f in FIELDS)
    del gr, static
    return ok

# ------------------------------------------------------------------ metrics
def compare(var, ref):
    out = {}
    for n in ref:
        a, r = var[n]["outs"], ref[n]["outs"]
        m = {"rel": {}, "max_abs": {}}
        for f in FIELDS:
            d = a[f] - r[f]
            m["rel"][f] = (d.norm() / r[f].norm().clamp_min(1e-12)).item()
            m["max_abs"][f] = d.abs().max().item()
        sa, sr = a["sequence_logits"], r["sequence_logits"]
        m["seq_top1"] = (sa.argmax(-1) == sr.argmax(-1)).float().mean().item()
        m["seq_kl"] = F.kl_div(F.log_softmax(sa, -1), F.log_softmax(sr, -1), log_target=True,
                               reduction="none").sum(-1).mean().item()
        pa, pr = a["structure_logits"], F.softmax(r["structure_logits"], -1)
        ia, ir = pa.argmax(-1), pr.argmax(-1)
        flip = ia != ir
        m["struct_top1"] = 1 - flip.float().mean().item()
        m["struct_top5"] = (pr.topk(5, -1).indices == ia.unsqueeze(-1)).any(-1).float().mean().item()
        if flip.any():
            gap = (pr.gather(-1, ir.unsqueeze(-1)) - pr.gather(-1, ia.unsqueeze(-1))).squeeze(-1)[flip]
            p1 = pr.gather(-1, ir.unsqueeze(-1)).squeeze(-1)[flip]
            m["struct_flip_gap_median"] = gap.median().item()
            m["struct_flip_frac_gap_lt_0.05"] = (gap < 0.05).float().mean().item()
            m["struct_flip_ref_p1_median"] = p1.median().item()
        cos = F.cosine_similarity(a["embeddings"], r["embeddings"], dim=-1)
        m["emb_cos_min"], m["emb_cos_mean"] = cos.min().item(), cos.mean().item()
        m["ce"], m["d_ce"] = var[n]["ce"], var[n]["ce"] - ref[n]["ce"]
        if var[n]["hidden"] and ref[n]["hidden"]:
            m["block_drift"] = [((x - y).norm() / y.norm()).item() for x, y in zip(var[n]["hidden"], ref[n]["hidden"])]
        out[n] = m
    return out

def judge(tier, M, yard_M, exact_ok=None):
    fails = []
    if tier == "exact":
        return [] if exact_ok else ["not byte-identical to baseline"]
    T = TIERS[tier]
    for n, m in M.items():
        for f, v in m["rel"].items():
            lim = T["rel"]
            if T["rel_yard"] is not None:
                lim = max(lim, T["rel_yard"] * yard_M[n]["rel"][f])
            if not v <= lim:
                fails.append(f"{n}: {f} rel {v:.2e} > {lim:.2e}")
        if m["seq_top1"] < T["seq"]:
            fails.append(f"{n}: seq top1 {m['seq_top1']:.3f} < {T['seq']}")
        if m["emb_cos_min"] < T["cos"]:
            fails.append(f"{n}: emb cos min {m['emb_cos_min']:.5f} < {T['cos']}")
        if abs(m["d_ce"]) > T["ce"]:
            fails.append(f"{n}: |dCE| {abs(m['d_ce']):.4f} > {T['ce']}")
        floor = T["struct"] if T["struct"] is not None else yard_M[n]["struct_top1"] - STRUCT_YARD_MARGIN
        if m["struct_top1"] < floor:
            fails.append(f"{n}: struct top1 {m['struct_top1']:.3f} < {floor:.3f}")
    return fails

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--rebuild-ref", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    a = ap.parse_args()
    t_start = time.time()

    cpu_model = ESM3.from_pretrained(device=torch.device("cpu")).eval()
    cases = build_cases(cpu_model.tokenizers.sequence)
    os.makedirs(REF_DIR, exist_ok=True)
    ref_path = f"{REF_DIR}/cpu_refs_v{REF_VERSION}.pt"
    if os.path.exists(ref_path) and not a.rebuild_ref:
        refs = torch.load(ref_path, weights_only=False)
        same_inputs = all(torch.equal(refs["inputs"][n], c["ids"]) for n, c in cases.items())
        assert same_inputs, "frozen reference was built from different inputs; use --rebuild-ref"
        print(f"loaded frozen references {ref_path} (built {refs['built']})", flush=True)
    else:
        _flags()
        t = time.time()
        fp32 = run_all(cpu_model, torch.device("cpu"), NO_AC, cases)
        cpu_model.to(torch.bfloat16)
        yard = run_all(cpu_model, torch.device("cpu"),
                       lambda: torch.autocast("cpu", dtype=torch.bfloat16), cases)
        refs = {"built": datetime.now().isoformat(timespec="seconds"), "fp32": fp32, "yard": yard,
                "inputs": {n: c["ids"] for n, c in cases.items()}}
        torch.save(refs, ref_path)
        print(f"built + froze references in {time.time() - t:.0f}s -> {ref_path}", flush=True)
    del cpu_model
    ref, yard_M = refs["fp32"], compare(refs["yard"], refs["fp32"])

    model = ESM3.from_pretrained(device=DEV).eval(); wait(DEV)
    order = sorted(a.variants, key=lambda v: VARIANTS[v][0] == "bf16")   # fp32 weights first
    results, raw = {}, {}
    for v in order:
        weights, flags, ctx, tier, base = VARIANTS[v]
        if weights == "bf16" and not getattr(model, "_bf16_linears", False):
            prepare_bf16_linears(model); model._bf16_linears = True
        _flags(**flags)
        t = time.time()
        raw[v] = run_all(model, DEV, ctx, cases)
        M = compare(raw[v], ref)
        exact_ok = None
        if tier == "exact":
            exact_ok = base in raw and all(torch.equal(raw[v][n]["outs"][f], raw[base][n]["outs"][f])
                                           for n in cases for f in FIELDS)
        g_ok = None if a.no_graph else graph_check(model, ctx, cases["lysozyme_129"])
        fails = judge(tier, M, yard_M, exact_ok)
        if g_ok is False:
            fails.append("graph replay != eager")
        results[v] = {"tier": tier, "baseline": base, "graph_eq_eager": g_ok, "fails": fails, "metrics": M,
                      "seconds": time.time() - t}
        print(f"ran {v:<18} {time.time() - t:5.1f}s  -> {'PASS' if not fails else 'FAIL'}", flush=True)
    _flags()

    # ---------------- report
    print(f"\n{'variant':<18}{'tier':<7}{'worst rel':>10}{'seq top1':>10}{'emb cos':>9}{'max|dCE|':>10}"
          f"{'struct t1':>10}{'t5':>7}{'graph':>7}  verdict")
    rows = [("cpu_bf16_upstream", "yard", yard_M)] + [(v, results[v]["tier"], results[v]["metrics"]) for v in order]
    for v, tier, M in rows:
        worst = max(max(m["rel"].values()) for m in M.values())
        g = results.get(v, {}).get("graph_eq_eager")
        verdict = "(yardstick)" if v not in results else ("PASS" if not results[v]["fails"] else "FAIL")
        print(f"{v:<18}{tier:<7}{worst:>10.2e}{min(m['seq_top1'] for m in M.values()) * 100:>9.1f}%"
              f"{min(m['emb_cos_min'] for m in M.values()):>9.5f}{max(abs(m['d_ce']) for m in M.values()):>10.4f}"
              f"{min(m['struct_top1'] for m in M.values()) * 100:>9.1f}%{min(m['struct_top5'] for m in M.values()) * 100:>6.1f}%"
              f"{'' if g is None else ('ok' if g else 'DIFF'):>7}  {verdict}")
    print("\nstructure-token flips (near-tie check), median FP32 prob gap at flipped positions / frac gap<0.05:")
    for v, tier, M in rows:
        parts = [f"{n.split('_')[0]}:{m['struct_flip_gap_median']:.3f}/{m['struct_flip_frac_gap_lt_0.05']:.2f}"
                 for n, m in M.items() if "struct_flip_gap_median" in m]
        print(f"  {v:<18} " + "  ".join(parts))
    print(f"\nper-block hidden-state drift ({TRACE_CASE}), rel L2 at blocks 1/12/24/36/48:")
    for v, tier, M in rows:
        d = M[TRACE_CASE].get("block_drift")
        if d:
            print(f"  {v:<18} " + "  ".join(f"{d[i]:.1e}" for i in (0, 11, 23, 35, 47)))
    for v in order:
        for f in results[v]["fails"][:8]:
            print(f"  FAIL {v}: {f}")

    p = f"{OUT}/accuracy_suite_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump({"time": datetime.now().isoformat(timespec="seconds"), "torch_musa": torch_musa.__version__,
               "reference": ref_path, "reference_built": refs["built"], "yardstick": yard_M,
               "variants": results, "seconds": time.time() - t_start}, open(p, "w"), indent=1, default=str)
    print(f"\nwrote {p}  ({time.time() - t_start:.0f}s)")
    sys.exit(1 if any(results[v]["fails"] for v in results) else 0)

if __name__ == "__main__":
    main()
