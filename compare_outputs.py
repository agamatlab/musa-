import torch
import torch.nn.functional as F


def run_comparision():
    ref = torch.load("outputs/cuda_L256.pt")
    musa = torch.load("outputs/musa_L256.pt")

    a = ref["embeddings"].float().flatten()
    b = musa["embeddings"].float().flatten()
    diff = a - b
    cos = F.cosine_similarity(a, b, dim=0)
    print("Logits Cosine similarity:", cos.item())
    print("Max ABS error:", diff.abs().max().item())
    print("Mean ABS error:", diff.abs().mean().item())

    ref_top1 = ref.argmax(dim=-1)
    musa_top1 = ref.argmax(dim=-1)
    agreement = (ref_top1 == musa_top1).float().mean()
    print("Agreement:", agreement)
