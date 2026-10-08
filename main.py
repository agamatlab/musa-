import os
import json

# Make esm read weights from ./data/weights instead of downloading from Hugging Face
os.environ.setdefault("INFRA_PROVIDER", "1")

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

import time
import torch

import argparse


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def get_device(name):
    if name.startswith("musa"):
        import torch_musa  # noqa: F401  registers the "musa" device with torch
    return torch.device(name)


def backend(device):
    # torch.cuda / torch.musa / torch.xpu / torch.mps, or None for cpu
    return getattr(torch, device.type, None) if device.type != "cpu" else None


def synchronize(device):
    mod = backend(device)
    if mod is not None:
        mod.synchronize()


def reset_peak_memory(device):
    mod = backend(device)
    if mod is not None and hasattr(mod, "reset_peak_memory_stats"):
        mod.reset_peak_memory_stats()


def peak_memory_gb(device):
    mod = backend(device)
    if mod is None or not hasattr(mod, "max_memory_allocated"):
        return None
    return mod.max_memory_allocated() / 1024**3


def load_model(device, precision="bf16"):
    model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
    model = model.to(DTYPES[precision])
    model.eval()
    return model


def get_protein(id):

    with open("inputs/test_sequences.jsonl") as f:
        for line in f:
            sample = json.loads(line)

            if sample["id"] == id:
                return sample

    raise ValueError("ProteinID is invalid")


def run_benchmark(device, model, n_runs, proteinID, n_warmup=2):
    sequence = get_protein(proteinID)["sequence"]

    protein = model.encode(ESMProtein(sequence=sequence))
    config = LogitsConfig(sequence=True, return_embeddings=True)

    with torch.no_grad():
        for _ in range(n_warmup):
            model.logits(protein, config)
        synchronize(device)

        reset_peak_memory(device)
        runtimes = []
        for _ in range(n_runs):
            synchronize(device)
            start = time.perf_counter()
            output = model.logits(protein, config)
            synchronize(device)
            runtimes.append(time.perf_counter() - start)

    return {
        "logits": output.logits.sequence.detach().cpu(),
        "embeddings": output.embeddings.detach().cpu(),
        "runtimes": runtimes,
        "avg_runtime": sum(runtimes) / len(runtimes),
        "peak_memory_gb": peak_memory_gb(device),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteinID", type=str, default="ubiquitin_human")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16", choices=DTYPES)
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(device, args.precision)
    print("Model Loaded")

    result = run_benchmark(
        device, model, args.runs, args.proteinID, args.warmup
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{device.type}_{args.proteinID}.pt"
    torch.save(
        {"logits": result["logits"], "embeddings": result["embeddings"]}, out_path
    )

    print("Logits shape:", tuple(result["logits"].shape))
    print("Embeddings shape:", tuple(result["embeddings"].shape))
    print(f"Avg runtime: {result['avg_runtime']:.4f} s over {args.runs} runs")
    if result["peak_memory_gb"] is not None:
        print(f"Peak memory: {result['peak_memory_gb']:.3f} GB")
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
