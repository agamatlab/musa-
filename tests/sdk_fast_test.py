#!/usr/bin/env python
"""
Validate ESM3.from_pretrained(..., musa_fast=True) on the end-user SDK paths.

  1. logits(): graph-cached path must be byte-identical to the same musa_fast config run eager
  2. batch_generate (seeded): graph vs eager musa_fast must give identical sequences
  3. timing vs the default FP32 SDK: logits() L=256 and batch_generate 4 prompts x 8 steps,
     first call (includes graph capture) and repeat call (cached graph)

  tools/wrun.sh python /workspace/agha/tests/sdk_fast_test.py
"""
import json, os, statistics, time
from datetime import datetime
import torch, torch_musa  # noqa: F401
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinTensor, GenerationConfig, LogitsConfig

os.environ["DISABLE_ITERATIVE_SAMPLING_TQDM"] = "1"
DEV = torch.device("musa:0")
KICK = torch.zeros(1, device=DEV)
def wait():
    _ = KICK.cpu()

CFG = LogitsConfig(sequence=True, structure=True, secondary_structure=True, sasa=True,
                   function=True, residue_annotations=True, return_embeddings=True)
def toks(L, seed=0):
    g = torch.Generator().manual_seed(L + seed)
    return torch.cat([torch.zeros(1, dtype=torch.long), torch.randint(4, 24, (L - 2,), generator=g),
                      torch.full((1,), 2, dtype=torch.long)]).to(DEV)

def flat(lo):
    t = lo.logits
    return {"sequence": t.sequence, "structure": t.structure, "ss8": t.secondary_structure, "sasa": t.sasa,
            "function": t.function, "residue": lo.residue_annotation_logits, "emb": lo.embeddings}

def timed(fn, reps=7):
    w = []
    for _ in range(reps):
        wait(); t0 = time.perf_counter(); fn(); wait(); w.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(w), w

g = torch.Generator().manual_seed(11)
BASE = ["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"]
PROMPTS = ["".join(c if k else "_" for c, k in zip(s, torch.rand(len(s), generator=g) > 0.3)) for s in BASE * 2]

def gen(model):
    torch.manual_seed(1234)
    out = model.batch_generate([ESMProtein(sequence=p) for p in PROMPTS],
                               [GenerationConfig(track="sequence", num_steps=8, temperature=0.7) for _ in PROMPTS])
    wait()
    return [o.sequence for o in out]

R = {"time": datetime.now().isoformat(timespec="seconds")}

# ---------------- default FP32 SDK (reference timings)
m32 = ESM3.from_pretrained(device=DEV, musa_fast=False).eval()
pt = ESMProteinTensor(sequence=toks(256))
m32.logits(pt, CFG); wait()
R["fp32_logits_ms"], _ = timed(lambda: m32.logits(pt, CFG))
gen(m32)
t = time.perf_counter(); gen(m32); R["fp32_generate_ms"] = (time.perf_counter() - t) * 1e3
del m32; torch.musa.empty_cache()

# ---------------- musa_fast
mf = ESM3.from_pretrained(device=DEV, musa_fast=True).eval()
print("musa_fast:", getattr(mf, "_musa_fast", False),
      "| param dtypes:", sorted({str(p.dtype) for p in mf.parameters()}), flush=True)

# 1. logits graph vs eager, byte-identical
t = time.perf_counter(); first = flat(mf.logits(pt, CFG)); wait(); R["fast_logits_first_ms"] = (time.perf_counter() - t) * 1e3
second = flat(mf.logits(pt, CFG)); wait()
graph_impl = mf._graph_forward
mf._graph_forward = lambda **kw: mf(**kw)                  # force eager, same config
eager = flat(mf.logits(pt, CFG)); wait()
R["fast_eager_logits_ms"], _ = timed(lambda: mf.logits(pt, CFG))
mf._graph_forward = graph_impl
bits = lambda x: x.view(torch.int32) if x.dtype == torch.float32 else x
R["logits_graph_eq_eager"] = {k: bool(torch.equal(bits(first[k]), bits(eager[k]))) for k in first}
R["logits_replay_eq_first"] = all(torch.equal(first[k], second[k]) for k in first)   # outputs not overwritten
R["fast_logits_ms"], _ = timed(lambda: mf.logits(pt, CFG))
other = ESMProteinTensor(sequence=toks(256, seed=5))                    # same shape, new content
o_g = flat(mf.logits(other, CFG)); wait()
mf._graph_forward = lambda **kw: mf(**kw)
o_e = flat(mf.logits(other, CFG)); wait()
mf._graph_forward = graph_impl
R["logits_new_input_eq"] = all(torch.equal(o_g[k], o_e[k]) for k in o_g)
R["first_still_intact"] = all(torch.equal(first[k], second[k]) for k in first)

# 2. generation: graph vs eager (same config, seeded)
t = time.perf_counter(); s_graph_first = gen(mf); R["fast_generate_first_ms"] = (time.perf_counter() - t) * 1e3
t = time.perf_counter(); s_graph = gen(mf); R["fast_generate_ms"] = (time.perf_counter() - t) * 1e3
mf._graph_forward = lambda **kw: mf(**kw)
gen(mf)
t = time.perf_counter(); s_eager = gen(mf); R["fast_eager_generate_ms"] = (time.perf_counter() - t) * 1e3
mf._graph_forward = graph_impl
R["generate_graph_eq_eager"] = s_graph == s_eager == s_graph_first
R["graph_cache_entries"] = len(getattr(mf, "_graph_cache", {}))

print(json.dumps({k: v for k, v in R.items()}, indent=1, default=str))
os.makedirs("/workspace/agha/results", exist_ok=True)
json.dump(R, open(f"/workspace/agha/results/sdk_fast_{datetime.now():%Y%m%d_%H%M%S}.json", "w"), indent=1, default=str)
