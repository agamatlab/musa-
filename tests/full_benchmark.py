#!/usr/bin/env python
"""
Full ESM3 benchmark for ONE configuration (run once per config, separate processes).
Works with both the original fork code (PYTHONPATH=/workspace/agha_baseline) and the current
code (PYTHONPATH=/workspace/agha); optimizations are only used when the code has them.

Use cases
  load        model load time, resident weight memory
  latency     one protein, model forward, L = 64..2048 (eager, or MUSA graph replay)
  sdk_logits  model.logits() (all heads + embeddings), L=256 - the end-user API, eager
  dataset     embed 64 proteins of realistic mixed length (50-1000 aa): proteins/s
              (eager: one by one; graph: length buckets, batched, padded via sequence_id)
  generate    batch_generate, 4 prompts (30% masked), sequence track, 8 steps
  max_len     longest single protein that fits: L = 2048, 4096 (eager)
  accuracy    vs the frozen CPU FP32 references of tests/accuracy_suite.py (9 cases)

Every timed item keeps all reps; "stalls" = items > 1 s and > 3x median (the ~10 s MUSA
submission stall). Waits use a 1-element device->host copy, never torch.musa.synchronize().

  python full_benchmark.py --label cur_bf16_graph --precision bf16 --graph --bf16-linears [--qkln]
"""
import argparse, contextlib, json, math, os, resource, statistics, sys, time, traceback
from datetime import datetime
import torch, torch_musa  # noqa: F401
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="fp32")
ap.add_argument("--graph", action="store_true")
ap.add_argument("--bf16-linears", action="store_true")
ap.add_argument("--qkln", action="store_true")
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", default="/workspace/agha/results/fullbench")
ap.add_argument("--ref", default="/workspace/agha/results/accuracy_ref/cpu_refs_v1.pt")
a = ap.parse_args()

import esm
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinTensor, GenerationConfig, LogitsConfig

DEV, MEM = torch.device("musa:0"), torch.musa
GiB = 2 ** 30
KICK = torch.zeros(1, device=DEV)
def wait():
    _ = KICK.cpu()

# ------------------------------------------------------------------ config
FEATURES = {}
try:
    import esm.layers.attention as A
    if hasattr(A, "USE_TILED_ATTENTION"):
        A.USE_TILED_ATTENTION = False; FEATURES["tiled_off"] = True
    if hasattr(A, "QK_LN_NATIVE_BF16"):
        A.QK_LN_NATIVE_BF16 = bool(a.qkln); FEATURES["qk_ln_native_bf16"] = bool(a.qkln)
    import esm.layers.rotary as RT
    FEATURES["fast_rotary"] = bool(getattr(RT, "FAST_ROTARY", False))
    import esm.models.esm3 as E
    FEATURES["geom_skip"] = bool(getattr(E, "SKIP_GEOM_WITHOUT_COORDS", False))
except Exception:
    pass
if a.qkln and not FEATURES.get("qk_ln_native_bf16"):
    sys.exit("--qkln requested but this code has no QK_LN_NATIVE_BF16")

with contextlib.suppress(Exception):
    torch.backends.musa.matmul.allow_tf32 = (a.precision == "tf32")
torch.set_float32_matmul_precision("high" if a.precision == "tf32" else "highest")
def ctx():
    return torch.autocast("musa", dtype=torch.bfloat16) if a.precision == "bf16" else contextlib.nullcontext()

R = {"label": a.label, "time": datetime.now().isoformat(timespec="seconds"), "args": vars(a),
     "torch": torch.__version__, "torch_musa": torch_musa.__version__,
     "musa": getattr(torch.version, "musa", None), "esm_path": os.path.dirname(esm.__file__),
     "features": FEATURES, "results": {}}
os.makedirs(a.out, exist_ok=True)
PATH = f"{a.out}/{a.label}.json"
def save():
    json.dump(R, open(PATH, "w"), indent=1, default=str)
def log(*x):
    print(f"[{a.label}]", *x, flush=True)

def stats(w):
    m = statistics.median(w)
    clean = [x for x in w if not (x > 1000 and x > 3 * m)] or w
    s = sorted(w)
    return {"reps": w, "median_ms": statistics.median(clean), "p90_ms": s[min(len(s) - 1, int(0.9 * len(s)))],
            "max_ms": max(w), "stalls": len(w) - len(clean)}

def timed(fn, reps=a.reps, warm=2):
    for _ in range(warm):
        fn(); wait()
    w = []
    for _ in range(reps):
        wait(); t0 = time.perf_counter(); fn(); wait(); w.append((time.perf_counter() - t0) * 1e3)
    return stats(w)

def mem_reset():
    wait(); MEM.reset_peak_memory_stats(DEV)
def mem_peak():
    return {"peak_alloc_GiB": MEM.max_memory_allocated(DEV) / GiB, "peak_reserved_GiB": MEM.max_memory_reserved(DEV) / GiB}

def tokens(L, seed=0, B=1):
    g = torch.Generator().manual_seed(1000 * L + seed)
    mid = torch.randint(4, 24, (B, L - 2), generator=g)
    return torch.cat([torch.zeros(B, 1, dtype=torch.long), mid, torch.full((B, 1), 2, dtype=torch.long)], 1).to(DEV)

def is_oom(e):
    return "out of memory" in str(e).lower()

# ------------------------------------------------------------------ load
mem_reset()
t0 = time.perf_counter()
model = ESM3.from_pretrained(device=DEV).eval()
wait()
load_s = time.perf_counter() - t0
if a.bf16_linears:
    from esm.utils.bf16_linears import prepare_bf16_linears
    prepare_bf16_linears(model)
torch.musa.empty_cache()
R["results"]["load"] = {"load_s": load_s, "weights_GiB": MEM.memory_allocated(DEV) / GiB,
                        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()})}
log(f"loaded in {load_s:.1f}s, weights {R['results']['load']['weights_GiB']:.2f} GiB, features {FEATURES}")
save()

def forward(x, **kw):
    with torch.inference_mode(), ctx():
        return model(sequence_tokens=x, **kw)

def capture(static_inputs):
    """MUSA graph of one forward; static_inputs is a dict of kwargs (tensors stay alive)."""
    s = torch.musa.Stream(); s.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(s):
        for _ in range(3):
            forward(**static_inputs)
    torch.musa.current_stream().wait_stream(s); wait()
    g = torch.musa.MUSAGraph()
    with torch.musa.stream(s), torch.inference_mode():
        g.capture_begin()
        try:
            with ctx():
                out = model(sequence_tokens=static_inputs["x"],
                            **{k: v for k, v in static_inputs.items() if k != "x"})
        finally:
            g.capture_end()
    torch.musa.current_stream().wait_stream(s); wait()
    return g, out

def free():
    import gc; gc.collect(); torch.musa.empty_cache()

# ------------------------------------------------------------------ latency
lat = {}
for L in (64, 128, 256, 512, 1024, 2048):
    try:
        x = tokens(L)
        mem_reset()
        if a.graph:
            t = time.perf_counter(); g, _ = capture({"x": x}); cap_s = time.perf_counter() - t
            st = timed(g.replay)
            st["capture_s"] = cap_s
            del g
        else:
            st = timed(lambda: forward(x))
        st.update(mem_peak())
        lat[L] = st
        log(f"latency L={L:<5} {st['median_ms']:8.2f} ms (max {st['max_ms']:.0f}, stalls {st['stalls']}) "
            f"peak {st['peak_alloc_GiB']:.2f} GiB")
    except Exception as e:
        lat[L] = {"error": "OOM" if is_oom(e) else repr(e)[:300]}
        log(f"latency L={L} FAILED {lat[L]['error'][:120]}")
    free()
R["results"]["latency"] = lat; save()

# ------------------------------------------------------------------ SDK logits
try:
    L = 256
    pt = ESMProteinTensor(sequence=tokens(L)[0])
    cfg = LogitsConfig(sequence=True, structure=True, secondary_structure=True, sasa=True,
                       function=True, residue_annotations=True, return_embeddings=True)
    def sdk():
        with ctx():
            return model.logits(pt, cfg)
    mem_reset()
    st = timed(sdk); st.update(mem_peak())
    R["results"]["sdk_logits"] = {"L": L, **st}
    log(f"sdk logits L=256 {st['median_ms']:.2f} ms (stalls {st['stalls']})")
except Exception as e:
    R["results"]["sdk_logits"] = {"error": repr(e)[:300]}
    log(f"sdk logits FAILED {repr(e)[:150]}")
free(); save()

# ------------------------------------------------------------------ dataset embedding throughput
g = torch.Generator().manual_seed(7)
lengths = [int(min(1000, max(50, round(math.exp(math.log(300) + 0.5 * torch.randn(1, generator=g).item())))))
           for _ in range(64)]
seqs = [tokens(n + 2, seed=i)[0] for i, n in enumerate(lengths)]
ds = {"n": len(seqs), "lengths": lengths, "tokens": sum(lengths)}
try:
    mem_reset()
    per = []
    if not a.graph:
        for x in seqs:                                    # warm a couple of shapes
            forward(x[None]); wait(); break
        t_all = time.perf_counter()
        for x in seqs:
            t = time.perf_counter(); forward(x[None]); wait(); per.append((time.perf_counter() - t) * 1e3)
        total = time.perf_counter() - t_all
        ds.update({"mode": "eager one-by-one", "capture_s": 0.0})
    else:
        BUCKETS = [(128, 16), (256, 8), (512, 4), (1024, 2)]      # (padded length, batch size)
        PAD = model.tokenizers.sequence.pad_token_id
        groups = {}
        for i, x in enumerate(seqs):
            Lb, Bb = next(b for b in BUCKETS if b[0] >= x.shape[0])
            groups.setdefault((Lb, Bb), []).append(i)
        graphs, t_cap = {}, time.perf_counter()
        for (Lb, Bb) in groups:
            sx = torch.full((Bb, Lb), PAD, dtype=torch.long, device=DEV)
            sid = torch.ones((Bb, Lb), dtype=torch.long, device=DEV)
            sx[:, 0], sx[:, -1], sid[:, :] = 0, 2, 0
            gr, out = capture({"x": sx, "sequence_id": sid})
            graphs[(Lb, Bb)] = (gr, sx, sid, out)
        cap_s = time.perf_counter() - t_cap
        emb = {}
        t_all = time.perf_counter()
        for (Lb, Bb), idx in groups.items():
            gr, sx, sid, out = graphs[(Lb, Bb)]
            for k in range(0, len(idx), Bb):
                chunk = idx[k:k + Bb]
                sx.fill_(PAD); sid.fill_(1)
                for j, i in enumerate(chunk):
                    n = seqs[i].shape[0]
                    sx[j, :n] = seqs[i]; sid[j, :n] = 0
                t = time.perf_counter(); gr.replay(); wait(); per.append((time.perf_counter() - t) * 1e3)
                for j, i in enumerate(chunk):
                    emb[i] = out.embeddings[j, :seqs[i].shape[0]].float().mean(0).cpu()
        total = time.perf_counter() - t_all
        # padded+batched vs single eager: mean-pooled embedding cosine for 4 proteins
        cos = []
        for i in (0, 1, 2, 3):
            e1 = forward(seqs[i][None]).embeddings[0].float().mean(0).cpu()
            cos.append(F.cosine_similarity(e1, emb[i], dim=0).item())
        ds.update({"mode": "graph, bucketed batches", "capture_s": cap_s, "batch_vs_single_cos": cos,
                   "buckets": {f"{k[0]}x{k[1]}": len(v) for k, v in groups.items()}})
        del graphs
    st = stats(per)
    ds.update({"total_s": total, "proteins_per_s": len(seqs) / total, "tokens_per_s": ds["tokens"] / total,
               "per_call": st, **mem_peak()})
    clean_total = sum(x for x in per if not (x > 1000 and x > 3 * st["median_ms"])) / 1e3
    ds["total_s_without_stalls"] = clean_total
    log(f"dataset 64 proteins: {total:.2f} s ({ds['proteins_per_s']:.1f} prot/s, {ds['tokens_per_s']:.0f} tok/s), "
        f"stalls {st['stalls']}, without stalls {clean_total:.2f} s, peak {ds['peak_alloc_GiB']:.2f} GiB")
except Exception as e:
    ds["error"] = repr(e)[:300] + " | " + traceback.format_exc()[-300:]
    log(f"dataset FAILED {repr(e)[:150]}")
R["results"]["dataset"] = ds; free(); save()

# ------------------------------------------------------------------ generation
try:
    gg = torch.Generator().manual_seed(11)
    base = ["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"]
    prompts = []
    for s in base * 2:
        keep = torch.rand(len(s), generator=gg) > 0.3
        prompts.append("".join(c if k else "_" for c, k in zip(s, keep)))
    os.environ["DISABLE_ITERATIVE_SAMPLING_TQDM"] = "1"
    def gen():
        torch.manual_seed(1234)
        with ctx():
            return model.batch_generate([ESMProtein(sequence=p) for p in prompts],
                                        [GenerationConfig(track="sequence", num_steps=8, temperature=0.7) for _ in prompts])
    gen(); wait()
    mem_reset()
    w = []
    for _ in range(3):
        wait(); t = time.perf_counter(); outs = gen(); wait(); w.append((time.perf_counter() - t) * 1e3)
    R["results"]["generate"] = {"prompts": len(prompts), "steps": 8, **stats(w), **mem_peak(),
                                "ok": all(hasattr(o, "sequence") and o.sequence for o in outs)}
    log(f"generate 4x8 steps {R['results']['generate']['median_ms']:.0f} ms")
except Exception as e:
    R["results"]["generate"] = {"error": repr(e)[:300]}
    log(f"generate FAILED {repr(e)[:150]}")
free(); save()

# ------------------------------------------------------------------ max length
ml = {}
for L in (2048, 4096):
    try:
        x = tokens(L); mem_reset()
        t = time.perf_counter(); forward(x); wait(); first = (time.perf_counter() - t) * 1e3
        t = time.perf_counter(); forward(x); wait(); second = (time.perf_counter() - t) * 1e3
        ml[L] = {"ok": True, "ms": second, "first_ms": first, **mem_peak()}
        log(f"max_len L={L}: ok {second:.0f} ms, peak {ml[L]['peak_alloc_GiB']:.1f} GiB")
    except Exception as e:
        ml[L] = {"ok": False, "error": "OOM" if is_oom(e) else repr(e)[:200]}
        log(f"max_len L={L}: {ml[L]['error'][:80]}")
    free()
R["results"]["max_len"] = ml; save()

# ------------------------------------------------------------------ accuracy vs frozen CPU FP32
def helix_coords(L):
    i = torch.arange(L, dtype=torch.float64)
    th = torch.deg2rad(100.0 * i)
    def at(r, dth, dz):
        return torch.stack([r * torch.cos(th + dth), r * torch.sin(th + dth), 1.5 * i + dz], -1)
    xyz = torch.stack([at(1.6, -0.45, -0.75), at(2.3, 0.0, 0.0), at(1.7, 0.45, 0.8)], 1).float()
    xyz[0] = float("nan"); xyz[-1] = float("nan")
    return xyz.unsqueeze(0)

FIELDS = ("sequence_logits", "structure_logits", "secondary_structure_logits",
          "sasa_logits", "function_logits", "residue_logits", "embeddings")
try:
    refs = torch.load(a.ref, weights_only=False)
    mask_id = model.tokenizers.sequence.mask_token_id
    acc = {}
    for name, ids in refs["inputs"].items():
        ref = refs["fp32"][name]
        kw = {"structure_coords": helix_coords(ids.shape[1]).to(DEV)} if name.startswith("coords") else {}
        o = forward(ids.to(DEV), **kw); wait()
        outs = {f: getattr(o, f).detach().float().cpu() for f in FIELDS}
        B, L = ids.shape
        gm = torch.Generator().manual_seed(L)
        pos = torch.randperm(L - 2, generator=gm)[: max(1, int(0.15 * (L - 2)))] + 1
        m = ids.clone(); m[:, pos] = mask_id
        lg = forward(m.to(DEV), **kw).sequence_logits.detach().float().cpu()[:, pos]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), ids[:, pos].reshape(-1)).item()
        r = ref["outs"]
        acc[name] = {
            "worst_rel": max(((outs[f] - r[f]).norm() / r[f].norm()).item() for f in FIELDS),
            "seq_top1": (outs["sequence_logits"].argmax(-1) == r["sequence_logits"].argmax(-1)).float().mean().item(),
            "struct_top1": (outs["structure_logits"].argmax(-1) == r["structure_logits"].argmax(-1)).float().mean().item(),
            "emb_cos_min": F.cosine_similarity(outs["embeddings"], r["embeddings"], dim=-1).min().item(),
            "d_ce": ce - ref["ce"]}
    R["results"]["accuracy"] = {
        "cases": acc, "worst_rel": max(v["worst_rel"] for v in acc.values()),
        "seq_top1_min": min(v["seq_top1"] for v in acc.values()),
        "struct_top1_min": min(v["struct_top1"] for v in acc.values()),
        "emb_cos_min": min(v["emb_cos_min"] for v in acc.values()),
        "max_abs_d_ce": max(abs(v["d_ce"]) for v in acc.values())}
    A_ = R["results"]["accuracy"]
    log(f"accuracy: worst rel {A_['worst_rel']:.2e}, seq top1 {A_['seq_top1_min']:.3f}, "
        f"emb cos {A_['emb_cos_min']:.5f}, |dCE| {A_['max_abs_d_ce']:.4f}")
except Exception as e:
    R["results"]["accuracy"] = {"error": repr(e)[:300]}
    log(f"accuracy FAILED {repr(e)[:150]}")

R["host_peak_rss_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
save()
log(f"done -> {PATH}")
