#!/usr/bin/env python3
"""
Charts + summary table for tests/full_benchmark.py results (run locally after pulling
~/agha/results/fullbench/*.json).   python3 tests/plot_full_benchmark.py [results_dir]
Writes fullbench_dashboard.png and fullbench_summary.md into the results dir.
"""
import json, os, sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/agha/results/fullbench")
ORDER = ["initial", "orig_newenv", "cur_fp32_eager", "cur_tf32_graph", "cur_bf16_graph", "cur_bf16_qkln_graph"]
NAMES = {"initial": "initial (old env, orig code, FP32 eager)",
         "orig_newenv": "orig code, new env, FP32 eager",
         "cur_fp32_eager": "current, FP32 eager",
         "cur_tf32_graph": "current, TF32 + graph",
         "cur_bf16_graph": "current, BF16 + graph",
         "cur_bf16_qkln_graph": "current, BF16 + graph + QK-LN"}
COL = dict(zip(ORDER, ["#7f7f7f", "#bcbd22", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]))
# distinct styles so configs with identical values (e.g. the two BF16 runs) stay visible
STY = dict(zip(ORDER, [dict(ls="-", marker="o", lw=3.5, ms=9), dict(ls="--", marker="s", lw=1.8, ms=5),
                       dict(ls="-", marker="o", lw=3.5, ms=9), dict(ls="--", marker="^", lw=1.8, ms=6),
                       dict(ls="-", marker="o", lw=3.5, ms=9), dict(ls=":", marker="D", lw=2.2, ms=5)]))
R = {}
for f in glob.glob(f"{D}/*.json"):
    r = json.load(open(f)); R[r["label"]] = r
cfgs = [c for c in ORDER if c in R]
res = lambda c: R[c]["results"]
Ls = [64, 128, 256, 512, 1024, 2048]

def lat(c, L):
    v = res(c).get("latency", {}).get(str(L), {})
    return v.get("median_ms")

fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
fig.suptitle("ESM3 on one MTT S5000 — initial vs current", fontsize=15, weight="bold")

a = ax[0, 0]                                                 # latency vs L
for c in cfgs:
    xs = [L for L in Ls if lat(c, L)]; ys = [lat(c, L) for L in xs]
    a.plot(xs, ys, color=COL[c], label=NAMES[c], **STY[c])
    for L in Ls:
        if res(c).get("latency", {}).get(str(L), {}).get("error"):
            a.plot([L], [a.get_ylim()[1] if a.lines else 1], "x", color=COL[c], ms=10)
a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xlabel("sequence length L"); a.set_ylabel("ms / forward (median)")
a.set_title("Single-protein forward latency (x = OOM)"); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)

a = ax[0, 1]                                                 # speedup vs initial
base = "initial" if "initial" in R else cfgs[0]
w = 0.8 / max(1, len(cfgs) - 1); k = 0
for c in cfgs:
    if c == base:
        continue
    sp = [(lat(base, L) / lat(c, L)) if (lat(base, L) and lat(c, L)) else 0 for L in Ls]
    a.bar([i + k * w for i in range(len(Ls))], sp, w, color=COL[c], label=NAMES[c]); k += 1
a.axhline(1, color="k", lw=.8); a.set_xticks([i + 0.4 - w / 2 for i in range(len(Ls))]); a.set_xticklabels(Ls)
a.set_xlabel("L"); a.set_ylabel(f"speed-up vs {base}"); a.set_title("Forward speed-up vs initial (0 = initial OOM)")
a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)

a = ax[0, 2]                                                 # dataset throughput
vals = [res(c).get("dataset", {}).get("proteins_per_s", 0) for c in cfgs]
clean = [(res(c)["dataset"]["n"] / res(c)["dataset"]["total_s_without_stalls"])
         if res(c).get("dataset", {}).get("total_s_without_stalls") else 0 for c in cfgs]
a.bar(range(len(cfgs)), vals, color=[COL[c] for c in cfgs], label="as measured")
if any(res(c).get("dataset", {}).get("per_call", {}).get("stalls", 0) for c in cfgs):
    a.scatter(range(len(cfgs)), clean, color="k", marker="_", s=400, zorder=3, label="excluding stalled calls")
for i, v in enumerate(vals):
    a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
a.set_xticks(range(len(cfgs))); a.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=8)
a.set_ylabel("proteins / s"); a.set_title("Embedding 64 proteins (50–1000 aa)"); a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)

a = ax[1, 0]                                                 # memory
for c in cfgs:
    xs = [L for L in Ls if res(c).get("latency", {}).get(str(L), {}).get("peak_alloc_GiB")]
    a.plot(xs, [res(c)["latency"][str(L)]["peak_alloc_GiB"] for L in xs], color=COL[c], label=NAMES[c], **STY[c])
    for L, v in res(c).get("max_len", {}).items():
        if v.get("ok"):
            a.plot([int(L)], [v["peak_alloc_GiB"]], "s", color=COL[c], ms=7, mfc="none")
a.set_xscale("log", base=2); a.set_xlabel("L"); a.set_ylabel("peak GPU memory (GiB)")
a.set_title("Peak GPU memory (□ = max-length run; overlaps = same value)"); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)

a = ax[1, 1]                                                 # SDK + generation
sdk = [res(c).get("sdk_logits", {}).get("median_ms", 0) for c in cfgs]
gen = [res(c).get("generate", {}).get("median_ms", 0) / 10 for c in cfgs]
x = range(len(cfgs))
a.bar([i - .2 for i in x], sdk, .4, color=[COL[c] for c in cfgs], label="SDK model.logits() L=256 (ms)")
a.bar([i + .2 for i in x], gen, .4, color=[COL[c] for c in cfgs], alpha=.45, hatch="//",
      label="batch_generate 4×8 steps (ms ÷ 10)")
a.set_xticks(list(x)); a.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=8)
a.set_title("End-user API calls (eager in every config)"); a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)

a = ax[1, 2]                                                 # accuracy
acc = [res(c).get("accuracy", {}) for c in cfgs]
a.bar(range(len(cfgs)), [v.get("worst_rel", 0) for v in acc], color=[COL[c] for c in cfgs])
a.set_yscale("log"); a.set_ylabel("worst relative error vs CPU FP32 (9 cases)")
for i, v in enumerate(acc):
    if v.get("worst_rel"):
        a.text(i, v["worst_rel"], f"seq {v['seq_top1_min'] * 100:.1f}%\ncos {v['emb_cos_min']:.4f}",
               ha="center", va="bottom", fontsize=7)
a.set_xticks(range(len(cfgs))); a.set_xticklabels([c.replace("_", "\n") for c in cfgs], fontsize=8)
a.set_title("Accuracy vs frozen CPU FP32 reference"); a.grid(alpha=.3, axis="y")
plt.tight_layout(rect=[0, 0, 1, .96])
png = f"{D}/fullbench_dashboard.png"; plt.savefig(png, dpi=130)

# ---------------- summary table
def f(v, fmt):
    return "—" if v is None else fmt.format(v)
lines = ["| metric | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
def row(name, fn):
    lines.append(f"| {name} | " + " | ".join(fn(c) for c in cfgs) + " |")
row("load (s) / weights (GiB)", lambda c: f"{res(c)['load']['load_s']:.0f} / {res(c)['load']['weights_GiB']:.1f}")
for L in Ls:
    row(f"forward L={L} (ms)", lambda c, L=L: (f(lat(c, L), "{:.1f}") if lat(c, L) else res(c)["latency"].get(str(L), {}).get("error", "—")[:12]))
row("forward stalls (all L)", lambda c: str(sum(v.get("stalls", 0) for v in res(c)["latency"].values())))
row("peak GPU mem L=1024 (GiB)", lambda c: f(res(c)["latency"].get("1024", {}).get("peak_alloc_GiB"), "{:.1f}"))
row("SDK logits L=256 (ms)", lambda c: f(res(c).get("sdk_logits", {}).get("median_ms"), "{:.1f}"))
row("dataset 64 prot: total s (prot/s)", lambda c: f"{res(c)['dataset'].get('total_s', 0):.1f} ({res(c)['dataset'].get('proteins_per_s', 0):.1f})" if "total_s" in res(c).get("dataset", {}) else "fail")
row("dataset: stalls / total w/o stalls (s)", lambda c: f"{res(c)['dataset'].get('per_call', {}).get('stalls', '—')} / {res(c)['dataset'].get('total_s_without_stalls', 0):.1f}" if "total_s" in res(c).get("dataset", {}) else "fail")
row("generate 4×8 steps (ms)", lambda c: f(res(c).get("generate", {}).get("median_ms"), "{:.0f}"))
row("max L 2048 / 4096", lambda c: " / ".join(("ok %.1fG" % v["peak_alloc_GiB"]) if v.get("ok") else v.get("error", "?")[:5]
                                            for v in (res(c)["max_len"].get("2048", {}), res(c)["max_len"].get("4096", {}))))
row("accuracy worst rel err", lambda c: f(res(c).get("accuracy", {}).get("worst_rel"), "{:.1e}"))
row("seq top-1 min / emb cos min", lambda c: f"{res(c)['accuracy']['seq_top1_min'] * 100:.1f}% / {res(c)['accuracy']['emb_cos_min']:.5f}" if "worst_rel" in res(c).get("accuracy", {}) else "—")
row("masked-LM max abs ΔCE", lambda c: f(res(c).get("accuracy", {}).get("max_abs_d_ce"), "{:.4f}"))
row("host peak RSS (GiB)", lambda c: f(R[c].get("host_peak_rss_GiB"), "{:.1f}"))
row("torch / musa", lambda c: f"{R[c]['torch']} / {R[c]['musa']}")
md = f"{D}/fullbench_summary.md"
open(md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\nwrote {png}\nwrote {md}")
