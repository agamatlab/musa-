#!/bin/bash
# Run tests/full_benchmark.py for every configuration, each in its own process, on GPU 3.
# Runs INSIDE the yzw_protein container. Results: /workspace/agha/results/fullbench/<label>.json
#   initial               old env (torch 2.9.0 / MUSA 4.3.5) + original fork code, eager FP32  (where we started)
#   orig_newenv           new env (torch 2.9.1 / MUSA 5.2.0)  + original fork code, eager FP32  (env effect only)
#   cur_fp32_eager        new env + current code, eager FP32 (current default path)
#   cur_tf32_graph        current code, TF32 + MUSA graph
#   cur_bf16_graph        current code, BF16 Linear weights + autocast + MUSA graph
#   cur_bf16_qkln_graph   ... + native BF16 Q/K LayerNorm (opt-in)
cd /workspace/models/esm3-sm-open-v1 || exit 1
export INFRA_PROVIDER=1 MTHREADS_VISIBLE_DEVICES=3 DISABLE_ITERATIVE_SAMPLING_TQDM=1
BENCH=/workspace/agha/tests/full_benchmark.py
NEWPY=/workspace/environments/musa-py312/bin/python
NEWLD=/workspace/mccl-stub:/workspace/musa-5.2.0/lib:$LD_LIBRARY_PATH
run_old() { timeout 3600 env PYTHONPATH=/workspace/agha_baseline python -u $BENCH "$@"; }
run_new() { timeout 3600 env MUSA_HOME=/workspace/musa-5.2.0 LD_LIBRARY_PATH=$NEWLD PYTHONPATH="$1" $NEWPY -u $BENCH "${@:2}"; }

echo "=== start $(date +%T)"
run_old --label initial --precision fp32
run_new /workspace/agha_baseline --label orig_newenv --precision fp32
run_new /workspace/agha --label cur_fp32_eager --precision fp32
run_new /workspace/agha --label cur_tf32_graph --precision tf32 --graph
run_new /workspace/agha --label cur_bf16_graph --precision bf16 --graph --bf16-linears
run_new /workspace/agha --label cur_bf16_qkln_graph --precision bf16 --graph --bf16-linears --qkln
echo "=== done $(date +%T)"
