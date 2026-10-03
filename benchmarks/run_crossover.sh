#!/usr/bin/env bash
# Crossover helper: distinguish scheduler effects from GPU variation.
# Usage:
#   MODEL=/path/to/model BRANCH_A=real BRANCH_B=dev ./benchmarks/run_crossover.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL=/path/to/model}"
WORKLOAD="${WORKLOAD:-mixed}"
NUM_REQUESTS="${NUM_REQUESTS:-256}"
SEED="${SEED:-42}"
RUNS="${RUNS:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run_one() {
  local branch="$1"
  local gpu="$2"
  local label="$3"
  local out="$4"
  git checkout "$branch"
  CUDA_VISIBLE_DEVICES="$gpu" python benchmarks/scheduler_bench.py \
    --workload "$WORKLOAD" \
    --model "$MODEL" \
    --num-requests "$NUM_REQUESTS" \
    --seed "$SEED" \
    --runs "$RUNS" \
    --output "$out" \
    --experiment-label "$label" \
    $EXTRA_ARGS
}

mkdir -p results

echo "=== Experiment A: real@GPU0, then switch and run dev@GPU1 manually in parallel if desired ==="
echo "This script runs sequentially. For true parallel runs, use two terminals."

run_one real 0 A_real_gpu0 "results/A_real_mixed.json"
run_one dev 1 A_dev_gpu1 "results/A_dev_mixed.json"

echo "=== Experiment B: crossover ==="
run_one real 1 B_real_gpu1 "results/B_real_mixed.json"
run_one dev 0 B_dev_gpu0 "results/B_dev_mixed.json"

echo "Compare matched seeds, e.g.:"
echo "  python benchmarks/compare_results.py results/A_real_mixed_run0_seed${SEED}.json results/A_dev_mixed_run0_seed${SEED}.json"
echo "  python benchmarks/compare_results.py results/B_real_mixed_run0_seed${SEED}.json results/B_dev_mixed_run0_seed${SEED}.json"
