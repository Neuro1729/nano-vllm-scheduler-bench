# Scheduler benchmark framework

Branch-neutral tooling to compare Nano-vLLM scheduler behavior under identical workloads.

Use the same commands on `real` (baseline) and `dev` (experiments). Do not change scheduling policy in the shared benchmark commit.

## Layout

```
benchmarks/
  scheduler_bench.py   # timed runner + JSON writer
  workloads.py         # deterministic synthetic traces
  metrics.py           # latency/throughput aggregation
  compare_results.py   # real vs dev table
  README.md
results/               # JSON outputs (local)
```

## Workloads

| Name | Intent |
|------|--------|
| `interactive` | short prompts/outputs, many concurrent requests |
| `rag` | long prompts, short outputs (scaled to `max_model_len`) |
| `long_generation` | moderate prompts, long outputs |
| `mixed` | 40% interactive / 25% rag / 25% long_generation / 10% heavy |

All prompts are synthetic token-id lists (seeded). Generation uses `ignore_eos=True` so output lengths match the request.

Note: Nano-vLLM rejects `temperature=0`. The benchmark defaults to `1e-5`.

## Single run

```bash
python benchmarks/scheduler_bench.py \
  --workload mixed \
  --model /path/to/model \
  --num-requests 256 \
  --seed 42 \
  --output results/mixed.json
```

## Multi-GPU comparison

Terminal A (`real`):

```bash
git checkout real
CUDA_VISIBLE_DEVICES=0 python benchmarks/scheduler_bench.py \
  --workload mixed --model /path/to/model --num-requests 256 --seed 42 \
  --runs 5 --output results/real_mixed.json --experiment-label A_real_gpu0
```

Terminal B (`dev`):

```bash
git checkout dev
CUDA_VISIBLE_DEVICES=1 python benchmarks/scheduler_bench.py \
  --workload mixed --model /path/to/model --num-requests 256 --seed 42 \
  --runs 5 --output results/dev_mixed.json --experiment-label A_dev_gpu1
```

Then crossover (swap GPUs) before trusting deltas:

```bash
# real on GPU1, dev on GPU0, same seed/workload
```

Helper scripts: `benchmarks/run_crossover.sh`, `benchmarks/run_crossover.ps1`.

## Compare

```bash
python benchmarks/compare_results.py results/real_mixed_run0_seed42.json results/dev_mixed_run0_seed42.json
```

## Fairness rules

- Identical `--seed`, `--num-requests`, model path, and engine limits
- Workload is generated before the timed region
- Model load + warmup are outside the timed region
- Prefix cache is cleared between runs; prompts are salted to avoid accidental reuse
- Two GPUs of the same SKU can still differ: always run the crossover

## Branches

| Branch | Meaning |
|--------|---------|
| `real` | Original Nano-vLLM baseline + instrumentation |
| `parity` | Modern vLLM-style scheduling semantics (running-first, shared token budget, mixed batches) |
| `dev` | parity + experimental waiting-admission policies (`--scheduler-policy`) |

On `dev`, default `--scheduler-policy fcfs` matches parity. Also: `sjf`, `sjf_aging`, `mlfq`. See `experimental_policies.md`.

On `parity`, expect `scheduler.mixed_iterations > 0` under `mixed` / concurrent decode+prefill workloads. On `real`, `mixed_iterations` stays 0.

## Instrumentation

Minimal hooks in:

- `nanovllm/engine/sequence.py` (per-request timestamps/counters)
- `nanovllm/engine/scheduler.py` (iteration/preempt/KV stats; parity also counts mixed/prefill/decode tokens)
- `nanovllm/engine/scheduler_metrics.py` (counter container)
- `nanovllm/engine/block_manager.py` (`clear_prefix_cache` helper)
- `nanovllm/engine/llm_engine.py` (optional request metadata kwargs)

Allocation metrics (waiting admit, `can_allocate == -1` only):

| Field | Meaning | Cross-branch |
|-------|---------|--------------|
| `allocation_failure_steps` | Iterations with ≥1 waiting alloc failure | Comparable |
| `allocation_failed_candidates` | Waiting requests that failed alloc | Parity may be higher (HOL skip) |
| `hol_skipped_requests` | Waiting requests skipped to continue scan | ~0 on real |
| `waiting_candidates_examined` | Waiting heads inspected | Policy-dependent |
| `allocation_failures` | Legacy alias of failed-candidate hits | Not comparable raw |

Prefer `allocation_failure_steps`, preemptions, recomputed tokens, and KV util for pressure comparisons.

TTFT / TPOT / E2E definitions are unchanged across branches.
