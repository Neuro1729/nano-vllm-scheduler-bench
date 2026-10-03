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

## Instrumentation

Minimal hooks in:

- `nanovllm/engine/sequence.py` (per-request timestamps/counters)
- `nanovllm/engine/scheduler.py` (iteration/preempt/KV stats only)
- `nanovllm/engine/scheduler_metrics.py` (counter container)
- `nanovllm/engine/block_manager.py` (`clear_prefix_cache` helper)
- `nanovllm/engine/llm_engine.py` (optional request metadata kwargs)

Scheduling control flow is intentionally unchanged.
