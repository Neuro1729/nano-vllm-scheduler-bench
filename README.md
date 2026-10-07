# Nano-vLLM Scheduler Bench

Fork of [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) for studying LLM inference **scheduler policies** under identical, reproducible workloads.

Upstream Nano-vLLM is a small (~1.2k LOC) offline inference engine with prefix caching, tensor parallelism, torch compile, and CUDA graphs. This repo keeps that engine and adds:

1. A fair **scheduler benchmark framework**
2. A **parity** scheduler closer to modern vLLM batching
3. **Experimental waiting-admission policies** (SJF, SJF+aging, MLFQ)
4. Recorded GPU experiment results on a mixed workload

Repo: https://github.com/Neuro1729/nano-vllm-scheduler-bench

## Branch model

| Branch | Role |
|--------|------|
| `real` | Original Nano-vLLM + instrumentation (frozen baseline) |
| `parity` | Modern scheduling semantics on top of instrumentation |
| `dev` | `parity` + pluggable waiting-admission policies |

Do not change scheduling policy in shared benchmark commits. Compare branches with the same seed, request count, model, and limits. See [`benchmarks/README.md`](benchmarks/README.md).

## What we improved (and how)

### 1. Fair measurement harness

Added `benchmarks/` so policies can be compared without changing the workload:

- Deterministic synthetic traces: `interactive`, `rag`, `long_generation`, `mixed`
- Timed region excludes model load and warmup
- Prefix cache cleared between runs; prompts salted against accidental reuse
- Metrics: throughput, wall time, TTFT / TPOT / E2E, preemptions, KV util, allocation pressure
- Optional two-GPU crossover helpers to reduce hardware bias

How to run: [`benchmarks/README.md`](benchmarks/README.md)

### 2. Parity scheduler (modern batching baseline)

On `parity` / `dev` (default policy `fcfs`), the scheduler was updated toward modern vLLM-style behavior:

| Change | Why |
|--------|-----|
| Track in-progress computed tokens per request | Correct chunked-prefill / recompute state |
| Mixed prefill + decode batches | Prefer serving running decode while admitting waiting prefill in the same step |
| Shared token budget across the batch | One token budget for mixed work instead of phase-isolated budgets |
| Running-first + HOL skip on waiting | Continue scanning waiting when a head cannot allocate |
| Scheduler instrumentation | `mixed_iterations`, prefill/decode token counts, alloc-failure steps |

On `real`, `mixed_iterations` stays 0. On `parity`/`dev` under concurrent decode+prefill, it is typically > 0.

### 3. Experimental policies (`dev`)

Selected with `--scheduler-policy` (default `fcfs` matches parity):

| Policy | Idea |
|--------|------|
| `fcfs` | Arrival-order waiting admission (parity baseline) |
| `sjf` | Admit shortest **known** remaining prefill/recompute work first |
| `sjf_aging` | SJF + promote overdue waiters after `--aging-threshold` steps |
| `mlfq` | Size-oblivious multilevel feedback (scheduled-token quanta + boost) |

SJF score (not classic total-runtime SJF; future generation length is unknown):

```text
remaining_compute = max(0, num_tokens - num_computed_tokens)
```

Policy details: [`benchmarks/experimental_policies.md`](benchmarks/experimental_policies.md)

## Experiment results so far

Hardware unless noted: **NVIDIA RTX A5000**, model **Qwen3-0.6B**, workload **`mixed`**, **256** requests, `max_model_len=4096`. Numbers below are **descriptive** multi-seed means, not significance tests.

### FCFS vs SJF (5 seeds: 42–46)

Full writeup: [`benchmarks/results/fcfs_vs_sjf_5seed.md`](benchmarks/results/fcfs_vs_sjf_5seed.md)

| Metric | FCFS | SJF | Approx. change |
|--------|------|-----|----------------|
| Output tok/s | 2439 | 2545 | **+4.3%** |
| Wall time | 58.9 s | 56.5 s | **-4.1%** |
| TTFT p50 | 3362 ms | 1029 ms | **-69%** |
| TTFT p99 | 17344 ms | 18973 ms | **+9% worse** |
| Preemptions | 64.2 | 18.6 | **-71%** |

Takeaway: SJF greatly improves median TTFT and cuts preemptions, but worsens tail TTFT (every seed). Per-class data on seed 42 showed interactive requests benefited a lot while RAG TTFT suffered. That motivated aging.

### SJF + aging (5 seeds)

Full writeup: [`benchmarks/results/sjf_aging_5seed.md`](benchmarks/results/sjf_aging_5seed.md)  
Threshold sweep (seed 42 only): [`benchmarks/results/sjf_aging_seed42_sweep.md`](benchmarks/results/sjf_aging_seed42_sweep.md)

| Policy | Out tok/s | TTFT p50 | TTFT p99 | Preemptions |
|--------|-----------|----------|----------|-------------|
| FCFS | 2439 | 3362 ms | 17344 ms | 64.2 |
| SJF | 2545 | 1029 ms | 18973 ms | 18.6 |
| SJF+Aging-64 | 2494 | 1023 ms | 17477 ms | 26.8 |
| SJF+Aging-256 | 2521 | 1048 ms | 18399 ms | 19.2 |

Takeaway: aging trades a little SJF throughput for better tail fairness. Threshold **64** nearly restores FCFS-level p99 while keeping most of SJF’s median-TTFT win. Threshold **256** stays closer to pure SJF. No universally optimal threshold is claimed.

### MLFQ

Implemented on `dev` (`--scheduler-policy mlfq`). Recommended first GPU baseline: Q0=256, Q1=1024, boost=256. **No multi-seed GPU comparison vs FCFS/SJF is recorded in-repo yet.** Do not claim MLFQ is better until those runs exist.

## Quick start (engine)

```bash
# install (this fork)
pip install -e .

# or upstream-style
# pip install git+https://github.com/Neuro1729/nano-vllm-scheduler-bench.git
```

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
print(outputs[0]["text"])
```

Model download example:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Run a scheduler experiment

On `dev`:

```bash
python benchmarks/scheduler_bench.py \
  --workload mixed \
  --model /path/to/Qwen3-0.6B \
  --num-requests 256 \
  --seed 42 \
  --scheduler-policy sjf_aging \
  --aging-threshold 64 \
  --output results/dev_sjf_aging_t64_seed42.json
```

Compare two JSON outputs:

```bash
python benchmarks/compare_results.py results/a.json results/b.json
```

## Upstream Nano-vLLM reference bench

Original offline throughput script: `bench.py` (not the scheduler A/B harness).

| Engine | Output tokens | Time (s) | Throughput (tok/s) |
|--------|---------------|----------|--------------------|
| vLLM | 133,966 | 98.37 | 1361.84 |
| Nano-vLLM | 133,966 | 93.41 | 1434.13 |

Config from upstream docs: RTX 4070 Laptop 8GB, Qwen3-0.6B, 256 sequences, random 100–1024 in/out lengths.

## Docs map

| Doc | Contents |
|-----|----------|
| [`benchmarks/README.md`](benchmarks/README.md) | Harness layout, fairness rules, run commands |
| [`benchmarks/experimental_policies.md`](benchmarks/experimental_policies.md) | FCFS / SJF / aging / MLFQ semantics |
| [`benchmarks/results/`](benchmarks/results/) | Recorded experiment summaries + CSVs |

## License

Same as upstream Nano-vLLM (see `LICENSE`).
