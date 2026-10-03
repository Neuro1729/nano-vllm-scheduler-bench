# Seed-42 SJF-aging threshold sweep (exploratory)

Single-seed exploratory sweep of `sjf_aging` thresholds under the mixed workload.
This file records **provided numerical summaries only**. Raw per-run JSON remains on the GPU machine until copied separately.

This is **not** a multi-seed final result and does **not** claim statistical significance.

## Code snapshot

| Role | Commit | Notes |
|------|--------|-------|
| GPU HEAD when sweep ran | `003f6ef` | includes aging impl + FCFS-vs-SJF docs |
| SJF-aging implementation | `b7dc177` | `experiment: add sjf_aging starvation-prevention admission` |

Related prior experiment (separate): `benchmarks/results/fcfs_vs_sjf_5seed.md` (do not overwrite).

## Hardware / configuration

```text
GPU: NVIDIA RTX A5000
CUDA_VISIBLE_DEVICES=1

Model: /models/Qwen3-0.6B
Workload: mixed
num_requests: 256
seed: 42
runs: 1
max_model_len: 4096

Policies: sjf, sjf_aging
Aging thresholds (scheduler waiting steps): 32, 64, 128, 256
```

### SJF-aging semantics

```text
Normally:
  choose WAITING request with minimum remaining_compute_tokens

If any waiting request has age >= aging_threshold:
  choose the oldest overdue request

Tie-break: sequence id / arrival order

Age: scheduler iterations in the current continuous WAITING spell
Preempted request re-entering WAITING: age restarts at 0
```

## Plain SJF reference (seed 42)

Baseline for relative comparisons in this note:

| Metric | Value |
|--------|-------|
| policy | sjf |
| output_tok_s | 2598.65 |
| wall_s | 53.135 |
| ttft_p50_ms | 1008.47 |
| ttft_p99_ms | 18793.74 |
| preemptions | 17 |
| preempted_requests | 11 |
| recomputed_tokens | 40659 |
| avg_kv_util | 0.6504 |
| peak_kv_util | 1.0 |

## Threshold sweep headline results

See also `sjf_aging_seed42_sweep.csv`.

| threshold | wall_s | output_tok_s | ttft_p50_ms | ttft_p99_ms | aging_promotions | max_waiting_age_steps | mean_age_at_admission | preemptions | recomputed_tokens |
|-----------|--------|--------------|-------------|-------------|------------------|-----------------------|-----------------------|-------------|-------------------|
| 32 | 54.094 | 2552.56 | 1146.16 | 16309.37 | 59 | 302 | 36.6416 | 23 | 57440 |
| 64 | 54.670 | 2525.69 | 996.16 | 16446.66 | 56 | 302 | 36.6416 | 23 | 57440 |
| 128 | 55.315 | 2496.24 | 1059.92 | 17006.50 | 43 | 327 | 36.3484 | 31 | 72088 |
| 256 | 54.617 | 2528.12 | 1054.05 | 17403.04 | 14 | 331 | 35.4094 | 20 | 51163 |

Aging promotions > 0 at every threshold: the starvation-prevention path was active.

## Relative to plain SJF (seed 42)

### Aging threshold 32

| Metric | SJF | t=32 | change |
|--------|-----|------|--------|
| Output tok/s | 2598.65 | 2552.56 | -1.8% |
| TTFT p50 (ms) | 1008.47 | 1146.16 | +13.7% worse |
| TTFT p99 (ms) | 18793.74 | 16309.37 | -13.2% better |
| Mean TTFT (ms) | 2766.24 | 2607.78 | -5.7% |
| TPOT p99 (ms) | 68.667 | 140.77 | +105.0% worse |
| E2E p99 (ms) | 52984.53 | 53804.78 | +1.5% worse |
| Preemptions | 17 | 23 | |
| Recomputed tokens | 40659 | 57440 | |
| Aging promotions | | 59 | |

RAG (important per-class effect):

| Metric | SJF | t=32 | change |
|--------|-----|------|--------|
| RAG TTFT p50 (ms) | 8764.75 | 6269.56 | -28.5% |
| RAG TTFT p99 (ms) | 19728.18 | 16992.80 | -13.9% |
| RAG TPOT p99 (ms) | 78.154 | 161.66 | +106.9% |
| RAG recomputed tokens | 24647 | 47455 | |

### Aging threshold 64

| Metric | SJF | t=64 | change |
|--------|-----|------|--------|
| Output tok/s | 2598.65 | 2525.69 | -2.8% |
| TTFT p50 (ms) | 1008.47 | 996.16 | -1.2% better |
| TTFT p99 (ms) | 18793.74 | 16446.66 | -12.5% better |
| Mean TTFT (ms) | 2766.24 | 2489.00 | -10.0% |
| TPOT p99 (ms) | 68.667 | 143.17 | +108.5% worse |
| E2E p99 (ms) | 52984.53 | 54361.21 | +2.6% worse |
| Preemptions | 17 | 23 | |
| Recomputed tokens | 40659 | 57440 | |
| Aging promotions | | 56 | |

RAG:

| Metric | SJF | t=64 | change |
|--------|-----|------|--------|
| RAG TTFT p50 (ms) | 8764.75 | 6246.40 | -28.7% |
| RAG TTFT p99 (ms) | 19728.18 | 17144.35 | -13.1% |
| RAG TPOT p99 (ms) | 78.154 | 163.84 | +109.6% |
| RAG recomputed tokens | 24647 | 47455 | |

In this seed, t=64 looks promising for TTFT fairness while keeping p50 near SJF, but the TPOT/recomputation cost is large. This is **not** a declared winner.

### Aging threshold 128

| Metric | SJF | t=128 | change |
|--------|-----|-------|--------|
| Output tok/s | 2598.65 | 2496.24 | -3.9% |
| TTFT p50 (ms) | 1008.47 | 1059.92 | +5.1% worse |
| TTFT p99 (ms) | 18793.74 | 17006.50 | -9.5% better |
| TPOT p99 (ms) | 68.667 | 118.24 | +72.2% worse |
| E2E p99 (ms) | 52984.53 | 55162.89 | +4.1% worse |
| Preemptions | 17 | 31 | |
| Recomputed tokens | 40659 | 72088 | |
| Aging promotions | | 43 | |

Among the four settings on this seed, t=128 incurred the highest preemption/recompute overhead while providing less TTFT-tail improvement than t=32/64. Not an attractive operating point from this single seed alone.

### Aging threshold 256

| Metric | SJF | t=256 | change |
|--------|-----|-------|--------|
| Output tok/s | 2598.65 | 2528.12 | -2.7% |
| TTFT p50 (ms) | 1008.47 | 1054.05 | +4.5% worse |
| TTFT p99 (ms) | 18793.74 | 17403.04 | -7.4% better |
| TPOT p99 (ms) | 68.667 | 69.912 | +1.8% worse |
| E2E p99 (ms) | 52984.53 | 54467.15 | +2.8% worse |
| Preemptions | 17 | 20 | |
| Recomputed tokens | 40659 | 51163 | |
| Aging promotions | | 14 | |

RAG:

| Metric | SJF | t=256 | change |
|--------|-----|-------|--------|
| RAG TTFT p99 (ms) | 19728.18 | 17943.01 | -9.0% |
| RAG TPOT p99 (ms) | 78.154 | 81.101 | +3.8% |
| RAG recomputed tokens | 24647 | 35151 | |

More conservative: smaller TTFT-tail improvement than t=64, but much less TPOT disruption. Not a declared winner from one seed.

## Interpretation (cautious)

The seed-42 aging sweep confirms that the starvation-prevention mechanism is active at every tested threshold.

Lower aging thresholds (32/64) substantially reduce the SJF tail-TTFT problem, particularly for RAG requests, but introduce significant decode/TPOT and recomputation costs.

Threshold 256 provides a smaller TTFT-tail improvement with much lower TPOT disruption.

Threshold 128 did not appear attractive in this seed because it incurred the highest preemption/recomputation overhead while providing less tail-TTFT improvement than thresholds 32/64.

These are single-seed exploratory observations only.
Multiple matched seeds are required before selecting a preferred threshold.

Do **not** treat threshold 64 or 256 as definitively best. No single winner is assigned here.

## Follow-up recommendation (multi-seed)

Carry these **two** distinct operating points into a multi-seed study (not a one-seed winner pick):

| Threshold | Rationale |
|-----------|-----------|
| **64** | Strong TTFT-tail/fairness correction while keeping p50 near SJF; high TPOT/recompute cost |
| **256** | Smaller TTFT-tail correction; much lower TPOT disruption |

## Raw JSON location (GPU machine only)

Not included in this repository commit. On the GPU host at the time of the runs:

```text
/results/dev_sjf_seed42.json

/results/dev_sjf_aging_t32_seed42.json
/results/dev_sjf_aging_t64_seed42.json
/results/dev_sjf_aging_t128_seed42.json
/results/dev_sjf_aging_t256_seed42.json
```

Upload/copy separately if bit-exact JSON archival is required.
