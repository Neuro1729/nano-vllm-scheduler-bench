# Five-seed SJF-aging experiment (descriptive)

Five matched seeds comparing FCFS, plain SJF, and SJF+aging at thresholds 64 and 256 under the mixed workload.

This file records **provided numerical summaries**. Related prior notes:

- `fcfs_vs_sjf_5seed.md` (FCFS vs plain SJF)
- `sjf_aging_seed42_sweep.md` (single-seed threshold sweep)

These are **descriptive** 5-seed results, not statistical significance claims. No universally optimal threshold is declared.

## Hardware / configuration

```text
GPU: NVIDIA RTX A5000
CUDA_VISIBLE_DEVICES=1
model: Qwen3-0.6B
workload: mixed
requests: 256
max_model_len: 4096
seeds: 42, 43, 44, 45, 46

policies:
  FCFS
  SJF
  SJF+Aging threshold 64
  SJF+Aging threshold 256
```

## Headline 5-seed descriptive means

| Policy | Out tok/s | Wall (s) | TTFT p50 (ms) | TTFT p99 (ms) | Preemptions |
|--------|-----------|----------|---------------|---------------|-------------|
| FCFS | 2439.1 | 58.89 | 3362 | 17344 | 64.2 |
| SJF | 2544.8 | 56.48 | 1029 | 18973 | 18.6 |
| SJF+Aging-64 | 2494.1 | 57.62 | 1023 | 17477 | 26.8 |
| SJF+Aging-256 | 2521.4 | 56.98 | 1048 | 18399 | 19.2 |

## Important comparisons

### SJF+Aging-64 vs FCFS

| Metric | change |
|--------|--------|
| throughput | +2.3% |
| wall time | -2.2% |
| TTFT p50 | -69.6% |
| TTFT p99 | +0.8% |
| preemptions | -58.3% |

### SJF+Aging-64 vs plain SJF

| Metric | change |
|--------|--------|
| throughput | -2.0% |
| wall time | +2.0% |
| TTFT p50 | -0.5% |
| TTFT p99 | -7.9% |
| preemptions | +44.1% |

### SJF+Aging-256 vs FCFS

| Metric | change |
|--------|--------|
| throughput | +3.4% |
| wall time | -3.2% |
| TTFT p50 | -68.8% |
| TTFT p99 | +6.1% |
| preemptions | -70.1% |

### SJF+Aging-256 vs plain SJF

| Metric | change |
|--------|--------|
| throughput | -0.9% |
| wall time | +0.9% |
| TTFT p50 | +1.8% |
| TTFT p99 | -3.0% |
| preemptions | +3.2% |

## Per-seed headlines: Aging-64

See also `sjf_aging_5seed.csv`.

| seed | wall_s | out_tok_s | p50_ms | p99_ms | preemptions |
|------|--------|-----------|--------|--------|-------------|
| 42 | 54.670 | 2525.69 | 996.16 | 16446.66 | 23 |
| 43 | 54.602 | 2531.1 | 1024.09 | 16347.32 | 24 |
| 44 | 58.883 | 2506.1 | 1053.63 | 15762.99 | 31 |
| 45 | 61.293 | 2435.4 | 1028.93 | 19359.09 | 36 |
| 46 | 58.628 | 2472.3 | 1014.38 | 19469.78 | 20 |

## Per-seed headlines: Aging-256

| seed | wall_s | out_tok_s | p50_ms | p99_ms | preemptions |
|------|--------|-----------|--------|--------|-------------|
| 42 | 54.617 | 2528.12 | 1054.05 | 17403.04 | 20 |
| 43 | 54.218 | 2549.0 | 1026.31 | 18582.92 | 17 |
| 44 | 57.645 | 2560.0 | 1051.23 | 17009.18 | 18 |
| 45 | 60.167 | 2481.0 | 1092.33 | 20465.95 | 23 |
| 46 | 58.243 | 2488.7 | 1016.08 | 18534.73 | 18 |

## Interpretation (cautious)

- SJF dramatically improves median TTFT and throughput but worsens p99 TTFT relative to FCFS.
- Aging successfully trades some SJF efficiency for improved tail fairness.
- Threshold 64 nearly restores FCFS-level p99 TTFT while preserving almost all of SJF's median-TTFT gain.
- Threshold 256 remains closer to pure SJF but provides less tail correction.
- Do **not** claim a universally optimal threshold.
- These are descriptive 5-seed results, not statistical significance claims.
