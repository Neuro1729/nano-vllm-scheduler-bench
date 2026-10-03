# Experimental scheduler policies (`dev`)

`real` and `parity` stay frozen. `dev` = parity + pluggable WAITING admission.

## Config

```text
--scheduler-policy fcfs|sjf   (default: fcfs)
```

`fcfs` must match parity waiting-queue order. `sjf` only reorders WAITING admission.

## SJF score (implemented)

```text
score = remaining_prefill_tokens
      = max(0, num_tokens - num_computed_tokens)   # while prefill/recompute
```

Known current KV work only. Not total request length. Future decode length is unknown.

Optional later (not default): `sjf_total_bound = remaining_prefill + remaining max_tokens`.

Tie-break: lower `seq_id`. No aging in v1 (starvation of long work is intentional to observe).

Non-preemptive: never preempt RUNNING just because a shorter request arrives (not SRTF).

## Round-robin analysis (not implemented)

Parity running-first decode already gives each scheduled decode sequence **one token per iteration** when it fits in the shared token budget and `max_num_seqs`. The running deque is drained front-to-back and survivors are appended again, so over iterations decode residents rotate through the batch much like **token-granularity RR** among RUNNING decodes.

A classic RR “time quantum” does not map cleanly to LLM inference: the natural quantum is already one decode token (or one prefill chunk up to the token budget). Implementing another RR layer would only matter if we introduced multi-token decode quanta, priority against prefill chunks, or per-request token caps inside a step. Until then, a separate RR policy is unlikely to be distinct from parity’s running loop.

## SRTF design (analysis only)

| | SJF (this milestone) | SRTF |
|--|----------------------|------|
| Scope | Choose shortest WAITING work at admit | Newly shorter work may outrank/preempt RUNNING |
| Preemption | KV-pressure recompute only | Also policy-driven preemption |

LLM cost of naive SRTF is high vs CPU context switch:

- Destroying KV forces recompute of prior tokens
- Prefill/recompute burns compute and can thrash under load
- Prefix cache may soften recompute but is not free and is workload-dependent
- Preempting a nearly-done decode can inflate E2E and recomputed_tokens

Any SRTF experiment should track preemptions, recomputed tokens, and tail E2E, and likely use non-destructive pause (keep KV) or high preemption thresholds rather than always deallocating.

## MLFQ design (analysis only)

Map OS multilevel feedback onto token-service, not workload class labels:

```text
Q0: new / little service consumed
Q1: moderate tokens scheduled so far
Q2: long-running (many tokens scheduled)
```

Service metric candidates: `num_scheduler_steps`, scheduled token sum, or decode tokens emitted.

Mechanics to consider later:

- Demote after consuming a token quantum
- Age/promote to prevent starvation in low-priority queues
- Prefill chunks vs decode tokens may need different quanta (prefill is bursty)
- Still separate from KV-pressure preemption

Do not implement MLFQ until SJF baseline results exist.
