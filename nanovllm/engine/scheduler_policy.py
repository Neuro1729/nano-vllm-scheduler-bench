"""Waiting-admission policy keys for experimental scheduling on `dev`.

Job-size note
-------------
LLM "job size" is ambiguous: future decode length is unknown at arrival.
Policies here must state which size proxy they use.

Current SJF score (known current work only)::

    remaining_prefill_tokens = max(0, num_tokens - num_computed_tokens)
    # while is_prefill / recompute; see Sequence.remaining_compute_tokens

This counts tokens that still need KV written for the request's *currently
known* token_ids (prompt + any already-generated tokens after recompute).
It is NOT total request length and does NOT include future max_tokens.

Optional future bound (not the default policy)::

    sjf_total_bound = remaining_prefill_tokens + remaining_max_tokens

FCFS uses waiting-queue order (parity control). SJF only reorders WAITING
admission; it does not preempt RUNNING work for a shorter arrival (not SRTF).
"""

from __future__ import annotations

from collections import deque
from typing import Literal

from nanovllm.engine.sequence import Sequence

SchedulerPolicyName = Literal["fcfs", "sjf"]
VALID_SCHEDULER_POLICIES: frozenset[str] = frozenset({"fcfs", "sjf"})


def normalize_scheduler_policy(name: str) -> SchedulerPolicyName:
    key = str(name).strip().lower()
    if key not in VALID_SCHEDULER_POLICIES:
        allowed = ", ".join(sorted(VALID_SCHEDULER_POLICIES))
        raise ValueError(f"unknown scheduler_policy={name!r}; expected one of: {allowed}")
    return key  # type: ignore[return-value]


def remaining_prefill_score(seq: Sequence) -> int:
    """SJF size: remaining prefill/recompute tokens (known current work)."""
    return seq.remaining_compute_tokens


def sjf_total_bound_score(seq: Sequence) -> int:
    """Optional future score: known prefill work + remaining max_tokens budget.

    Not used by the default ``sjf`` policy; kept for later experiments.
    """
    remaining_out = max(0, seq.max_tokens - seq.num_completion_tokens)
    return remaining_prefill_score(seq) + remaining_out


def waiting_selection_key(policy: SchedulerPolicyName, seq: Sequence, queue_index: int) -> tuple:
    """Ascending sort key for WAITING admission (lower wins)."""
    if policy == "fcfs":
        return (queue_index, seq.seq_id)
    if policy == "sjf":
        return (remaining_prefill_score(seq), seq.seq_id)
    raise ValueError(f"unsupported scheduler_policy={policy!r}")


def select_waiting_index(policy: SchedulerPolicyName, waiting: deque[Sequence]) -> int:
    """Index of the next WAITING candidate under ``policy``."""
    if not waiting:
        raise IndexError("waiting queue is empty")
    if policy == "fcfs":
        return 0
    # Size-aware policies (SJF): lowest waiting_selection_key wins.
    best_i = 0
    best_key = waiting_selection_key(policy, waiting[0], 0)
    for i in range(1, len(waiting)):
        key = waiting_selection_key(policy, waiting[i], i)
        if key < best_key:
            best_i = i
            best_key = key
    return best_i


def pop_waiting_at(waiting: deque[Sequence], index: int) -> Sequence:
    """Remove and return ``waiting[index]``, preserving relative order of others."""
    if index < 0 or index >= len(waiting):
        raise IndexError(index)
    waiting.rotate(-index)
    seq = waiting.popleft()
    waiting.rotate(index)
    return seq
