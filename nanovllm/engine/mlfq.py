"""Multi-Level Feedback Queue helpers for experimental `mlfq` scheduling on `dev`.

Service is measured in scheduled tokens (prefill + decode). Demotion changes
priority only; it does not free KV or force recompute.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence as TypingSequence

from nanovllm.engine.sequence import Sequence


def mlfq_quantum(level: int, q0: int, q1: int) -> int | None:
    """Return quantum for level, or None for bottom queue (no demotion)."""
    if level == 0:
        return q0
    if level == 1:
        return q1
    return None


def remaining_quantum(seq: Sequence, q0: int, q1: int) -> int:
    """Tokens still allowed at the current level before demotion."""
    quantum = mlfq_quantum(seq.mlfq_level, q0, q1)
    if quantum is None:
        return 2**31 - 1
    return max(0, quantum - seq.mlfq_service_in_level)


def detach_from_queues(queues: list[deque[Sequence]], seq: Sequence) -> None:
    for q in queues:
        try:
            q.remove(seq)
        except ValueError:
            continue


def place_on_level(queues: list[deque[Sequence]], seq: Sequence) -> None:
    detach_from_queues(queues, seq)
    level = max(0, min(2, seq.mlfq_level))
    seq.mlfq_level = level
    queues[level].append(seq)


def demote(seq: Sequence, queues: list[deque[Sequence]], q0: int, q1: int) -> bool:
    """Demote one level if a finite quantum was exhausted. Returns True if demoted."""
    quantum = mlfq_quantum(seq.mlfq_level, q0, q1)
    if quantum is None:
        return False
    if seq.mlfq_service_in_level < quantum:
        return False
    old = seq.mlfq_level
    seq.mlfq_level = min(2, old + 1)
    seq.mlfq_service_in_level = 0
    place_on_level(queues, seq)
    return seq.mlfq_level != old


def charge_service(
    seq: Sequence,
    tokens: int,
    queues: list[deque[Sequence]],
    q0: int,
    q1: int,
) -> bool:
    """Add scheduled tokens and demote if the level quantum is exhausted."""
    if tokens <= 0:
        return False
    seq.mlfq_service_in_level += tokens
    return demote(seq, queues, q0, q1)


def boost_all(
    queues: list[deque[Sequence]],
    active: TypingSequence[Sequence],
) -> int:
    """Promote all unfinished active requests to Q0. Returns number boosted."""
    unfinished = [s for s in active if not s.is_finished]
    for q in queues:
        q.clear()
    unfinished.sort(key=lambda s: s.seq_id)
    boosted = 0
    for seq in unfinished:
        if seq.mlfq_level != 0 or seq.mlfq_service_in_level != 0:
            boosted += 1
        seq.mlfq_level = 0
        seq.mlfq_service_in_level = 0
        queues[0].append(seq)
    return boosted
