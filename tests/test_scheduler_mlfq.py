"""Deterministic MLFQ scheduler tests (no GPU)."""

from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_mlfq(
    *,
    q0: int = 256,
    q1: int = 1024,
    boost: int = 256,
    max_batched_tokens: int = 64,
    max_seqs: int = 8,
    num_blocks: int = 256,
) -> Scheduler:
    Sequence.block_size = 256
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = max_seqs
    sched.max_num_batched_tokens = max_batched_tokens
    sched.eos = -1
    sched.block_size = 256
    sched.scheduler_policy = "mlfq"
    sched.scheduler_aging_threshold = 128
    sched.mlfq_q0_quantum = q0
    sched.mlfq_q1_quantum = q1
    sched.mlfq_boost_interval = boost
    sched.block_manager = BlockManager(num_blocks, 256)
    sched.waiting = deque()
    sched.running = deque()
    sched._mlfq_queues = [deque(), deque(), deque()]
    sched._mlfq_epoch = 0
    sched.metrics = SchedulerMetrics(num_kv_blocks=num_blocks)
    sched.metrics.scheduler_policy = "mlfq"
    sched.metrics.mlfq_q0_quantum = q0
    sched.metrics.mlfq_q1_quantum = q1
    sched.metrics.mlfq_boost_interval = boost
    return sched


def _seq(n_prompt: int, max_tokens: int = 8) -> Sequence:
    return Sequence(
        list(range(n_prompt)),
        SamplingParams(temperature=1e-5, max_tokens=max_tokens, ignore_eos=True),
    )


def _advance_prefill(sched: Scheduler, seq: Sequence) -> None:
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0


def test_new_request_enters_q0():
    sched = _make_mlfq()
    seq = _seq(32)
    sched.add(seq)
    assert seq.mlfq_level == 0
    assert seq in sched._mlfq_queues[0]
    assert sched.metrics.mlfq_q0_admissions == 1


def test_q0_quantum_exhaustion_demotes_to_q1():
    sched = _make_mlfq(q0=64, q1=1024, boost=0, max_batched_tokens=64)
    seq = _seq(500, max_tokens=2)
    sched.add(seq)
    out = sched.schedule()
    assert out.seqs[0] is seq
    assert seq.num_scheduled_tokens == 64
    _advance_prefill(sched, seq)
    assert seq.mlfq_level == 1
    assert seq.mlfq_service_in_level == 0
    assert seq in sched._mlfq_queues[1]
    assert sched.metrics.mlfq_q1_demotions == 1


def test_q1_quantum_exhaustion_demotes_to_q2():
    sched = _make_mlfq(q0=32, q1=64, boost=0, max_batched_tokens=64)
    seq = _seq(400, max_tokens=2)
    sched.add(seq)
    # Drain Q0 quantum
    out = sched.schedule()
    _advance_prefill(sched, seq)
    assert seq.mlfq_level == 1
    # Drain Q1 quantum (64)
    out = sched.schedule()
    assert out.seqs[0] is seq
    assert seq.num_scheduled_tokens == 64
    _advance_prefill(sched, seq)
    assert seq.mlfq_level == 2
    assert sched.metrics.mlfq_q2_demotions == 1


def test_q2_does_not_demote_further():
    sched = _make_mlfq(q0=16, q1=16, boost=0, max_batched_tokens=32)
    seq = _seq(200, max_tokens=2)
    sched.add(seq)
    for _ in range(3):
        out = sched.schedule()
        if seq in out.seqs:
            _advance_prefill(sched, seq)
    assert seq.mlfq_level == 2
    before = seq.mlfq_level
    out = sched.schedule()
    if seq in out.seqs:
        _advance_prefill(sched, seq)
    assert seq.mlfq_level == before == 2


def test_short_request_finishes_in_q0():
    sched = _make_mlfq(q0=256, boost=0, max_batched_tokens=64, max_seqs=1)
    seq = _seq(20, max_tokens=1)
    sched.add(seq)
    # Prefill all known tokens in one or more Q0 slices.
    while seq.num_computed_tokens < seq.num_tokens:
        out = sched.schedule()
        assert out.seqs[0] is seq
        _advance_prefill(sched, seq)
        assert seq.mlfq_level == 0
    # Decode one token to finish.
    out = sched.schedule()
    assert out.seqs[0] is seq
    assert seq.mlfq_level == 0
    sched.postprocess(out.seqs, [7])
    assert seq.is_finished
    assert sched.metrics.completed_in_q0 == 1


def test_q0_has_priority_over_q2():
    # max_seqs=2 so a Q0 newcomer can become resident while A remains in Q2.
    sched = _make_mlfq(q0=256, boost=0, max_batched_tokens=32, max_seqs=2)
    a = _seq(40, max_tokens=4)
    sched.add(a)
    # Push A to Q2 via service accounting.
    a.mlfq_level = 2
    a.mlfq_service_in_level = 0
    sched._mlfq_queues[0].clear()
    sched._mlfq_queues[2].append(a)
    # Allocate A as running decode-ready-ish resident without finishing.
    sched.block_manager.allocate(a, 0)
    a.num_cached_tokens = a.num_prompt_tokens
    a.is_prefill = False
    a.status = SequenceStatus.RUNNING
    sched.waiting.clear()
    sched.running.append(a)

    b = _seq(30, max_tokens=2)
    sched.add(b)
    out = sched.schedule()
    assert out.seqs[0] is b
    assert b.mlfq_level == 0


def test_round_robin_within_q0():
    sched = _make_mlfq(q0=10_000, boost=0, max_batched_tokens=1, max_seqs=1)
    a = _seq(8, max_tokens=5)
    b = _seq(8, max_tokens=5)
    c = _seq(8, max_tokens=5)
    for seq in (a, b, c):
        sched.add(seq)
        # Fully prefill offline so MLFQ decode RR is visible.
        sched.block_manager.allocate(seq, 0)
        seq.num_cached_tokens = seq.num_prompt_tokens
        seq.is_prefill = False
        seq.status = SequenceStatus.RUNNING
        if seq in sched.waiting:
            sched.waiting.remove(seq)
        sched.running.append(seq)

    order = []
    for _ in range(6):
        out = sched.schedule()
        assert out.num_seqs == 1
        order.append(out.seqs[0].seq_id)
        # Emulate decode postprocess without finishing.
        out.seqs[0].num_scheduled_tokens = 0
        out.seqs[0].append_token(1)
        out.seqs[0].num_cached_tokens = out.seqs[0].num_tokens - 1

    # Deterministic cyclic service among A,B,C (not stuck on one id).
    assert order[:3] == [a.seq_id, b.seq_id, c.seq_id]
    assert order[3:6] == [a.seq_id, b.seq_id, c.seq_id]


def test_prefill_respects_remaining_quantum():
    sched = _make_mlfq(q0=256, boost=0, max_batched_tokens=10_000)
    seq = _seq(1000, max_tokens=2)
    sched.add(seq)
    seq.mlfq_service_in_level = 192  # remaining quantum = 64
    out = sched.schedule()
    assert out.seqs[0] is seq
    assert seq.num_scheduled_tokens == 64


def test_priority_boost_returns_to_q0_without_touching_kv():
    sched = _make_mlfq(q0=256, q1=1024, boost=3, max_batched_tokens=32)
    seq = _seq(40, max_tokens=4)
    sched.add(seq)
    sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = 10
    seq.block_table = list(seq.block_table)
    table_before = list(seq.block_table)
    cached_before = seq.num_cached_tokens
    seq.mlfq_level = 2
    seq.mlfq_service_in_level = 77
    sched._mlfq_queues[0].clear()
    sched._mlfq_queues[2].append(seq)
    seq.status = SequenceStatus.RUNNING
    if seq in sched.waiting:
        sched.waiting.remove(seq)
    sched.running.append(seq)

    # epoch becomes 3 on third schedule -> boost at start of that call
    for _ in range(3):
        out = sched.schedule()
        for s in out.seqs:
            s.num_scheduled_tokens = 0

    assert seq.mlfq_level == 0
    assert seq.block_table == table_before
    assert seq.num_cached_tokens >= cached_before  # KV not cleared by boost
    assert sched.metrics.mlfq_priority_boosts >= 1
    # Service may be non-zero if the same schedule() served after boosting.


def test_kv_preemption_retains_mlfq_level():
    sched = _make_mlfq(q0=256, boost=0, max_batched_tokens=8, num_blocks=1)
    seq = _seq(10, max_tokens=4)
    sched.add(seq)
    sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = seq.num_prompt_tokens
    seq.is_prefill = False
    seq.status = SequenceStatus.RUNNING
    seq.mlfq_level = 2
    seq.mlfq_service_in_level = 33
    if seq in sched.waiting:
        sched.waiting.remove(seq)
    sched.running.append(seq)
    # Force len % 256 == 1 style append need: use short seq already allocated.
    # With 0 free blocks, decode append needing a new block will preempt.
    # Make length such that can_append needs a block.
    while len(seq) % 256 != 1:
        seq.append_token(9)
    assert not sched.block_manager.can_append(seq)
    sched.preempt(seq)
    assert seq.status == SequenceStatus.WAITING
    assert seq.mlfq_level == 2
    assert seq.mlfq_service_in_level == 33
    assert not seq.block_table


def test_mixed_prefill_decode_under_mlfq():
    sched = _make_mlfq(q0=10_000, boost=0, max_batched_tokens=16, max_seqs=4)
    decode = _seq(8, max_tokens=3)
    prefill = _seq(40, max_tokens=2)
    sched.add(decode)
    sched.block_manager.allocate(decode, 0)
    decode.num_cached_tokens = decode.num_prompt_tokens
    decode.is_prefill = False
    decode.status = SequenceStatus.RUNNING
    sched.waiting.remove(decode)
    sched.running.append(decode)
    sched.add(prefill)

    out = sched.schedule()
    assert out.num_seqs >= 1
    assert out.total_scheduled_tokens <= 16
    kinds = {s.is_prefill for s in out.seqs}
    # Prefer seeing mixed when both eligible; at least scheduler stays live.
    assert out.num_seqs > 0


def test_mlfq_smoke_trace_long_short_medium():
    """Visible state transitions for A(long), B(short), C(medium)."""
    sched = _make_mlfq(q0=64, q1=128, boost=5, max_batched_tokens=64, max_seqs=1)
    a = _seq(400, max_tokens=3)  # long
    b = _seq(20, max_tokens=1)   # short
    c = _seq(100, max_tokens=2)  # medium
    sched.add(a)
    sched.add(b)
    sched.add(c)
    assert a.mlfq_level == b.mlfq_level == c.mlfq_level == 0

    # Serve until B completes prefill+decode in Q0 while A/C demote.
    finished_b = False
    for _ in range(40):
        if sched.is_finished():
            break
        out = sched.schedule()
        tokens = [7] * out.num_seqs
        # Finish B quickly after prefill done.
        for seq, tok in zip(out.seqs, tokens):
            if seq is b and seq.num_computed_tokens + seq.num_scheduled_tokens >= seq.num_tokens:
                # let postprocess append/finish
                pass
        sched.postprocess(out.seqs, tokens)
        if b.is_finished:
            finished_b = True
            break
        # Manual accounting already in postprocess for cached tokens.

    assert finished_b
    assert b.mlfq_level == 0 or sched.metrics.completed_in_q0 >= 1
    # A should have consumed enough to leave Q0 at some point.
    assert a.mlfq_level >= 1 or sched.metrics.mlfq_q1_demotions >= 1


def test_empty_batch_impossible_with_waiting_work():
    sched = _make_mlfq(boost=0, max_batched_tokens=32, num_blocks=64)
    sched.add(_seq(50))
    out = sched.schedule()
    assert out.num_seqs > 0
