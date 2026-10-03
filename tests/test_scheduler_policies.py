"""Deterministic FCFS / SJF / SJF+aging waiting-admission tests (no GPU)."""

from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.scheduler_policy import (
    remaining_prefill_score,
    select_waiting_choice,
    select_waiting_index,
    sjf_total_bound_score,
    waiting_selection_key,
)
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(
    policy: str = "fcfs",
    max_batched_tokens: int = 64,
    max_seqs: int = 8,
    num_blocks: int = 256,
    aging_threshold: int = 128,
) -> Scheduler:
    Sequence.block_size = 256
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = max_seqs
    sched.max_num_batched_tokens = max_batched_tokens
    sched.eos = -1
    sched.block_size = 256
    sched.scheduler_policy = policy
    sched.scheduler_aging_threshold = aging_threshold
    sched.block_manager = BlockManager(num_blocks, 256)
    sched.waiting = deque()
    sched.running = deque()
    sched.metrics = SchedulerMetrics(num_kv_blocks=num_blocks)
    sched.metrics.scheduler_policy = policy
    sched.metrics.scheduler_aging_threshold = aging_threshold
    return sched


def _seq(n_prompt: int, max_tokens: int = 4) -> Sequence:
    return Sequence(
        list(range(n_prompt)),
        SamplingParams(temperature=1e-5, max_tokens=max_tokens, ignore_eos=True),
    )


def _admission_order(policy: str, aging_threshold: int = 128) -> list[int]:
    """Return prompt lengths in the order requests are first admitted."""
    sched = _make_scheduler(
        policy=policy, max_batched_tokens=32, max_seqs=1, aging_threshold=aging_threshold
    )
    a = _seq(4000)
    b = _seq(100)
    c = _seq(500)
    sched.add(a)
    sched.add(b)
    sched.add(c)

    order: list[int] = []
    seen: set[int] = set()
    for _ in range(5000):
        if len(order) == 3:
            break
        out = sched.schedule()
        assert out.num_seqs == 1
        seq = out.seqs[0]
        if seq.seq_id not in seen:
            seen.add(seq.seq_id)
            order.append(seq.num_prompt_tokens)

        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if seq.num_computed_tokens >= seq.num_tokens:
            seq.is_prefill = False
            seq.status = SequenceStatus.FINISHED
            sched.block_manager.deallocate(seq)
            sched.running.clear()
    assert len(order) == 3
    return order


def test_fcfs_admits_in_arrival_order():
    """Control: policy=fcfs preserves parity-style A -> B -> C admission."""
    assert _admission_order("fcfs") == [4000, 100, 500]


def test_sjf_admits_shortest_remaining_prefill_first():
    """SJF over remaining_prefill_tokens: B(100) -> C(500) -> A(4000)."""
    assert _admission_order("sjf") == [100, 500, 4000]


def test_sjf_aging_below_threshold_matches_sjf():
    """Test 1: none overdue => same as SJF (B -> C -> A)."""
    assert _admission_order("sjf_aging", aging_threshold=10_000) == [100, 500, 4000]


def test_sjf_aging_promotes_overdue_long_request():
    """Test 2: A aged past threshold beats shorter B/C."""
    sched = _make_scheduler(policy="sjf_aging", max_batched_tokens=32, max_seqs=1, aging_threshold=8)
    a = _seq(4000)
    b = _seq(100)
    c = _seq(500)
    sched.add(a)
    sched.add(b)
    sched.add(c)
    # After schedule()'s bump, age becomes threshold => overdue.
    a.waiting_age_steps = 8
    b.waiting_age_steps = 1
    c.waiting_age_steps = 2
    out = sched.schedule()
    assert out.seqs[0] is a
    assert sched.metrics.aging_promotions == 1


def test_sjf_aging_multiple_overdue_picks_oldest():
    """Test 3: among overdue, highest age wins; tie-break seq_id."""
    a = _seq(4000)
    b = _seq(100)
    c = _seq(500)
    a.waiting_age_steps = 20
    b.waiting_age_steps = 50
    c.waiting_age_steps = 50
    waiting = deque([a, b, c])
    choice = select_waiting_choice("sjf_aging", waiting, aging_threshold=10)
    # b and c both age 50; lower seq_id wins among them.
    assert choice.aging_promoted is True
    assert waiting[choice.index] is b
    assert b.seq_id < c.seq_id


def test_waiting_age_increments_while_remaining_waiting():
    """Test 4: age +1 per schedule() while still WAITING."""
    sched = _make_scheduler(policy="sjf_aging", max_batched_tokens=32, max_seqs=1, aging_threshold=1000)
    a = _seq(4000)
    b = _seq(100)
    sched.add(a)
    sched.add(b)
    assert a.waiting_age_steps == 0
    out = sched.schedule()
    assert out.seqs[0] is b
    assert a in sched.waiting
    assert a.waiting_age_steps == 1
    # Keep B running without finishing; A should keep aging.
    b.num_cached_tokens += b.num_scheduled_tokens
    b.num_scheduled_tokens = 0
    sched.schedule()
    assert a.waiting_age_steps == 2
    assert sched.metrics.max_waiting_age_steps >= 2


def test_sjf_deterministic_tie_break_by_seq_id():
    sched = _make_scheduler(policy="sjf", max_batched_tokens=16, max_seqs=1)
    a = _seq(200)
    b = _seq(200)
    assert remaining_prefill_score(a) == remaining_prefill_score(b) == 200
    assert a.seq_id < b.seq_id
    sched.add(a)
    sched.add(b)
    out = sched.schedule()
    assert out.seqs[0] is a
    assert waiting_selection_key("sjf", a, 0) < waiting_selection_key("sjf", b, 1)


def test_sjf_selection_index_prefers_shortest():
    a = _seq(4000)
    b = _seq(100)
    c = _seq(500)
    waiting = deque([a, b, c])
    assert select_waiting_index("fcfs", waiting) == 0
    assert select_waiting_index("sjf", waiting) == 1
    waiting = deque([a, c])
    assert select_waiting_index("sjf", waiting) == 1


def test_sjf_total_bound_helper_exists_but_unused_by_default():
    seq = _seq(100, max_tokens=50)
    assert remaining_prefill_score(seq) == 100
    assert sjf_total_bound_score(seq) == 150


def test_fcfs_does_not_count_waiting_reorders():
    sched = _make_scheduler(policy="fcfs", max_batched_tokens=32)
    for n in (4000, 100, 500):
        sched.add(_seq(n))
    sched.schedule()
    assert sched.metrics.waiting_reorders == 0
    assert sched.metrics.to_dict()["scheduler_policy"] == "fcfs"


def test_sjf_counts_waiting_reorders_when_head_is_long():
    sched = _make_scheduler(policy="sjf", max_batched_tokens=32, max_seqs=1)
    sched.add(_seq(4000))
    sched.add(_seq(100))
    out = sched.schedule()
    assert out.seqs[0].num_prompt_tokens == 100
    assert sched.metrics.waiting_reorders >= 1


def test_preempt_restarts_waiting_age():
    """Test 6: recompute preemption starts a new waiting spell (age -> 0)."""
    sched = _make_scheduler(policy="sjf_aging", max_batched_tokens=16, aging_threshold=64)
    seq = _seq(20)
    sched.block_manager.allocate(seq, 0)
    seq.num_cached_tokens = seq.num_prompt_tokens
    seq.status = SequenceStatus.RUNNING
    seq.is_prefill = False
    seq.waiting_age_steps = 99  # stale value must not survive preemption
    sched.running.append(seq)
    sched.running.remove(seq)
    sched.preempt(seq)
    assert seq.status == SequenceStatus.WAITING
    assert seq.waiting_age_steps == 0
    assert seq in sched.waiting
