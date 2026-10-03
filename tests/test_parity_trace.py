"""Example mixed-iteration scheduler trace for parity validation."""

from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(max_batched_tokens: int = 8) -> Scheduler:
    Sequence.block_size = 256
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = 8
    sched.max_num_batched_tokens = max_batched_tokens
    sched.eos = -1
    sched.block_size = 256
    sched.block_manager = BlockManager(64, 256)
    sched.waiting = deque()
    sched.running = deque()
    sched.scheduler_policy = "fcfs"
    sched.metrics = SchedulerMetrics(num_kv_blocks=64)
    sched.metrics.scheduler_policy = "fcfs"
    return sched


def test_example_mixed_iteration_trace(capsys):
    sched = _make_scheduler(max_batched_tokens=8)

    decode = Sequence([1, 2, 3, 4], SamplingParams(temperature=1e-5, max_tokens=4, ignore_eos=True))
    sched.block_manager.allocate(decode, 0)
    decode.append_token(50)
    decode.num_cached_tokens = len(decode) - 1
    decode.status = SequenceStatus.RUNNING
    decode.is_prefill = False
    sched.running.append(decode)

    prefill = Sequence(list(range(100, 120)), SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    sched.add(prefill)

    out = sched.schedule()
    lines = [
        "PARITY_SCHEDULER_TRACE",
        f"mixed={out.is_mixed} total_tokens={out.total_scheduled_tokens} budget={sched.max_num_batched_tokens}",
    ]
    for seq in out.seqs:
        kind = "prefill" if seq.is_prefill else "decode"
        lines.append(
            f"  seq={seq.seq_id} kind={kind} scheduled={seq.num_scheduled_tokens} "
            f"computed={seq.num_computed_tokens} prompt={seq.num_prompt_tokens}"
        )
    lines.append(
        f"metrics: mixed_iterations={sched.metrics.mixed_iterations} "
        f"prefill_tok={sched.metrics.scheduled_prefill_tokens} "
        f"decode_tok={sched.metrics.scheduled_decode_tokens}"
    )
    print("\n".join(lines))

    assert out.is_mixed
    assert out.total_scheduled_tokens <= 8
    assert sched.metrics.mixed_iterations > 0
    assert sched.metrics.scheduled_decode_tokens == 1
    assert sched.metrics.scheduled_prefill_tokens == 7
