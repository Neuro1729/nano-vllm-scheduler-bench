"""SchedulerOutput metadata tests."""

from nanovllm.engine.scheduler_output import SchedulerOutput
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _seq(prefill: bool, scheduled: int) -> Sequence:
    Sequence.block_size = 256
    seq = Sequence([1, 2, 3], SamplingParams(temperature=1e-5, max_tokens=2, ignore_eos=True))
    seq.is_prefill = prefill
    seq.num_scheduled_tokens = scheduled
    return seq


def test_scheduler_output_flags():
    prefill = _seq(True, 8)
    decode = _seq(False, 1)
    only_p = SchedulerOutput([prefill])
    assert only_p.is_prefill_only and not only_p.is_mixed
    assert only_p.num_prefill_tokens == 8
    only_d = SchedulerOutput([decode])
    assert only_d.is_decode_only and only_d.num_decode_tokens == 1
    mixed = SchedulerOutput([prefill, decode])
    assert mixed.is_mixed
    assert mixed.total_scheduled_tokens == 9
