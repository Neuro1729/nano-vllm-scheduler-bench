"""Per-request scheduler batch metadata (parity)."""

from __future__ import annotations

from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence


@dataclass
class SchedulerOutput:
    seqs: list[Sequence]

    @property
    def num_seqs(self) -> int:
        return len(self.seqs)

    @property
    def total_scheduled_tokens(self) -> int:
        return sum(seq.num_scheduled_tokens for seq in self.seqs)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(seq.num_scheduled_tokens for seq in self.seqs if seq.is_prefill)

    @property
    def num_decode_tokens(self) -> int:
        return sum(seq.num_scheduled_tokens for seq in self.seqs if not seq.is_prefill)

    @property
    def num_prefill_seqs(self) -> int:
        return sum(1 for seq in self.seqs if seq.is_prefill)

    @property
    def num_decode_seqs(self) -> int:
        return sum(1 for seq in self.seqs if not seq.is_prefill)

    @property
    def is_prefill_only(self) -> bool:
        return self.num_seqs > 0 and self.num_decode_seqs == 0

    @property
    def is_decode_only(self) -> bool:
        return self.num_seqs > 0 and self.num_prefill_seqs == 0

    @property
    def is_mixed(self) -> bool:
        return self.num_prefill_seqs > 0 and self.num_decode_seqs > 0

    @property
    def is_prefill(self) -> bool:
        """Backward-compatible: True when the batch has any prefill work and no decode."""
        return self.is_prefill_only
