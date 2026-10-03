from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler_metrics import SchedulerMetrics
from nanovllm.engine.scheduler_output import SchedulerOutput

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # INSTRUMENTATION-ONLY
        self.metrics = SchedulerMetrics(num_kv_blocks=max(config.num_kvcache_blocks, 1))

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _instrument_on_schedule(self, seq: Sequence) -> None:
        # INSTRUMENTATION-ONLY: counters/timestamps; no control-flow impact
        now = perf_counter()
        if not seq._instrument_admitted:
            seq.first_admission_time = now
            seq._instrument_admitted = True
        seq.num_scheduler_steps += 1

    def schedule(self) -> SchedulerOutput:
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0

        # INSTRUMENTATION-ONLY
        self.metrics.observe_queues(
            len(self.waiting),
            len(self.running),
            len(self.block_manager.used_block_ids),
        )

        # Prefill: continue in-progress running chunks, then admit from waiting.
        # Partial prefills live in `running` (parity accounting); exclusive phase
        # still returns before decode (mixed batches arrive in a later milestone).
        running_prefills = deque(seq for seq in self.running if seq.is_prefill_chunk)
        self.running = deque(seq for seq in self.running if not seq.is_prefill_chunk)

        while running_prefills and len(scheduled_seqs) < self.max_num_seqs:
            seq = running_prefills[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            num_tokens = seq.remaining_prefill_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            running_prefills.popleft()
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            self.running.append(seq)
            self._instrument_on_schedule(seq)
            scheduled_seqs.append(seq)

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # INSTRUMENTATION-ONLY
                    self.metrics.allocation_failures += 1
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.remaining_prefill_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            self.waiting.popleft()
            # Admitted prefills (full or partial) become running immediately.
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            self._instrument_on_schedule(seq)
            scheduled_seqs.append(seq)

        # Keep unscheduled in-progress prefills in running.
        self.running.extend(running_prefills)

        if scheduled_seqs:
            # INSTRUMENTATION-ONLY
            self.metrics.scheduler_iterations += 1
            self.metrics.prefill_iterations += 1
            return SchedulerOutput(scheduled_seqs)

        # decode (only sequences that finished prompt prefill)
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if seq.is_prefill_chunk:
                # Should not happen after the prefill pass drained chunks; keep safe.
                self.running.append(seq)
                break
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                self._instrument_on_schedule(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        # INSTRUMENTATION-ONLY
        self.metrics.scheduler_iterations += 1
        self.metrics.decode_iterations += 1
        return SchedulerOutput(scheduled_seqs)

    def preempt(self, seq: Sequence):
        # INSTRUMENTATION-ONLY
        self.metrics.preemption_count += 1
        self.metrics.preempted_request_ids.add(seq.seq_id)
        recomputed = seq.num_cached_tokens
        self.metrics.recomputed_tokens += recomputed
        seq.num_preemptions += 1
        seq.num_recomputed_tokens += recomputed

        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        now = perf_counter()  # INSTRUMENTATION-ONLY timestamp
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.is_prefill_chunk:
                continue
            seq.is_prefill = False
            seq.append_token(token_id)
            # INSTRUMENTATION-ONLY
            if seq.first_token_time == 0.0:
                seq.first_token_time = now
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # INSTRUMENTATION-ONLY
                seq.finish_time = now
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
