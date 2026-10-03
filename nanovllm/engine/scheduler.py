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
    """vLLM-V1-style shared token-budget scheduler (parity baseline)."""

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

    def _record_iteration_metrics(self, output: SchedulerOutput) -> None:
        self.metrics.scheduler_iterations += 1
        self.metrics.scheduled_prefill_tokens += output.num_prefill_tokens
        self.metrics.scheduled_decode_tokens += output.num_decode_tokens
        self.metrics.chunked_prefill_count += sum(
            1
            for seq in output.seqs
            if seq.is_prefill
            and seq.num_computed_tokens + seq.num_scheduled_tokens < seq.num_prompt_tokens
        )
        if output.is_mixed:
            self.metrics.mixed_iterations += 1
        elif output.is_prefill_only:
            self.metrics.prefill_iterations += 1
        elif output.is_decode_only:
            self.metrics.decode_iterations += 1

    def schedule(self) -> SchedulerOutput:
        scheduled_seqs: list[Sequence] = []
        token_budget = self.max_num_batched_tokens
        preempted_this_step = False

        # INSTRUMENTATION-ONLY
        self.metrics.observe_queues(
            len(self.waiting),
            len(self.running),
            len(self.block_manager.used_block_ids),
        )

        # ------------------------------------------------------------------
        # 1) Schedule RUNNING first (decode + in-progress prefill chunks).
        # ------------------------------------------------------------------
        pending_running = self.running
        self.running = deque()

        while pending_running and len(scheduled_seqs) < self.max_num_seqs and token_budget > 0:
            seq = pending_running.popleft()

            if seq.is_prefill_chunk:
                take = min(seq.remaining_prefill_tokens, token_budget)
                if take <= 0:
                    self.running.append(seq)
                    continue
                seq.num_scheduled_tokens = take
                seq.is_prefill = True
                token_budget -= take
                scheduled_seqs.append(seq)
                self.running.append(seq)
                self._instrument_on_schedule(seq)
                continue

            # Decode: need one token of budget and possibly a new KV block.
            while not self.block_manager.can_append(seq):
                if pending_running:
                    self.preempt(pending_running.pop())
                    preempted_this_step = True
                else:
                    self.preempt(seq)
                    preempted_this_step = True
                    seq = None
                    break
            if seq is None:
                continue
            if token_budget < 1:
                self.running.append(seq)
                break
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            token_budget -= 1
            scheduled_seqs.append(seq)
            self.running.append(seq)
            self._instrument_on_schedule(seq)

        # Preserve unscheduled running requests (FCFS tail).
        self.running.extend(pending_running)

        # ------------------------------------------------------------------
        # 2) Fill remaining budget from WAITING (HOL skip on alloc failure).
        #    Skip waiting admission if we preempted this step (avoid thrash).
        # ------------------------------------------------------------------
        if not preempted_this_step:
            skipped: deque[Sequence] = deque()
            while (
                self.waiting
                and token_budget > 0
                and len(scheduled_seqs) < self.max_num_seqs
                and len(self.running) < self.max_num_seqs
            ):
                seq = self.waiting[0]
                if not seq.block_table:
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                    if num_cached_blocks == -1:
                        # HOL skip: try later waiting requests.
                        self.metrics.allocation_failures += 1
                        skipped.append(self.waiting.popleft())
                        continue
                    self.block_manager.allocate(seq, num_cached_blocks)

                need = seq.remaining_prefill_tokens
                take = min(need, token_budget)
                if take <= 0:
                    break

                seq.num_scheduled_tokens = take
                seq.is_prefill = True
                token_budget -= take
                self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_seqs.append(seq)
                self._instrument_on_schedule(seq)

            # Re-queue skipped requests at the front, preserving their relative order.
            for seq in reversed(skipped):
                self.waiting.appendleft(seq)

        assert scheduled_seqs, "scheduler produced an empty batch"
        assert sum(s.num_scheduled_tokens for s in scheduled_seqs) <= self.max_num_batched_tokens

        output = SchedulerOutput(scheduled_seqs)
        self._record_iteration_metrics(output)
        return output

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
