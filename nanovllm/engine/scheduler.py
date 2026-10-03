from __future__ import annotations

import os
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

    def _debug_enabled(self) -> bool:
        return os.environ.get("NANOVLLM_SCHED_DEBUG", "").strip() not in {"", "0", "false", "False"}

    def _format_schedule_debug(
        self,
        *,
        token_budget: int,
        scheduled_seqs: list[Sequence],
        preempted_this_step: bool,
        num_preempted: int,
        num_skipped_waiting: int,
    ) -> str:
        return (
            "scheduler empty-batch debug: "
            f"token_budget={token_budget} "
            f"len(running)={len(self.running)} "
            f"len(waiting)={len(self.waiting)} "
            f"free_kv_blocks={len(self.block_manager.free_block_ids)} "
            f"used_kv_blocks={len(self.block_manager.used_block_ids)} "
            f"scheduled_seqs={len(scheduled_seqs)} "
            f"preempted_this_step={preempted_this_step} "
            f"num_preempted={num_preempted} "
            f"num_skipped_waiting={num_skipped_waiting} "
            f"allocation_failures={self.metrics.allocation_failures}"
        )

    def _admit_from_waiting(
        self,
        scheduled_seqs: list[Sequence],
        token_budget: int,
    ) -> tuple[list[Sequence], int, int]:
        """FCFS waiting admission with HOL skip. Returns skipped count."""
        skipped: deque[Sequence] = deque()
        num_skipped = 0
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
                    self.metrics.allocation_failures += 1
                    skipped.append(self.waiting.popleft())
                    num_skipped += 1
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

        for seq in reversed(skipped):
            self.waiting.appendleft(seq)
        return scheduled_seqs, token_budget, num_skipped

    def schedule(self) -> SchedulerOutput:
        scheduled_seqs: list[Sequence] = []
        token_budget = self.max_num_batched_tokens
        preempted_this_step = False
        num_preempted = 0
        num_skipped_waiting = 0

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
            # Preemption frees blocks; the while-loop retries can_append.
            while not self.block_manager.can_append(seq):
                if pending_running:
                    self.preempt(pending_running.pop())
                    preempted_this_step = True
                    num_preempted += 1
                else:
                    # Last resort: preempt self. KV is freed and seq returns to
                    # waiting for recompute; do not count it as scheduled work.
                    self.preempt(seq)
                    preempted_this_step = True
                    num_preempted += 1
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
        #
        # Anti-thrash: after a preemption that already produced model work,
        # defer new waiting admits to the next iteration (vLLM-like).
        #
        # Liveness: if the batch is still empty after preemption, we MUST
        # admit from waiting in this same step so freed KV can be used for
        # recompute. Otherwise the engine would hit an empty-batch dead end.
        # ------------------------------------------------------------------
        # Anti-thrash: after preemption that already scheduled model work, defer
        # new waiting admits (vLLM-like). Liveness: if the batch is still empty,
        # always admit waiting so freed KV can be used for recompute immediately.
        admit_waiting = (not preempted_this_step) or (not scheduled_seqs)
        if admit_waiting:
            scheduled_seqs, token_budget, num_skipped_waiting = self._admit_from_waiting(
                scheduled_seqs, token_budget
            )

        if not scheduled_seqs:
            msg = self._format_schedule_debug(
                token_budget=token_budget,
                scheduled_seqs=scheduled_seqs,
                preempted_this_step=preempted_this_step,
                num_preempted=num_preempted,
                num_skipped_waiting=num_skipped_waiting,
            )
            if self._debug_enabled():
                print(msg)
            # Prefer a loud failure over an engine spin on perpetual empty batches.
            raise RuntimeError(
                "scheduler produced an empty batch with no schedulable tokens; " + msg
            )

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
