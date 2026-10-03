import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Experimental waiting-admission policy (`dev`). Default `fcfs` matches parity.
    # Validated in Scheduler via scheduler_policy.normalize_scheduler_policy.
    scheduler_policy: str = "fcfs"
    # For scheduler_policy=sjf_aging: promote overdue WAITING requests after this
    # many continuous waiting scheduler iterations. Ignored by fcfs/sjf.
    scheduler_aging_threshold: int = 128
    # MLFQ parameters (scheduler_policy=mlfq). Quanta are scheduled tokens.
    mlfq_q0_quantum: int = 256
    mlfq_q1_quantum: int = 1024
    # Periodic boost to Q0 every N scheduler iterations; 0 disables boost.
    mlfq_boost_interval: int = 256

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.scheduler_policy = str(self.scheduler_policy).strip().lower()
        self.scheduler_aging_threshold = int(self.scheduler_aging_threshold)
        assert self.scheduler_aging_threshold >= 1
        self.mlfq_q0_quantum = int(self.mlfq_q0_quantum)
        self.mlfq_q1_quantum = int(self.mlfq_q1_quantum)
        self.mlfq_boost_interval = int(self.mlfq_boost_interval)
        assert self.mlfq_q0_quantum >= 1
        assert self.mlfq_q1_quantum >= 1
        assert self.mlfq_boost_interval >= 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
