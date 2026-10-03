"""Deterministic synthetic workloads for scheduler benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    workload_class: str
    prompt_token_ids: list[int]
    max_tokens: int

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "workload_class": self.workload_class,
            "prompt_tokens": self.prompt_tokens,
            "requested_output_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class LengthRange:
    prompt_min: int
    prompt_max: int
    output_min: int
    output_max: int


WORKLOAD_RANGES: dict[str, LengthRange] = {
    "interactive": LengthRange(64, 512, 64, 256),
    "rag": LengthRange(2048, 8192, 32, 256),
    "long_generation": LengthRange(128, 1024, 512, 2048),
    "heavy": LengthRange(2048, 8192, 512, 2048),
}

MIXED_MIX: list[tuple[str, float]] = [
    ("interactive", 0.40),
    ("rag", 0.25),
    ("long_generation", 0.25),
    ("heavy", 0.10),
]


def scale_range(length_range: LengthRange, max_model_len: int) -> LengthRange:
    """Scale prompt/output ranges so prompt_max + output_max fits in context."""
    prompt_min = max(1, length_range.prompt_min)
    prompt_max = max(prompt_min, length_range.prompt_max)
    output_min = max(1, length_range.output_min)
    output_max = max(output_min, length_range.output_max)

    # Leave at least 1 token for output.
    max_prompt = max(1, max_model_len - 1)
    if prompt_max > max_prompt:
        scale = max_prompt / prompt_max
        prompt_min = max(1, int(prompt_min * scale))
        prompt_max = max(prompt_min, int(prompt_max * scale))

    max_output = max(1, max_model_len - prompt_min)
    if output_max > max_output:
        scale = max_output / output_max
        output_min = max(1, int(output_min * scale))
        output_max = max(output_min, int(output_max * scale))

    # Final safety: ensure worst-case pair fits.
    if prompt_max + output_max > max_model_len:
        overflow = prompt_max + output_max - max_model_len
        # Prefer shrinking prompt first for decode-heavy classes, else balanced.
        shrink_prompt = min(overflow, max(0, prompt_max - prompt_min))
        prompt_max -= shrink_prompt
        overflow -= shrink_prompt
        if overflow > 0:
            output_max = max(output_min, output_max - overflow)
        if prompt_max + output_max > max_model_len:
            output_max = max(1, max_model_len - prompt_max)
            output_min = min(output_min, output_max)
            prompt_min = min(prompt_min, prompt_max)

    return LengthRange(prompt_min, prompt_max, output_min, output_max)


def _sample_lengths(rng: random.Random, length_range: LengthRange, max_model_len: int) -> tuple[int, int]:
    scaled = scale_range(length_range, max_model_len)
    prompt_len = rng.randint(scaled.prompt_min, scaled.prompt_max)
    max_out = max(1, max_model_len - prompt_len)
    out_min = min(scaled.output_min, max_out)
    out_max = min(scaled.output_max, max_out)
    output_len = rng.randint(out_min, out_max)
    return prompt_len, output_len


def _make_prompt_tokens(rng: random.Random, prompt_len: int, vocab_size: int, salt: int) -> list[int]:
    """Unique-looking prompts to avoid accidental prefix-cache hits across requests."""
    # Keep token ids in a stable synthetic range independent of model vocab quirks.
    hi = max(2, min(vocab_size - 1, 32000))
    tokens = [rng.randint(1, hi) for _ in range(prompt_len)]
    # Embed salt into the first few tokens so repeated trials do not share prefixes.
    if tokens:
        tokens[0] = 1 + ((tokens[0] + salt * 131) % hi)
    if len(tokens) > 1:
        tokens[1] = 1 + ((tokens[1] + salt * 17) % hi)
    return tokens


def _class_counts(num_requests: int, mix: list[tuple[str, float]]) -> dict[str, int]:
    raw = {name: int(num_requests * weight) for name, weight in mix}
    assigned = sum(raw.values())
    # Distribute remainder to largest mix weights first for determinism.
    order = [name for name, _ in sorted(mix, key=lambda x: (-x[1], x[0]))]
    idx = 0
    while assigned < num_requests:
        raw[order[idx % len(order)]] += 1
        assigned += 1
        idx += 1
    return raw


def generate_workload(
    name: str,
    num_requests: int,
    seed: int,
    max_model_len: int,
    vocab_size: int = 32000,
    run_id: int = 0,
) -> list[RequestSpec]:
    if num_requests < 1:
        raise ValueError("num_requests must be >= 1")
    if name not in {"interactive", "rag", "long_generation", "mixed"}:
        raise ValueError(f"unknown workload: {name}")

    rng = random.Random(seed + run_id * 1_000_003)
    specs: list[RequestSpec] = []

    if name == "mixed":
        counts = _class_counts(num_requests, MIXED_MIX)
        class_list: list[str] = []
        for cls, count in counts.items():
            class_list.extend([cls] * count)
        rng.shuffle(class_list)
        for req_id, cls in enumerate(class_list):
            prompt_len, output_len = _sample_lengths(rng, WORKLOAD_RANGES[cls], max_model_len)
            prompt = _make_prompt_tokens(rng, prompt_len, vocab_size, salt=seed + run_id * 10_000 + req_id)
            specs.append(RequestSpec(req_id, cls, prompt, output_len))
        return specs

    length_range = WORKLOAD_RANGES[name]
    for req_id in range(num_requests):
        prompt_len, output_len = _sample_lengths(rng, length_range, max_model_len)
        prompt = _make_prompt_tokens(rng, prompt_len, vocab_size, salt=seed + run_id * 10_000 + req_id)
        specs.append(RequestSpec(req_id, name, prompt, output_len))
    return specs


def describe_scaled_ranges(max_model_len: int) -> dict[str, dict]:
    return {
        name: asdict(scale_range(length_range, max_model_len))
        for name, length_range in WORKLOAD_RANGES.items()
    }
