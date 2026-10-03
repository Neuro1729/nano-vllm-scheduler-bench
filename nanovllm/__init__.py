"""Nano-vLLM public API (lazy imports keep lightweight modules testable without GPU stacks)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["LLM", "SamplingParams"]

if TYPE_CHECKING:
    from nanovllm.llm import LLM as LLM
    from nanovllm.sampling_params import SamplingParams as SamplingParams


def __getattr__(name: str) -> Any:
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
