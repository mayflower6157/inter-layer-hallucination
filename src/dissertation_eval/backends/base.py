from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import BackendOutput, DecodingConfig


class GenerationBackend(ABC):
    """Interface shared by baseline and future layer-aware decoders."""

    @abstractmethod
    def generate(self, prompt: str, image_path: str | None, config: DecodingConfig, seed: int) -> BackendOutput:
        raise NotImplementedError

    def peak_memory_bytes(self) -> int | None:
        return None

    def close(self) -> None:
        return None
