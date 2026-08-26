from __future__ import annotations

import hashlib

from ..schema import BackendOutput, DecodingConfig
from .base import GenerationBackend


class MockBackend(GenerationBackend):
    """Deterministic backend for integration tests and pipeline smoke tests."""

    def __init__(self, prefix: str = "MOCK") -> None:
        self.prefix = prefix

    def generate(self, prompt: str, image_path: str | None, config: DecodingConfig, seed: int) -> BackendOutput:
        digest = hashlib.sha256(f"{prompt}|{image_path}|{seed}".encode()).hexdigest()[:12]
        text = f"{self.prefix} reasoning [{digest}]\nFINAL: mock-answer"
        token_ids = list(text.encode("utf-8"))
        return BackendOutput(text=text, finish_reason="stop", prompt_tokens=len(prompt.split()),
                             generated_tokens=len(text.split()), token_ids=token_ids,
                             metrics={"mock": True})
