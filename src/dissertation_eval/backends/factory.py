from __future__ import annotations

from typing import Any

from .base import GenerationBackend
from .mock import MockBackend


def create_backend(name: str, model_id: str, revision: str, options: dict[str, Any]) -> GenerationBackend:
    if name == "mock":
        return MockBackend(prefix=str(options.get("prefix", "MOCK")))
    if name == "transformers":
        from .transformers import TransformersBackend
        return TransformersBackend(model_id, revision, options)
    raise ValueError(f"Unknown backend {name!r}; expected mock or transformers")
