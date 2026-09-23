from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .errors import ConfigurationError


class Modality(str, Enum):
    TEXT = "text"
    VISION_LANGUAGE = "vision_language"


class GenerationStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class DecodingStrategy(str, Enum):
    GREEDY = "greedy"
    SAMPLE = "sample"
    BEAM = "beam"


@dataclass(frozen=True)
class FieldMapping:
    id: str
    prompt: str
    reference_answer: str | None = None
    image: str | None = None
    group_id: str | None = None
    metadata: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FieldMapping":
        for required in ("id", "prompt"):
            if not isinstance(value.get(required), str) or not value[required]:
                raise ConfigurationError(f"fields.{required} must be a non-empty string")
        metadata = value.get("metadata", [])
        if not isinstance(metadata, list) or not all(isinstance(item, str) for item in metadata):
            raise ConfigurationError("fields.metadata must be a list of field names")
        return cls(
            id=value["id"],
            prompt=value["prompt"],
            reference_answer=value.get("reference_answer"),
            image=value.get("image"),
            group_id=value.get("group_id"),
            metadata=metadata,
        )


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: str
    format: str = "jsonl"
    sha256: str | None = None
    count: int | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SplitSpec":
        if not value.get("name") or not value.get("path"):
            raise ConfigurationError("Each split requires name and path")
        file_format = value.get("format", "jsonl")
        if file_format not in {"json", "jsonl"}:
            raise ConfigurationError("split.format must be json or jsonl")
        return cls(
            name=str(value["name"]), path=str(value["path"]), format=file_format,
            sha256=value.get("sha256"), count=value.get("count"),
        )


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    dataset_id: str
    display_name: str
    modality: Modality
    source_url: str
    revision: str
    license: str
    fields: FieldMapping
    splits: list[SplitSpec]
    adapter: str | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetManifest":
        required = ("schema_version", "dataset_id", "display_name", "modality", "source_url", "revision", "license")
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ConfigurationError(f"Manifest missing fields: {', '.join(missing)}")
        adapter = value.get("adapter")
        if adapter is not None and (not isinstance(adapter, str) or not adapter):
            raise ConfigurationError("Manifest adapter must be a non-empty string or null")
        splits = [SplitSpec.from_dict(item) for item in value.get("splits", [])]
        if not splits:
            raise ConfigurationError("Manifest must declare at least one split")
        if len({split.name for split in splits}) != len(splits):
            raise ConfigurationError("Split names must be unique")
        return cls(
            schema_version=str(value["schema_version"]), dataset_id=str(value["dataset_id"]),
            display_name=str(value["display_name"]), modality=Modality(value["modality"]),
            source_url=str(value["source_url"]), revision=str(value["revision"]),
            license=str(value["license"]), fields=FieldMapping.from_dict(value.get("fields", {})),
            splits=splits, adapter=adapter, notes=str(value.get("notes", "")),
        )

    def split(self, name: str) -> SplitSpec:
        try:
            return next(item for item in self.splits if item.name == name)
        except StopIteration as exc:
            raise ConfigurationError(f"Unknown split {name!r} for {self.dataset_id}") from exc


@dataclass(frozen=True)
class DatasetExample:
    sample_id: str
    prompt: str
    reference_answer: str | None
    image_path: str | None
    group_id: str | None
    metadata: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class DecodingConfig:
    strategy: DecodingStrategy = DecodingStrategy.GREEDY
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1
    num_generations: int = 1
    seeds: list[int] = field(default_factory=lambda: [0])

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecodingConfig":
        config = cls(
            strategy=DecodingStrategy(value.get("strategy", "greedy")),
            max_new_tokens=int(value.get("max_new_tokens", 512)),
            temperature=float(value.get("temperature", 0.0)), top_p=float(value.get("top_p", 1.0)),
            num_beams=int(value.get("num_beams", 1)), num_generations=int(value.get("num_generations", 1)),
            seeds=[int(seed) for seed in value.get("seeds", [0])],
        )
        if config.max_new_tokens < 1 or config.num_generations < 1 or config.num_beams < 1:
            raise ConfigurationError("Token, generation, and beam counts must be positive")
        if config.strategy == DecodingStrategy.GREEDY and config.num_generations != 1:
            raise ConfigurationError("Greedy decoding must use num_generations=1")
        if config.strategy == DecodingStrategy.BEAM and config.num_beams < 2:
            raise ConfigurationError("Beam decoding requires num_beams>=2")
        if not config.seeds:
            raise ConfigurationError("At least one seed is required")
        return config


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    dataset_manifest: str
    split: str
    backend: str
    model_id: str
    model_revision: str
    prompt_template: str
    output_path: str
    decoding: DecodingConfig
    backend_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunConfig":
        required = ("run_id", "dataset_manifest", "split", "backend", "model_id", "model_revision", "prompt_template", "output_path")
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ConfigurationError(f"Run config missing fields: {', '.join(missing)}")
        return cls(**{key: str(value[key]) for key in required},
                   decoding=DecodingConfig.from_dict(value.get("decoding", {})),
                   backend_options=dict(value.get("backend_options", {})))


@dataclass
class BackendOutput:
    text: str
    finish_reason: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    token_ids: list[int] = field(default_factory=list)
    token_timestamps_ms: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRecord:
    schema_version: str
    run_id: str
    record_id: str
    dataset_id: str
    split: str
    sample_id: str
    group_id: str | None
    generation_index: int
    seed: int
    modality: str
    model_id: str
    model_revision: str
    backend: str
    decoding: dict[str, Any]
    prompt_template: str
    rendered_prompt: str
    image_path: str | None
    reference_answer: str | None
    status: str
    response_text: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    generated_tokens: int | None
    token_ids: list[int]
    token_timestamps_ms: list[float]
    latency_ms: float
    peak_memory_bytes: int | None
    backend_metrics: dict[str, Any]
    sample_metadata: dict[str, Any]
    error: dict[str, str] | None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
