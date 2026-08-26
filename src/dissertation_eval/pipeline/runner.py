from __future__ import annotations

import hashlib
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..backends import create_backend
from ..datasets import load_split
from ..io import append_jsonl, atomic_write_json, load_processed_keys, read_json, sha256_file
from ..schema import GenerationRecord, GenerationStatus, RunConfig


def load_run_config(path: Path) -> RunConfig:
    return RunConfig.from_dict(read_json(path))


def _record_id(run_id: str, sample_id: str, generation_index: int) -> str:
    raw = f"{run_id}\0{sample_id}\0{generation_index}".encode()
    return hashlib.sha256(raw).hexdigest()


def _render_prompt(template: str, prompt: str) -> str:
    try:
        return template.format(prompt=prompt)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("prompt_template must contain a valid {prompt} placeholder") from exc


def _run_manifest(config_path: Path, config: RunConfig, dataset_id: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": config.run_id,
        "dataset_id": dataset_id,
        "run_config_path": str(config_path.resolve()),
        "run_config_sha256": sha256_file(config_path.resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "config": {**asdict(config), "decoding": asdict(config.decoding)},
    }


def run_generation(config_path: Path, *, limit: int | None = None, fail_fast: bool = False) -> dict[str, int]:
    config_path = config_path.resolve()
    config = load_run_config(config_path)
    manifest_path = (config_path.parent / config.dataset_manifest).resolve()
    output_path = (config_path.parent / config.output_path).resolve()
    manifest, examples = load_split(manifest_path, config.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_path.with_suffix(output_path.suffix + ".run.json")
    current_manifest = _run_manifest(config_path, config, manifest.dataset_id)
    if run_manifest_path.exists():
        previous = read_json(run_manifest_path)
        if previous.get("run_config_sha256") != current_manifest["run_config_sha256"]:
            raise RuntimeError("Refusing to resume: run config changed for the existing output")
    else:
        atomic_write_json(run_manifest_path, current_manifest)

    processed = load_processed_keys(output_path)
    backend = create_backend(config.backend, config.model_id, config.model_revision, config.backend_options)
    counters = {"success": 0, "error": 0, "skipped": 0}
    visited = 0
    try:
        for example in examples:
            if limit is not None and visited >= limit:
                break
            visited += 1
            for generation_index in range(config.decoding.num_generations):
                key = (example.sample_id, generation_index)
                if key in processed:
                    counters["skipped"] += 1
                    continue
                seed = config.decoding.seeds[generation_index % len(config.decoding.seeds)]
                rendered = _render_prompt(config.prompt_template, example.prompt)
                started = time.perf_counter()
                output = None
                error = None
                status = GenerationStatus.SUCCESS
                try:
                    output = backend.generate(rendered, example.image_path, config.decoding, seed)
                except Exception as exc:  # each failed sample remains auditable and resumable
                    status = GenerationStatus.ERROR
                    error = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": "".join(traceback.format_exception(exc))[-8000:]}
                    if fail_fast:
                        raise
                latency_ms = (time.perf_counter() - started) * 1000
                record = GenerationRecord(
                    schema_version="1.0", run_id=config.run_id,
                    record_id=_record_id(config.run_id, example.sample_id, generation_index),
                    dataset_id=manifest.dataset_id, split=config.split, sample_id=example.sample_id,
                    group_id=example.group_id, generation_index=generation_index, seed=seed,
                    modality=manifest.modality.value, model_id=config.model_id,
                    model_revision=config.model_revision, backend=config.backend,
                    decoding=asdict(config.decoding), prompt_template=config.prompt_template,
                    rendered_prompt=rendered, image_path=example.image_path,
                    reference_answer=example.reference_answer, status=status.value,
                    response_text=output.text if output else None,
                    finish_reason=output.finish_reason if output else None,
                    prompt_tokens=output.prompt_tokens if output else None,
                    generated_tokens=output.generated_tokens if output else None,
                    token_ids=output.token_ids if output else [],
                    token_timestamps_ms=output.token_timestamps_ms if output else [],
                    latency_ms=latency_ms, peak_memory_bytes=backend.peak_memory_bytes(),
                    backend_metrics=output.metrics if output else {}, sample_metadata=example.metadata,
                    error=error,
                )
                append_jsonl(output_path, record.to_dict())
                processed.add(key)
                counters[status.value] += 1
    finally:
        backend.close()
    return counters
