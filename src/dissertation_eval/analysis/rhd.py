from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ..backends.transformers import TransformersBackend
from ..io import append_jsonl, atomic_write_json, iter_jsonl, read_json, sha256_file


RHD_DIAGNOSTIC_VERSION = "1.1.0"
DEFAULT_RELATIVE_LAYERS = (0.5, 0.625, 0.75, 0.875)


def resolve_candidate_layers(
    num_hidden_layers: int, relative_layers: list[float] | tuple[float, ...]
) -> list[int]:
    """Map normalized depths to unique one-based block indices below the final layer."""

    if num_hidden_layers < 2:
        raise ValueError("RHD diagnostics require at least two transformer layers")
    if not relative_layers:
        raise ValueError("At least one relative candidate layer is required")
    resolved: list[int] = []
    for depth in relative_layers:
        if not 0 < depth < 1:
            raise ValueError("Relative candidate layers must be strictly between 0 and 1")
        layer = min(num_hidden_layers - 1, max(1, math.floor(depth * num_hidden_layers + 0.5)))
        if layer not in resolved:
            resolved.append(layer)
    return sorted(resolved)


def _atomic_save_npz(path: Path, arrays: dict[str, Any]) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("RHD diagnostics require NumPy; install the hf dependencies") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run_config(generations_path: Path) -> dict[str, Any]:
    run_manifest_path = generations_path.with_suffix(generations_path.suffix + ".run.json")
    if not run_manifest_path.is_file():
        raise ValueError(f"Generation run manifest not found: {run_manifest_path}")
    run_manifest = read_json(run_manifest_path)
    config = run_manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Generation run manifest has no config: {run_manifest_path}")
    if config.get("backend") != "transformers":
        raise ValueError("RHD diagnostics currently require a Transformers generation run")
    return config


def _model_parts(model: Any) -> tuple[Any, Any]:
    base = getattr(model, "model", None)
    norm = getattr(base, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if norm is None or lm_head is None:
        raise RuntimeError("Unsupported model structure: expected model.norm and lm_head")
    return norm, lm_head


def _analyse_record(
    backend: TransformersBackend,
    row: dict[str, Any],
    layers: list[int],
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = backend.torch
    token_ids = row.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(isinstance(x, int) for x in token_ids)
    ):
        raise ValueError(f"Record {row.get('record_id')} has no generated token IDs")
    inputs = backend._inputs(str(row.get("rendered_prompt", "")), row.get("image_path"))
    prompt_length = int(inputs["input_ids"].shape[-1])
    if prompt_length != row.get("prompt_tokens"):
        raise ValueError(
            f"Prompt reconstruction mismatch for {row.get('record_id')}: "
            f"expected {row.get('prompt_tokens')}, found {prompt_length}"
        )
    generated = torch.tensor(
        [token_ids], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device
    )
    input_ids = torch.cat((inputs["input_ids"], generated), dim=-1)
    attention_mask = torch.ones_like(input_ids)
    started = time.perf_counter()
    base_model = getattr(backend.model, "model", None)
    if base_model is None:
        raise RuntimeError("Unsupported model structure: expected a base model")
    with torch.inference_mode():
        # Call the base transformer rather than the causal-LM wrapper so a full
        # sequence-by-vocabulary logits tensor is never materialized.
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states
    num_hidden_layers = len(hidden_states) - 1
    if layers[-1] >= num_hidden_layers:
        raise ValueError(f"Candidate layer {layers[-1]} is invalid for {num_hidden_layers} layers")
    norm, lm_head = _model_parts(backend.model)
    positions = torch.arange(
        prompt_length - 1,
        prompt_length + len(token_ids) - 1,
        device=input_ids.device,
    )
    chosen = generated[0]
    arrays: dict[str, list[Any]] = {
        "final_entropy": [],
        "final_chosen_logprob": [],
        "final_chosen_rank": [],
        "final_top_token_id": [],
    }
    for layer in layers:
        arrays[f"layer_{layer}_jsd"] = []
        arrays[f"layer_{layer}_reference_divergence"] = []
        arrays[f"layer_{layer}_l1"] = []
        arrays[f"layer_{layer}_chosen_logprob"] = []
        arrays[f"layer_{layer}_chosen_rank"] = []
        arrays[f"layer_{layer}_top_token_id"] = []
        arrays[f"layer_{layer}_top_agrees_final"] = []

    log_two = math.log(2.0)
    with torch.inference_mode():
        for start in range(0, len(token_ids), chunk_size):
            stop = min(len(token_ids), start + chunk_size)
            index = positions[start:stop]
            target = chosen[start:stop]
            final_logits = lm_head(hidden_states[-1][0, index]).float()
            final_logp = torch.log_softmax(final_logits, dim=-1)
            final_p = final_logp.exp()
            target_column = target.unsqueeze(-1)
            final_chosen = final_logp.gather(-1, target_column).squeeze(-1)
            final_top = final_logits.argmax(dim=-1)
            arrays["final_entropy"].extend((-(final_p * final_logp).sum(-1)).cpu().tolist())
            arrays["final_chosen_logprob"].extend(final_chosen.cpu().tolist())
            arrays["final_chosen_rank"].extend(
                (final_logits > final_logits.gather(-1, target_column))
                .sum(-1)
                .add(1)
                .cpu()
                .tolist()
            )
            arrays["final_top_token_id"].extend(final_top.cpu().tolist())
            for layer in layers:
                layer_logits = lm_head(norm(hidden_states[layer][0, index])).float()
                layer_logp = torch.log_softmax(layer_logits, dim=-1)
                layer_p = layer_logp.exp()
                mixture_logp = torch.logaddexp(layer_logp, final_logp) - log_two
                jsd = 0.5 * (
                    (layer_p * (layer_logp - mixture_logp)).sum(-1)
                    + (final_p * (final_logp - mixture_logp)).sum(-1)
                )
                # Reproduce the released RHD implementation's numerical statistic in
                # addition to standard JSD. Its torch.kl_div calls use M as the target,
                # average over the vocabulary, and the later step score is scaled by
                # 1e5. Keeping both metrics prevents the reference implementation's
                # quantity from being mislabeled as mathematically standard JSD.
                mixture_p = 0.5 * (layer_p + final_p)
                reference_divergence = 0.5 * (
                    (mixture_p * (mixture_logp - final_logp)).mean(-1)
                    + (mixture_p * (mixture_logp - layer_logp)).mean(-1)
                )
                if not bool(torch.isfinite(jsd).all()):
                    raise RuntimeError(f"Non-finite JSD at layer {layer}")
                if float(jsd.min()) < -1e-6 or float(jsd.max()) > log_two + 1e-5:
                    raise RuntimeError(f"JSD outside [0, ln(2)] at layer {layer}")
                layer_top = layer_logits.argmax(dim=-1)
                layer_chosen = layer_logp.gather(-1, target_column).squeeze(-1)
                prefix = f"layer_{layer}"
                arrays[f"{prefix}_jsd"].extend(jsd.cpu().tolist())
                arrays[f"{prefix}_reference_divergence"].extend(
                    reference_divergence.cpu().tolist()
                )
                arrays[f"{prefix}_l1"].extend((layer_p - final_p).abs().sum(-1).cpu().tolist())
                arrays[f"{prefix}_chosen_logprob"].extend(layer_chosen.cpu().tolist())
                arrays[f"{prefix}_chosen_rank"].extend(
                    (layer_logits > layer_logits.gather(-1, target_column))
                    .sum(-1)
                    .add(1)
                    .cpu()
                    .tolist()
                )
                arrays[f"{prefix}_top_token_id"].extend(layer_top.cpu().tolist())
                arrays[f"{prefix}_top_agrees_final"].extend((layer_top == final_top).cpu().tolist())

    reasoning_score = [
        mean(float(arrays[f"layer_{layer}_jsd"][index]) for layer in layers)
        for index in range(len(token_ids))
    ]
    arrays["reasoning_score"] = reasoning_score
    reference_reasoning_score = [
        1e5
        * mean(
            float(arrays[f"layer_{layer}_reference_divergence"][index])
            for layer in layers
        )
        for index in range(len(token_ids))
    ]
    arrays["reference_reasoning_score"] = reference_reasoning_score
    arrays["token_ids"] = token_ids
    elapsed_ms = (time.perf_counter() - started) * 1000
    summary = {
        "record_id": row.get("record_id"),
        "run_id": row.get("run_id"),
        "sample_id": row.get("sample_id"),
        "generation_index": row.get("generation_index"),
        "token_count": len(token_ids),
        "num_hidden_layers": num_hidden_layers,
        "candidate_layers": layers,
        "candidate_relative_depths": [layer / num_hidden_layers for layer in layers],
        "reasoning_score_mean": mean(reasoning_score),
        "reasoning_score_std": pstdev(reasoning_score),
        "reasoning_score_max": max(reasoning_score),
        "reasoning_score_min": min(reasoning_score),
        "reference_reasoning_score_mean": mean(reference_reasoning_score),
        "reference_reasoning_score_std": pstdev(reference_reasoning_score),
        "final_entropy_mean": mean(float(x) for x in arrays["final_entropy"]),
        "top_token_agreement_by_layer": {
            str(layer): mean(float(x) for x in arrays[f"layer_{layer}_top_agrees_final"])
            for layer in layers
        },
        "diagnostic_latency_ms": elapsed_ms,
    }
    del outputs, hidden_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, arrays


def diagnose_rhd_generations(
    generations_path: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    relative_layers: list[float] | tuple[float, ...] = DEFAULT_RELATIVE_LAYERS,
    chunk_size: int = 16,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    generations_path = generations_path.resolve()
    output_dir = output_dir.resolve()
    rows = list(iter_jsonl(generations_path))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"Generation file contains no selected records: {generations_path}")
    config = _run_config(generations_path)
    options = dict(config.get("backend_options", {}))
    if options.get("multimodal", False):
        raise ValueError("RHD diagnostic v1 currently supports text-only runs")
    backend = TransformersBackend(str(config["model_id"]), str(config["model_revision"]), options)
    num_hidden_layers = int(backend.model.config.num_hidden_layers)
    layers = resolve_candidate_layers(num_hidden_layers, relative_layers)
    summary_path = output_dir / "records.jsonl"
    existing = (
        {str(row.get("record_id")): row for row in iter_jsonl(summary_path)}
        if summary_path.exists()
        else {}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "diagnostic_version": RHD_DIAGNOSTIC_VERSION,
        "mode": "post_hoc_teacher_forced_read_only",
        "generation_path": str(generations_path),
        "generation_sha256": sha256_file(generations_path),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "num_hidden_layers": num_hidden_layers,
        "candidate_layers": layers,
        "candidate_relative_depths": [layer / num_hidden_layers for layer in layers],
        "anchor_layer": num_hidden_layers,
        "relative_layer_request": list(relative_layers),
        "chunk_size": chunk_size,
        "labels_available_to_extractor": False,
        "changes_generation": False,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError("Refusing to resume: RHD diagnostic configuration changed")
    atomic_write_json(manifest_path, manifest)
    created = skipped = 0
    try:
        for row in rows:
            record_id = str(row.get("record_id", ""))
            if not record_id:
                raise ValueError("Every generation record must have a record_id")
            if record_id in existing:
                skipped += 1
                continue
            summary, arrays = _analyse_record(backend, row, layers, chunk_size)
            sidecar = output_dir / "sidecars" / f"{record_id}.npz"
            _atomic_save_npz(sidecar, arrays)
            summary.update(
                {
                    "schema_version": "1.0",
                    "diagnostic_version": RHD_DIAGNOSTIC_VERSION,
                    "sidecar_path": str(sidecar.relative_to(output_dir)),
                    "sidecar_sha256": sha256_file(sidecar),
                }
            )
            append_jsonl(summary_path, summary)
            existing[record_id] = summary
            created += 1
    finally:
        backend.close()
    result_rows = list(existing.values())
    audit = {
        "created": created,
        "skipped": skipped,
        "records": len(result_rows),
        "reasoning_score_mean": mean(float(row["reasoning_score_mean"]) for row in result_rows),
        "reasoning_score_range": [
            min(float(row["reasoning_score_min"]) for row in result_rows),
            max(float(row["reasoning_score_max"]) for row in result_rows),
        ],
        "nonconstant_records": sum(
            float(row["reasoning_score_std"]) > 0 for row in result_rows
        ),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(output_dir / "summary.json", audit)
    return audit
