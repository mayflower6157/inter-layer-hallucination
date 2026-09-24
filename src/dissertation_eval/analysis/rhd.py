from __future__ import annotations

import math
import os
import re
import tempfile
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ..backends.transformers import TransformersBackend
from ..io import append_jsonl, atomic_write_json, iter_jsonl, read_json, sha256_file


RHD_DIAGNOSTIC_VERSION = "1.1.0"
DEFAULT_RELATIVE_LAYERS = (0.5, 0.625, 0.75, 0.875)


def segment_reasoning_steps(text: str) -> list[tuple[int, int]]:
    """Return immutable paragraph-based reasoning-step character spans.

    A step boundary is two or more newlines, optionally containing whitespace.
    Empty regions are discarded; no semantic or answer-label information is used.
    """

    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n+", text):
        left, right = start, match.start()
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = match.end()
    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans or ([(0, len(text))] if text else [])


def official_trace_features(
    step_scores: list[float], step_entropies: list[float], attention_matrix: list[list[float]]
) -> dict[str, float]:
    """Compute the four released RHD features without labels or fitted weights."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("RHD diagnostics require NumPy; install the hf dependencies") from exc
    r = np.asarray(step_scores, dtype=float)
    entropy = np.asarray(step_entropies, dtype=float)
    if r.size == 0 or entropy.size != r.size:
        raise ValueError("RHD trace features require aligned non-empty step arrays")
    early = r[: max(1, r.size // 2)]
    early_mean = float(early.mean())
    cv = float(early.std() / early_mean) if early_mean != 0 else 0.0
    attention = np.asarray(attention_matrix, dtype=float)
    if attention.shape != (r.size, r.size):
        raise ValueError("attention_matrix must be square and aligned to steps")
    start = int(np.floor(r.size * 0.8))
    threshold = float(np.quantile(r, 0.2))
    late_scores: list[float] = []
    for index in range(start, r.size):
        prior = attention[index, :start]
        top_k = min(5, prior.size)
        if top_k:
            top = np.argsort(prior)[-top_k:]
            late_scores.append(float(np.mean((r[top] < threshold) | (r[top] > 4))))
    attention_score = float(np.mean(late_scores)) if late_scores else 0.0

    def minmax(values: Any) -> Any:
        low, high = float(values.min()), float(values.max())
        return (values - low) / (high - low) if high > low else np.zeros_like(values)

    dr = np.diff(minmax(r))
    de = np.diff(minmax(entropy))
    if dr.size >= 2 and float(dr.std()) > 0 and float(de.std()) > 0:
        pcc = float(np.corrcoef(dr, de)[0, 1])
    else:
        pcc = 0.0
    return {
        "attention_score": attention_score,
        "cv_score": cv,
        "reasoning_score_mean": float(r.mean()),
        "pcc": pcc,
    }


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


def _token_step_indices(tokenizer: Any, token_ids: list[int]) -> tuple[str, list[int]]:
    pieces = [
        tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]
    text = "".join(pieces)
    spans = segment_reasoning_steps(text)
    indices: list[int] = []
    cursor = 0
    step = 0
    for piece in pieces:
        midpoint = cursor + max(0, len(piece) - 1) / 2
        while step + 1 < len(spans) and midpoint >= spans[step][1]:
            step += 1
        indices.append(step)
        cursor += len(piece)
    return text, indices


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
            output_attentions=True,
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
    decoded_text, step_index = _token_step_indices(backend.tokenizer, token_ids)
    arrays["step_index"] = step_index
    step_count = max(step_index) + 1
    step_reference_scores = [
        mean(reference_reasoning_score[i] for i, value in enumerate(step_index) if value == step)
        for step in range(step_count)
    ]
    step_entropies = [
        mean(float(arrays["final_entropy"][i]) for i, value in enumerate(step_index) if value == step)
        for step in range(step_count)
    ]
    attention_matrix = [[0.0 for _ in range(step_count)] for _ in range(step_count)]
    if outputs.attentions:
        generated_positions = torch.arange(
            prompt_length, prompt_length + len(token_ids), device=input_ids.device
        )
        token_attention = None
        for layer_attention in outputs.attentions:
            selected = layer_attention[0].float().mean(0).index_select(0, generated_positions)
            selected = selected.index_select(1, generated_positions)
            token_attention = selected if token_attention is None else token_attention + selected
        token_attention = token_attention / len(outputs.attentions)
        for current_step in range(step_count):
            current = torch.tensor(
                [i for i, value in enumerate(step_index) if value == current_step],
                device=input_ids.device,
            )
            for previous_step in range(current_step):
                previous = torch.tensor(
                    [i for i, value in enumerate(step_index) if value == previous_step],
                    device=input_ids.device,
                )
                attention_matrix[current_step][previous_step] = float(
                    token_attention.index_select(0, current).index_select(1, previous).mean()
                )
        del token_attention
    features = official_trace_features(step_reference_scores, step_entropies, attention_matrix)
    arrays["step_reference_reasoning_score"] = step_reference_scores
    arrays["step_entropy"] = step_entropies
    arrays["step_attention_matrix"] = attention_matrix
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
        "step_count": step_count,
        "step_segmentation": "paragraph_blank_line_v1",
        "decoded_text_matches_record": decoded_text == str(row.get("response_text", "")),
        "official_features": features,
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
