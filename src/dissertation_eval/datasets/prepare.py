from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError, DataIntegrityError
from ..io import atomic_write_json, atomic_write_jsonl, iter_jsonl, read_json, sha256_file
from .loader import load_manifest, load_split


RETRUTHQA_EXTRACTOR_VERSION = "1.1.0"
_PREPARED_METADATA = [
    "source_model",
    "domain",
    "task",
    "positive_path_count",
    "negative_path_count",
    "positive_step_counts",
    "negative_step_counts",
    "reference_answer_source",
    "reference_answer_candidates",
    "reference_answer_normalized",
    "reference_answer_normalized_candidates",
    "reference_answer_candidate_counts",
    "reference_answer_selection",
    "reference_answer_confidence",
    "answer_status",
    "answer_type",
    "extractor_version",
    "source_file_sha256",
]


def normalize_reference_answer(answer: str, domain: str) -> str:
    value = unicodedata.normalize("NFKC", answer).strip()
    domain_key = domain.casefold()
    if domain_key == "math":
        value = value.replace("\\dfrac", "\\frac")
        return re.sub(r"\s+", "", value)
    if domain_key == "science":
        wrapper = re.fullmatch(r"\\(?:text|mathrm)\s*\{\s*([A-Za-z])\s*\}", value)
        if wrapper:
            value = wrapper.group(1)
        value = value.strip(" \t\r\n$()[]{}.:;")
        if re.fullmatch(r"[A-Za-z]", value):
            return value.upper()
        return re.sub(r"\s+", " ", value).casefold()
    if domain_key == "multihopqa":
        value = value.replace("’", "'").replace("‘", "'")
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n\"'`.,:;!?()[]{}")
        return value.casefold()
    return re.sub(r"\s+", " ", value).casefold()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _prepared_record(example: Any, source_sha256: str) -> dict[str, Any]:
    metadata = dict(example.metadata)
    candidates = list(metadata["reference_answer_candidates"])
    domain = str(metadata["domain"])
    normalized_per_path = [normalize_reference_answer(candidate, domain) for candidate in candidates]
    normalized_candidates = _unique(normalized_per_path)
    counts = Counter(normalized_per_path)
    winner, winner_count = counts.most_common(1)[0]
    if len(counts) == 1:
        status = "consensus"
        selection = "normalized_consensus"
    elif winner_count > len(candidates) / 2:
        status = "majority"
        selection = "strict_majority_of_positive_paths"
    else:
        status = "ambiguous"
        selection = "none"
    selected_answer = next(
        (raw for raw, normalized in zip(candidates, normalized_per_path) if normalized == winner),
        None,
    )
    usable = status != "ambiguous"
    metadata.update({
        "reference_answer_normalized": winner if usable else None,
        "reference_answer_normalized_candidates": normalized_candidates,
        "reference_answer_candidate_counts": dict(sorted(counts.items())),
        "reference_answer_selection": selection,
        "reference_answer_confidence": winner_count / len(candidates),
        "answer_status": status,
        "answer_type": domain.casefold(),
        "extractor_version": RETRUTHQA_EXTRACTOR_VERSION,
        "source_file_sha256": source_sha256,
    })
    return {
        "sample_id": example.sample_id,
        "prompt": example.prompt,
        "reference_answer": selected_answer if usable else None,
        "group_id": example.group_id,
        **metadata,
    }


def _cache_is_current(output_dir: Path, source_manifest_sha256: str) -> bool:
    audit_path = output_dir / "preparation-manifest.json"
    manifest_path = output_dir / "manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        return False
    try:
        audit = read_json(audit_path)
        if audit.get("extractor_version") != RETRUTHQA_EXTRACTOR_VERSION:
            return False
        if audit.get("source_manifest_sha256") != source_manifest_sha256:
            return False
        if sha256_file(manifest_path) != audit.get("prepared_manifest_sha256"):
            return False
        for split in audit.get("splits", []):
            path = output_dir / split["path"]
            if not path.is_file() or sha256_file(path) != split["sha256"]:
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def prepare_retruthqa(
    source_manifest_path: Path, output_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    source_manifest_path = source_manifest_path.resolve()
    output_dir = output_dir.resolve()
    source_manifest = load_manifest(source_manifest_path)
    if source_manifest.dataset_id != "retruthqa" or source_manifest.adapter != "retruthqa":
        raise ConfigurationError("prepare-retruthqa requires the raw ReTruthQA manifest")
    source_manifest_sha256 = sha256_file(source_manifest_path)
    if not force and _cache_is_current(output_dir, source_manifest_sha256):
        audit = read_json(output_dir / "preparation-manifest.json")
        return {"status": "reused", **audit}

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_splits = []
    audit_splits = []
    total_records = 0
    total_ambiguous = 0
    for split in source_manifest.splits:
        _, examples = load_split(source_manifest_path, split.name)
        records = [_prepared_record(example, split.sha256 or "") for example in examples]
        output_path = output_dir / f"{split.name}.jsonl"
        atomic_write_jsonl(output_path, records)
        output_sha256 = sha256_file(output_path)
        ambiguous = sum(row["answer_status"] == "ambiguous" for row in records)
        majority = sum(row["answer_status"] == "majority" for row in records)
        total_records += len(records)
        total_ambiguous += ambiguous
        prepared_splits.append({
            "name": split.name,
            "path": output_path.name,
            "format": "jsonl",
            "sha256": output_sha256,
            "count": len(records),
        })
        audit_splits.append({
            "name": split.name,
            "path": output_path.name,
            "sha256": output_sha256,
            "count": len(records),
            "ambiguous": ambiguous,
            "majority": majority,
            "source_path": split.path,
            "source_sha256": split.sha256,
        })

    prepared_manifest = {
        "schema_version": "1.0",
        "dataset_id": "retruthqa",
        "display_name": "ReTruthQA prepared generation inputs",
        "modality": "text",
        "source_url": source_manifest.source_url,
        "revision": f"{source_manifest.revision}+extractor-{RETRUTHQA_EXTRACTOR_VERSION}",
        "license": source_manifest.license,
        "fields": {
            "id": "sample_id",
            "prompt": "prompt",
            "reference_answer": "reference_answer",
            "group_id": "group_id",
            "metadata": _PREPARED_METADATA,
        },
        "splits": prepared_splits,
        "notes": (
            "Generated from the immutable raw manifest. Answers are extracted once from "
            "positive reasoning paths and normalized by domain. See preparation-manifest.json."
        ),
    }
    atomic_write_json(output_dir / "manifest.json", prepared_manifest)
    audit = {
        "schema_version": "1.0",
        "dataset_id": "retruthqa",
        "extractor_version": RETRUTHQA_EXTRACTOR_VERSION,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "prepared_manifest": "manifest.json",
        "prepared_manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "total_records": total_records,
        "total_ambiguous": total_ambiguous,
        "total_majority": sum(split["majority"] for split in audit_splits),
        "splits": audit_splits,
    }
    atomic_write_json(output_dir / "preparation-manifest.json", audit)
    return {"status": "prepared", **audit}


def validate_prepared_retruthqa(path: Path) -> dict[str, Any]:
    path = path.resolve()
    output_dir = path if path.is_dir() else path.parent
    manifest_path = output_dir / "manifest.json" if path.is_dir() else path
    audit_path = output_dir / "preparation-manifest.json"
    if not audit_path.is_file():
        raise DataIntegrityError(f"Preparation manifest does not exist: {audit_path}")
    audit = read_json(audit_path)
    if audit.get("extractor_version") != RETRUTHQA_EXTRACTOR_VERSION:
        raise DataIntegrityError("Prepared data uses an unsupported extractor version")
    if sha256_file(manifest_path) != audit.get("prepared_manifest_sha256"):
        raise DataIntegrityError("Prepared dataset manifest checksum mismatch")

    manifest = load_manifest(manifest_path)
    counts: dict[str, int] = {}
    ambiguous = 0
    majority = 0
    for split in manifest.splits:
        _, examples = load_split(manifest_path, split.name)
        rows = list(examples)
        counts[split.name] = len(rows)
        split_path = (manifest_path.parent / split.path).resolve()
        for row in iter_jsonl(split_path):
            if row.get("answer_status") == "ambiguous":
                ambiguous += 1
            elif row.get("answer_status") == "majority":
                majority += 1
            elif not row.get("reference_answer") or not row.get("reference_answer_normalized"):
                raise DataIntegrityError(
                    f"Prepared consensus record lacks an answer in {split_path}"
                )
    if ambiguous:
        raise DataIntegrityError(f"Prepared ReTruthQA contains {ambiguous} ambiguous answers")
    return {
        "dataset_id": manifest.dataset_id,
        "extractor_version": RETRUTHQA_EXTRACTOR_VERSION,
        "splits": counts,
        "total_records": sum(counts.values()),
        "ambiguous": ambiguous,
        "majority": majority,
    }
