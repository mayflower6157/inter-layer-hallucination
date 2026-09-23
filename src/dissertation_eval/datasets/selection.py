from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError, DataIntegrityError
from ..io import atomic_write_json, atomic_write_jsonl, read_json, sha256_file
from .loader import load_manifest, load_split


RETRUTHQA_SELECTION_VERSION = "1.0.0"
RETRUTHQA_SELECTION_ID = "retruthqa-240-v1"
RETRUTHQA_SELECTION_SEED = 20260924
PILOT_SPLIT = "r1-7b_math_task1"

DEVELOPMENT_QUOTAS = {
    PILOT_SPLIT: 20,
    "r1-7b_multihopqa_task1": 5,
    "r1-7b_multihopqa_task2": 5,
    "r1-14b_multihopqa_task1": 5,
    "r1-14b_multihopqa_task2": 5,
    "r1-7b_science_task1": 5,
    "r1-7b_science_task2": 5,
    "r1-14b_science_task1": 5,
    "r1-14b_science_task2": 5,
}

TEST_QUOTAS = {
    "r1-7b_math_task2": 20,
    "r1-14b_math_task1": 20,
    "r1-14b_math_task2": 20,
    "r1-7b_multihopqa_task1": 15,
    "r1-7b_multihopqa_task2": 15,
    "r1-14b_multihopqa_task1": 15,
    "r1-14b_multihopqa_task2": 15,
    "r1-7b_science_task1": 15,
    "r1-7b_science_task2": 15,
    "r1-14b_science_task1": 15,
    "r1-14b_science_task2": 15,
}


def _stratum_seed(seed: int, role: str, split: str) -> int:
    payload = f"{seed}\0{role}\0{split}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _select(rows: list[Any], count: int, seed: int, role: str, split: str) -> list[Any]:
    if len(rows) < count:
        raise DataIntegrityError(f"Selection requests {count} from {split}, found {len(rows)}")
    ordered = sorted(rows, key=lambda row: row.sample_id)
    random.Random(_stratum_seed(seed, role, split)).shuffle(ordered)
    return ordered[:count]


def _counts(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in records).items()))


def create_retruthqa_selection(
    prepared_manifest_path: Path,
    selection_manifest_path: Path,
    output_dir: Path,
    *,
    seed: int = RETRUTHQA_SELECTION_SEED,
) -> dict[str, Any]:
    prepared_manifest_path = prepared_manifest_path.resolve()
    selection_manifest_path = selection_manifest_path.resolve()
    output_dir = output_dir.resolve()
    source_manifest = load_manifest(prepared_manifest_path)
    if source_manifest.dataset_id != "retruthqa":
        raise ConfigurationError("The selection source must be a prepared ReTruthQA manifest")

    by_split: dict[str, list[Any]] = {}
    raw_by_id: dict[str, dict[str, Any]] = {}
    for split in source_manifest.splits:
        _, examples = load_split(prepared_manifest_path, split.name)
        rows = list(examples)
        by_split[split.name] = rows
        for row in rows:
            if row.sample_id in raw_by_id:
                raise DataIntegrityError(f"Duplicate sample ID across splits: {row.sample_id}")
            raw_by_id[row.sample_id] = row.raw

    missing = sorted((set(DEVELOPMENT_QUOTAS) | set(TEST_QUOTAS)) - set(by_split))
    if missing:
        raise DataIntegrityError(f"Prepared manifest lacks required splits: {missing}")

    selected: dict[str, list[Any]] = {"development": [], "test": []}
    for split, count in DEVELOPMENT_QUOTAS.items():
        selected["development"].extend(
            _select(by_split[split], count, seed, "development", split)
        )
    development_ids = {row.sample_id for row in selected["development"]}
    for split, count in TEST_QUOTAS.items():
        available = [row for row in by_split[split] if row.sample_id not in development_ids]
        selected["test"].extend(_select(available, count, seed, "test", split))

    test_ids = {row.sample_id for row in selected["test"]}
    pilot_ids = {row.sample_id for row in by_split[PILOT_SPLIT]}
    if development_ids & test_ids:
        raise DataIntegrityError("Development and test selections overlap")
    if pilot_ids & test_ids:
        raise DataIntegrityError("Observed pilot samples must not enter the held-out test")
    if len(development_ids) != 60 or len(test_ids) != 180:
        raise DataIntegrityError("Expected exactly 60 development and 180 test samples")

    selection_records: list[dict[str, Any]] = []
    for role in ("development", "test"):
        for row in sorted(selected[role], key=lambda item: item.sample_id):
            selection_records.append({
                "role": role,
                "sample_id": row.sample_id,
                "source_split": next(
                    split for split, candidates in by_split.items()
                    if any(candidate.sample_id == row.sample_id for candidate in candidates)
                ),
                "domain": str(row.metadata["domain"]),
                "source_model": str(row.metadata["source_model"]),
                "task": str(row.metadata["task"]),
            })

    roles: dict[str, Any] = {}
    for role in ("development", "test"):
        records = [row for row in selection_records if row["role"] == role]
        roles[role] = {
            "count": len(records),
            "domain_counts": _counts(records, "domain"),
            "source_model_counts": _counts(records, "source_model"),
            "task_counts": _counts(records, "task"),
            "source_split_counts": _counts(records, "source_split"),
        }

    selection_manifest = {
        "schema_version": "1.0",
        "selection_version": RETRUTHQA_SELECTION_VERSION,
        "selection_id": RETRUTHQA_SELECTION_ID,
        "dataset_id": "retruthqa",
        "seed": seed,
        "source_revision": source_manifest.revision,
        "source_manifest_sha256": sha256_file(prepared_manifest_path),
        "policy": {
            "development_quotas": DEVELOPMENT_QUOTAS,
            "test_quotas": TEST_QUOTAS,
            "within_split_order": "sort sample_id, then shuffle with SHA-256-derived stratum seed",
            "observed_pilot_split": PILOT_SPLIT,
            "held_out_test_excludes_all_observed_pilot_samples": True,
        },
        "observed_pilot": {
            "split": PILOT_SPLIT,
            "count": len(pilot_ids),
            "sample_ids": sorted(pilot_ids),
        },
        "roles": roles,
        "records": selection_records,
    }
    if selection_manifest_path.exists() and read_json(selection_manifest_path) != selection_manifest:
        raise DataIntegrityError(
            f"Existing selection manifest differs from deterministic selection: "
            f"{selection_manifest_path}"
        )
    atomic_write_json(selection_manifest_path, selection_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_specs = []
    for role in ("development", "test"):
        materialized = []
        for selection in (row for row in selection_records if row["role"] == role):
            source = dict(raw_by_id[selection["sample_id"]])
            source["selection_role"] = role
            source["source_split"] = selection["source_split"]
            materialized.append(source)
        path = output_dir / f"{role}.jsonl"
        atomic_write_jsonl(path, materialized)
        split_specs.append({
            "name": role,
            "path": path.name,
            "format": "jsonl",
            "sha256": sha256_file(path),
            "count": len(materialized),
        })

    metadata = list(source_manifest.fields.metadata)
    for name in ("selection_role", "source_split"):
        if name not in metadata:
            metadata.append(name)
    generated_manifest = {
        "schema_version": "1.0",
        "dataset_id": "retruthqa",
        "display_name": "ReTruthQA frozen 240-example dissertation selection",
        "modality": source_manifest.modality.value,
        "source_url": source_manifest.source_url,
        "revision": f"{source_manifest.revision}+selection-{RETRUTHQA_SELECTION_VERSION}",
        "license": source_manifest.license,
        "fields": {
            "id": source_manifest.fields.id,
            "prompt": source_manifest.fields.prompt,
            "reference_answer": source_manifest.fields.reference_answer,
            "group_id": source_manifest.fields.group_id,
            "metadata": metadata,
        },
        "splits": split_specs,
        "notes": (
            "Materialized from the committed selection manifest "
            f"{selection_manifest_path.name}; held-out test excludes the observed pilot split."
        ),
    }
    atomic_write_json(output_dir / "manifest.json", generated_manifest)
    audit = {
        "selection_id": RETRUTHQA_SELECTION_ID,
        "selection_manifest": str(selection_manifest_path),
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "output_manifest": str(output_dir / "manifest.json"),
        "output_manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "roles": roles,
    }
    atomic_write_json(output_dir / "selection-audit.json", audit)
    return audit
