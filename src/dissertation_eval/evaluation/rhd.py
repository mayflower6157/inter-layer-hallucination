from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import atomic_write_json, iter_jsonl, read_json, sha256_file


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    wins = 0.0
    for label, score in zip(labels, scores):
        if not label:
            continue
        for other_label, other_score in zip(labels, scores):
            if other_label:
                continue
            wins += 1.0 if score > other_score else 0.5 if score == other_score else 0.0
    return wins / (positives * negatives)


def evaluate_rhd_diagnostics(
    evaluation_path: Path, diagnostic_dirs: dict[str, Path], output_path: Path
) -> dict[str, Any]:
    """Compare frozen diagnostic configurations using development labels only."""

    evaluation_path = evaluation_path.resolve()
    labels = {
        str(row["record_id"]): 0 if row.get("exact_match") is True else 1
        for row in iter_jsonl(evaluation_path)
        if row.get("exact_match") is not None
    }
    configurations: dict[str, Any] = {}
    for name, directory in diagnostic_dirs.items():
        directory = directory.resolve()
        manifest = read_json(directory / "manifest.json")
        rows = list(iter_jsonl(directory / "records.jsonl"))
        matched = [row for row in rows if str(row.get("record_id")) in labels]
        y = [labels[str(row["record_id"])] for row in matched]
        feature_names = ("attention_score", "cv_score", "reasoning_score_mean", "pcc")
        aucs = {
            feature: _auc(y, [float(row["official_features"][feature]) for row in matched])
            for feature in feature_names
        }
        configurations[name] = {
            "records": len(matched),
            "correct": y.count(0),
            "incorrect": y.count(1),
            "candidate_layers": manifest["candidate_layers"],
            "feature_auroc": aucs,
            "best_univariate_auroc": max(
                (value for value in aucs.values() if value is not None), default=None
            ),
            "diagnostic_manifest_sha256": sha256_file(directory / "manifest.json"),
        }
    result = {
        "schema_version": "1.0",
        "scope": "development_only_layer_grid_comparison",
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "configurations": configurations,
    }
    atomic_write_json(output_path, result)
    return result
