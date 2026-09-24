from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from ..io import atomic_write_json, iter_jsonl, read_json, sha256_file
from .rhd import _auc


RHD_FEATURES = ("attention_score", "cv_score", "reasoning_score_mean", "pcc")
MODEL_FEATURES = {
    "length_step_baseline": ("token_count", "step_count"),
    "rhd_only": RHD_FEATURES,
    "rhd_plus_length_step": RHD_FEATURES + ("token_count", "step_count"),
}


def domain_stratified_folds(rows: list[dict[str, Any]], folds: int = 4) -> list[list[int]]:
    """Deterministically distribute every domain×label stratum across folds."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    sample_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        sample_groups[str(row.get("sample_id", row["record_id"]))].append(index)
    result: list[list[int]] = [[] for _ in range(folds)]
    if any(len(indices) > 1 for indices in sample_groups.values()):
        # Sampling produces several traces for the same question. Keep every trace
        # from a question in one fold to prevent prompt/reference leakage, while
        # distributing question groups independently inside each domain.
        by_domain: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
        for sample_id, indices in sample_groups.items():
            domains = {str(rows[index]["domain"]) for index in indices}
            if len(domains) != 1:
                raise ValueError(f"Sample {sample_id} spans multiple domains")
            by_domain[domains.pop()].append((sample_id, indices))
        for domain, groups in sorted(by_domain.items()):
            ordered = sorted(
                groups,
                key=lambda item: hashlib.sha256(
                    f"20260924\0{domain}\0{item[0]}".encode()
                ).digest(),
            )
            for offset, (_, indices) in enumerate(ordered):
                result[offset % folds].extend(indices)
        return [sorted(indices) for indices in result]

    strata: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(str(row["domain"]), int(row["label"]))].append(index)
    for key, indices in sorted(strata.items()):
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"20260924\0{key}\0{rows[index]['record_id']}".encode()
            ).digest(),
        )
        for offset, index in enumerate(ordered):
            result[offset % folds].append(index)
    return [sorted(indices) for indices in result]


def _fit_logistic(x: Any, y: Any, regularization: float = 1.0) -> tuple[Any, Any, Any]:
    import numpy as np

    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (x - center) / scale
    design = np.column_stack([np.ones(len(x)), standardized])
    weights = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    for _ in range(100):
        logits = np.clip(design @ weights, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + penalty
        gradient = design.T @ (probabilities - y) + penalty @ weights
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return weights, center, scale


def _predict_logistic(x: Any, weights: Any, center: Any, scale: Any) -> Any:
    import numpy as np

    design = np.column_stack([np.ones(len(x)), (x - center) / scale])
    logits = np.clip(design @ weights, -30, 30)
    return 1.0 / (1.0 + np.exp(-logits))


def _threshold_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    positives, negatives = sum(labels), len(labels) - sum(labels)
    recall = tp / positives if positives else 0.0
    specificity = tn / negatives if negatives else 0.0
    return {
        "balanced_accuracy": (recall + specificity) / 2,
        "recall": recall,
        "specificity": specificity,
    }


def select_training_threshold(labels: list[int], scores: list[float]) -> float:
    candidates = sorted({0.0, 1.0, *scores})
    candidates += [(left + right) / 2 for left, right in zip(candidates, candidates[1:])]
    return max(
        candidates,
        key=lambda value: (
            _threshold_metrics(labels, scores, value)["balanced_accuracy"],
            _threshold_metrics(labels, scores, value)["recall"],
            value,
        ),
    )


def _average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    found = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            found += 1
            total += found / rank
    return total / positives


def cross_validate_rhd(
    evaluation_path: Path,
    diagnostic_dir: Path,
    output_dir: Path,
    *,
    folds: int = 4,
) -> dict[str, Any]:
    """Fit preprocessing, directions, weights, and thresholds inside each fold."""

    import numpy as np

    evaluation_path = evaluation_path.resolve()
    diagnostic_dir = diagnostic_dir.resolve()
    output_dir = output_dir.resolve()
    evaluations = {str(row["record_id"]): row for row in iter_jsonl(evaluation_path)}
    rows: list[dict[str, Any]] = []
    for diagnostic in iter_jsonl(diagnostic_dir / "records.jsonl"):
        evaluation = evaluations.get(str(diagnostic["record_id"]))
        if evaluation is None or evaluation.get("exact_match") is None:
            continue
        values = {
            name: float(diagnostic["official_features"][name]) for name in RHD_FEATURES
        }
        values.update(
            token_count=float(diagnostic["token_count"]),
            step_count=float(diagnostic["step_count"]),
        )
        rows.append(
            {
                "record_id": str(diagnostic["record_id"]),
                "sample_id": str(diagnostic["sample_id"]),
                "domain": str(evaluation["domain"]),
                "label": 0 if evaluation["exact_match"] is True else 1,
                "values": values,
            }
        )
    split = domain_stratified_folds(rows, folds)
    models: dict[str, Any] = {}
    all_indices = set(range(len(rows)))
    for model_name, features in MODEL_FEATURES.items():
        fold_results = []
        oof: dict[int, tuple[float, int]] = {}
        for fold_index, validation_indices in enumerate(split):
            training_indices = sorted(all_indices - set(validation_indices))
            x_train = np.asarray(
                [[rows[index]["values"][name] for name in features] for index in training_indices]
            )
            y_train = np.asarray([rows[index]["label"] for index in training_indices], dtype=float)
            x_validation = np.asarray(
                [[rows[index]["values"][name] for name in features] for index in validation_indices]
            )
            y_validation = [rows[index]["label"] for index in validation_indices]
            weights, center, scale = _fit_logistic(x_train, y_train)
            training_scores = _predict_logistic(x_train, weights, center, scale).tolist()
            threshold = select_training_threshold(y_train.astype(int).tolist(), training_scores)
            validation_scores = _predict_logistic(x_validation, weights, center, scale).tolist()
            validation_metrics = _threshold_metrics(y_validation, validation_scores, threshold)
            validation_metrics.update(
                auroc=_auc(y_validation, validation_scores),
                auprc=_average_precision(y_validation, validation_scores),
            )
            for index, score in zip(validation_indices, validation_scores):
                oof[index] = (score, int(score >= threshold))
            fold_results.append(
                {
                    "fold": fold_index,
                    "train_n": len(training_indices),
                    "validation_n": len(validation_indices),
                    "validation_domains": dict(
                        sorted(defaultdict(int, {
                            domain: sum(rows[index]["domain"] == domain for index in validation_indices)
                            for domain in {rows[index]["domain"] for index in validation_indices}
                        }).items())
                    ),
                    "training_threshold": threshold,
                    "standardized_weights": dict(zip(features, weights[1:].tolist())),
                    "validation": validation_metrics,
                }
            )
        ordered_scores = [oof[index][0] for index in range(len(rows))]
        ordered_predictions = [oof[index][1] for index in range(len(rows))]
        labels = [row["label"] for row in rows]
        tp = sum(y == 1 and p == 1 for y, p in zip(labels, ordered_predictions))
        tn = sum(y == 0 and p == 0 for y, p in zip(labels, ordered_predictions))
        oof_metrics = {
            "auroc": _auc(labels, ordered_scores),
            "auprc": _average_precision(labels, ordered_scores),
            "balanced_accuracy": 0.5 * (tp / sum(labels) + tn / (len(labels) - sum(labels))),
        }
        models[model_name] = {
            "features": list(features),
            "folds": fold_results,
            "oof": oof_metrics,
            "fold_mean": {
                metric: mean(float(fold["validation"][metric]) for fold in fold_results)
                for metric in ("auroc", "auprc", "balanced_accuracy")
            },
            "fold_std": {
                metric: pstdev(float(fold["validation"][metric]) for fold in fold_results)
                for metric in ("auroc", "auprc", "balanced_accuracy")
            },
        }
    result = {
        "schema_version": "1.0",
        "scope": "development_only_domain_stratified_cross_validation",
        "folds": folds,
        "records": len(rows),
        "correct": sum(row["label"] == 0 for row in rows),
        "incorrect": sum(row["label"] == 1 for row in rows),
        "fold_record_ids": [[rows[index]["record_id"] for index in indices] for indices in split],
        "evaluation_sha256": sha256_file(evaluation_path),
        "diagnostic_manifest": read_json(diagnostic_dir / "manifest.json"),
        "models": models,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "cross_validation.json", result)
    lines = [
        "# RHD domain-stratified development cross-validation",
        "",
        f"Four folds over {len(rows)} scored development traces. Scaling, feature direction, "
        "weights, and threshold are fitted using each training fold only.",
        "",
        "| Model | OOF AUROC | OOF AUPRC | OOF balanced accuracy | Fold AUROC mean ± std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, model in models.items():
        lines.append(
            f"| {name} | {model['oof']['auroc']:.3f} | {model['oof']['auprc']:.3f} | "
            f"{model['oof']['balanced_accuracy']:.3f} | {model['fold_mean']['auroc']:.3f} ± "
            f"{model['fold_std']['auroc']:.3f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
