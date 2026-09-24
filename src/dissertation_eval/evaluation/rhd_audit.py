from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from ..io import atomic_write_json, iter_jsonl, read_json, sha256_file
from .rhd import _auc


FEATURES = ("attention_score", "cv_score", "reasoning_score_mean", "pcc")


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = (
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    ) ** 0.5
    return numerator / denominator if denominator else None


def _trajectory_bins(values: list[float], bins: int = 10) -> list[float | None]:
    if not values:
        return [None] * bins
    grouped: list[list[float]] = [[] for _ in range(bins)]
    size = len(values)
    for index, value in enumerate(values):
        target = min(bins - 1, int(index * bins / size))
        grouped[target].append(float(value))
    return [mean(items) if items else None for items in grouped]


def audit_rhd_diagnostics(
    generations_path: Path,
    evaluation_path: Path,
    diagnostic_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Produce a development-only segmentation, confound, and trajectory audit."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("RHD audit requires NumPy") from exc
    generations_path = generations_path.resolve()
    evaluation_path = evaluation_path.resolve()
    diagnostic_dir = diagnostic_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generations = {str(row["record_id"]): row for row in iter_jsonl(generations_path)}
    evaluations = {str(row["record_id"]): row for row in iter_jsonl(evaluation_path)}
    diagnostics = list(iter_jsonl(diagnostic_dir / "records.jsonl"))
    joined: list[dict[str, Any]] = []
    step_lengths: list[int] = []
    trajectories: dict[str, list[list[float | None]]] = defaultdict(list)
    for diagnostic in diagnostics:
        record_id = str(diagnostic["record_id"])
        generation = generations.get(record_id)
        evaluation = evaluations.get(record_id)
        if generation is None or evaluation is None:
            continue
        sidecar = diagnostic_dir / str(diagnostic["sidecar_path"])
        if sha256_file(sidecar) != diagnostic["sidecar_sha256"]:
            raise RuntimeError(f"RHD sidecar checksum mismatch: {sidecar}")
        with np.load(sidecar) as arrays:
            step_index = [int(value) for value in arrays["step_index"]]
            counts = Counter(step_index)
            step_lengths.extend(counts[index] for index in sorted(counts))
            trajectory = _trajectory_bins(arrays["reference_reasoning_score"].tolist())
        exact = evaluation.get("exact_match")
        correctness = "correct" if exact is True else "incorrect" if exact is False else "unscored"
        trajectories[correctness].append(trajectory)
        joined.append(
            {
                "record_id": record_id,
                "domain": str(evaluation.get("domain")),
                "correctness": correctness,
                "finish_reason": generation.get("finish_reason"),
                "token_count": int(diagnostic["token_count"]),
                "step_count": int(diagnostic["step_count"]),
                "features": {name: float(diagnostic["official_features"][name]) for name in FEATURES},
            }
        )

    scored = [row for row in joined if row["correctness"] != "unscored"]
    token_counts = sorted(row["token_count"] for row in scored)
    q1 = token_counts[len(token_counts) // 3] if token_counts else 0
    q2 = token_counts[(2 * len(token_counts)) // 3] if token_counts else 0
    for row in scored:
        row["length_group"] = "short" if row["token_count"] <= q1 else "medium" if row["token_count"] <= q2 else "long"

    feature_distributions: dict[str, Any] = {}
    for feature in FEATURES:
        feature_distributions[feature] = {
            correctness: _describe(
                [row["features"][feature] for row in scored if row["correctness"] == correctness]
            )
            for correctness in ("correct", "incorrect")
        }
    labels = [1 if row["correctness"] == "incorrect" else 0 for row in scored]
    overall_auc = {
        feature: _auc(labels, [row["features"][feature] for row in scored]) for feature in FEATURES
    }
    stratified_auc: dict[str, Any] = {}
    for field in ("domain", "length_group"):
        stratified_auc[field] = {}
        for value in sorted({str(row[field]) for row in scored}):
            subset = [row for row in scored if row[field] == value]
            subset_labels = [1 if row["correctness"] == "incorrect" else 0 for row in subset]
            stratified_auc[field][value] = {
                "n": len(subset),
                "correct": subset_labels.count(0),
                "incorrect": subset_labels.count(1),
                "feature_auroc": {
                    feature: _auc(subset_labels, [row["features"][feature] for row in subset])
                    for feature in FEATURES
                },
            }
    trajectory_summary: dict[str, list[float | None]] = {}
    for correctness, traces in trajectories.items():
        trajectory_summary[correctness] = [
            mean([trace[index] for trace in traces if trace[index] is not None])
            if any(trace[index] is not None for trace in traces)
            else None
            for index in range(10)
        ]
    confounds = {
        feature: {
            "pearson_token_count": _pearson(
                [row["features"][feature] for row in scored],
                [float(row["token_count"]) for row in scored],
            ),
            "pearson_step_count": _pearson(
                [row["features"][feature] for row in scored],
                [float(row["step_count"]) for row in scored],
            ),
        }
        for feature in FEATURES
    }
    result = {
        "schema_version": "1.0",
        "scope": "development_only_rhd_audit",
        "inputs": {
            "generations_sha256": sha256_file(generations_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "diagnostic_manifest": read_json(diagnostic_dir / "manifest.json"),
        },
        "records": len(joined),
        "scored_records": len(scored),
        "correctness_counts": dict(Counter(row["correctness"] for row in joined)),
        "segmentation": {
            "method": "paragraph_blank_line_v1",
            "step_count": _describe([float(row["step_count"]) for row in joined]),
            "step_token_count": _describe([float(value) for value in step_lengths]),
            "one_or_two_step_fraction": sum(row["step_count"] <= 2 for row in joined) / len(joined),
        },
        "length_boundaries_tokens": [q1, q2],
        "feature_distributions": feature_distributions,
        "overall_feature_auroc": overall_auc,
        "stratified_auroc": stratified_auc,
        "feature_confounds": confounds,
        "normalized_reference_reasoning_trajectory": trajectory_summary,
    }
    atomic_write_json(output_dir / "audit.json", result)
    lines = [
        "# RHD development audit",
        "",
        f"Records: {len(joined)} ({len(scored)} scored). This report uses development data only.",
        "",
        "## Step segmentation",
        "",
        f"Method: `paragraph_blank_line_v1`; one-or-two-step fraction: "
        f"{result['segmentation']['one_or_two_step_fraction']:.3f}.",
        "",
        "| Quantity | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|",
        f"| Steps per trace | {result['segmentation']['step_count']['mean']:.3f} | "
        f"{result['segmentation']['step_count']['median']:.3f} | "
        f"{result['segmentation']['step_count']['min']:.0f} | "
        f"{result['segmentation']['step_count']['max']:.0f} |",
        f"| Tokens per step | {result['segmentation']['step_token_count']['mean']:.3f} | "
        f"{result['segmentation']['step_token_count']['median']:.3f} | "
        f"{result['segmentation']['step_token_count']['min']:.0f} | "
        f"{result['segmentation']['step_token_count']['max']:.0f} |",
        "",
        "## Overall feature discrimination",
        "",
        "| Feature | AUROC | Corr. tokens | Corr. steps |",
        "|---|---:|---:|---:|",
    ]
    for feature in FEATURES:
        token_corr = confounds[feature]["pearson_token_count"]
        step_corr = confounds[feature]["pearson_step_count"]
        lines.append(
            f"| {feature} | {overall_auc[feature]:.3f} | "
            f"{token_corr:.3f} | {step_corr:.3f} |"
        )
    lines.extend(["", "## Stratified AUROC", ""])
    for field, groups in stratified_auc.items():
        lines.extend(
            [
                f"### {field}",
                "",
                "| Group | N | Correct | Incorrect | Attention | CV | Mean-R | PCC |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, values in groups.items():
            aucs = values["feature_auroc"]
            rendered = ["NA" if aucs[key] is None else f"{aucs[key]:.3f}" for key in FEATURES]
            lines.append(
                f"| {name} | {values['n']} | {values['correct']} | {values['incorrect']} | "
                + " | ".join(rendered)
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Normalized token-position trajectory",
            "",
            "Values are the mean reference-compatible Reasoning Score in ten normalized "
            "token-position bins.",
            "",
            "| Outcome | " + " | ".join(f"B{i + 1}" for i in range(10)) + " |",
            "|---|" + "---:|" * 10,
        ]
    )
    for name, values in trajectory_summary.items():
        lines.append(
            f"| {name} | "
            + " | ".join("NA" if value is None else f"{value:.3f}" for value in values)
            + " |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
