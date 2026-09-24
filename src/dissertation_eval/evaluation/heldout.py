from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..io import atomic_write_json, atomic_write_jsonl, iter_jsonl, read_json, sha256_file
from .rhd import _auc
from .rhd_cv import MODEL_FEATURES, RHD_FEATURES, _average_precision, _threshold_metrics
from .rhd_cv import score_frozen_calibration


def _feature_values(diagnostic: dict[str, Any]) -> dict[str, float]:
    values = {
        name: float(diagnostic["official_features"][name]) for name in RHD_FEATURES
    }
    values.update(
        token_count=float(diagnostic["token_count"]),
        step_count=float(diagnostic["step_count"]),
    )
    return values


def _metrics(rows: list[dict[str, Any]], model_name: str, threshold: float) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["scores"][model_name]) for row in rows]
    threshold_metrics = _threshold_metrics(labels, scores, threshold) if rows else {}
    return {
        "records": len(rows),
        "questions": len({str(row["sample_id"]) for row in rows}),
        "correct": labels.count(0),
        "incorrect": labels.count(1),
        "incorrect_rate": sum(labels) / len(labels) if labels else None,
        "auroc": _auc(labels, scores),
        "auprc": _average_precision(labels, scores) if labels else None,
        "balanced_accuracy": threshold_metrics.get("balanced_accuracy"),
        "recall": threshold_metrics.get("recall"),
        "specificity": threshold_metrics.get("specificity"),
        "threshold": threshold,
    }


def _validate_calibration(calibration: dict[str, Any], manifest: dict[str, Any]) -> None:
    if calibration.get("scope") != "development_only_frozen_calibration":
        raise ValueError("Calibration was not frozen from development data")
    if calibration.get("held_out_policy") != "apply_only_no_refit_no_threshold_selection":
        raise ValueError("Calibration does not declare the frozen held-out policy")
    expected = calibration.get("diagnostic_manifest", {})
    for field in ("model_id", "model_revision", "candidate_layers", "anchor_layer"):
        if manifest.get(field) != expected.get(field):
            raise ValueError(f"Held-out diagnostic mismatch for {field}")
    if set(calibration.get("models", {})) != set(MODEL_FEATURES):
        raise ValueError("Calibration does not contain the three frozen model families")


def apply_frozen_calibration(
    evaluation_path: Path,
    diagnostic_dir: Path,
    calibration_path: Path,
) -> dict[str, Any]:
    """Score one held-out cell without fitting or selecting any parameter."""

    evaluation_path = evaluation_path.resolve()
    diagnostic_dir = diagnostic_dir.resolve()
    calibration_path = calibration_path.resolve()
    calibration = read_json(calibration_path)
    manifest = read_json(diagnostic_dir / "manifest.json")
    _validate_calibration(calibration, manifest)
    evaluations = {str(row["record_id"]): row for row in iter_jsonl(evaluation_path)}
    diagnostics = list(iter_jsonl(diagnostic_dir / "records.jsonl"))
    diagnostic_ids = {str(row["record_id"]) for row in diagnostics}
    if len(diagnostic_ids) != len(diagnostics):
        raise ValueError("Duplicate record_id in held-out diagnostics")
    missing = sorted(set(evaluations) - diagnostic_ids)
    unexpected = sorted(diagnostic_ids - set(evaluations))
    if missing or unexpected:
        raise ValueError(
            f"Evaluation/diagnostic record mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    rows: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        record_id = str(diagnostic["record_id"])
        evaluation = evaluations[record_id]
        if evaluation.get("exact_match") is None:
            continue
        values = _feature_values(diagnostic)
        scores = {
            name: score_frozen_calibration(model, values)
            for name, model in calibration["models"].items()
        }
        rows.append(
            {
                "record_id": record_id,
                "sample_id": str(diagnostic["sample_id"]),
                "generation_index": diagnostic.get("generation_index"),
                "domain": str(evaluation["domain"]),
                "exact_match": bool(evaluation["exact_match"]),
                "label": 0 if evaluation["exact_match"] is True else 1,
                "values": values,
                "scores": scores,
            }
        )

    domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        domains[row["domain"]].append(row)
    models: dict[str, Any] = {}
    for name, model in calibration["models"].items():
        threshold = float(model["threshold"])
        models[name] = {
            "features": list(model["features"]),
            "overall": _metrics(rows, name, threshold),
            "by_domain": {
                domain: _metrics(domain_rows, name, threshold)
                for domain, domain_rows in sorted(domains.items())
            },
        }
    generation_status_counts: dict[str, int] = defaultdict(int)
    for evaluation in evaluations.values():
        generation_status_counts[str(evaluation["evaluation_status"])] += 1
    return {
        "schema_version": "1.0",
        "scope": "held_out_apply_only_frozen_development_calibration",
        "held_out_policy": calibration["held_out_policy"],
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "diagnostic_dir": str(diagnostic_dir),
        "diagnostic_manifest_sha256": sha256_file(diagnostic_dir / "manifest.json"),
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "total_records": len(evaluations),
        "scored_records": len(rows),
        "excluded_records": len(evaluations) - len(rows),
        "evaluation_status_counts": dict(sorted(generation_status_counts.items())),
        "models": models,
        "records": rows,
    }


def _format(value: Any) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_heldout_matrix(
    conditions: dict[str, tuple[Path, Path, Path]], output_dir: Path
) -> dict[str, Any]:
    """Apply six frozen calibrations and render machine- and paper-readable tables."""

    if not conditions:
        raise ValueError("At least one held-out condition is required")
    output_dir = output_dir.resolve()
    condition_results: dict[str, Any] = {}
    scored_rows: list[dict[str, Any]] = []
    for label, (evaluation, diagnostic, calibration) in conditions.items():
        result = apply_frozen_calibration(evaluation, diagnostic, calibration)
        rows = result.pop("records")
        for row in rows:
            scored_rows.append({"condition": label, **row})
        condition_results[label] = result

    summary_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    for label, result in condition_results.items():
        for model_name, model in result["models"].items():
            overall = model["overall"]
            summary_rows.append(
                {"condition": label, "model": model_name, **overall}
            )
            for domain, metrics in model["by_domain"].items():
                domain_rows.append(
                    {"condition": label, "domain": domain, "model": model_name, **metrics}
                )

    result = {
        "schema_version": "1.0",
        "scope": "held_out_apply_only_frozen_development_calibrations",
        "conditions": condition_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "heldout_results.json", result)
    atomic_write_jsonl(output_dir / "scored_records.jsonl", scored_rows)
    fields = [
        "condition", "model", "records", "questions", "correct", "incorrect",
        "incorrect_rate", "auroc", "auprc", "balanced_accuracy", "recall",
        "specificity", "threshold",
    ]
    _write_csv(output_dir / "heldout_summary.csv", fields, summary_rows)
    _write_csv(
        output_dir / "domain_breakdown.csv",
        ["condition", "domain", *fields[1:]],
        domain_rows,
    )

    by_condition = {
        label: {row["model"]: row for row in summary_rows if row["condition"] == label}
        for label in condition_results
    }
    paper_rows = []
    for label, models in by_condition.items():
        baseline = models["length_step_baseline"]
        rhd = models["rhd_only"]
        combined = models["rhd_plus_length_step"]
        paper_rows.append(
            {
                "condition": label,
                "records": rhd["records"],
                "incorrect": rhd["incorrect"],
                "baseline_auroc": baseline["auroc"],
                "rhd_auroc": rhd["auroc"],
                "combined_auroc": combined["auroc"],
                "combined_gain": (
                    combined["auroc"] - baseline["auroc"]
                    if combined["auroc"] is not None and baseline["auroc"] is not None
                    else None
                ),
                "rhd_auprc": rhd["auprc"],
                "rhd_balanced_accuracy": rhd["balanced_accuracy"],
            }
        )
    markdown = [
        "# Frozen held-out RHD evaluation",
        "",
        "All transforms, weights, and thresholds are loaded from development-only "
        "calibrations. No held-out fitting or threshold selection is performed.",
        "",
        "| Condition | Scored | Incorrect | Length/step AUROC | RHD AUROC | "
        "Combined AUROC | Gain | RHD AUPRC | RHD balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper_rows:
        markdown.append(
            f"| {row['condition']} | {row['records']} | {row['incorrect']} | "
            f"{_format(row['baseline_auroc'])} | {_format(row['rhd_auroc'])} | "
            f"{_format(row['combined_auroc'])} | {_format(row['combined_gain'])} | "
            f"{_format(row['rhd_auprc'])} | {_format(row['rhd_balanced_accuracy'])} |"
        )
    (output_dir / "heldout_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    latex = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Condition & Scored & Incorrect & Base AUC & RHD AUC & Comb. AUC & "
        r"$\Delta$ & RHD AP & RHD BA \\",
        r"\midrule",
    ]
    for row in paper_rows:
        latex.append(
            f"{_latex_escape(row['condition'])} & {row['records']} & {row['incorrect']} & "
            f"{_format(row['baseline_auroc'])} & {_format(row['rhd_auroc'])} & "
            f"{_format(row['combined_auroc'])} & {_format(row['combined_gain'])} & "
            f"{_format(row['rhd_auprc'])} & {_format(row['rhd_balanced_accuracy'])} " + r"\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "heldout_summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return result
