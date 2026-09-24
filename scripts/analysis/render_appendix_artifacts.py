#!/usr/bin/env python3
"""Render reproducibility tables that belong in the dissertation appendix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = (
    ("Qwen3-2B", "Greedy", "qwen3-2b-greedy.json"),
    ("Qwen3-2B", "Beam-4", "qwen3-2b-beam4.json"),
    ("Qwen3-2B", "Sampling-5", "qwen3-2b-sampling5.json"),
    ("Qwen3-4B", "Greedy", "qwen3-4b-greedy.json"),
    ("Qwen3-4B", "Beam-4", "qwen3-4b-beam4.json"),
    ("Qwen3-4B", "Sampling-5", "qwen3-4b-sampling5.json"),
)
FEATURE_LABELS = {
    "attention_score": "Attention",
    "cv_score": "CV",
    "reasoning_score_mean": "Reasoning score",
    "pcc": "PCC",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def esc(value: str) -> str:
    return value.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def calibration_rows(calibration_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model, strategy, filename in CONDITIONS:
        payload = json.loads((calibration_dir / filename).read_text(encoding="utf-8"))
        rhd = payload["models"]["rhd_only"]
        layers = payload["diagnostic_manifest"]["candidate_layers"]
        anchor = payload["diagnostic_manifest"]["anchor_layer"]
        for feature, centre, scale, weight in zip(
            rhd["features"], rhd["center"], rhd["scale"], rhd["weights"]
        ):
            feature_rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "feature": FEATURE_LABELS.get(feature, feature),
                    "direction": "+" if float(weight) >= 0 else "-",
                    "center": float(centre),
                    "scale": float(scale),
                    "weight": float(weight),
                }
            )
        summary_rows.append(
            {
                "model": model,
                "strategy": strategy,
                "candidate_layers": ", ".join(map(str, layers)),
                "anchor_layer": anchor,
                "intercept": float(rhd["intercept"]),
                "threshold": float(rhd["threshold"]),
                "development_records": int(payload["records"]),
            }
        )
    return feature_rows, summary_rows


def render_calibration_feature_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{longtable}{lllr@{\hspace{0.7em}}rrr}",
        r"\caption{Frozen development-only RHD feature transformations and logistic weights. Direction is the sign of the fitted weight after standardisation.}\label{tab:appendix-calibration-features}\\",
        r"\toprule",
        r"Model & Decoder & Feature & Dir. & Centre & Scale & Weight \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\small\itshape Table~\thetable\ continued from previous page}\\",
        r"\toprule",
        r"Model & Decoder & Feature & Dir. & Centre & Scale & Weight \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{\small\itshape Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    previous = None
    for row in rows:
        key = (row["model"], row["strategy"])
        model = esc(row["model"]) if key != previous else ""
        strategy = esc(row["strategy"]) if key != previous else ""
        lines.append(
            f"{model} & {strategy} & {esc(row['feature'])} & {row['direction']} & "
            f"{row['center']:.4f} & {row['scale']:.4f} & {row['weight']:.4f} \\\\"
        )
        previous = key
    lines.extend([r"\end{longtable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def render_calibration_summary_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Frozen RHD layer configuration and operating parameters. The layer list excludes the final anchor layer.}",
        r"\label{tab:appendix-calibration-summary}",
        r"\scriptsize",
        r"\begin{tabular}{lllr@{\hspace{0.7em}}rrr}",
        r"\toprule",
        r"Model & Decoder & Candidate layers & Anchor & Intercept & Threshold & $n_{dev}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['model'])} & {esc(row['strategy'])} & {row['candidate_layers']} & "
            f"{row['anchor_layer']} & {row['intercept']:.4f} & {row['threshold']:.4f} & "
            f"{row['development_records']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def is_truncated(row: dict[str, Any]) -> bool:
    return (
        row.get("finish_reason") == "length"
        or row.get("generation_status") == "truncated"
        or row.get("evaluation_status") == "truncated"
    )


def truncation_rows(model: str, records_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(records_path):
        by_question[row["sample_id"]].append(row)
    detailed: list[dict[str, Any]] = []
    for sample_id, rows in sorted(by_question.items()):
        truncated = sum(is_truncated(row) for row in rows)
        detailed.append(
            {
                "model": model,
                "sample_id": sample_id,
                "domain": rows[0]["domain"],
                "truncated_traces": truncated,
                "total_traces": len(rows),
                "at_least_one_truncated": int(truncated > 0),
                "all_truncated": int(truncated == len(rows)),
            }
        )
    distribution = Counter(row["truncated_traces"] for row in detailed)
    summary = {
        "model": model,
        "questions": len(detailed),
        "zero": distribution[0],
        "at_least_one": sum(row["at_least_one_truncated"] for row in detailed),
        "all_five": sum(row["all_truncated"] for row in detailed),
        **{f"exactly_{value}": distribution[value] for value in range(1, 6)},
    }
    return detailed, summary


def render_truncation_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Question-level Sampling-5 truncation. Counts in columns 4--8 are the number of questions with exactly that many truncated generations.}",
        r"\label{tab:appendix-sampling-question-truncation}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Model & Questions & $\geq 1$ & 1 & 2 & 3 & 4 & 5 & All five \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{esc(row['model'])} & {row['questions']} & {row['at_least_one']} & "
            f"{row['exactly_1']} & {row['exactly_2']} & {row['exactly_3']} & "
            f"{row['exactly_4']} & {row['exactly_5']} & {row['all_five']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--sampling-2b-records", type=Path, required=True)
    parser.add_argument("--sampling-4b-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_rows, summary_rows = calibration_rows(args.calibration_dir)
    write_csv(args.output_dir / "calibration_feature_parameters.csv", feature_rows)
    write_csv(args.output_dir / "calibration_operating_parameters.csv", summary_rows)
    render_calibration_feature_table(feature_rows, args.output_dir / "calibration_feature_parameters.tex")
    render_calibration_summary_table(summary_rows, args.output_dir / "calibration_operating_parameters.tex")

    details_2b, summary_2b = truncation_rows("Qwen3-2B", args.sampling_2b_records)
    details_4b, summary_4b = truncation_rows("Qwen3-4B", args.sampling_4b_records)
    write_csv(args.output_dir / "sampling_question_truncation.csv", details_2b + details_4b)
    write_csv(args.output_dir / "sampling_question_truncation_summary.csv", [summary_2b, summary_4b])
    render_truncation_table([summary_2b, summary_4b], args.output_dir / "sampling_question_truncation.tex")
    manifest = {
        "schema_version": "1.0",
        "calibration_scope": "development_only_frozen_calibration",
        "truncation_unit": "sample_id",
        "sampling_generations_per_question": 5,
        "files": sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
    }
    (args.output_dir / "appendix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
