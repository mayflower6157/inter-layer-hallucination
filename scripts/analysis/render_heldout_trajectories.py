#!/usr/bin/env python3
"""Select and render held-out trajectory examples by a fixed, auditable rule."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


CASES = (
    ("TP", 1, True, "incorrect, detected"),
    ("FP", 0, True, "correct, flagged"),
    ("FN", 1, False, "incorrect, missed"),
    ("TN", 0, False, "correct, not flagged"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def svg_text(x: float, y: float, value: str, **attributes: Any) -> str:
    options = {"font-family": "Arial, Helvetica, sans-serif", "font-size": 14}
    options.update({key.replace("_", "-"): item for key, item in attributes.items()})
    payload = " ".join(f'{key}="{item}"' for key, item in options.items())
    return f'<text x="{x}" y="{y}" {payload}>{html.escape(value)}</text>'


def select_cases(
    condition: str,
    scored_rows: list[dict[str, Any]],
    diagnostic_dir: Path,
    threshold: float,
) -> list[dict[str, Any]]:
    diagnostics = {row["record_id"]: row for row in read_jsonl(diagnostic_dir / "records.jsonl")}
    candidates = []
    for row in scored_rows:
        if row["condition"] != condition or row["record_id"] not in diagnostics:
            continue
        score = float(row["scores"]["rhd_only"])
        predicted = score >= threshold
        candidates.append({**row, "predicted": predicted, "margin": abs(score - threshold)})
    selected = []
    for case, label, predicted, description in CASES:
        eligible = [row for row in candidates if row["label"] == label and row["predicted"] is predicted]
        if not eligible:
            raise RuntimeError(f"No {case} candidate for {condition}")
        eligible.sort(key=lambda row: (-row["margin"], row["sample_id"], row["generation_index"]))
        row = eligible[0]
        diagnostic = diagnostics[row["record_id"]]
        selected.append(
            {
                **row,
                "case": case,
                "description": description,
                "threshold": threshold,
                "diagnostic": diagnostic,
                "diagnostic_dir": diagnostic_dir,
            }
        )
    return selected


def binned_series(values: np.ndarray, maximum: int = 260) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= maximum:
        return np.arange(len(values), dtype=float), values.astype(float)
    edges = np.linspace(0, len(values), maximum + 1, dtype=int)
    x = np.array([(edges[i] + edges[i + 1] - 1) / 2 for i in range(maximum)])
    y = np.array([values[edges[i] : edges[i + 1]].mean() for i in range(maximum)])
    return x, y


def render(condition: str, selected: list[dict[str, Any]], output: Path) -> None:
    width, height = 1240, 860
    panel_w, panel_h = 570, 340
    positions = ((45, 100), (635, 100), (45, 475), (635, 475))
    colour = "#d45d5d" if "2B" in condition else "#4f78b8"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(35, 42, f"Fixed-rule trajectory examples: {condition}", font_size=25, font_weight="bold"),
        svg_text(35, 69, "Within each outcome cell, select the trace with the largest absolute margin from the frozen RHD threshold", font_size=14, fill="#444"),
    ]
    for item, (left, top) in zip(selected, positions):
        diagnostic = item["diagnostic"]
        sidecar = item["diagnostic_dir"] / diagnostic["sidecar_path"]
        with np.load(sidecar) as archive:
            raw = np.asarray(archive["reference_reasoning_score"], dtype=float)
            step_index = np.asarray(archive["step_index"], dtype=int)
        centre = float(raw.mean())
        scale = float(raw.std()) or 1.0
        normalized = np.clip((raw - centre) / scale, -3.0, 3.0)
        x_values, y_values = binned_series(normalized)
        plot_left, plot_top = left + 55, top + 82
        plot_w, plot_h = panel_w - 82, panel_h - 118
        x_map = lambda value: plot_left + (float(value) / max(1, len(raw) - 1)) * plot_w
        y_map = lambda value: plot_top + (3.0 - float(value)) / 6.0 * plot_h
        parts.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" rx="8" fill="#fafafa" stroke="#c7ccd3"/>')
        score = float(item["scores"]["rhd_only"])
        parts.append(svg_text(left + 18, top + 30, f"{item['case']}: {item['description']}", font_size=18, font_weight="bold"))
        parts.append(svg_text(left + 18, top + 54, f"{item['domain']} | RHD probability {score:.3f} | threshold {item['threshold']:.3f}", font_size=13, fill="#333"))
        short_id = item["sample_id"].split("/")[-1][:16]
        parts.append(svg_text(left + panel_w - 18, top + 30, short_id, text_anchor="end", font_size=12, fill="#666"))
        for tick in (-3, -2, -1, 0, 1, 2, 3):
            y = y_map(tick)
            parts.append(f'<line x1="{plot_left}" y1="{y}" x2="{plot_left + plot_w}" y2="{y}" stroke="#e4e7eb"/>')
            if tick in (-3, 0, 3):
                parts.append(svg_text(plot_left - 8, y + 4, str(tick), text_anchor="end", font_size=11, fill="#555"))
        for boundary in np.flatnonzero(np.diff(step_index) != 0) + 1:
            x = x_map(boundary)
            parts.append(f'<line x1="{x}" y1="{plot_top}" x2="{x}" y2="{plot_top + plot_h}" stroke="#aeb5bd" stroke-dasharray="4 4"/>')
        points = " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(x_values, y_values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2.5"/>')
        for step in sorted(set(step_index.tolist())):
            mask = step_index == step
            midpoint = np.flatnonzero(mask).mean()
            mean_value = normalized[mask].mean()
            parts.append(f'<circle cx="{x_map(midpoint)}" cy="{y_map(mean_value)}" r="5" fill="#222" stroke="white" stroke-width="1.5"/>')
        parts.append(svg_text(plot_left + plot_w / 2, top + panel_h - 12, f"generated token position (n={len(raw)}; steps={diagnostic['step_count']})", text_anchor="middle", font_size=12, fill="#444"))
        parts.append(svg_text(left + 12, plot_top + plot_h / 2, "normalized score", transform=f"rotate(-90 {left + 12} {plot_top + plot_h / 2})", text_anchor="middle", font_size=11, fill="#444"))
    parts.extend(
        [
            f'<line x1="45" y1="838" x2="85" y2="838" stroke="{colour}" stroke-width="3"/>',
            svg_text(94, 843, "token-level reference Reasoning Score (within-trace z-score)", font_size=12),
            '<circle cx="520" cy="838" r="5" fill="#222"/>',
            svg_text(532, 843, "step mean", font_size=12),
            svg_text(1165, 843, "Dashed lines: frozen paragraph step boundaries", text_anchor="end", font_size=12, fill="#555"),
        ]
    )
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-results", type=Path, required=True)
    parser.add_argument("--scored-records", type=Path, required=True)
    parser.add_argument("--diagnostic-2b-greedy", type=Path, required=True)
    parser.add_argument("--diagnostic-4b-greedy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = json.loads(args.heldout_results.read_text(encoding="utf-8"))
    scored_rows = read_jsonl(args.scored_records)
    specifications = (
        ("Qwen3-2B Greedy", args.diagnostic_2b_greedy, "qwen3-2b-greedy-trajectories.svg"),
        ("Qwen3-4B Greedy", args.diagnostic_4b_greedy, "qwen3-4b-greedy-trajectories.svg"),
    )
    manifest_rows = []
    for condition, diagnostic_dir, filename in specifications:
        threshold = results["conditions"][condition]["models"]["rhd_only"]["overall"]["threshold"]
        selected = select_cases(condition, scored_rows, diagnostic_dir, threshold)
        render(condition, selected, args.output_dir / filename)
        for item in selected:
            manifest_rows.append(
                {
                    "condition": condition, "case": item["case"], "description": item["description"],
                    "sample_id": item["sample_id"], "record_id": item["record_id"],
                    "generation_index": item["generation_index"], "domain": item["domain"],
                    "label": item["label"], "predicted": item["predicted"],
                    "rhd_probability": item["scores"]["rhd_only"], "frozen_threshold": threshold,
                    "absolute_margin": item["margin"], "token_count": item["values"]["token_count"],
                    "step_count": item["values"]["step_count"], "sidecar_sha256": item["diagnostic"]["sidecar_sha256"],
                }
            )
    with (args.output_dir / "trajectory_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest = {
        "schema_version": "1.0",
        "scope": "heldout_completed_greedy_explanatory_examples",
        "selection_rule": "For each model and TP/FP/FN/TN cell, select the completed trace with the greatest absolute margin between frozen RHD probability and frozen threshold; break ties by sample_id then generation_index.",
        "score_display": "Token-level reference Reasoning Score standardized within trace; black points are per-step means; dashed lines are frozen paragraph boundaries.",
        "confirmatory_use": False,
        "truncation_policy": "Truncated traces have no correctness label or RHD sidecar under the frozen protocol and are represented only in aggregate truncation analyses.",
        "figures": [item[2] for item in specifications],
        "selections": manifest_rows,
    }
    (args.output_dir / "trajectory_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
