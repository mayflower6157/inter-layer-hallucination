#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


CONDITIONS = {
    ("2B", "Greedy"): "rhd-qwen3-2b-greedy-development-cv-v1/cross_validation.json",
    ("2B", "Beam-4"): "rhd-qwen3-2b-beam4-development-cv-v1/cross_validation.json",
    ("2B", "Sampling-5"): "rhd-qwen3-2b-sampling5-development-cv-v1/cross_validation.json",
    ("4B", "Greedy"): "rhd-qwen3-4b-development-cv-v1/cross_validation.json",
    ("4B", "Beam-4"): "rhd-qwen3-4b-beam4-development-cv-v1/cross_validation.json",
    ("4B", "Sampling-5"): "rhd-qwen3-4b-sampling5-development-cv-v1/cross_validation.json",
}


def colour(value: float) -> str:
    strength = max(0.0, min(1.0, (value - 0.4) / 0.4))
    red = round(241 - 125 * strength)
    green = round(245 - 67 * strength)
    blue = round(249 - 48 * strength)
    return f"#{red:02x}{green:02x}{blue:02x}"


def auroc_matrix(results_root: Path, output: Path) -> list[dict[str, object]]:
    rows = []
    for (model, strategy), relative in CONDITIONS.items():
        data = json.loads((results_root / relative).read_text())
        baseline = float(data["models"]["length_step_baseline"]["oof"]["auroc"])
        rhd = float(data["models"]["rhd_only"]["oof"]["auroc"])
        combined = float(data["models"]["rhd_plus_length_step"]["oof"]["auroc"])
        rows.append({
            "model": model, "strategy": strategy, "scored_traces": int(data["records"]),
            "length_step_auroc": baseline, "rhd_auroc": rhd,
            "rhd_minus_baseline": rhd - baseline, "combined_auroc": combined,
        })
    with (output / "auroc_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    width, height = 940, 430
    cell_w, cell_h = 235, 125
    left, top = 190, 100
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="28" y="38" font-family="sans-serif" font-size="25" font-weight="bold">Frozen development RHD AUROC matrix</text>',
        '<text x="28" y="66" font-family="sans-serif" font-size="14" fill="#444">Cell: RHD-only AUROC; parenthesis: Δ versus token/step-count baseline</text>',
    ]
    strategies = ["Greedy", "Beam-4", "Sampling-5"]
    models = ["2B", "4B"]
    lookup = {(row["model"], row["strategy"]): row for row in rows}
    for column, strategy in enumerate(strategies):
        parts.append(f'<text x="{left + column * cell_w + cell_w / 2}" y="90" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">{strategy}</text>')
    for row_index, model in enumerate(models):
        y = top + row_index * cell_h
        parts.append(f'<text x="155" y="{y + 67}" text-anchor="end" font-family="sans-serif" font-size="22" font-weight="bold">Qwen3-{model}</text>')
        for column, strategy in enumerate(strategies):
            item = lookup[(model, strategy)]
            x = left + column * cell_w
            rhd, delta = float(item["rhd_auroc"]), float(item["rhd_minus_baseline"])
            parts.extend([
                f'<rect x="{x}" y="{y}" width="{cell_w - 12}" height="{cell_h - 12}" rx="8" fill="{colour(rhd)}" stroke="#ccd2dc"/>',
                f'<text x="{x + (cell_w - 12) / 2}" y="{y + 52}" text-anchor="middle" font-family="sans-serif" font-size="31" font-weight="bold">{rhd:.3f}</text>',
                f'<text x="{x + (cell_w - 12) / 2}" y="{y + 82}" text-anchor="middle" font-family="sans-serif" font-size="17">Δ {delta:+.3f}</text>',
                f'<text x="{x + (cell_w - 12) / 2}" y="{y + 104}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">n={item["scored_traces"]}</text>',
            ])
    parts.append('</svg>')
    (output / "auroc_matrix.svg").write_text("\n".join(parts) + "\n")
    return rows


def truncation_figure(results_root: Path, output: Path) -> None:
    audit = json.loads((results_root / "rhd-sampling5-truncation-audit-v1/audit.json").read_text())
    metrics = []
    for label, data in audit["conditions"].items():
        metrics.append((label, "Trace truncation", float(data["truncation_rate"])))
        metrics.append((label, "Questions ≥1", data["questions_any_truncated"] / data["questions"]))
        metrics.append((label, "Questions all 5", data["questions_all_truncated"] / data["questions"]))
    width, height = 940, 455
    left, top, chart_w, chart_h = 130, 85, 720, 280
    colours = {"2B": "#d45d5d", "4B": "#4f78b8"}
    categories = ["Trace truncation", "Questions ≥1", "Questions all 5"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="28" y="38" font-family="sans-serif" font-size="25" font-weight="bold">Sampling-5 truncation comparison</text>',
        '<text x="28" y="65" font-family="sans-serif" font-size="14" fill="#444">60 development questions × 5 generations per model</text>',
    ]
    for tick in range(0, 6):
        rate = tick / 10
        y = top + chart_h - rate / 0.5 * chart_h
        parts.extend([
            f'<line x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}" stroke="#e3e6eb"/>',
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-family="sans-serif" font-size="13">{100*rate:.0f}%</text>',
        ])
    group_w = chart_w / len(categories)
    bar_w = 60
    lookup = {(model, category): value for model, category, value in metrics}
    for index, category in enumerate(categories):
        center = left + group_w * (index + 0.5)
        for offset, model in ((-36, "2B"), (36, "4B")):
            value = lookup[(model, category)]
            height_bar = value / 0.5 * chart_h
            x = center + offset - bar_w / 2
            y = top + chart_h - height_bar
            parts.extend([
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{height_bar}" fill="{colours[model]}"/>',
                f'<text x="{x + bar_w/2}" y="{y - 8}" text-anchor="middle" font-family="sans-serif" font-size="13">{100*value:.1f}%</text>',
            ])
        parts.append(f'<text x="{center}" y="{top + chart_h + 30}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(category)}</text>')
    parts.extend([
        '<rect x="680" y="24" width="16" height="16" fill="#d45d5d"/><text x="703" y="37" font-family="sans-serif" font-size="14">Qwen3-2B</text>',
        '<rect x="790" y="24" width="16" height="16" fill="#4f78b8"/><text x="813" y="37" font-family="sans-serif" font-size="14">Qwen3-4B</text>',
        '</svg>',
    ])
    (output / "sampling_truncation_comparison.svg").write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("docs/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/rhd-frozen-protocol-v1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    auroc_matrix(args.results_root, args.output_dir)
    truncation_figure(args.results_root, args.output_dir)


if __name__ == "__main__":
    main()
