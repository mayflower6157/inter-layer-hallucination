#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


ORDER = [
    "Qwen3-2B Greedy",
    "Qwen3-2B Beam-4",
    "Qwen3-2B Sampling-5",
    "Qwen3-4B Greedy",
    "Qwen3-4B Beam-4",
    "Qwen3-4B Sampling-5",
]
STRATEGIES = ["Greedy", "Beam-4", "Sampling-5"]
DOMAINS = ["MATH", "Science", "MultiHopQA"]
MODEL_COLOURS = {"2B": "#d45d5d", "4B": "#4f78b8"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def pct(value: Any) -> str:
    return "--" if value is None else f"{100 * float(value):.1f}"


def esc(value: str) -> str:
    return value.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def svg_text(x: float, y: float, value: str, **attributes: Any) -> str:
    options = {"font-family": "Arial, Helvetica, sans-serif", "font-size": 14}
    options.update({key.replace("_", "-"): item for key, item in attributes.items()})
    payload = " ".join(f'{key.replace("_", "-")}="{item}"' for key, item in options.items())
    return f'<text x="{x}" y="{y}" {payload}>{html.escape(value)}</text>'


def svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(28, 38, title, font_size=25, font_weight="bold"),
        svg_text(28, 64, subtitle, font_size=14, fill="#444"),
    ]


def write_svg(path: Path, parts: list[str]) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def condition_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition in ORDER:
        item = results["conditions"][condition]
        baseline = item["models"]["length_step_baseline"]["overall"]
        rhd = item["models"]["rhd_only"]["overall"]
        combined = item["models"]["rhd_plus_length_step"]["overall"]
        rows.append(
            {
                "condition": condition,
                "model": "2B" if "2B" in condition else "4B",
                "strategy": condition.split()[-1],
                "scored": rhd["records"],
                "incorrect": rhd["incorrect"],
                "baseline": baseline["auroc"],
                "rhd": rhd["auroc"],
                "rhd_ci": rhd["confidence_95"]["auroc"],
                "combined": combined["auroc"],
                "gain": combined["auroc"] - baseline["auroc"],
                "gain_ci": item["bootstrap"]["combined_minus_baseline_auroc"],
                "auprc": rhd["auprc"],
                "auprc_ci": rhd["confidence_95"]["auprc"],
                "balanced_accuracy": rhd["balanced_accuracy"],
                "balanced_accuracy_ci": rhd["confidence_95"]["balanced_accuracy"],
            }
        )
    return rows


def auroc_forest(rows: list[dict[str, Any]], output: Path) -> None:
    width, height = 1260, 570
    left, right, top, row_h = 270, 1010, 100, 70
    low, high = 0.35, 0.80
    x = lambda value: left + (float(value) - low) / (high - low) * (right - left)
    parts = svg_start(
        width,
        height,
        "Held-out discrimination under frozen calibration",
        "RHD-only 95% clustered bootstrap CI; baseline and combined estimates shown as points",
    )
    for tick in (0.4, 0.5, 0.6, 0.7, 0.8):
        tx = x(tick)
        parts.append(f'<line x1="{tx}" y1="82" x2="{tx}" y2="520" stroke="#e1e5ea"/>')
        parts.append(svg_text(tx, 545, f"{tick:.1f}", text_anchor="middle", font_size=13))
    parts.append(f'<line x1="{x(0.5)}" y1="82" x2="{x(0.5)}" y2="520" stroke="#555" stroke-dasharray="6 5"/>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        colour = MODEL_COLOURS[row["model"]]
        parts.append(svg_text(left - 18, y + 5, row["condition"], text_anchor="end", font_size=15))
        lo, hi = row["rhd_ci"]
        parts.extend(
            [
                f'<line x1="{x(lo)}" y1="{y}" x2="{x(hi)}" y2="{y}" stroke="{colour}" stroke-width="5"/>',
                f'<circle cx="{x(row["rhd"])}" cy="{y}" r="7" fill="{colour}"/>',
                f'<rect x="{x(row["baseline"])-5}" y="{y-5}" width="10" height="10" fill="#777"/>',
                f'<path d="M {x(row["combined"])} {y-7} L {x(row["combined"])+7} {y} L {x(row["combined"])} {y+7} L {x(row["combined"])-7} {y} Z" fill="#222"/>',
            ]
        )
        parts.append(svg_text(right + 12, y + 5, f"{row['rhd']:.3f} [{lo:.3f}, {hi:.3f}]", font_size=13))
    parts.extend(
        [
            '<circle cx="300" cy="505" r="6" fill="#4f78b8"/>',
            svg_text(314, 510, "RHD-only (line: 95% CI)", font_size=13),
            '<rect x="490" y="499" width="10" height="10" fill="#777"/>',
            svg_text(506, 510, "Length/step baseline", font_size=13),
            '<path d="M 690 498 L 697 505 L 690 512 L 683 505 Z" fill="#222"/>',
            svg_text(706, 510, "Combined", font_size=13),
        ]
    )
    write_svg(output / "heldout_auroc_ci.svg", parts)


def gain_forest(rows: list[dict[str, Any]], output: Path) -> None:
    width, height = 1180, 560
    left, right, top, row_h = 270, 900, 100, 68
    low, high = -0.10, 0.25
    x = lambda value: left + (float(value) - low) / (high - low) * (right - left)
    parts = svg_start(
        width,
        height,
        "Incremental AUROC beyond length and step count",
        "Combined minus baseline; paired domain-stratified sample_id bootstrap",
    )
    for tick in (-0.1, 0.0, 0.1, 0.2):
        tx = x(tick)
        parts.append(f'<line x1="{tx}" y1="82" x2="{tx}" y2="505" stroke="#e1e5ea"/>')
        parts.append(svg_text(tx, 535, f"{tick:+.1f}", text_anchor="middle", font_size=13))
    parts.append(f'<line x1="{x(0)}" y1="82" x2="{x(0)}" y2="505" stroke="#222" stroke-width="2"/>')
    for index, row in enumerate(rows):
        y = top + index * row_h
        colour = MODEL_COLOURS[row["model"]]
        lo, hi = row["gain_ci"]
        parts.append(svg_text(left - 18, y + 5, row["condition"], text_anchor="end", font_size=15))
        parts.append(f'<line x1="{x(lo)}" y1="{y}" x2="{x(hi)}" y2="{y}" stroke="{colour}" stroke-width="5"/>')
        parts.append(f'<circle cx="{x(row["gain"])}" cy="{y}" r="7" fill="{colour}"/>')
        parts.append(svg_text(right + 12, y + 5, f"{row['gain']:+.3f} [{lo:+.3f}, {hi:+.3f}]", font_size=13))
    write_svg(output / "heldout_incremental_gain.svg", parts)


def development_comparison(rows: list[dict[str, Any]], development_csv: Path, output: Path) -> None:
    development = {
        (row["model"], row["strategy"]): float(row["rhd_auroc"])
        for row in csv.DictReader(development_csv.open(encoding="utf-8"))
    }
    width, height = 1050, 560
    left, right, top, row_h = 270, 930, 100, 68
    low, high = 0.4, 0.82
    x = lambda value: left + (float(value) - low) / (high - low) * (right - left)
    parts = svg_start(
        width,
        height,
        "Development-to-held-out RHD AUROC",
        "Development: four-fold OOF estimate; held-out: frozen-calibration estimate",
    )
    for tick in (0.4, 0.5, 0.6, 0.7, 0.8):
        tx = x(tick)
        parts.append(f'<line x1="{tx}" y1="82" x2="{tx}" y2="505" stroke="#e1e5ea"/>')
        parts.append(svg_text(tx, 535, f"{tick:.1f}", text_anchor="middle", font_size=13))
    for index, row in enumerate(rows):
        y = top + index * row_h
        dev = development[(row["model"], row["strategy"])]
        test = row["rhd"]
        colour = MODEL_COLOURS[row["model"]]
        parts.append(svg_text(left - 18, y + 5, row["condition"], text_anchor="end", font_size=15))
        parts.append(f'<line x1="{x(dev)}" y1="{y}" x2="{x(test)}" y2="{y}" stroke="#aab1ba" stroke-width="3"/>')
        parts.append(f'<circle cx="{x(dev)}" cy="{y}" r="7" fill="white" stroke="{colour}" stroke-width="3"/>')
        parts.append(f'<circle cx="{x(test)}" cy="{y}" r="7" fill="{colour}"/>')
        parts.append(svg_text(right + 12, y + 5, f"{dev:.3f} → {test:.3f}", font_size=13))
    parts.extend(
        [
            '<circle cx="350" cy="505" r="7" fill="white" stroke="#4f78b8" stroke-width="3"/>',
            svg_text(365, 510, "Development", font_size=13),
            '<circle cx="500" cy="505" r="7" fill="#4f78b8"/>',
            svg_text(515, 510, "Held-out", font_size=13),
        ]
    )
    write_svg(output / "development_heldout_auroc.svg", parts)


def heat_colour(value: float) -> str:
    strength = max(0.0, min(1.0, (value - 0.4) / 0.35))
    return "#{:02x}{:02x}{:02x}".format(
        round(243 - 116 * strength),
        round(246 - 70 * strength),
        round(250 - 38 * strength),
    )


def domain_heatmap(results: dict[str, Any], output: Path) -> None:
    width, height = 930, 650
    left, top, cell_w, cell_h = 285, 110, 190, 72
    parts = svg_start(
        width,
        height,
        "Held-out RHD AUROC by domain",
        "Secondary analysis; fixed development calibration, no domain-specific refitting",
    )
    for column, domain in enumerate(DOMAINS):
        parts.append(svg_text(left + column * cell_w + 85, 95, domain, text_anchor="middle", font_size=16, font_weight="bold"))
    for index, condition in enumerate(ORDER):
        y = top + index * cell_h
        parts.append(svg_text(left - 15, y + 40, condition, text_anchor="end", font_size=15))
        model = results["conditions"][condition]["models"]["rhd_only"]
        for column, domain in enumerate(DOMAINS):
            metric = model["by_domain"][domain]
            value = float(metric["auroc"])
            x = left + column * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="170" height="58" rx="7" fill="{heat_colour(value)}" stroke="#ccd2dc"/>')
            parts.append(svg_text(x + 85, y + 28, f"{value:.3f}", text_anchor="middle", font_size=22, font_weight="bold"))
            parts.append(svg_text(x + 85, y + 48, f"n={metric['records']}", text_anchor="middle", font_size=12, fill="#333"))
    write_svg(output / "heldout_domain_auroc.svg", parts)


def generation_rows(comparisons: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in ("2B", "4B"):
        comparison = comparisons[model]
        for strategy in STRATEGIES:
            item = comparison["strategies"][strategy]
            evaluation, efficiency = item["evaluation"], item["efficiency"]
            records = int(evaluation["records"])
            truncation = evaluation["status_counts"].get("truncated", 0)
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "records": records,
                    "completed": evaluation["completed"] / records,
                    "truncation": truncation / records,
                    "parse_coverage": evaluation["parse_coverage_completed"],
                    "accuracy_scored": evaluation["exact_match_rate_scored"],
                    "accuracy_all": evaluation["exact_match_rate_all"],
                    "mean_tokens": efficiency["generated_tokens_mean"],
                    "median_latency_s": efficiency["latency_ms_median"] / 1000,
                    "peak_gpu_gib": efficiency["peak_memory_bytes_max"] / 1024**3,
                }
            )
    return rows


def truncation_figure(rows: list[dict[str, Any]], output: Path) -> None:
    width, height = 980, 500
    left, top, chart_w, chart_h = 100, 90, 790, 310
    parts = svg_start(
        width,
        height,
        "Held-out truncation rate",
        "180 questions per model/strategy; Sampling-5 contains 900 traces",
    )
    for tick in range(0, 5):
        value = tick / 10
        y = top + chart_h - value / 0.4 * chart_h
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + chart_w}" y2="{y}" stroke="#e1e5ea"/>')
        parts.append(svg_text(left - 10, y + 5, f"{100*value:.0f}%", text_anchor="end", font_size=13))
    group_w, bar_w = chart_w / 3, 64
    lookup = {(row["model"], row["strategy"]): row for row in rows}
    for index, strategy in enumerate(STRATEGIES):
        center = left + group_w * (index + 0.5)
        for offset, model in ((-40, "2B"), (40, "4B")):
            value = lookup[(model, strategy)]["truncation"]
            bar_h = value / 0.4 * chart_h
            x = center + offset - bar_w / 2
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{MODEL_COLOURS[model]}"/>')
            parts.append(svg_text(x + bar_w / 2, y - 8, f"{100*value:.1f}%", text_anchor="middle", font_size=13))
        parts.append(svg_text(center, top + chart_h + 30, strategy, text_anchor="middle", font_size=15))
    parts.extend(
        [
            '<rect x="700" y="25" width="16" height="16" fill="#d45d5d"/>',
            svg_text(723, 38, "Qwen3-2B", font_size=14),
            '<rect x="820" y="25" width="16" height="16" fill="#4f78b8"/>',
            svg_text(843, 38, "Qwen3-4B", font_size=14),
        ]
    )
    write_svg(output / "heldout_truncation.svg", parts)


def write_tables(
    condition_data: list[dict[str, Any]],
    generation_data: list[dict[str, Any]],
    results: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    with (output / "heldout_generation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(generation_data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(generation_data)
    generation_tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Held-out generation performance and efficiency. Accuracy is reported over completed, parsed traces.}",
        r"\label{tab:heldout-generation}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
        r"Model & Decoding & Records & Complete (\%) & Trunc. (\%) & Parse (\%) & Acc. (\%) & Tokens & Latency (s) & GPU (GiB) \\",
        r"\midrule",
    ]
    for row in generation_data:
        generation_tex.append(
            f"Qwen3-{row['model']} & {row['strategy']} & {row['records']} & {pct(row['completed'])} & "
            f"{pct(row['truncation'])} & {pct(row['parse_coverage'])} & {pct(row['accuracy_scored'])} & "
            f"{row['mean_tokens']:.1f} & {row['median_latency_s']:.1f} & {row['peak_gpu_gib']:.2f} " + r"\\"
        )
    generation_tex.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (output / "heldout_generation_results.tex").write_text("\n".join(generation_tex) + "\n")

    rhd_tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Held-out RHD discrimination under frozen development calibration. Confidence intervals use 2,000 domain-stratified, sample-clustered bootstrap replicates.}",
        r"\label{tab:heldout-rhd}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Model & Decoding & Scored & Base AUC & RHD AUC [95\% CI] & Combined AUC & $\Delta_{\mathrm{inc}}$ [95\% CI] & RHD BA \\",
        r"\midrule",
    ]
    for row in condition_data:
        lo, hi = row["rhd_ci"]
        gain_lo, gain_hi = row["gain_ci"]
        rhd_tex.append(
            f"Qwen3-{row['model']} & {row['strategy']} & {row['scored']} & {row['baseline']:.3f} & "
            f"{row['rhd']:.3f} [{lo:.3f}, {hi:.3f}] & {row['combined']:.3f} & "
            f"{row['gain']:+.3f} [{gain_lo:+.3f}, {gain_hi:+.3f}] & "
            f"{row['balanced_accuracy']:.3f} " + r"\\"
        )
    rhd_tex.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (output / "heldout_rhd_results.tex").write_text("\n".join(rhd_tex) + "\n")

    operating_rows = []
    operating_tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Held-out class prevalence, AUPRC, and frozen-threshold performance. The positive class is an incorrect final answer; confidence intervals use the same clustered bootstrap as AUROC.}",
        r"\label{tab:heldout-operating}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Model & Decoding & Scored & Incorrect (\%) & RHD AUPRC [95\% CI] & RHD BA [95\% CI] & Threshold \\",
        r"\midrule",
    ]
    for row in condition_data:
        auprc_lo, auprc_hi = row["auprc_ci"]
        ba_lo, ba_hi = row["balanced_accuracy_ci"]
        threshold = results["conditions"][row["condition"]]["models"]["rhd_only"]["overall"]["threshold"]
        prevalence = row["incorrect"] / row["scored"]
        operating_rows.append(
            {
                "model": row["model"], "strategy": row["strategy"], "scored": row["scored"],
                "incorrect_prevalence": prevalence, "rhd_auprc": row["auprc"],
                "rhd_auprc_ci_low": auprc_lo, "rhd_auprc_ci_high": auprc_hi,
                "rhd_balanced_accuracy": row["balanced_accuracy"],
                "rhd_balanced_accuracy_ci_low": ba_lo, "rhd_balanced_accuracy_ci_high": ba_hi,
                "frozen_threshold": threshold,
            }
        )
        operating_tex.append(
            f"Qwen3-{row['model']} & {row['strategy']} & {row['scored']} & {100*prevalence:.1f} & "
            f"{row['auprc']:.3f} [{auprc_lo:.3f}, {auprc_hi:.3f}] & "
            f"{row['balanced_accuracy']:.3f} [{ba_lo:.3f}, {ba_hi:.3f}] & {threshold:.3f} " + r"\\"
        )
    operating_tex.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (output / "heldout_operating_results.tex").write_text("\n".join(operating_tex) + "\n")
    with (output / "heldout_operating_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operating_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(operating_rows)

    self_consistency_rows = []
    self_consistency_tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Question-level Sampling-5 self-consistency on the held-out split. Accuracy over all questions treats uncovered questions as unanswered; predicted accuracy is conditional on a valid majority-vote prediction.}",
        r"\label{tab:heldout-self-consistency}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Model & Questions & Coverage (\%) & Accuracy all (\%) & Accuracy predicted (\%) & Strict majority & Ties \\",
        r"\midrule",
    ]
    for model in ("2B", "4B"):
        values = comparisons[model]["self_consistency"]
        row = {
            "model": model, "questions": values["samples"],
            "prediction_coverage": values["prediction_coverage"],
            "accuracy_all": values["exact_match_rate_all"],
            "accuracy_predicted": values["exact_match_rate_predicted"],
            "strict_majority_count": values["strict_majority_count"], "tie_count": values["tie_count"],
        }
        self_consistency_rows.append(row)
        self_consistency_tex.append(
            f"Qwen3-{model} & {row['questions']} & {100*row['prediction_coverage']:.1f} & "
            f"{100*row['accuracy_all']:.1f} & {100*row['accuracy_predicted']:.1f} & "
            f"{row['strict_majority_count']} & {row['tie_count']} " + r"\\"
        )
    self_consistency_tex.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (output / "heldout_self_consistency.tex").write_text("\n".join(self_consistency_tex) + "\n")
    with (output / "heldout_self_consistency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(self_consistency_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(self_consistency_rows)

    domain_rows = []
    domain_tex = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Domain-stratified held-out AUROC. Domain results are secondary and use no domain-specific refitting.}",
        r"\label{tab:heldout-domain}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrrr}", r"\toprule",
        r"Model & Decoding & Domain & Scored & Base AUC & RHD AUC & Combined AUC \\", r"\midrule",
    ]
    for condition in ORDER:
        model = "2B" if "2B" in condition else "4B"
        strategy = condition.split()[-1]
        condition_result = results["conditions"][condition]
        for domain in DOMAINS:
            baseline = condition_result["models"]["length_step_baseline"]["by_domain"][domain]
            rhd = condition_result["models"]["rhd_only"]["by_domain"][domain]
            combined = condition_result["models"]["rhd_plus_length_step"]["by_domain"][domain]
            row = {
                "model": model, "strategy": strategy, "domain": domain,
                "scored": rhd["records"], "baseline_auroc": baseline["auroc"],
                "rhd_auroc": rhd["auroc"], "combined_auroc": combined["auroc"],
            }
            domain_rows.append(row)
            domain_tex.append(
                f"Qwen3-{model} & {strategy} & {domain} & {row['scored']} & "
                f"{fmt(row['baseline_auroc'])} & {fmt(row['rhd_auroc'])} & "
                f"{fmt(row['combined_auroc'])} " + r"\\"
            )
        domain_tex.append(r"\addlinespace")
    domain_tex.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (output / "heldout_domain_results.tex").write_text("\n".join(domain_tex) + "\n")
    with (output / "heldout_domain_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(domain_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(domain_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-results", type=Path, required=True)
    parser.add_argument("--development-matrix", type=Path, required=True)
    parser.add_argument("--comparison-2b", type=Path, required=True)
    parser.add_argument("--comparison-4b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = load(args.heldout_results)
    rows = condition_rows(results)
    comparisons = {"2B": load(args.comparison_2b), "4B": load(args.comparison_4b)}
    generations = generation_rows(comparisons)
    auroc_forest(rows, args.output_dir)
    gain_forest(rows, args.output_dir)
    development_comparison(rows, args.development_matrix, args.output_dir)
    domain_heatmap(results, args.output_dir)
    truncation_figure(generations, args.output_dir)
    write_tables(rows, generations, results, comparisons, args.output_dir)
    manifest = {
        "schema_version": "1.0",
        "heldout_results": str(args.heldout_results.resolve()),
        "development_matrix": str(args.development_matrix.resolve()),
        "comparison_2b": str(args.comparison_2b.resolve()),
        "comparison_4b": str(args.comparison_4b.resolve()),
        "figures": [
            "heldout_auroc_ci.svg", "heldout_incremental_gain.svg",
            "development_heldout_auroc.svg", "heldout_domain_auroc.svg",
            "heldout_truncation.svg",
        ],
        "tables": [
            "heldout_generation_results.tex", "heldout_rhd_results.tex",
            "heldout_domain_results.tex", "heldout_generation_results.csv",
            "heldout_domain_results.csv", "heldout_operating_results.tex",
            "heldout_operating_results.csv", "heldout_self_consistency.tex",
            "heldout_self_consistency.csv",
        ],
    }
    (args.output_dir / "publication_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
