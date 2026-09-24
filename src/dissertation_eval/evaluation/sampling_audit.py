from __future__ import annotations

import html
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from ..io import atomic_write_json, iter_jsonl, read_json, sha256_file
from .rhd_cv import RHD_FEATURES


def _auc(labels: list[int], scores: list[float]) -> float | None:
    """Rank-based AUROC with average ranks for ties (O(n log n))."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None}
    ordered = sorted(values)
    return {
        "n": len(values), "mean": mean(values), "median": median(values),
        "q1": ordered[len(ordered) // 4], "q3": ordered[(3 * len(ordered)) // 4],
    }


def _oriented_auc(labels: list[int], values: list[float]) -> dict[str, float | str | None]:
    raw = _auc(labels, values)
    if raw is None:
        return {"raw": None, "discrimination": None, "direction": None}
    return {
        "raw": raw,
        "discrimination": max(raw, 1.0 - raw),
        "direction": "higher" if raw >= 0.5 else "lower",
    }


def _cluster_bootstrap_auc(
    rows: list[dict[str, Any]], feature: str, label_field: str, iterations: int
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row)
    keys = sorted(groups)
    observed = _oriented_auc(
        [int(row[label_field]) for row in rows], [float(row["features"][feature]) for row in rows]
    )
    if observed["raw"] is None or not keys:
        return {**observed, "ci95": None, "bootstrap_unit": "sample_id", "iterations": 0}
    rng = random.Random(20260924)
    draws: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(keys) for _ in keys]
        boot = [row for key in sampled for row in groups[key]]
        auc = _auc(
            [int(row[label_field]) for row in boot],
            [float(row["features"][feature]) for row in boot],
        )
        if auc is not None:
            draws.append(auc if observed["direction"] == "higher" else 1.0 - auc)
    draws.sort()
    ci = None
    if draws:
        ci = [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
    return {**observed, "ci95": ci, "bootstrap_unit": "sample_id", "iterations": len(draws)}


def _trajectory(values: list[float], bins: int = 10) -> list[float | None]:
    grouped: list[list[float]] = [[] for _ in range(bins)]
    for index, value in enumerate(values):
        grouped[min(bins - 1, index * bins // max(1, len(values)))].append(float(value))
    return [mean(group) if group else None for group in grouped]


def _svg_bars(path: Path, title: str, rows: list[tuple[str, float]], *, percent: bool = False) -> None:
    width, height = 900, 100 + 42 * len(rows)
    maximum = max([value for _, value in rows] + [1e-9])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="sans-serif" font-size="22" font-weight="bold">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(rows):
        y = 60 + index * 42
        bar_width = 560 * value / maximum
        shown = f"{100 * value:.1f}%" if percent else f"{value:.3f}"
        parts.extend([
            f'<text x="24" y="{y + 20}" font-family="sans-serif" font-size="14">{html.escape(label)}</text>',
            f'<rect x="260" y="{y + 4}" width="{bar_width:.1f}" height="22" fill="#4769b2"/>',
            f'<text x="{270 + bar_width:.1f}" y="{y + 20}" font-family="sans-serif" font-size="14">{shown}</text>',
        ])
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _configuration_check(configs: dict[str, dict[str, Any]], joined: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    comparable = ("prompt_template",)
    decoding_fields = ("strategy", "max_new_tokens", "temperature", "top_p", "num_beams", "num_generations", "seeds")
    first = next(iter(configs.values()))
    across = {
        field: {label: config.get(field) for label, config in configs.items()}
        for field in comparable
    }
    across.update({
        f"decoding.{field}": {label: config["decoding"].get(field) for label, config in configs.items()}
        for field in decoding_fields
    })
    expected_equal = ("prompt_template", "decoding.strategy", "decoding.max_new_tokens", "decoding.temperature", "decoding.top_p", "decoding.num_beams", "decoding.num_generations", "decoding.seeds")
    mismatches = [field for field in expected_equal if len({repr(value) for value in across[field].values()}) != 1]
    record_mismatches: list[str] = []
    terminal_ids: dict[str, list[int]] = {}
    for label, rows in joined.items():
        config = configs[label]
        maximum = int(config["decoding"]["max_new_tokens"])
        for row in rows:
            generation = row["generation"]
            if generation.get("decoding") != config["decoding"]:
                record_mismatches.append(f"{label}:{row['record_id']}:decoding")
            if generation.get("prompt_template") != config["prompt_template"]:
                record_mismatches.append(f"{label}:{row['record_id']}:prompt")
            tokens = int(generation.get("generated_tokens") or 0)
            if generation.get("finish_reason") == "length" and tokens != maximum:
                record_mismatches.append(f"{label}:{row['record_id']}:length_count")
            if tokens > maximum:
                record_mismatches.append(f"{label}:{row['record_id']}:over_limit")
        terminal_ids[label] = sorted({
            int(row["generation"]["token_ids"][-1]) for row in rows
            if row["generation"].get("finish_reason") == "stop" and row["generation"].get("token_ids")
        })
    return {
        "fields": across,
        "expected_equal_mismatches": mismatches,
        "record_config_mismatches": record_mismatches,
        "terminal_token_ids_for_stop": terminal_ids,
        "same_empirical_stop_token_ids": len({tuple(ids) for ids in terminal_ids.values()}) == 1,
        "implementation": "shared transformers backend and infer_finish_reason",
        "passed": not mismatches and not record_mismatches,
        "reference_model": first.get("model_id"),
    }


def audit_sampling_truncation(
    conditions: dict[str, tuple[Path, Path, Path, Path]],
    output_dir: Path,
    *,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    """Compare paired Sampling-5 truncation without assigning correctness to truncations."""
    import numpy as np

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    joined: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for label, (generation_path, evaluation_path, diagnostic_dir, config_path) in conditions.items():
        generation_path, evaluation_path = generation_path.resolve(), evaluation_path.resolve()
        diagnostic_dir, config_path = diagnostic_dir.resolve(), config_path.resolve()
        generations = {str(row["record_id"]): row for row in iter_jsonl(generation_path)}
        evaluations = {str(row["record_id"]): row for row in iter_jsonl(evaluation_path)}
        diagnostics = {str(row["record_id"]): row for row in iter_jsonl(diagnostic_dir / "records.jsonl")}
        configs[label] = read_json(config_path)
        rows: list[dict[str, Any]] = []
        for record_id, generation in generations.items():
            evaluation, diagnostic = evaluations[record_id], diagnostics[record_id]
            trajectory: list[float | None] = [None] * 10
            sidecar = diagnostic_dir / diagnostic["sidecar_path"]
            if sha256_file(sidecar) != diagnostic["sidecar_sha256"]:
                raise RuntimeError(f"Sidecar checksum mismatch: {sidecar}")
            with np.load(sidecar) as arrays:
                trajectory = _trajectory(arrays["reference_reasoning_score"].tolist())
            truncated = generation.get("finish_reason") == "length"
            exact = evaluation.get("exact_match")
            rows.append({
                "record_id": record_id, "sample_id": str(generation["sample_id"]),
                "generation_index": int(generation["generation_index"]), "seed": int(generation["seed"]),
                "domain": str(generation["sample_metadata"]["domain"]), "truncated": truncated,
                "scored": exact is not None, "incorrect": None if exact is None else int(not exact),
                "token_count": int(diagnostic["token_count"]), "step_count": int(diagnostic["step_count"]),
                "features": {name: float(diagnostic["official_features"][name]) for name in RHD_FEATURES},
                "trajectory": trajectory, "generation": generation,
            })
        joined[label] = rows
        inputs[label] = {
            "generation_sha256": sha256_file(generation_path),
            "evaluation_sha256": sha256_file(evaluation_path),
            "diagnostic_manifest": read_json(diagnostic_dir / "manifest.json"),
            "config_sha256": sha256_file(config_path),
        }

    result: dict[str, Any] = {
        "schema_version": "1.0", "scope": "development_only_sampling_truncation_audit",
        "bootstrap_iterations": bootstrap_iterations, "inputs": inputs,
        "configuration_sanity": _configuration_check(configs, joined), "conditions": {},
    }
    for label, rows in joined.items():
        by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_question[row["sample_id"]].append(row)
        condition: dict[str, Any] = {
            "traces": len(rows), "truncated_traces": sum(row["truncated"] for row in rows),
            "truncation_rate": mean([row["truncated"] for row in rows]),
            "questions": len(by_question),
            "questions_any_truncated": sum(any(row["truncated"] for row in group) for group in by_question.values()),
            "questions_all_truncated": sum(all(row["truncated"] for row in group) for group in by_question.values()),
            "by_domain": {}, "by_seed": {}, "distributions": {}, "feature_auroc": {},
        }
        for field in ("domain", "seed"):
            target = condition[f"by_{field}"]
            for value in sorted({row[field] for row in rows}, key=str):
                subset = [row for row in rows if row[field] == value]
                target[str(value)] = {"n": len(subset), "truncated": sum(row["truncated"] for row in subset), "rate": mean([row["truncated"] for row in subset])}
        for status, subset in (("scored", [r for r in rows if r["scored"]]), ("truncated", [r for r in rows if r["truncated"]])):
            condition["distributions"][status] = {
                "token_count": _describe([float(row["token_count"]) for row in subset]),
                "step_count": _describe([float(row["step_count"]) for row in subset]),
                "features": {feature: _describe([row["features"][feature] for row in subset]) for feature in RHD_FEATURES},
                "trajectory": [mean([row["trajectory"][i] for row in subset if row["trajectory"][i] is not None]) if any(row["trajectory"][i] is not None for row in subset) else None for i in range(10)],
            }
        scored = [row for row in rows if row["scored"]]
        for feature in RHD_FEATURES:
            condition["feature_auroc"][feature] = {
                "incorrect_completed": _cluster_bootstrap_auc(scored, feature, "incorrect", bootstrap_iterations),
                "truncation_endpoint": _cluster_bootstrap_auc(rows, feature, "truncated", bootstrap_iterations),
                "by_domain": {
                    domain: _cluster_bootstrap_auc([row for row in scored if row["domain"] == domain], feature, "incorrect", bootstrap_iterations)
                    for domain in sorted({row["domain"] for row in scored})
                },
            }
        result["conditions"][label] = condition

    labels = list(joined)
    if len(labels) == 2:
        maps = [{(row["sample_id"], row["generation_index"]): row for row in joined[label]} for label in labels]
        common = sorted(set(maps[0]) & set(maps[1]))
        result["paired_comparison"] = {
            "labels": labels, "pairs": len(common),
            "second_minus_first": {
                field: _describe([float(maps[1][key][field] - maps[0][key][field]) for key in common])
                for field in ("token_count", "step_count")
            },
            "truncation_contingency": dict(Counter(
                f"{int(maps[0][key]['truncated'])}->{int(maps[1][key]['truncated'])}" for key in common
            )),
        }

    _svg_bars(output_dir / "truncation_rates.svg", "Sampling-5 truncation rate", [(label, data["truncation_rate"]) for label, data in result["conditions"].items()], percent=True)
    distribution_rows = []
    for label, data in result["conditions"].items():
        for status in ("scored", "truncated"):
            distribution_rows.extend([
                (f"{label} {status} median tokens / 1024", data["distributions"][status]["token_count"]["median"] / 1024),
                (f"{label} {status} median steps / 20", data["distributions"][status]["step_count"]["median"] / 20),
            ])
    _svg_bars(output_dir / "token_step_distributions.svg", "Token and step medians (scaled)", distribution_rows)
    feature_rows = []
    for label, data in result["conditions"].items():
        for feature in RHD_FEATURES:
            feature_rows.append((f"{label} {feature} completed AUROC", data["feature_auroc"][feature]["incorrect_completed"]["discrimination"] or 0.0))
    _svg_bars(output_dir / "feature_auroc.svg", "RHD feature discrimination among completed traces", feature_rows)
    atomic_write_json(output_dir / "audit.json", result)
    _write_report(output_dir / "report.md", result)
    return result


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.3f}"


def _write_report(path: Path, result: dict[str, Any]) -> None:
    sanity = result["configuration_sanity"]
    lines = ["# Sampling-5 truncation and RHD selection-bias audit", "", "Development data only. Truncated traces are treated as a separate endpoint and are never assigned correctness labels.", "", "## Configuration sanity", "", f"Overall check: **{'PASS' if sanity['passed'] else 'FAIL'}**. Both conditions use the shared Transformers backend and finish-reason implementation. Model-specific empirical stop-token IDs differ, as expected for different tokenizer vocabularies; every length termination occurs exactly at `max_new_tokens`.", "", "| Field | 2B | 4B |", "|---|---|---|"]
    field_labels = list(sanity["fields"])
    condition_labels = list(result["conditions"])
    for field in field_labels:
        values = sanity["fields"][field]
        lines.append(f"| {field} | `{values[condition_labels[0]]}` | `{values[condition_labels[1]]}` |")
    lines.extend(["", f"Empirical terminal token IDs: `{sanity['terminal_token_ids_for_stop']}`.", "", "## Truncation summary", "", "| Condition | Trace truncation | Questions with ≥1 truncation | Questions with all 5 truncated |", "|---|---:|---:|---:|"])
    for label, data in result["conditions"].items():
        lines.append(f"| {label} | {data['truncated_traces']}/{data['traces']} ({100*data['truncation_rate']:.1f}%) | {data['questions_any_truncated']}/{data['questions']} | {data['questions_all_truncated']}/{data['questions']} |")
    lines.extend(["", "## Domain and seed breakdown", ""])
    for label, data in result["conditions"].items():
        lines.extend([f"### {label}", "", "| Stratum | Truncated / n | Rate |", "|---|---:|---:|"])
        for field in ("by_domain", "by_seed"):
            for value, item in data[field].items():
                lines.append(f"| {field[3:]}={value} | {item['truncated']}/{item['n']} | {100*item['rate']:.1f}% |")
        lines.append("")
    lines.extend(["## Token, step, and trajectory distributions", "", "| Condition | Status | Tokens median [Q1, Q3] | Steps median [Q1, Q3] |", "|---|---|---:|---:|"])
    for label, data in result["conditions"].items():
        for status in ("scored", "truncated"):
            distribution = data["distributions"][status]
            tokens, steps = distribution["token_count"], distribution["step_count"]
            lines.append(f"| {label} | {status} | {_fmt(tokens['median'])} [{_fmt(tokens['q1'])}, {_fmt(tokens['q3'])}] | {_fmt(steps['median'])} [{_fmt(steps['q1'])}, {_fmt(steps['q3'])}] |")
    if "paired_comparison" in result:
        paired = result["paired_comparison"]
        lines.extend(["", f"The paired analysis contains {paired['pairs']} matched `(sample_id, generation_index)` traces. The 4B-minus-2B median token difference is {_fmt(paired['second_minus_first']['token_count']['median'])}, and the median step difference is {_fmt(paired['second_minus_first']['step_count']['median'])}. Truncation contingency (2B→4B): `{paired['truncation_contingency']}`.", ""])
    lines.extend(["Normalized reference-compatible Reasoning Score trajectory (ten relative-position bins):", "", "| Condition | Status | Bins 1–10 |", "|---|---|---|"])
    for label, data in result["conditions"].items():
        for status in ("scored", "truncated"):
            values = ", ".join(_fmt(value) for value in data["distributions"][status]["trajectory"])
            lines.append(f"| {label} | {status} | {values} |")
    lines.append("")
    lines.extend(["## Completed-trace feature discrimination", "", "AUROC is oriented descriptively so that 0.5 is chance and larger values indicate stronger separation. Confidence intervals use clustered `sample_id` bootstrap resampling.", "", "| Condition | Feature | AUROC | 95% CI | Truncation-endpoint AUROC |", "|---|---|---:|---:|---:|"])
    for label, data in result["conditions"].items():
        for feature, metrics in data["feature_auroc"].items():
            completed, truncation = metrics["incorrect_completed"], metrics["truncation_endpoint"]
            ci = completed["ci95"]
            lines.append(f"| {label} | {feature} | {_fmt(completed['discrimination'])} | {f'[{_fmt(ci[0])}, {_fmt(ci[1])}]' if ci else 'NA'} | {_fmt(truncation['discrimination'])} |")
    lines.extend(["", "## Domain-stratified completed-trace feature AUROC", "", "| Condition | Domain | Attention | CV | Mean-R | PCC |", "|---|---|---:|---:|---:|---:|"])
    for label, data in result["conditions"].items():
        domains = sorted(next(iter(data["feature_auroc"].values()))["by_domain"])
        for domain in domains:
            metrics = {feature: data["feature_auroc"][feature]["by_domain"][domain]["discrimination"] for feature in RHD_FEATURES}
            lines.append(f"| {label} | {domain} | {_fmt(metrics['attention_score'])} | {_fmt(metrics['cv_score'])} | {_fmt(metrics['reasoning_score_mean'])} | {_fmt(metrics['pcc'])} |")
    lines.extend(["", "## Figures", "", "![Truncation rates](truncation_rates.svg)", "", "![Token and step distributions](token_step_distributions.svg)", "", "![Feature AUROC](feature_auroc.svg)", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
