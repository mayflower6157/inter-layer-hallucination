from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from ..io import atomic_write_json, atomic_write_jsonl, iter_jsonl
from .retruthqa import evaluate_record, evaluate_retruthqa_generations


def _numeric(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]


def _efficiency(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    tokens = _numeric(rows, "generated_tokens")
    latency = _numeric(rows, "latency_ms")
    memory = _numeric(rows, "peak_memory_bytes")
    return {
        "generated_tokens_mean": mean(tokens) if tokens else None,
        "generated_tokens_median": median(tokens) if tokens else None,
        "latency_ms_mean": mean(latency) if latency else None,
        "latency_ms_median": median(latency) if latency else None,
        "peak_memory_bytes_mean": mean(memory) if memory else None,
        "peak_memory_bytes_max": int(max(memory)) if memory else None,
    }


def _self_consistency(
    raw_rows: list[dict[str, Any]], evaluated_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_by_record = {str(row.get("record_id")): row for row in raw_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        grouped[str(row.get("sample_id"))].append(row)

    results: list[dict[str, Any]] = []
    for sample_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row.get("generation_index") or 0))
        valid = [
            row
            for row in rows
            if row["evaluation_status"] in {"exact_match", "mismatch"}
            and row.get("prediction_normalized") is not None
        ]
        counts = Counter(str(row["prediction_normalized"]) for row in valid)
        winner = None
        tied = False
        strict_majority = False
        if counts:
            highest = max(counts.values())
            leaders = {answer for answer, count in counts.items() if count == highest}
            tied = len(leaders) > 1
            # Deterministic tie rule: take the tied answer produced by the lowest
            # generation index. The tie remains explicitly reported for sensitivity checks.
            winner = next(
                str(row["prediction_normalized"])
                for row in valid
                if str(row["prediction_normalized"]) in leaders
            )
            strict_majority = highest > len(valid) / 2
        reference = next(
            (row.get("reference_normalized") for row in rows if row.get("reference_normalized")),
            None,
        )
        exact_match = winner == reference if winner is not None and reference is not None else None
        source_rows = [raw_by_record[str(row.get("record_id"))] for row in rows]
        latencies = _numeric(source_rows, "latency_ms")
        tokens = _numeric(source_rows, "generated_tokens")
        memory = _numeric(source_rows, "peak_memory_bytes")
        results.append(
            {
                "sample_id": sample_id,
                "generation_count": len(rows),
                "valid_vote_count": len(valid),
                "vote_counts": dict(sorted(counts.items())),
                "prediction_normalized": winner,
                "reference_normalized": reference,
                "exact_match": exact_match,
                "tie_broken_by_lowest_generation_index": tied,
                "strict_majority": strict_majority,
                "generated_tokens_total": int(sum(tokens)),
                "latency_ms_total": sum(latencies),
                "peak_memory_bytes_max": int(max(memory)) if memory else None,
            }
        )

    sample_count = len(results)
    predictions = sum(row["prediction_normalized"] is not None for row in results)
    correct = sum(row["exact_match"] is True for row in results)
    summary = {
        "samples": sample_count,
        "predictions": predictions,
        "prediction_coverage": predictions / sample_count if sample_count else None,
        "exact_matches": correct,
        "exact_match_rate_all": correct / sample_count if sample_count else None,
        "exact_match_rate_predicted": correct / predictions if predictions else None,
        "tie_count": sum(row["tie_broken_by_lowest_generation_index"] for row in results),
        "strict_majority_count": sum(row["strict_majority"] for row in results),
        "generations_per_sample_min": min(
            (row["generation_count"] for row in results), default=None
        ),
        "generations_per_sample_max": max(
            (row["generation_count"] for row in results), default=None
        ),
        "generated_tokens_total_mean": mean(
            row["generated_tokens_total"] for row in results
        ) if results else None,
        "latency_ms_total_mean": mean(row["latency_ms_total"] for row in results)
        if results else None,
        "latency_ms_total_median": median(row["latency_ms_total"] for row in results)
        if results else None,
        "peak_memory_bytes_max": max(
            (row["peak_memory_bytes_max"] for row in results if row["peak_memory_bytes_max"]),
            default=None,
        ),
        "tie_rule": "lowest generation_index among answers tied for the highest vote count",
    }
    return summary, results


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _number(value: float | int | None, divisor: float = 1.0, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value) / divisor:.{digits}f}"


def _markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Strategy | Records | Samples | Completion | Truncation | Parse coverage | Exact match (scored) | Exact match (all) | Mean tokens | Median latency (s) | Peak GPU (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in summary["strategies"].items():
        evaluation = row["evaluation"]
        efficiency = row["efficiency"]
        records = evaluation["records"]
        truncation = evaluation["status_counts"].get("truncated", 0) / records
        completion = evaluation["completed"] / records
        lines.append(
            f"| {label} | {records} | {row['unique_samples']} | {_percent(completion)} | "
            f"{_percent(truncation)} | {_percent(evaluation['parse_coverage_completed'])} | "
            f"{_percent(evaluation['exact_match_rate_scored'])} | "
            f"{_percent(evaluation['exact_match_rate_all'])} | "
            f"{_number(efficiency['generated_tokens_mean'])} | "
            f"{_number(efficiency['latency_ms_median'], 1000)} | "
            f"{_number(efficiency['peak_memory_bytes_max'], 1024 ** 3, 2)} |"
        )
    if "self_consistency" in summary:
        row = summary["self_consistency"]
        lines.append(
            f"| {row['source_strategy']} self-consistency | — | {row['samples']} | — | — | "
            f"{_percent(row['prediction_coverage'])} | "
            f"{_percent(row['exact_match_rate_predicted'])} | "
            f"{_percent(row['exact_match_rate_all'])} | "
            f"{_number(row['generated_tokens_total_mean'])} | "
            f"{_number(row['latency_ms_total_median'], 1000)} | "
            f"{_number(row['peak_memory_bytes_max'], 1024 ** 3, 2)} |"
        )
    return "\n".join(lines) + "\n"


def compare_retruthqa_runs(
    runs: dict[str, Path], output_dir: Path, *, self_consistency: str | None = None
) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one labelled generation run is required")
    output_dir = output_dir.resolve()
    strategies: dict[str, Any] = {}
    cached: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for label, path in runs.items():
        raw_rows = list(iter_jsonl(path.resolve()))
        if not raw_rows:
            raise ValueError(f"Generation file contains no records: {path}")
        evaluation = evaluate_retruthqa_generations(path, output_dir / label)
        evaluated = [evaluate_record(row) for row in raw_rows]
        cached[label] = raw_rows, evaluated
        strategies[label] = {
            "input_path": str(path.resolve()),
            "unique_samples": len({str(row.get("sample_id")) for row in raw_rows}),
            "evaluation": evaluation,
            "efficiency": _efficiency(raw_rows),
        }

    summary: dict[str, Any] = {"schema_version": "1.0", "strategies": strategies}
    if self_consistency is not None:
        if self_consistency not in cached:
            raise ValueError(f"Unknown self-consistency run label: {self_consistency}")
        sc_summary, sc_rows = _self_consistency(*cached[self_consistency])
        sc_summary["source_strategy"] = self_consistency
        summary["self_consistency"] = sc_summary
        atomic_write_jsonl(output_dir / "self-consistency-records.jsonl", sc_rows)

    atomic_write_json(output_dir / "comparison.json", summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.md").write_text(_markdown_table(summary), encoding="utf-8")
    return summary
