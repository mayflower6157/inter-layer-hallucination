from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..datasets.prepare import normalize_reference_answer
from ..io import atomic_write_json, atomic_write_jsonl, iter_jsonl, sha256_file


EVALUATOR_VERSION = "1.0.1"
_BOXED_START = re.compile(r"\\boxed\s*\{")
_FINAL_MARKER = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?\*{0,2}final(?:\s+answer)?\*{0,2}\s*[:：]?\s*",
    re.IGNORECASE,
)


def extract_boxed_answers(text: str) -> list[str]:
    """Return every complete balanced ``\\boxed{...}`` value.

    Incomplete trailing boxes are ignored because generated text may be truncated;
    earlier complete answers remain auditable.
    """

    answers: list[str] = []
    cursor = 0
    while match := _BOXED_START.search(text, cursor):
        value_start = match.end()
        depth = 1
        index = value_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            break
        answer = text[value_start:index - 1].strip()
        if answer:
            answers.append(answer)
        cursor = index
    return answers


def _clean_final_value(value: str) -> str | None:
    line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    line = line.strip("` ")
    for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if line.startswith(left) and line.endswith(right) and len(line) > len(left) + len(right):
            line = line[len(left):-len(right)].strip()
    return line or None


def extract_prediction(text: str) -> tuple[str | None, str | None]:
    boxed = extract_boxed_answers(text)
    if boxed:
        return boxed[-1], "last_boxed_answer"
    markers = list(_FINAL_MARKER.finditer(text))
    if markers:
        value = _clean_final_value(text[markers[-1].end():])
        if value:
            return value, "final_marker"
    return None, None


def normalize_evaluation_answer(answer: str, domain: str) -> str:
    value = normalize_reference_answer(answer, domain)
    if domain.casefold() == "math":
        wrapper = re.fullmatch(r"\\(?:text|mathrm)\{(.+)\}", value)
        if wrapper:
            return wrapper.group(1)
    return value


def _evaluation_id(record_id: str) -> str:
    return hashlib.sha256(f"{EVALUATOR_VERSION}\0{record_id}".encode()).hexdigest()


def evaluate_record(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response_text") or ""
    reference = row.get("reference_answer")
    domain = str(row.get("sample_metadata", {}).get("domain", ""))
    prediction, prediction_source = extract_prediction(response)
    prediction_normalized = (
        normalize_evaluation_answer(prediction, domain) if prediction is not None else None
    )
    reference_normalized = (
        normalize_evaluation_answer(str(reference), domain) if reference is not None else None
    )

    if row.get("status") != "success":
        evaluation_status = "generation_error"
        exact_match = None
    elif row.get("finish_reason") == "length":
        evaluation_status = "truncated"
        exact_match = None
    elif reference_normalized is None:
        evaluation_status = "missing_reference"
        exact_match = None
    elif prediction_normalized is None:
        evaluation_status = "parse_failure"
        exact_match = None
    else:
        exact_match = prediction_normalized == reference_normalized
        evaluation_status = "exact_match" if exact_match else "mismatch"

    record_id = str(row.get("record_id", ""))
    return {
        "schema_version": "1.0",
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_id": _evaluation_id(record_id),
        "record_id": record_id,
        "run_id": row.get("run_id"),
        "sample_id": row.get("sample_id"),
        "generation_index": row.get("generation_index"),
        "domain": domain,
        "generation_status": row.get("status"),
        "finish_reason": row.get("finish_reason"),
        "generated_tokens": row.get("generated_tokens"),
        "prediction": prediction,
        "prediction_source": prediction_source,
        "prediction_normalized": prediction_normalized,
        "reference_answer": reference,
        "reference_normalized": reference_normalized,
        "evaluation_status": evaluation_status,
        "exact_match": exact_match,
        "needs_manual_review": evaluation_status in {"truncated", "parse_failure", "mismatch"},
    }


def evaluate_retruthqa_generations(input_path: Path, output_dir: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    rows = list(iter_jsonl(input_path))
    if not rows:
        raise ValueError(f"Generation file contains no records: {input_path}")
    wrong_dataset = sorted({str(row.get("dataset_id")) for row in rows} - {"retruthqa"})
    if wrong_dataset:
        raise ValueError(f"Expected ReTruthQA generation records, found: {wrong_dataset}")

    evaluations = [evaluate_record(row) for row in rows]
    statuses = Counter(row["evaluation_status"] for row in evaluations)
    completed = sum(
        row["generation_status"] == "success" and row["finish_reason"] != "length"
        for row in evaluations
    )
    parsed_completed = statuses["exact_match"] + statuses["mismatch"]
    exact_matches = statuses["exact_match"]
    summary = {
        "schema_version": "1.0",
        "evaluator_version": EVALUATOR_VERSION,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "records": len(evaluations),
        "completed": completed,
        "parsed_completed": parsed_completed,
        "exact_matches": exact_matches,
        "status_counts": dict(sorted(statuses.items())),
        "parse_coverage_completed": parsed_completed / completed if completed else None,
        "exact_match_rate_scored": exact_matches / parsed_completed if parsed_completed else None,
        "exact_match_rate_all": exact_matches / len(evaluations),
        "manual_review_count": sum(row["needs_manual_review"] for row in evaluations),
    }
    atomic_write_jsonl(output_dir / "records.jsonl", evaluations)
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
