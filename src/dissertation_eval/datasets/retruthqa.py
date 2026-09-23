from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ..errors import DataIntegrityError
from ..schema import DatasetExample, DatasetManifest


_FILENAME = re.compile(r"^(?P<domain>.+)_(?P<task>task[12])\.jsonl$")


def _required(row: dict[str, Any], name: str, source: Path) -> Any:
    if name not in row or row[name] in (None, ""):
        raise DataIntegrityError(f"Missing required ReTruthQA field {name!r} in {source}")
    return row[name]


def _reasoning_paths(row: dict[str, Any], name: str, source: Path) -> list[list[str]]:
    value = _required(row, name, source)
    if not isinstance(value, list) or not value:
        raise DataIntegrityError(f"ReTruthQA field {name!r} must be a non-empty list")
    if any(
        not isinstance(path, list)
        or not path
        or any(not isinstance(step, str) or not step for step in path)
        for path in value
    ):
        raise DataIntegrityError(
            f"ReTruthQA field {name!r} must contain non-empty lists of text steps"
        )
    return value


def _boxed_answers(text: str) -> list[str]:
    """Extract balanced ``\boxed{...}`` values, including nested LaTeX braces."""

    marker = "\\boxed{"
    answers: list[str] = []
    cursor = 0
    while True:
        marker_start = text.find(marker, cursor)
        if marker_start < 0:
            return answers
        value_start = marker_start + len(marker)
        depth = 1
        index = value_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            raise DataIntegrityError("Unbalanced \\boxed answer in a ReTruthQA positive path")
        answer = text[value_start:index - 1].strip()
        if not answer:
            raise DataIntegrityError("Empty \\boxed answer in a ReTruthQA positive path")
        answers.append(answer)
        cursor = index


def _positive_answers(paths: list[list[str]], source: Path) -> list[str]:
    answers = []
    for path in paths:
        boxed = _boxed_answers("\n".join(path))
        if not boxed:
            raise DataIntegrityError(
                f"Positive ReTruthQA path has no \\boxed answer in {source}"
            )
        answers.append(boxed[-1])
    return answers


def _source_metadata(data_path: Path) -> tuple[str, str, str]:
    match = _FILENAME.match(data_path.name)
    if not match or data_path.parent.name != "hallucination":
        raise DataIntegrityError(f"Unexpected ReTruthQA file layout: {data_path}")
    source_model = data_path.parent.parent.name
    if not source_model:
        raise DataIntegrityError(f"Cannot infer source model from {data_path}")
    return source_model, match.group("domain"), match.group("task")


def normalize_retruthqa(
    row: dict[str, Any], data_path: Path, manifest: DatasetManifest
) -> DatasetExample:
    """Normalize a ReTruthQA reasoning-trace record without copying traces to outputs."""

    question = str(_required(row, manifest.fields.prompt, data_path))
    positive = _reasoning_paths(row, "pos_reasoning_paths_split", data_path)
    negative = _reasoning_paths(row, "neg_reasoning_paths_split", data_path)
    positive_answers = _positive_answers(positive, data_path)
    source_model, domain, task = _source_metadata(data_path)
    question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
    sample_id = f"{source_model}/{domain}/{task}/{question_hash}"
    return DatasetExample(
        sample_id=sample_id,
        prompt=question,
        reference_answer=positive_answers[0],
        image_path=None,
        group_id=question_hash,
        metadata={
            "source_model": source_model,
            "domain": domain,
            "task": task,
            "positive_path_count": len(positive),
            "negative_path_count": len(negative),
            "positive_step_counts": [len(path) for path in positive],
            "negative_step_counts": [len(path) for path in negative],
            "reference_answer_source": "last_boxed_answer_in_positive_path",
            "reference_answer_candidates": positive_answers,
        },
        raw=row,
    )
