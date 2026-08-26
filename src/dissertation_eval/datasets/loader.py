from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ..errors import DataIntegrityError
from ..io import iter_jsonl, read_json, sha256_file
from ..schema import DatasetExample, DatasetManifest


def load_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.from_dict(read_json(path))


def _rows(path: Path, file_format: str) -> Iterator[dict[str, Any]]:
    if file_format == "jsonl":
        yield from iter_jsonl(path)
        return
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict):
        value = value.get("data", value.get("examples"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise DataIntegrityError(f"Expected a JSON list (or data/examples list) in {path}")
    yield from value


def _required(row: dict[str, Any], field_name: str, source: Path) -> Any:
    if field_name not in row or row[field_name] in (None, ""):
        raise DataIntegrityError(f"Missing required field {field_name!r} in {source}")
    return row[field_name]


def load_split(manifest_path: Path, split_name: str) -> tuple[DatasetManifest, Iterator[DatasetExample]]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    split = manifest.split(split_name)
    data_path = (manifest_path.parent / split.path).resolve()
    if not data_path.is_file():
        raise DataIntegrityError(f"Dataset file does not exist: {data_path}")
    if split.sha256 and sha256_file(data_path) != split.sha256:
        raise DataIntegrityError(f"SHA-256 mismatch for {data_path}")

    def examples() -> Iterator[DatasetExample]:
        count = 0
        seen: set[str] = set()
        for row in _rows(data_path, split.format):
            count += 1
            sample_id = str(_required(row, manifest.fields.id, data_path))
            if sample_id in seen:
                raise DataIntegrityError(f"Duplicate sample id {sample_id!r} in {data_path}")
            seen.add(sample_id)
            prompt = str(_required(row, manifest.fields.prompt, data_path))
            image_path = None
            if manifest.fields.image and row.get(manifest.fields.image):
                image_path = str((data_path.parent / str(row[manifest.fields.image])).resolve())
            reference = row.get(manifest.fields.reference_answer) if manifest.fields.reference_answer else None
            group_id = row.get(manifest.fields.group_id) if manifest.fields.group_id else None
            metadata = {name: row.get(name) for name in manifest.fields.metadata}
            yield DatasetExample(sample_id, prompt, None if reference is None else str(reference),
                                 image_path, None if group_id is None else str(group_id), metadata, row)
        if split.count is not None and count != split.count:
            raise DataIntegrityError(f"Expected {split.count} rows in {data_path}, found {count}")

    return manifest, examples()
