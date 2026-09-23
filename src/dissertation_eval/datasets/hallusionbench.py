from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import DataIntegrityError
from ..schema import DatasetExample, DatasetManifest


_ID_FIELDS = ("category", "subcategory", "set_id", "figure_id", "question_id")
_GROUP_FIELDS = ("category", "subcategory", "set_id")


def _required(row: dict[str, Any], name: str, source: Path) -> Any:
    if name not in row or row[name] in (None, ""):
        raise DataIntegrityError(f"Missing required HallusionBench field {name!r} in {source}")
    return row[name]


def _stable_key(row: dict[str, Any], fields: tuple[str, ...], source: Path) -> str:
    return "/".join(f"{name}={_required(row, name, source)}" for name in fields)


def _image_path(row: dict[str, Any], data_path: Path) -> str | None:
    filename = row.get("filename")
    if filename in (None, ""):
        return None

    root = data_path.parent.resolve()
    resolved = (root / str(filename)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DataIntegrityError(
            f"HallusionBench image path escapes dataset directory: {filename!r}"
        ) from exc
    if not resolved.is_file():
        raise DataIntegrityError(f"HallusionBench image does not exist: {resolved}")
    return str(resolved)


def normalize_hallusionbench(
    row: dict[str, Any], data_path: Path, manifest: DatasetManifest
) -> DatasetExample:
    """Normalize one official HallusionBench JSON record.

    HallusionBench identifiers are only unique as a composite of category,
    subcategory, set, figure, and question identifiers. Text-only control
    questions intentionally have no filename.
    """

    sample_id = _stable_key(row, _ID_FIELDS, data_path)
    group_id = _stable_key(row, _GROUP_FIELDS, data_path)
    prompt = str(_required(row, manifest.fields.prompt, data_path))
    reference = _required(row, manifest.fields.reference_answer or "gt_answer", data_path)
    metadata = {name: row.get(name) for name in manifest.fields.metadata}
    return DatasetExample(
        sample_id=sample_id,
        prompt=prompt,
        reference_answer=str(reference),
        image_path=_image_path(row, data_path),
        group_id=group_id,
        metadata=metadata,
        raw=row,
    )
