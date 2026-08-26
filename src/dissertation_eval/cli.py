from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import load_manifest, load_split
from .pipeline.runner import run_generation


def _validate_manifest(path: Path, split: str | None) -> int:
    manifest = load_manifest(path)
    names = [split] if split else [item.name for item in manifest.splits]
    counts: dict[str, int] = {}
    for name in names:
        _, examples = load_split(path, name)
        counts[name] = sum(1 for _ in examples)
    print(json.dumps({"dataset_id": manifest.dataset_id, "splits": counts}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dissertation-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-manifest", help="Validate manifest and input files")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--split")
    generate = subparsers.add_parser("generate", help="Run or resume baseline generation")
    generate.add_argument("config", type=Path)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-manifest":
        return _validate_manifest(args.manifest, args.split)
    counters = run_generation(args.config, limit=args.limit, fail_fast=args.fail_fast)
    print(json.dumps(counters, indent=2, sort_keys=True))
    return 0 if counters["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
