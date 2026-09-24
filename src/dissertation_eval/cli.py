from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import diagnose_rhd_generations
from .datasets import load_manifest, load_split
from .datasets.prepare import prepare_retruthqa, validate_prepared_retruthqa
from .datasets.selection import create_retruthqa_selection
from .evaluation import compare_retruthqa_runs, evaluate_retruthqa_generations, evaluate_rhd_diagnostics
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
    generate.add_argument(
        "--sample-seed",
        type=int,
        help="Deterministically shuffle examples before applying --limit",
    )
    generate.add_argument("--fail-fast", action="store_true")
    prepare = subparsers.add_parser(
        "prepare-retruthqa", help="Materialize normalized ReTruthQA generation inputs"
    )
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--force", action="store_true")
    validate_prepared = subparsers.add_parser(
        "validate-prepared", help="Validate a prepared ReTruthQA directory or manifest"
    )
    validate_prepared.add_argument("path", type=Path)
    select = subparsers.add_parser(
        "select-retruthqa", help="Create the frozen 240-example dissertation selection"
    )
    select.add_argument("prepared_manifest", type=Path)
    select.add_argument("--selection-manifest", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    evaluate = subparsers.add_parser(
        "evaluate-retruthqa", help="Extract and score answers from ReTruthQA generations"
    )
    evaluate.add_argument("generations", type=Path)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    compare = subparsers.add_parser(
        "compare-retruthqa", help="Compare labelled ReTruthQA generation runs"
    )
    compare.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Labelled generation JSONL; repeat for each decoding strategy",
    )
    compare.add_argument("--self-consistency", metavar="LABEL")
    compare.add_argument("--output-dir", type=Path, required=True)
    diagnose = subparsers.add_parser(
        "diagnose-rhd", help="Extract read-only token-level inter-layer diagnostics"
    )
    diagnose.add_argument("generations", type=Path)
    diagnose.add_argument("--output-dir", type=Path, required=True)
    diagnose.add_argument("--limit", type=int)
    diagnose.add_argument("--relative-layer", type=float, action="append")
    diagnose.add_argument("--chunk-size", type=int, default=16)
    evaluate_rhd = subparsers.add_parser(
        "evaluate-rhd", help="Compare development-only RHD diagnostic configurations"
    )
    evaluate_rhd.add_argument("evaluation", type=Path)
    evaluate_rhd.add_argument("--diagnostic", action="append", required=True, metavar="LABEL=DIR")
    evaluate_rhd.add_argument("--output", type=Path, required=True)
    evaluate_rhd.add_argument("--skip-records", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-manifest":
        return _validate_manifest(args.manifest, args.split)
    if args.command == "prepare-retruthqa":
        result = prepare_retruthqa(args.manifest, args.output_dir, force=args.force)
        summary = {
            key: result[key]
            for key in (
                "status", "extractor_version", "total_records",
                "total_ambiguous", "total_majority",
            )
        }
        summary["output_dir"] = str(args.output_dir.resolve())
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-prepared":
        result = validate_prepared_retruthqa(args.path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "select-retruthqa":
        result = create_retruthqa_selection(
            args.prepared_manifest, args.selection_manifest, args.output_dir
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate-retruthqa":
        result = evaluate_retruthqa_generations(args.generations, args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "compare-retruthqa":
        runs: dict[str, Path] = {}
        for item in args.run:
            if "=" not in item:
                raise SystemExit(f"--run must use LABEL=PATH, received: {item!r}")
            label, raw_path = item.split("=", 1)
            if not label or not raw_path or label in runs:
                raise SystemExit(f"Invalid or duplicate --run value: {item!r}")
            runs[label] = Path(raw_path)
        result = compare_retruthqa_runs(
            runs, args.output_dir, self_consistency=args.self_consistency
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "diagnose-rhd":
        kwargs = {
            "limit": args.limit,
            "chunk_size": args.chunk_size,
        }
        if args.relative_layer is not None:
            kwargs["relative_layers"] = args.relative_layer
        result = diagnose_rhd_generations(args.generations, args.output_dir, **kwargs)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate-rhd":
        diagnostics: dict[str, Path] = {}
        for item in args.diagnostic:
            if "=" not in item:
                raise SystemExit(f"--diagnostic must use LABEL=DIR, received: {item!r}")
            label, raw_path = item.split("=", 1)
            if not label or not raw_path or label in diagnostics:
                raise SystemExit(f"Invalid or duplicate --diagnostic value: {item!r}")
            diagnostics[label] = Path(raw_path)
        result = evaluate_rhd_diagnostics(
            args.evaluation, diagnostics, args.output, skip_records=args.skip_records
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    counters = run_generation(
        args.config,
        limit=args.limit,
        fail_fast=args.fail_fast,
        sample_seed=args.sample_seed,
    )
    print(json.dumps(counters, indent=2, sort_keys=True))
    return 0 if counters["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
