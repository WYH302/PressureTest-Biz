"""Validate completed archived human-audit sheets and report reliability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from build_archived_human_validation_pack import (
    PACK_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    analyze_dual_annotations,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "archived_human_validation_v2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pack_provenance(input_dir: Path, *, allow_legacy_pack: bool = False) -> None:
    manifest_path = input_dir / "pack_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = str(manifest.get("pack_schema_version", "")).strip()
    if schema != PACK_SCHEMA_VERSION:
        if not allow_legacy_pack:
            raise ValueError(
                f"Pack schema must be {PACK_SCHEMA_VERSION}; pass --allow-legacy-pack only for an archived pre-v2 pack"
            )
        if schema:
            raise ValueError(f"Unsupported nonempty pack schema: {schema}")
    elif str(manifest.get("protocol_version", "")).strip() != PROTOCOL_VERSION:
        raise ValueError(f"Pack protocol must be {PROTOCOL_VERSION}")
    artifacts = manifest.get("artifact_sha256", {})
    immutable_paths = {
        "private_pair_manifest": input_dir / "private" / "private_pair_manifest.jsonl",
        "private_response_manifest": input_dir / "private" / "private_response_manifest.jsonl",
    }
    if not isinstance(artifacts, dict) or not set(immutable_paths).issubset(artifacts):
        raise ValueError("Pack manifest lacks immutable private-manifest hashes")
    for key, path in immutable_paths.items():
        if not path.is_file() or _sha256_file(path) != str(artifacts[key]).strip().lower():
            raise ValueError(f"Pack provenance hash mismatch for {key}")

    sources = manifest.get("source_sha256")
    if sources is None and allow_legacy_pack and not schema:
        return
    if not isinstance(sources, dict):
        raise ValueError("Pack source_sha256 must be an object")
    fixed_sources = {
        "cases": ROOT
        / "boundary_recovery_study"
        / "archived_boundary_analysis"
        / "structural_alignment_crossing_cases.csv",
        "prompts": ROOT / "main_dataset" / "main_prompts_v2_detailed_action_choice.jsonl",
        "matrix": ROOT / "results" / "four_by_five_crossed_matrix_submission_20260716" / "matrix_status.csv",
        "builder_script": ROOT / "scripts" / "build_archived_human_validation_pack.py",
        "analyzer_script": ROOT / "scripts" / "analyze_archived_human_validation.py",
        "annotation_protocol": ROOT / "archived_human_validation_v2" / "ANNOTATION_PROTOCOL.md",
    }
    missing_source_keys = set(fixed_sources) - set(sources)
    if missing_source_keys:
        raise ValueError(f"Pack lacks required source hash keys: {sorted(missing_source_keys)}")
    for key, path in fixed_sources.items():
        if key not in sources or not path.is_file() or _sha256_file(path) != str(sources[key]).strip().lower():
            raise ValueError(f"Pack source hash mismatch for {key}")
    target_sources = sources.get("target_response_files", {})
    if not isinstance(target_sources, dict) or not target_sources:
        raise ValueError("Pack manifest lacks target response source hashes")
    for relative, expected in target_sources.items():
        path = (ROOT / str(relative)).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Pack target response path escapes the project root: {relative}")
        if not path.is_file() or _sha256_file(path) != str(expected).strip().lower():
            raise ValueError(f"Pack target response hash mismatch for {relative}")


def _confusion_rows(pair_rows: list[dict[str, Any]], *, dimension: str) -> list[dict[str, Any]]:
    """Create complete 2x2 raw-count tables within each requested dimension."""
    if dimension not in {"model", "cue_variant"}:
        raise ValueError("dimension must be 'model' or 'cue_variant'")
    values = sorted({str(row[dimension]) for row in pair_rows})
    counts: dict[tuple[str, int, int], int] = {}
    for row in pair_rows:
        exact_anchor = int(str(row["sampling_stratum"]) == "exact_anchor_crossing")
        human = int(row["human_crossing"])
        key = (str(row[dimension]), exact_anchor, human)
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            dimension: value,
            "exact_anchor_crossing": exact_anchor,
            "human_crossing": human,
            "count": counts.get((value, exact_anchor, human), 0),
        }
        for value in values
        for exact_anchor in (0, 1)
        for human in (0, 1)
    ]


def validate_calibration_chain(calibration_dir: Path) -> dict[str, str]:
    """Require a completed calibration and return role-bound pseudonyms."""
    from build_human_calibration_packs import validate_calibration_pack, validate_personnel_log

    status = validate_calibration_pack(calibration_dir)
    if status.get("status") != "calibration_complete":
        raise ValueError("Archived-response calibration must be completed and frozen before formal analysis")
    personnel = validate_personnel_log(calibration_dir / "personnel_log_template.csv")
    return {
        "A": str(personnel["annotator_A"]["pseudonym"]).strip(),
        "B": str(personnel["annotator_B"]["pseudonym"]).strip(),
        "adjudicator": str(personnel["adjudicator"]["pseudonym"]).strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-legacy-pack",
        action="store_true",
        help="Explicitly allow a pre-v2 pack without source hashes; never use for confirmatory v2 analysis.",
    )
    args = parser.parse_args()
    try:
        expected_pseudonyms = validate_calibration_chain(args.calibration_dir or args.input_dir / "calibration")
        validate_pack_provenance(args.input_dir, allow_legacy_pack=args.allow_legacy_pack)
        result = analyze_dual_annotations(
            _read_jsonl(args.input_dir / "private" / "private_response_manifest.jsonl"),
            _read_csv(args.input_dir / "annotator_delivery" / "annotator_A" / "annotation.csv"),
            _read_csv(args.input_dir / "annotator_delivery" / "annotator_B" / "annotation.csv"),
            _read_csv(args.input_dir / "private" / "adjudication.csv"),
            expected_pseudonyms=expected_pseudonyms,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    pair_rows = result.pop("pair_rows", [])
    if pair_rows:
        _write_csv(args.input_dir / "final_pair_labels.csv", pair_rows)
        _write_csv(
            args.input_dir / "per_model_confusion_counts.csv",
            _confusion_rows(pair_rows, dimension="model"),
        )
        _write_csv(
            args.input_dir / "per_cue_confusion_counts.csv",
            _confusion_rows(pair_rows, dimension="cue_variant"),
        )
    (args.input_dir / "reliability_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
