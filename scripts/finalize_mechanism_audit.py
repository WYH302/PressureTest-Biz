"""Filter mechanism prompts through the completed independent scenario-audit gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from build_boundary_recovery_dataset import registered_prompt_set_sha256_from_rows
from build_mechanism_decomposition_dataset import MECHANISM_CONDITIONS, OUTPUT_DIR, PROMPT_VERSION, STUDY_KIND


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_AUDIT = ROOT / "boundary_recovery_study" / "scenario_audit_final.csv"
DEFAULT_BOUNDARY_MANIFEST = ROOT / "boundary_recovery_study" / "accepted_only_prompts.manifest.json"
DEFAULT_BOUNDARY_SOURCE_PATHS = {
    "annotator_A": ROOT / "boundary_recovery_study" / "scenario_audit_annotator_A.csv",
    "annotator_B": ROOT / "boundary_recovery_study" / "scenario_audit_annotator_B.csv",
    "adjudication": ROOT / "boundary_recovery_study" / "scenario_audit_adjudication.csv",
    "candidate_prompts": ROOT / "boundary_recovery_study" / "candidate_prompts_1200.jsonl",
}
DEFAULT_CANDIDATES = OUTPUT_DIR / "candidate_prompts_1600.jsonl"
DEFAULT_ACCEPTED = OUTPUT_DIR / "accepted_only_prompts.jsonl"
AUDIT_GATE_VERSION = "boundary_audit_gate_v1"
EMPTY_PREFIX_SHA256 = hashlib.sha256(b"").hexdigest()
APPROVED_PREFIX_POLICY_BY_MODEL = {
    "deepseek-chat": {
        "system_prefix_sha256": EMPTY_PREFIX_SHA256,
        "user_prefix_sha256": EMPTY_PREFIX_SHA256,
    },
    "llama-3.1-8b-instruct-q8_0": {
        "system_prefix_sha256": EMPTY_PREFIX_SHA256,
        "user_prefix_sha256": EMPTY_PREFIX_SHA256,
    },
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_boundary_source_files(
    boundary_gate_manifest: Mapping[str, Any],
    source_paths: Mapping[str, Path],
) -> None:
    required = set(DEFAULT_BOUNDARY_SOURCE_PATHS)
    declared = boundary_gate_manifest.get("source_sha256", {})
    if not isinstance(declared, dict) or set(source_paths) != required or not required.issubset(declared):
        raise ValueError("Boundary source inventory is incomplete")
    for key in sorted(required):
        path = Path(source_paths[key])
        if not path.is_file():
            raise ValueError(f"Boundary source file does not exist for {key}: {path}")
        observed = sha256_file(path)
        if observed != str(declared[key]).strip().lower():
            raise ValueError(f"Boundary source hash mismatch for {key}")


def filter_accepted_mechanism_prompts(
    final_audit_rows: Iterable[dict[str, Any]],
    candidate_prompts: Iterable[dict[str, Any]],
    *,
    boundary_gate_manifest: dict[str, Any],
    final_audit_sha256: str,
    minimum_accepted: int = 160,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return hash-bound accepted-only prompt blocks without changing audited content."""
    final_rows = [dict(row) for row in final_audit_rows]
    candidates = [dict(row) for row in candidate_prompts]
    if not final_rows or not candidates:
        raise ValueError("Final audit rows and candidate prompts are required")
    final_by_id = {str(row.get("audit_id", "")).strip(): row for row in final_rows}
    if "" in final_by_id or len(final_by_id) != len(final_rows):
        raise ValueError("Final audit IDs must be nonempty and unique")
    accepted_ids = {
        audit_id
        for audit_id, row in final_by_id.items()
        if str(row.get("accepted_all_six_criteria", "")).strip() == "1"
        and str(row.get("final_overall_accept", "")).strip() == "1"
    }
    required_source_hashes = {"annotator_A", "annotator_B", "adjudication", "candidate_prompts"}
    source_hashes = boundary_gate_manifest.get("source_sha256", {})
    if (
        boundary_gate_manifest.get("schema_version") != AUDIT_GATE_VERSION
        or boundary_gate_manifest.get("status") != "accepted_only"
        or str(boundary_gate_manifest.get("final_audit_sha256", "")) != final_audit_sha256
        or int(boundary_gate_manifest.get("audited_scenarios", -1)) != len(final_rows)
        or int(boundary_gate_manifest.get("accepted_scenarios", -1)) != len(accepted_ids)
        or not isinstance(source_hashes, dict)
        or not required_source_hashes.issubset(source_hashes)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(source_hashes[key]).strip().lower()) is None
            for key in required_source_hashes
        )
    ):
        raise ValueError("Final audit is not bound to a complete boundary-audit gate manifest")
    if len(accepted_ids) < minimum_accepted:
        raise RuntimeError(
            f"STOP: only {len(accepted_ids)} scenarios passed; at least {minimum_accepted} are required"
        )
    prompt_ids = [str(row.get("prompt_id", "")).strip() for row in candidates]
    if any(not prompt_id for prompt_id in prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Candidate prompt IDs must be nonempty and unique")
    by_audit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    registered = set(MECHANISM_CONDITIONS)
    for row in candidates:
        frozen_id = str(row.get("audit_id", "")).strip()
        if frozen_id not in final_by_id:
            raise ValueError(f"Candidate prompt has an unknown audit ID: {frozen_id!r}")
        condition = str(row.get("condition_id", ""))
        if (
            row.get("prompt_version") != PROMPT_VERSION
            or row.get("study_kind") != STUDY_KIND
            or row.get("audit_required") is not True
            or row.get("variant") != condition
            or row.get("pressure_condition") != condition
        ):
            raise ValueError(f"Invalid mechanism prompt schema for {row.get('prompt_id')}")
        by_audit[frozen_id].append(row)
    if set(by_audit) != set(final_by_id):
        raise ValueError("Candidate prompt and final-audit scenario inventories differ")
    for frozen_id, rows in by_audit.items():
        conditions = [str(row["condition_id"]) for row in rows]
        if len(conditions) != len(registered) or set(conditions) != registered:
            raise ValueError(f"Incomplete mechanism arm block for {frozen_id}")
        final_hash = str(final_by_id[frozen_id].get("audited_scenario_sha256", ""))
        if not final_hash or any(str(row.get("audited_scenario_sha256", "")) != final_hash for row in rows):
            raise ValueError(f"Candidate scenario content is not bound to final audit {frozen_id}")
        registered_hash = str(rows[0].get("registered_prompt_set_sha256", ""))
        if (
            not registered_hash
            or any(str(row.get("registered_prompt_set_sha256", "")) != registered_hash for row in rows)
            or registered_prompt_set_sha256_from_rows(rows) != registered_hash
        ):
            raise ValueError(f"Candidate mechanism prompt content changed for {frozen_id}")

    accepted = [
        {
            **row,
            "audit_gate_passed": True,
            "accepted_only_source": True,
            "audit_gate_version": AUDIT_GATE_VERSION,
            "final_audit_record_sha256": _canonical_sha256(final_by_id[str(row["audit_id"])]),
        }
        for row in candidates
        if str(row["audit_id"]) in accepted_ids
    ]
    manifest = {
        "schema_version": AUDIT_GATE_VERSION,
        "status": "accepted_only",
        "study_kind": STUDY_KIND,
        "minimum_accepted_scenarios": minimum_accepted,
        "audited_scenarios": len(final_rows),
        "accepted_scenarios": len(accepted_ids),
        "rejected_scenarios": len(final_rows) - len(accepted_ids),
        "registered_conditions": list(MECHANISM_CONDITIONS),
        "arms_per_scenario": len(MECHANISM_CONDITIONS),
        "accepted_prompt_rows": len(accepted),
        "accepted_audit_ids_sha256": _canonical_sha256(sorted(accepted_ids)),
        "accepted_prompt_ids_sha256": _canonical_sha256(sorted(str(row["prompt_id"]) for row in accepted)),
        "boundary_final_audit_sha256": final_audit_sha256,
        "boundary_gate_manifest_sha256": _canonical_sha256(boundary_gate_manifest),
        "approved_prefix_policy_by_model": APPROVED_PREFIX_POLICY_BY_MODEL,
    }
    return sorted(accepted, key=lambda row: str(row["prompt_id"])), manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-audit", type=Path, default=DEFAULT_FINAL_AUDIT)
    parser.add_argument("--boundary-manifest", type=Path, default=DEFAULT_BOUNDARY_MANIFEST)
    parser.add_argument("--boundary-annotator-a", type=Path, default=DEFAULT_BOUNDARY_SOURCE_PATHS["annotator_A"])
    parser.add_argument("--boundary-annotator-b", type=Path, default=DEFAULT_BOUNDARY_SOURCE_PATHS["annotator_B"])
    parser.add_argument("--boundary-adjudication", type=Path, default=DEFAULT_BOUNDARY_SOURCE_PATHS["adjudication"])
    parser.add_argument(
        "--boundary-candidate-prompts",
        type=Path,
        default=DEFAULT_BOUNDARY_SOURCE_PATHS["candidate_prompts"],
    )
    parser.add_argument("--candidate-prompts", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--accepted-prompts", type=Path, default=DEFAULT_ACCEPTED)
    args = parser.parse_args()
    try:
        input_paths = {
            args.final_audit.resolve(),
            args.boundary_manifest.resolve(),
            args.candidate_prompts.resolve(),
            args.boundary_annotator_a.resolve(),
            args.boundary_annotator_b.resolve(),
            args.boundary_adjudication.resolve(),
            args.boundary_candidate_prompts.resolve(),
        }
        output_paths = {
            args.accepted_prompts.resolve(),
            args.accepted_prompts.with_suffix(".manifest.json").resolve(),
        }
        if len(output_paths) != 2 or input_paths & output_paths:
            raise ValueError("Mechanism gate outputs must be distinct and must not overwrite input evidence")
        final_audit_sha256 = sha256_file(args.final_audit)
        candidate_prompts_sha256 = sha256_file(args.candidate_prompts)
        boundary_manifest_sha256 = sha256_file(args.boundary_manifest)
        boundary_gate_manifest = _read_json(args.boundary_manifest)
        validate_boundary_source_files(
            boundary_gate_manifest,
            {
                "annotator_A": args.boundary_annotator_a,
                "annotator_B": args.boundary_annotator_b,
                "adjudication": args.boundary_adjudication,
                "candidate_prompts": args.boundary_candidate_prompts,
            },
        )
        accepted, manifest = filter_accepted_mechanism_prompts(
            _read_csv(args.final_audit),
            _read_jsonl(args.candidate_prompts),
            boundary_gate_manifest=boundary_gate_manifest,
            final_audit_sha256=final_audit_sha256,
        )
        _write_jsonl(args.accepted_prompts, accepted)
        manifest = {
            **manifest,
            "accepted_prompts_sha256": sha256_file(args.accepted_prompts),
            "source_sha256": {
                "final_audit": sha256_file(args.final_audit),
                "candidate_prompts": candidate_prompts_sha256,
                "boundary_manifest": boundary_manifest_sha256,
            },
        }
        args.accepted_prompts.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print(
        f"Mechanism gate passed: {manifest['accepted_scenarios']} scenarios and "
        f"{manifest['accepted_prompt_rows']} prompt rows released."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
