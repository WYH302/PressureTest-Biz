"""Finalize the blinded boundary-study audit and release accepted-only prompts.

This is a hard pre-execution gate.  It never creates an accepted prompt file
unless two complete independent audits cover the same IDs, all disagreements
are adjudicated, and at least 160 scenarios pass all six final criteria.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from build_boundary_recovery_dataset import (
    AUDIT_CONTENT_FIELDS,
    AUDIT_CRITERIA,
    OUTPUT_DIR,
    audited_scenario_sha256_from_row,
    registered_prompt_set_sha256_from_rows,
)


AUDIT_GATE_VERSION = "boundary_audit_gate_v1"
DEFAULT_MIN_ACCEPTED = 160
CONFIRMATORY_CONDITIONS = frozenset(
    {
        "B0_baseline",
        "C0_neutral_length",
        "P1_cognitive_load",
        "P2_resource_scarcity",
        "P3_combined_pressure",
        "R3_constraint_regrounding",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{label} table is empty")
    return rows


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {label} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} file is empty")
    return rows


def _index_unique(rows: Iterable[dict[str, Any]], *, label: str, key: str = "audit_id") -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        value = str(row.get(key, "")).strip()
        if not value:
            raise ValueError(f"{label} row {row_number} has a missing {key}")
        if value in indexed:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        indexed[value] = dict(row)
    return indexed


def _binary(row: dict[str, Any], field: str, *, label: str, allow_blank: bool = False) -> int | None:
    raw = str(row.get(field, "")).strip()
    if allow_blank and not raw:
        return None
    if raw not in {"0", "1"}:
        detail = "missing" if not raw else f"invalid value {raw!r}"
        raise ValueError(f"{label} field {field} is {detail}; expected 0 or 1")
    return int(raw)


def _validate_annotation_rows(rows: list[dict[str, str]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed = _index_unique(rows, label=label)
    annotator_ids: set[str] = set()
    for audit_id, row in indexed.items():
        annotator_id = str(row.get("annotator_id", "")).strip()
        if not annotator_id:
            raise ValueError(f"{label} {audit_id} has a missing annotator_id")
        annotator_ids.add(annotator_id)
        for criterion in AUDIT_CRITERIA:
            _binary(row, criterion, label=f"{label} {audit_id}")
        _binary(row, "overall_accept", label=f"{label} {audit_id}")
        confidence = str(row.get("confidence_1_to_5", "")).strip()
        if confidence not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"{label} {audit_id} has missing or invalid confidence_1_to_5")
        for field in AUDIT_CONTENT_FIELDS:
            if not str(row.get(field, "")).strip():
                raise ValueError(f"{label} {audit_id} has a missing frozen audit field: {field}")
        expected_scenario_hash = audited_scenario_sha256_from_row(row)
        if str(row.get("audited_scenario_sha256", "")) != expected_scenario_hash:
            raise ValueError(f"{label} {audit_id} audited_scenario_sha256 does not match its frozen content")
        if not str(row.get("registered_prompt_set_sha256", "")).strip():
            raise ValueError(f"{label} {audit_id} has a missing registered_prompt_set_sha256")
    if len(annotator_ids) != 1:
        raise ValueError(f"{label} must contain exactly one annotator_id")
    return indexed


def _resolved_final_value(
    *,
    a_value: int,
    b_value: int,
    adjudication_row: dict[str, Any],
    final_field: str,
    audit_id: str,
) -> tuple[int, bool]:
    supplied = _binary(
        adjudication_row,
        final_field,
        label=f"adjudication {audit_id}",
        allow_blank=True,
    )
    disagreed = a_value != b_value
    if disagreed and supplied is None:
        raise ValueError(f"adjudication {audit_id} is missing {final_field} for an A/B disagreement")
    if not disagreed and supplied is not None and supplied != a_value:
        raise ValueError(f"adjudication {audit_id} has {final_field} inconsistent with A/B agreement")
    return (a_value if supplied is None else supplied), disagreed


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty final audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finalize_boundary_audit(
    *,
    annotator_a_path: Path,
    annotator_b_path: Path,
    adjudication_path: Path,
    candidate_prompts_path: Path,
    accepted_prompts_path: Path,
    final_audit_path: Path | None = None,
    minimum_accepted: int = DEFAULT_MIN_ACCEPTED,
) -> dict[str, Any]:
    """Validate, merge, and finalize the two-person audit."""
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    final_audit_path = final_audit_path or accepted_prompts_path.with_name("scenario_audit_final.csv")
    manifest_path = accepted_prompts_path.with_suffix(".manifest.json")
    inputs = {
        path.resolve()
        for path in (annotator_a_path, annotator_b_path, adjudication_path, candidate_prompts_path)
    }
    outputs = [accepted_prompts_path.resolve(), final_audit_path.resolve(), manifest_path.resolve()]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Accepted prompts, final audit, and manifest must use distinct output paths")
    if any(path in inputs for path in outputs):
        raise ValueError("Audit outputs must not overwrite any input evidence file")
    a_rows = _validate_annotation_rows(_read_csv(annotator_a_path, label="annotator A"), label="annotator A")
    b_rows = _validate_annotation_rows(_read_csv(annotator_b_path, label="annotator B"), label="annotator B")
    a_annotator_ids = {str(row["annotator_id"]).strip() for row in a_rows.values()}
    b_annotator_ids = {str(row["annotator_id"]).strip() for row in b_rows.values()}
    if a_annotator_ids == b_annotator_ids:
        raise ValueError("Annotator A and annotator B must be independent identities")
    adjudication_rows = _index_unique(_read_csv(adjudication_path, label="adjudication"), label="adjudication")
    id_sets = {"annotator A": set(a_rows), "annotator B": set(b_rows), "adjudication": set(adjudication_rows)}
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        sizes = {label: len(ids) for label, ids in id_sets.items()}
        raise ValueError(f"Audit ID sets do not match across files: {sizes}")
    for frozen_audit_id in a_rows:
        fields_to_match = (
            *AUDIT_CONTENT_FIELDS,
            "audited_scenario_sha256",
            "registered_prompt_set_sha256",
        )
        if any(str(a_rows[frozen_audit_id].get(field, "")) != str(b_rows[frozen_audit_id].get(field, "")) for field in fields_to_match):
            raise ValueError(f"Annotator A/B frozen audit content differs for {frozen_audit_id}")

    final_rows: list[dict[str, Any]] = []
    accepted_ids: set[str] = set()
    for audit_id in sorted(a_rows):
        a_row, b_row, adjudication_row = a_rows[audit_id], b_rows[audit_id], adjudication_rows[audit_id]
        final_values: dict[str, int] = {}
        any_disagreement = False
        for criterion in AUDIT_CRITERIA:
            resolved, disagreed = _resolved_final_value(
                a_value=int(_binary(a_row, criterion, label=f"annotator A {audit_id}")),
                b_value=int(_binary(b_row, criterion, label=f"annotator B {audit_id}")),
                adjudication_row=adjudication_row,
                final_field=f"final_{criterion}",
                audit_id=audit_id,
            )
            final_values[criterion] = resolved
            any_disagreement = any_disagreement or disagreed
        final_overall, overall_disagreement = _resolved_final_value(
            a_value=int(_binary(a_row, "overall_accept", label=f"annotator A {audit_id}")),
            b_value=int(_binary(b_row, "overall_accept", label=f"annotator B {audit_id}")),
            adjudication_row=adjudication_row,
            final_field="final_overall_accept",
            audit_id=audit_id,
        )
        any_disagreement = any_disagreement or overall_disagreement
        adjudicator_id = str(adjudication_row.get("adjudicator_id", "")).strip()
        adjudication_reason = str(adjudication_row.get("adjudication_reason", "")).strip()
        if any_disagreement and (not adjudicator_id or not adjudication_reason):
            raise ValueError(f"adjudication {audit_id} is missing adjudicator_id or adjudication_reason")
        accepted = all(final_values[criterion] == 1 for criterion in AUDIT_CRITERIA)
        if accepted:
            accepted_ids.add(audit_id)
        final_rows.append(
            {
                "audit_id": audit_id,
                **{f"annotator_A_{criterion}": a_row[criterion] for criterion in AUDIT_CRITERIA},
                **{f"annotator_B_{criterion}": b_row[criterion] for criterion in AUDIT_CRITERIA},
                **{f"final_{criterion}": final_values[criterion] for criterion in AUDIT_CRITERIA},
                "annotator_A_overall_accept": a_row["overall_accept"],
                "annotator_B_overall_accept": b_row["overall_accept"],
                "final_overall_accept": final_overall,
                "accepted_all_six_criteria": int(accepted),
                "audited_scenario_sha256": a_row["audited_scenario_sha256"],
                "registered_prompt_set_sha256": a_row["registered_prompt_set_sha256"],
                "adjudicator_id": adjudicator_id,
                "adjudication_reason": adjudication_reason,
            }
        )

    if len(accepted_ids) < minimum_accepted:
        raise RuntimeError(
            f"STOP: only {len(accepted_ids)} scenarios passed all six final criteria; at least {minimum_accepted} are required"
        )

    candidate_prompts = _read_jsonl(candidate_prompts_path, label="candidate prompts")
    prompt_ids = _index_unique(candidate_prompts, label="candidate prompts", key="prompt_id")
    del prompt_ids  # uniqueness validation only
    candidate_audit_ids = {str(row.get("audit_id", "")).strip() for row in candidate_prompts}
    if "" in candidate_audit_ids or not candidate_audit_ids.issubset(a_rows):
        raise ValueError("Candidate prompts contain missing or unknown audit_id values")
    arms_by_audit: dict[str, list[str]] = {}
    for row in candidate_prompts:
        if (
            row.get("prompt_version") != "boundary_recovery_v1"
            or row.get("study_kind") != "boundary_confirmatory_v1"
            or row.get("audit_required") is not True
        ):
            raise ValueError("Candidate prompts contain missing or invalid boundary-confirmatory schema markers")
        audit_id = str(row["audit_id"])
        condition_id = str(row.get("condition_id", ""))
        if row.get("variant") != condition_id or row.get("pressure_condition") != condition_id:
            raise ValueError("Candidate prompt condition_id, variant, and pressure_condition must agree")
        arms_by_audit.setdefault(audit_id, []).append(condition_id)
    if set(arms_by_audit) != set(a_rows) or any(
        len(arms) != len(CONFIRMATORY_CONDITIONS) or set(arms) != CONFIRMATORY_CONDITIONS
        for arms in arms_by_audit.values()
    ):
        raise ValueError("Candidate prompts must contain each of the six registered conditions exactly once per audit_id")
    prompts_by_audit = {
        frozen_audit_id: [row for row in candidate_prompts if str(row["audit_id"]) == frozen_audit_id]
        for frozen_audit_id in arms_by_audit
    }
    for frozen_audit_id, rows in prompts_by_audit.items():
        expected_scenario_hash = str(a_rows[frozen_audit_id]["audited_scenario_sha256"])
        expected_prompt_set_hash = str(a_rows[frozen_audit_id]["registered_prompt_set_sha256"])
        if any(str(row.get("audited_scenario_sha256", "")) != expected_scenario_hash for row in rows):
            raise ValueError(f"Candidate prompt scenario content is not bound to audit {frozen_audit_id}")
        if any(str(row.get("registered_prompt_set_sha256", "")) != expected_prompt_set_hash for row in rows):
            raise ValueError(f"Candidate prompt-set marker is not bound to audit {frozen_audit_id}")
        if registered_prompt_set_sha256_from_rows(rows) != expected_prompt_set_hash:
            raise ValueError(f"Candidate prompt content changed after audit registration for {frozen_audit_id}")

    final_by_id = {row["audit_id"]: row for row in final_rows}
    accepted_prompts: list[dict[str, Any]] = []
    for prompt in candidate_prompts:
        audit_id = str(prompt["audit_id"])
        if audit_id not in accepted_ids:
            continue
        final_record_hash = _canonical_sha256(final_by_id[audit_id])
        accepted_prompts.append(
            {
                **prompt,
                "audit_required": True,
                "audit_gate_passed": True,
                "accepted_only_source": True,
                "audit_gate_version": AUDIT_GATE_VERSION,
                "final_audit_record_sha256": final_record_hash,
            }
        )

    _write_csv(final_audit_path, final_rows)
    _write_jsonl(accepted_prompts_path, accepted_prompts)
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_GATE_VERSION,
        "status": "accepted_only",
        "minimum_accepted_scenarios": minimum_accepted,
        "audited_scenarios": len(final_rows),
        "accepted_scenarios": len(accepted_ids),
        "rejected_scenarios": len(final_rows) - len(accepted_ids),
        "accepted_prompt_rows": len(accepted_prompts),
        "accepted_audit_ids_sha256": _canonical_sha256(sorted(accepted_ids)),
        "accepted_prompt_ids_sha256": _canonical_sha256(sorted(str(row["prompt_id"]) for row in accepted_prompts)),
        "accepted_scenario_bindings_sha256": _canonical_sha256(
            sorted(
                (
                    frozen_audit_id,
                    str(a_rows[frozen_audit_id]["audited_scenario_sha256"]),
                    str(a_rows[frozen_audit_id]["registered_prompt_set_sha256"]),
                )
                for frozen_audit_id in accepted_ids
            )
        ),
        "accepted_prompts_sha256": sha256_file(accepted_prompts_path),
        "final_audit_sha256": sha256_file(final_audit_path),
        "source_sha256": {
            "annotator_A": sha256_file(annotator_a_path),
            "annotator_B": sha256_file(annotator_b_path),
            "adjudication": sha256_file(adjudication_path),
            "candidate_prompts": sha256_file(candidate_prompts_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-a", type=Path, default=OUTPUT_DIR / "scenario_audit_annotator_A.csv")
    parser.add_argument("--annotator-b", type=Path, default=OUTPUT_DIR / "scenario_audit_annotator_B.csv")
    parser.add_argument("--adjudication", type=Path, default=OUTPUT_DIR / "scenario_audit_adjudication.csv")
    parser.add_argument("--candidate-prompts", type=Path, default=OUTPUT_DIR / "candidate_prompts_1200.jsonl")
    parser.add_argument("--accepted-prompts", type=Path, default=OUTPUT_DIR / "accepted_only_prompts.jsonl")
    parser.add_argument("--final-audit", type=Path, default=OUTPUT_DIR / "scenario_audit_final.csv")
    args = parser.parse_args()
    try:
        result = finalize_boundary_audit(
            annotator_a_path=args.annotator_a,
            annotator_b_path=args.annotator_b,
            adjudication_path=args.adjudication,
            candidate_prompts_path=args.candidate_prompts,
            accepted_prompts_path=args.accepted_prompts,
            final_audit_path=args.final_audit,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print(
        f"Audit gate passed: {result['accepted_scenarios']} scenarios and "
        f"{result['accepted_prompt_rows']} prompt rows released."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
