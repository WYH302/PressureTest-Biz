"""Combine immutable round-level audit evidence under the unchanged 160 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from finalize_boundary_audit import (
    APPROVED_PREFIX_POLICY_BY_MODEL,
    AUDIT_GATE_VERSION,
    CONFIRMATORY_CONDITIONS,
    DEFAULT_MIN_ACCEPTED,
    ROUND_EVIDENCE_VERSION,
    sha256_file,
)


RELEASE_KIND = "combined_fixed_rounds_v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} has invalid JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _read_csv(path: Path, *, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _safe_sibling(manifest_path: Path, filename: Any, *, field: str) -> Path:
    value = str(filename or "").strip()
    if not value or Path(value).name != value:
        raise ValueError(f"Round manifest has invalid {field}")
    return manifest_path.parent / value


def _unique_values(rows: Iterable[dict[str, Any]], field: str, *, label: str) -> set[str]:
    values: set[str] = set()
    for index, row in enumerate(rows, 1):
        value = str(row.get(field, "")).strip()
        if not value:
            raise ValueError(f"{label} row {index} has missing {field}")
        if value in values:
            raise ValueError(f"{label} contains duplicate {field}: {value}")
        values.add(value)
    return values


def _load_round(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(manifest_path, label="round evidence manifest")
    if (
        manifest.get("schema_version") != ROUND_EVIDENCE_VERSION
        or manifest.get("status") != "round_audit_finalized"
        or manifest.get("round_evidence_only") is not True
        or manifest.get("release_gate_applied") is not False
        or manifest.get("combined_release_minimum_accepted_scenarios") != DEFAULT_MIN_ACCEPTED
    ):
        raise ValueError("Round manifest is not finalized non-runnable evidence under the unchanged 160 gate")
    round_id = str(manifest.get("round_id", "")).strip()
    if not round_id:
        raise ValueError("Round evidence manifest has a missing round_id")
    evidence_path = _safe_sibling(manifest_path, manifest.get("evidence_prompts_file"), field="evidence_prompts_file")
    final_audit_path = _safe_sibling(manifest_path, manifest.get("final_audit_file"), field="final_audit_file")
    if sha256_file(evidence_path) != manifest.get("accepted_prompts_sha256"):
        raise ValueError(f"Round {round_id} evidence prompt hash does not match its manifest")
    if sha256_file(final_audit_path) != manifest.get("final_audit_sha256"):
        raise ValueError(f"Round {round_id} final-audit hash does not match its manifest")

    final_rows = _read_csv(final_audit_path, label=f"round {round_id} final audit")
    final_ids = _unique_values(final_rows, "audit_id", label=f"round {round_id} final audit")
    accepted_ids = {
        str(row["audit_id"])
        for row in final_rows
        if str(row.get("accepted_all_six_criteria", "")).strip() == "1"
    }
    if len(final_ids) != manifest.get("audited_scenarios") or len(accepted_ids) != manifest.get("accepted_scenarios"):
        raise ValueError(f"Round {round_id} final-audit counts do not match its manifest")
    if _canonical_sha256(sorted(accepted_ids)) != manifest.get("accepted_audit_ids_sha256"):
        raise ValueError(f"Round {round_id} accepted audit-ID hash does not match its final audit")
    final_by_id = {str(row["audit_id"]): row for row in final_rows}
    bindings = sorted(
        (
            audit_id,
            str(final_by_id[audit_id].get("audited_scenario_sha256", "")),
            str(final_by_id[audit_id].get("registered_prompt_set_sha256", "")),
        )
        for audit_id in accepted_ids
    )
    if any(not scenario_hash or not prompt_hash for _, scenario_hash, prompt_hash in bindings):
        raise ValueError(f"Round {round_id} final audit has a missing accepted scenario binding")
    if _canonical_sha256(bindings) != manifest.get("accepted_scenario_bindings_sha256"):
        raise ValueError(f"Round {round_id} accepted scenario-binding hash does not match its final audit")

    prompts = _read_jsonl(evidence_path, label=f"round {round_id} accepted-candidate evidence")
    prompt_ids = _unique_values(prompts, "prompt_id", label=f"round {round_id} accepted-candidate evidence")
    if _canonical_sha256(sorted(prompt_ids)) != manifest.get("accepted_prompt_ids_sha256"):
        raise ValueError(f"Round {round_id} accepted prompt-ID hash does not match its evidence")
    if len(prompts) != manifest.get("accepted_prompt_rows"):
        raise ValueError(f"Round {round_id} evidence prompt count does not match its manifest")
    if any(
        row.get("round_id") != round_id
        or row.get("round_evidence_only") is not True
        or row.get("audit_gate_passed") is not False
        or row.get("accepted_only_source") is not False
        or row.get("audit_gate_version") != ROUND_EVIDENCE_VERSION
        for row in prompts
    ):
        raise ValueError(f"Round {round_id} evidence rows have invalid non-runnable markers")
    prompt_audit_ids = {str(row.get("audit_id", "")).strip() for row in prompts}
    if "" in prompt_audit_ids or prompt_audit_ids != accepted_ids:
        raise ValueError(f"Round {round_id} evidence prompts do not match accepted final-audit IDs")
    for row in prompts:
        final_row = final_by_id[str(row["audit_id"])]
        if (
            str(row.get("audited_scenario_sha256", "")) != str(final_row["audited_scenario_sha256"])
            or str(row.get("registered_prompt_set_sha256", "")) != str(final_row["registered_prompt_set_sha256"])
        ):
            raise ValueError(f"Round {round_id} evidence prompt has a scenario-binding mismatch")
    arms: dict[str, list[str]] = defaultdict(list)
    for row in prompts:
        arms[str(row["audit_id"])].append(str(row.get("condition_id", "")))
    if any(len(values) != len(CONFIRMATORY_CONDITIONS) or set(values) != CONFIRMATORY_CONDITIONS for values in arms.values()):
        raise ValueError(f"Round {round_id} evidence does not retain every registered condition exactly once")
    return manifest, prompts


def combine_boundary_rounds(
    *,
    round_manifest_paths: list[Path],
    accepted_prompts_path: Path,
) -> dict[str, Any]:
    """Release the union of accepted fixed rounds only when at least 160 pass."""
    if len(round_manifest_paths) != 2:
        raise ValueError("Combined release requires exactly two fixed round manifests: round1 and round2")
    resolved_manifests = [Path(path).resolve() for path in round_manifest_paths]
    if len(set(resolved_manifests)) != len(resolved_manifests):
        raise ValueError("Combined release contains a duplicate round manifest")
    accepted_prompts_path = Path(accepted_prompts_path)
    output_manifest_path = accepted_prompts_path.with_suffix(".manifest.json")
    if accepted_prompts_path.exists() or output_manifest_path.exists():
        raise RuntimeError("Combined accepted-only output already exists and will not be overwritten")

    round_data = [_load_round(path) for path in resolved_manifests]
    round_ids = [str(manifest["round_id"]) for manifest, _ in round_data]
    if set(round_ids) != {"round1", "round2"}:
        raise ValueError("Combined release requires exactly the fixed round1 and round2 evidence bundles")
    indexed_rounds = {str(manifest["round_id"]): (path, manifest, prompts) for path, (manifest, prompts) in zip(resolved_manifests, round_data)}
    resolved_manifests = [indexed_rounds[round_id][0] for round_id in ("round1", "round2")]
    round_data = [(indexed_rounds[round_id][1], indexed_rounds[round_id][2]) for round_id in ("round1", "round2")]
    round_ids = ["round1", "round2"]
    round1_manifest, round2_manifest = round_data[0][0], round_data[1][0]
    if round1_manifest.get("audited_scenarios") != 200 or round1_manifest.get("accepted_scenarios") != 84:
        raise ValueError("Frozen Round 1 evidence must remain exactly 84 accepted of 200 audited scenarios")
    if round2_manifest.get("audited_scenarios") != 240:
        raise ValueError("Fixed Round 2 evidence must contain exactly 240 audited scenarios")
    all_rows = [row for _, prompts in round_data for row in prompts]
    audit_ids = [str(row["audit_id"]) for row in all_rows]
    prompt_ids = [str(row["prompt_id"]) for row in all_rows]
    base_ids = [str(row.get("base_scenario_id", "")).strip() for row in all_rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Combined release contains duplicate prompt IDs across rounds")
    if "" in base_ids:
        raise ValueError("Combined release contains a missing base_scenario_id")
    audit_round_pairs = {(str(row["round_id"]), str(row["audit_id"])) for row in all_rows}
    unique_audit_ids = set(audit_ids)
    if len(unique_audit_ids) != sum(int(manifest["accepted_scenarios"]) for manifest, _ in round_data):
        raise ValueError("Combined release contains duplicate audit IDs across rounds")
    base_round_pairs = {(str(row["round_id"]), str(row["base_scenario_id"])) for row in all_rows}
    if len({base_id for base_id in base_ids}) != len(base_round_pairs):
        raise ValueError("Combined release contains duplicate base scenario IDs across rounds")
    del audit_round_pairs
    accepted_scenarios = len(unique_audit_ids)
    if accepted_scenarios < DEFAULT_MIN_ACCEPTED:
        raise RuntimeError(
            f"STOP: only {accepted_scenarios} unique scenarios passed across fixed rounds; "
            f"at least {DEFAULT_MIN_ACCEPTED} are required"
        )

    released: list[dict[str, Any]] = []
    for row in all_rows:
        cleaned = {key: value for key, value in row.items() if key not in {"round_id", "round_evidence_only"}}
        released.append(
            {
                **cleaned,
                "source_round_id": str(row["round_id"]),
                "audit_gate_passed": True,
                "accepted_only_source": True,
                "audit_gate_version": AUDIT_GATE_VERSION,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_GATE_VERSION,
        "status": "accepted_only",
        "release_kind": RELEASE_KIND,
        "minimum_accepted_scenarios": DEFAULT_MIN_ACCEPTED,
        "accepted_scenarios": accepted_scenarios,
        "accepted_prompt_rows": len(released),
        "source_rounds": round_ids,
        "registered_conditions": sorted(CONFIRMATORY_CONDITIONS),
        "approved_prefix_policy_by_model": APPROVED_PREFIX_POLICY_BY_MODEL,
        "accepted_audit_ids_sha256": _canonical_sha256(sorted(unique_audit_ids)),
        "accepted_prompt_ids_sha256": _canonical_sha256(sorted(prompt_ids)),
        "accepted_scenario_bindings_sha256": _canonical_sha256(
            sorted(
                {
                    (
                        str(row["audit_id"]),
                        str(row.get("audited_scenario_sha256", "")),
                        str(row.get("registered_prompt_set_sha256", "")),
                    )
                    for row in released
                }
            )
        ),
        "source_round_manifests": [
            {
                "round_id": str(source["round_id"]),
                "path": os.path.relpath(path, accepted_prompts_path.parent),
                "sha256": sha256_file(path),
                "accepted_scenarios": int(source["accepted_scenarios"]),
            }
            for path, (source, _) in zip(resolved_manifests, round_data)
        ],
    }
    accepted_prompts_path.parent.mkdir(parents=True, exist_ok=True)
    created_outputs: list[Path] = []
    try:
        with accepted_prompts_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in released:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        created_outputs.append(accepted_prompts_path)
        manifest["accepted_prompts_sha256"] = sha256_file(accepted_prompts_path)
        with output_manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        created_outputs.append(output_manifest_path)
    except FileExistsError as exc:
        for created_path in created_outputs:
            created_path.unlink(missing_ok=True)
        raise RuntimeError(f"Combined accepted-only output already exists due to a concurrent writer: {exc.filename}") from exc
    except Exception:
        for created_path in created_outputs:
            created_path.unlink(missing_ok=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-manifest", type=Path, action="append", required=True)
    parser.add_argument("--accepted-prompts", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = combine_boundary_rounds(
            round_manifest_paths=args.round_manifest,
            accepted_prompts_path=args.accepted_prompts,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print(
        f"Combined gate passed: {manifest['accepted_scenarios']} scenarios and "
        f"{manifest['accepted_prompt_rows']} prompts released."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
