"""Build the eight-arm mechanism release from the fixed Round 1 + Round 2 gate."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from build_boundary_recovery_dataset import (
    audit_content_fields,
    audited_scenario_sha256_from_row,
    bind_archived_v0_context,
    load_archived_v0_prompts,
)
from build_main_dataset import build_main_scenarios
from build_mechanism_decomposition_dataset import (
    MECHANISM_CONDITIONS,
    OUTPUT_DIR,
    STUDY_KIND,
    build_mechanism_records,
)
from finalize_boundary_audit import AUDIT_CRITERIA
from finalize_mechanism_audit import APPROVED_PREFIX_POLICY_BY_MODEL, AUDIT_GATE_VERSION


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DIR = ROOT / "boundary_recovery_study"
DEFAULT_BOUNDARY_PROMPTS = BOUNDARY_DIR / "accepted_only_prompts.jsonl"
DEFAULT_BOUNDARY_MANIFEST = BOUNDARY_DIR / "accepted_only_prompts.manifest.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "accepted_only_prompts.jsonl"
RELEASE_KIND = "combined_fixed_rounds_mechanism_v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _group_accepted_evidence(
    rows: Iterable[dict[str, Any]],
    *,
    expected_scenarios: int,
    registered_conditions: set[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        frozen_id = str(row.get("audit_id", "")).strip()
        if not frozen_id:
            raise ValueError("Round evidence contains an empty audit ID")
        if (
            row.get("audit_gate_version") != "boundary_round_audit_v1"
            or row.get("round_evidence_only") is not True
        ):
            raise ValueError(f"Round evidence has an invalid evidence-only marker for {frozen_id}")
        grouped[frozen_id].append(dict(row))
    if len(grouped) != expected_scenarios:
        raise ValueError("Round evidence scenario count does not match its manifest")
    for frozen_id, group in grouped.items():
        observed = [str(row.get("condition_id", "")) for row in group]
        if len(observed) != len(registered_conditions) or set(observed) != registered_conditions:
            raise ValueError(f"Round evidence has an incomplete condition block for {frozen_id}")
        content_hashes = {str(row.get("audited_scenario_sha256", "")) for row in group}
        final_hashes = {str(row.get("final_audit_record_sha256", "")) for row in group}
        if len(content_hashes) != 1 or "" in content_hashes or len(final_hashes) != 1 or "" in final_hashes:
            raise ValueError(f"Round evidence has inconsistent audit bindings for {frozen_id}")
    return grouped


def _normalized_final_audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    numeric_fields = [f"final_{criterion}" for criterion in AUDIT_CRITERIA]
    numeric_fields.extend(["final_overall_accept", "accepted_all_six_criteria"])
    for field in numeric_fields:
        normalized[field] = int(str(normalized.get(field, "")).strip())
    return normalized


def _validate_final_audit(
    *,
    round_manifest: Mapping[str, Any],
    round_manifest_path: Path,
    evidence_groups: Mapping[str, list[dict[str, Any]]],
) -> None:
    final_path = round_manifest_path.parent / str(round_manifest.get("final_audit_file", ""))
    if not final_path.is_file() or _sha256_file(final_path) != str(round_manifest.get("final_audit_sha256", "")):
        raise ValueError(f"Final audit hash mismatch for {round_manifest.get('round_id')}")
    final_rows = _read_csv(final_path)
    final_by_id = {str(row.get("audit_id", "")).strip(): row for row in final_rows}
    if "" in final_by_id or len(final_by_id) != len(final_rows):
        raise ValueError("Final audit IDs must be nonempty and unique")
    if len(final_rows) != int(round_manifest.get("audited_scenarios", -1)):
        raise ValueError("Final audit scenario count does not match its manifest")
    accepted_ids = {
        frozen_id
        for frozen_id, row in final_by_id.items()
        if all(str(row.get(f"final_{criterion}", "")).strip() == "1" for criterion in AUDIT_CRITERIA)
        and str(row.get("final_overall_accept", "")).strip() == "1"
        and str(row.get("accepted_all_six_criteria", "")).strip() == "1"
    }
    if accepted_ids != set(evidence_groups):
        raise ValueError("Final audit accepted IDs differ from the round evidence")
    if _canonical_sha256(sorted(accepted_ids)) != str(round_manifest.get("accepted_audit_ids_sha256", "")):
        raise ValueError("Final audit accepted-ID hash differs from the round manifest")
    bindings: list[tuple[str, str, str]] = []
    for frozen_id in sorted(accepted_ids):
        final_row = final_by_id[frozen_id]
        evidence = evidence_groups[frozen_id][0]
        if (
            str(final_row.get("audited_scenario_sha256", "")) != str(evidence.get("audited_scenario_sha256", ""))
            or str(final_row.get("registered_prompt_set_sha256", "")) != str(evidence.get("registered_prompt_set_sha256", ""))
        ):
            raise ValueError(f"Final audit binding differs from evidence for {frozen_id}")
        if _canonical_sha256(_normalized_final_audit_row(final_row)) != str(evidence.get("final_audit_record_sha256", "")):
            raise ValueError(f"Final audit record hash differs from evidence for {frozen_id}")
        bindings.append(
            (
                frozen_id,
                str(final_row["audited_scenario_sha256"]),
                str(final_row["registered_prompt_set_sha256"]),
            )
        )
    if _canonical_sha256(bindings) != str(round_manifest.get("accepted_scenario_bindings_sha256", "")):
        raise ValueError("Final audit scenario-binding hash differs from the round manifest")


def _load_round_evidence(
    *,
    boundary_dir: Path,
    combined_manifest: Mapping[str, Any],
    round_manifest_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    entries = combined_manifest.get("source_round_manifests")
    if not isinstance(entries, list) or [entry.get("round_id") for entry in entries] != ["round1", "round2"]:
        raise ValueError("Combined boundary manifest must bind fixed Round 1 and Round 2")
    registered = set(str(value) for value in combined_manifest.get("registered_conditions", []))
    if not registered:
        raise ValueError("Combined boundary manifest has no registered conditions")
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for entry in entries:
        round_id = str(entry["round_id"])
        manifest_path = boundary_dir / str(entry.get("path", ""))
        if not manifest_path.is_file() or _sha256_file(manifest_path) != str(entry.get("sha256", "")):
            raise ValueError(f"Source manifest hash mismatch for {round_id}")
        round_manifest = dict(
            (round_manifest_overrides or {}).get(round_id, _read_json(manifest_path))
        )
        expected = int(entry.get("accepted_scenarios", -1))
        if (
            round_manifest.get("status") != "round_audit_finalized"
            or round_manifest.get("round_id") != round_id
            or int(round_manifest.get("accepted_scenarios", -1)) != expected
        ):
            raise ValueError(f"Source manifest content mismatch for {round_id}")
        evidence_path = manifest_path.parent / str(round_manifest.get("evidence_prompts_file", ""))
        if not evidence_path.is_file() or _sha256_file(evidence_path) != str(round_manifest.get("accepted_prompts_sha256", "")):
            raise ValueError(f"Accepted evidence hash mismatch for {round_id}")
        groups = _group_accepted_evidence(
            _read_jsonl(evidence_path),
            expected_scenarios=expected,
            registered_conditions=registered,
        )
        if any(str(row.get("round_id", "")) != round_id for group in groups.values() for row in group):
            raise ValueError(f"Round evidence carries the wrong round ID for {round_id}")
        _validate_final_audit(
            round_manifest=round_manifest,
            round_manifest_path=manifest_path,
            evidence_groups=groups,
        )
        output[round_id] = groups
    return output


def _scenario_from_round2(row: Mapping[str, Any]) -> dict[str, Any]:
    scenario = {**row, "scenario_id": str(row["base_scenario_id"]), "source_round_id": "round2"}
    observed = audited_scenario_sha256_from_row(audit_content_fields(scenario))
    if observed != str(row.get("audited_scenario_sha256", "")):
        raise ValueError(f"Round 2 scenario content hash mismatch for {row.get('audit_id')}")
    return scenario


def _prepare_scenarios(
    round_evidence: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    round1_groups = round_evidence["round1"]
    main_by_id = {str(row["scenario_id"]): row for row in build_main_scenarios()}
    round1_representatives = [group[0] for group in round1_groups.values()]
    missing = {str(row["base_scenario_id"]) for row in round1_representatives} - set(main_by_id)
    if missing:
        raise ValueError(f"Round 1 evidence references unknown base scenarios: {sorted(missing)[:3]}")
    bound_round1 = bind_archived_v0_context(
        [main_by_id[str(row["base_scenario_id"])] for row in round1_representatives],
        load_archived_v0_prompts(),
    )
    evidence_by_base = {str(row["base_scenario_id"]): row for row in round1_representatives}
    scenarios: list[dict[str, Any]] = []
    final_hash_by_audit: dict[str, str] = {}
    for source in bound_round1:
        evidence = evidence_by_base[str(source["scenario_id"])]
        scenario = {
            **source,
            "audit_id": evidence["audit_id"],
            "audited_scenario_sha256": evidence["audited_scenario_sha256"],
            "ethical_option": evidence["ethical_option"],
            "source_round_id": "round1",
        }
        if audited_scenario_sha256_from_row(audit_content_fields(scenario)) != evidence["audited_scenario_sha256"]:
            raise ValueError(f"Round 1 scenario content hash mismatch for {evidence['audit_id']}")
        scenarios.append(scenario)
        final_hash_by_audit[str(evidence["audit_id"])] = str(evidence["final_audit_record_sha256"])
    for group in round_evidence["round2"].values():
        evidence = group[0]
        scenario = _scenario_from_round2(evidence)
        scenarios.append(scenario)
        final_hash_by_audit[str(evidence["audit_id"])] = str(evidence["final_audit_record_sha256"])
    if len({str(row["scenario_id"]) for row in scenarios}) != len(scenarios):
        raise ValueError("Combined mechanism scenarios have duplicate base IDs")
    if len(final_hash_by_audit) != len(scenarios):
        raise ValueError("Combined mechanism scenarios have duplicate audit IDs")
    return scenarios, final_hash_by_audit


def prepare_combined_mechanism_release(
    root: Path = ROOT,
    *,
    boundary_manifest: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    boundary_dir = root / "boundary_recovery_study"
    boundary_prompts_path = boundary_dir / "accepted_only_prompts.jsonl"
    boundary_manifest_path = boundary_dir / "accepted_only_prompts.manifest.json"
    manifest = dict(boundary_manifest) if boundary_manifest is not None else _read_json(boundary_manifest_path)
    if (
        manifest.get("schema_version") != AUDIT_GATE_VERSION
        or manifest.get("status") != "accepted_only"
        or manifest.get("release_kind") != "combined_fixed_rounds_v1"
        or manifest.get("source_rounds") != ["round1", "round2"]
    ):
        raise ValueError("Boundary input is not the combined fixed-round accepted-only release")
    if _sha256_file(boundary_prompts_path) != str(manifest.get("accepted_prompts_sha256", "")):
        raise ValueError("Combined accepted-only prompt hash does not match its manifest")
    boundary_prompts = _read_jsonl(boundary_prompts_path)
    if len(boundary_prompts) != int(manifest.get("accepted_prompt_rows", -1)):
        raise ValueError("Combined accepted-only prompt count does not match its manifest")
    round_evidence = _load_round_evidence(boundary_dir=boundary_dir, combined_manifest=manifest)
    scenarios, final_hash_by_audit = _prepare_scenarios(round_evidence)
    if len(scenarios) != int(manifest.get("accepted_scenarios", -1)):
        raise ValueError("Combined accepted scenario count does not match its manifest")
    mechanism_rows = build_mechanism_records(scenarios)
    accepted = [
        {
            **row,
            "audit_gate_passed": True,
            "accepted_only_source": True,
            "audit_gate_version": AUDIT_GATE_VERSION,
            "final_audit_record_sha256": final_hash_by_audit[str(row["audit_id"])],
        }
        for row in mechanism_rows
    ]
    audit_ids = sorted({str(row["audit_id"]) for row in accepted})
    prompt_ids = sorted(str(row["prompt_id"]) for row in accepted)
    release_manifest = {
        "schema_version": AUDIT_GATE_VERSION,
        "status": "accepted_only",
        "release_kind": RELEASE_KIND,
        "study_kind": STUDY_KIND,
        "source_rounds": ["round1", "round2"],
        "minimum_accepted_scenarios": 160,
        "audited_scenarios": 440,
        "accepted_scenarios": len(audit_ids),
        "rejected_scenarios": 116,
        "registered_conditions": list(MECHANISM_CONDITIONS),
        "arms_per_scenario": len(MECHANISM_CONDITIONS),
        "accepted_prompt_rows": len(accepted),
        "accepted_audit_ids_sha256": _canonical_sha256(audit_ids),
        "accepted_prompt_ids_sha256": _canonical_sha256(prompt_ids),
        "source_boundary_manifest_sha256": _sha256_file(boundary_manifest_path),
        "source_boundary_prompts_sha256": _sha256_file(boundary_prompts_path),
        "source_boundary_manifest_path": "../boundary_recovery_study/accepted_only_prompts.manifest.json",
        "source_boundary_prompts_path": "../boundary_recovery_study/accepted_only_prompts.jsonl",
        "source_round_manifests": [
            {**entry, "path": f"../boundary_recovery_study/{entry['path']}"}
            for entry in manifest["source_round_manifests"]
        ],
        "approved_prefix_policy_by_model": APPROVED_PREFIX_POLICY_BY_MODEL,
    }
    return sorted(accepted, key=lambda row: str(row["prompt_id"])), release_manifest


def main() -> int:
    try:
        rows, manifest = prepare_combined_mechanism_release(ROOT)
        output = DEFAULT_OUTPUT
        _write_jsonl(output, rows)
        completed_manifest = {
            **manifest,
            "accepted_prompts_sha256": _sha256_file(output),
        }
        output.with_suffix(".manifest.json").write_text(
            json.dumps(completed_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    print(
        f"Combined mechanism gate passed: {manifest['accepted_scenarios']} scenarios and "
        f"{manifest['accepted_prompt_rows']} prompt rows released."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
