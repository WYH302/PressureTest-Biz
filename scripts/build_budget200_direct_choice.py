"""Build the frozen 32-scenario/192-prompt low-cost direct-choice panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / "boundary_recovery_study" / "accepted_only_prompts.jsonl"
DEFAULT_OUTPUT = (
    ROOT
    / "boundary_recovery_study"
    / "direct_choice_budget200"
    / "prompts_32x6.jsonl"
)
SEED = "pressuretest-biz-direct-choice-budget200-v1"
CONDITIONS = (
    "B0_baseline",
    "C0_neutral_length",
    "P1_cognitive_load",
    "P2_resource_scarcity",
    "P3_combined_pressure",
    "R3_constraint_regrounding",
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank(*parts: str) -> str:
    return hashlib.sha256("\x1f".join((SEED, *parts)).encode("utf-8")).hexdigest()


def _portable_reference(path: Path, *, relative_to: Path) -> str:
    try:
        return os.path.relpath(path, relative_to)
    except ValueError:
        return str(path)


def _scenario_rows(prompts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prompts:
        if row.get("source_round_id") == "round2":
            grouped[str(row.get("audit_id", ""))].append(row)
    expected = set(CONDITIONS)
    for audit_id, arms in grouped.items():
        if not audit_id or len(arms) != len(CONDITIONS) or {str(row.get("condition_id")) for row in arms} != expected:
            raise ValueError(f"Round-2 scenario {audit_id!r} does not contain the complete six-condition block")
    return dict(grouped)


def _select_audit_ids(prompts: list[dict[str, Any]]) -> list[str]:
    scenarios = _scenario_rows(prompts)
    exemplars = {audit_id: arms[0] for audit_id, arms in scenarios.items()}
    subdomains = sorted({str(row.get("subdomain", "")) for row in exemplars.values()})
    if len(subdomains) != 8 or "" in subdomains:
        raise ValueError("Budget panel requires exactly eight nonempty Round-2 subdomains")

    selected: list[str] = []
    for subdomain in subdomains:
        audit_ids = [
            audit_id
            for audit_id, row in exemplars.items()
            if row.get("subdomain") == subdomain
        ]
        by_family: dict[str, list[str]] = defaultdict(list)
        for audit_id in audit_ids:
            by_family[str(exemplars[audit_id].get("action_family_id", ""))].append(audit_id)
        if len(by_family) != 5 or "" in by_family:
            raise ValueError(f"Subdomain {subdomain!r} must contain exactly five Round-2 action families")

        chosen_families = sorted(by_family, key=lambda family: _rank(subdomain, "family", family))[:4]
        option_targets = {
            family: ("A" if index < 2 else "B")
            for index, family in enumerate(chosen_families)
        }
        for family in chosen_families:
            target = option_targets[family]
            candidates = [
                audit_id
                for audit_id in by_family[family]
                if exemplars[audit_id].get("ethical_option") == target
            ]
            if not candidates:
                raise ValueError(
                    f"Subdomain {subdomain!r}, family {family!r} lacks ethical-option {target}"
                )
            selected.append(min(candidates, key=lambda audit_id: _rank(subdomain, family, target, audit_id)))

    if len(selected) != 32 or len(set(selected)) != 32:
        raise ValueError("Budget-panel selection did not yield 32 unique scenarios")
    return sorted(selected, key=lambda audit_id: _rank("panel-order", audit_id))


def _validate_parent(parent_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_path = parent_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Frozen parent prompt manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "boundary_audit_gate_v1"
        or manifest.get("status") != "accepted_only"
        or manifest.get("release_kind") != "combined_fixed_rounds_v1"
        or manifest.get("accepted_scenarios") != 324
        or manifest.get("accepted_prompt_rows") != 1944
        or manifest.get("accepted_prompts_sha256") != _file_sha256(parent_path)
    ):
        raise ValueError("Frozen parent is not the verified 324-scenario accepted-only release")
    if len(rows) != 1944:
        raise ValueError("Frozen parent prompt row count is not 1,944")
    return manifest


def build_budget_panel(parent_path: Path, output_path: Path) -> dict[str, Any]:
    """Create the deterministic Round-2-only 32x6 panel and its sidecar manifest."""
    parent_path = parent_path.resolve()
    output_path = output_path.resolve()
    parent_rows = _load_jsonl(parent_path)
    parent_manifest = _validate_parent(parent_path, parent_rows)
    selected_audit_ids = _select_audit_ids(parent_rows)
    selected_set = set(selected_audit_ids)
    panel_rows = [row for row in parent_rows if str(row.get("audit_id", "")) in selected_set]
    if len(panel_rows) != 192:
        raise ValueError("Budget panel must contain exactly 192 prompt rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in panel_rows),
        encoding="utf-8",
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    parent_manifest_path = parent_path.with_suffix(".manifest.json")
    exemplars = {}
    for row in panel_rows:
        exemplars.setdefault(str(row["audit_id"]), row)
    scenario_rows = list(exemplars.values())
    excluded_family_by_subdomain: dict[str, str] = {}
    parent_round2 = _scenario_rows(parent_rows)
    for subdomain in sorted({str(row["subdomain"]) for row in scenario_rows}):
        all_families = {
            str(arms[0]["action_family_id"])
            for arms in parent_round2.values()
            if arms[0]["subdomain"] == subdomain
        }
        chosen_families = {
            str(row["action_family_id"])
            for row in scenario_rows
            if row["subdomain"] == subdomain
        }
        excluded = sorted(all_families - chosen_families)
        if len(excluded) != 1:
            raise ValueError(f"Subdomain {subdomain!r} must exclude exactly one action family")
        excluded_family_by_subdomain[subdomain] = excluded[0]

    manifest = {
        "schema_version": "boundary_budget200_panel_v1",
        "status": "frozen_budget_panel",
        "study_kind": "boundary_direct_choice_budget200_v1",
        "selection_seed": SEED,
        "selection_uses_target_outputs": False,
        "source_round_id": "round2",
        "parent_prompts_path": _portable_reference(parent_path, relative_to=manifest_path.parent),
        "parent_prompts_sha256": _file_sha256(parent_path),
        "parent_manifest_path": _portable_reference(parent_manifest_path, relative_to=manifest_path.parent),
        "parent_manifest_sha256": _file_sha256(parent_manifest_path),
        "parent_manifest_declared_prompts_sha256": parent_manifest["accepted_prompts_sha256"],
        "accepted_parent_scenarios": parent_manifest["accepted_scenarios"],
        "selected_scenarios": 32,
        "selected_prompt_rows": 192,
        "registered_conditions": list(CONDITIONS),
        "selected_audit_ids_sha256": _canonical_sha256(sorted(selected_audit_ids)),
        "selected_prompt_ids_sha256": _canonical_sha256(sorted(str(row["prompt_id"]) for row in panel_rows)),
        "panel_prompts_sha256": _file_sha256(output_path),
        "quotas": {
            "subdomains": dict(sorted(Counter(str(row["subdomain"]) for row in scenario_rows).items())),
            "ethical_option": dict(sorted(Counter(str(row["ethical_option"]) for row in scenario_rows).items())),
            "unique_action_families": len({str(row["action_family_id"]) for row in scenario_rows}),
            "conditions": dict(sorted(Counter(str(row["condition_id"]) for row in panel_rows).items())),
        },
        "excluded_action_family_by_subdomain": excluded_family_by_subdomain,
        "generation_policy": {
            "formal_prompt_rows_per_model": 192,
            "temperature": 0.0,
            "replicate_id": "r01",
            "maximum_output_tokens": 8,
            "api_retries": 0,
            "formal_provider_request_budget_per_model": 192,
            "hosted_smoke_request_budget_per_model": 1,
        },
        "approved_prefix_policy_by_model": parent_manifest["approved_prefix_policy_by_model"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_budget_panel(args.parent, args.output)
    print(
        f"Frozen budget panel: {manifest['selected_scenarios']} scenarios, "
        f"{manifest['selected_prompt_rows']} prompts -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
