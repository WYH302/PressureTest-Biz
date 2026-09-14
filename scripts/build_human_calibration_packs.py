"""Build separate, hash-bound 20-item calibration packs for both human audits.

Calibration items are training material only.  They are selected from the
complement of the confirmatory inventories and never enter reliability or
effect estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from build_archived_human_validation_pack import (
    PUBLIC_STIMULUS_FIELDS,
    canonical_stimulus_sha256 as response_stimulus_sha256,
)
from build_archived_human_validation_pack import _blind_scenario, _response_action
from build_boundary_recovery_dataset import AUDIT_CONTENT_FIELDS, AUDIT_CRITERIA, audit_content_fields


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_OUTPUT = ROOT / "boundary_recovery_study" / "calibration"
RESPONSE_OUTPUT = ROOT / "archived_human_validation_v2" / "calibration"
SCENARIO_SOURCE = ROOT / "main_dataset" / "mother_scenarios_1000_v2_detailed.jsonl"
SCENARIO_FORMAL_INVENTORY = ROOT / "boundary_recovery_study" / "scenario_sample_manifest_200.jsonl"
PROMPT_SOURCE = ROOT / "main_dataset" / "main_prompts_v2_detailed_action_choice.jsonl"
RESPONSE_FORMAL_INVENTORY = ROOT / "archived_human_validation_v2" / "private" / "private_response_manifest.jsonl"
MATRIX_STATUS = ROOT / "results" / "four_by_five_crossed_matrix_submission_20260716" / "matrix_status.csv"
SAMPLE_SIZE = 20
SCENARIO_PROTOCOL = "scenario_calibration_v1"
RESPONSE_PROTOCOL = "archived_response_calibration_v1"
SCENARIO_SEED = "pressuretest-biz-scenario-calibration-v1"
RESPONSE_SEED = "pressuretest-biz-archived-response-calibration-v1"
PERSONNEL_FIELDS = ("pseudonym", "role", "training_date", "codebook_version")


def _canonical_hash(namespace: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{namespace}\x1f{payload}".encode("utf-8")).hexdigest()


def _rank(seed: str, *parts: Any) -> str:
    return hashlib.sha256("\x1f".join((seed, *map(str, parts))).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_stimulus_sha256(row: dict[str, Any]) -> str:
    fields = ("calibration_id", "protocol_version", *AUDIT_CONTENT_FIELDS)
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"Scenario calibration stimulus is missing fields: {', '.join(missing)}")
    return _canonical_hash("scenario-calibration-stimulus-v1", {field: str(row[field]) for field in fields})


def _annotation_rows(public_rows: Sequence[dict[str, Any]], *, kind: str, annotator_id: str) -> list[dict[str, Any]]:
    if kind == "scenario":
        rows = [
            {
                "annotator_id": annotator_id,
                "annotator_pseudonym": "",
                **public,
                **{criterion: "" for criterion in AUDIT_CRITERIA},
                "overall_accept": "",
                "confidence_1_to_5": "",
                "notes": "",
            }
            for public in public_rows
        ]
        id_field, seed = "calibration_id", SCENARIO_SEED
    elif kind == "archived_response":
        rows = [
            {
                "annotator_id": annotator_id,
                "annotator_pseudonym": "",
                **public,
                "selected_action_label": "",
                "norm_deviation_0_to_3": "",
                "confidence_1_to_5": "",
                "notes": "",
            }
            for public in public_rows
        ]
        id_field, seed = "response_audit_id", RESPONSE_SEED
    else:
        raise ValueError(f"Unknown calibration kind: {kind}")
    return sorted(rows, key=lambda row: _rank(seed, f"annotator-{annotator_id}", row[id_field]))


def _balanced_scenario_complement(candidates: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    by_subdomain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_subdomain[str(row.get("subdomain", ""))].append(row)
    if not by_subdomain:
        raise ValueError("No eligible scenario calibration records remain outside the formal inventory")
    groups = sorted(by_subdomain)
    base, remainder = divmod(sample_size, len(groups))
    selected: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        quota = base + int(index < remainder)
        ranked = sorted(by_subdomain[group], key=lambda row: _rank(SCENARIO_SEED, row["scenario_id"]))
        if len(ranked) < quota:
            raise ValueError(f"Subdomain {group!r} has fewer than {quota} calibration candidates")
        selected.extend(ranked[:quota])
    if len(selected) != sample_size:
        raise AssertionError("Internal scenario calibration sampling error")
    return selected


def build_scenario_calibration_pack(
    scenario_rows: Iterable[dict[str, Any]],
    *,
    formal_ids: set[str],
    sample_size: int = SAMPLE_SIZE,
) -> dict[str, list[dict[str, Any]]]:
    if sample_size != SAMPLE_SIZE:
        raise ValueError("The registered calibration size is exactly 20")
    scenarios = [dict(row) for row in scenario_rows]
    ids = [str(row.get("scenario_id", "")).strip() for row in scenarios]
    if not scenarios or "" in ids or len(set(ids)) != len(ids):
        raise ValueError("Scenario source IDs must be nonempty and unique")
    candidates = [row for row in scenarios if str(row["scenario_id"]) not in formal_ids]
    selected = _balanced_scenario_complement(candidates, sample_size)
    private_manifest: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for source in selected:
        source_id = str(source["scenario_id"])
        calibration_id = f"CAL-SC-{_rank(SCENARIO_SEED, source_id)[:12].upper()}"
        public = {
            "calibration_id": calibration_id,
            "protocol_version": SCENARIO_PROTOCOL,
            **audit_content_fields(source),
        }
        public = {**public, "stimulus_sha256": scenario_stimulus_sha256(public)}
        public_rows.append(public)
        private_manifest.append(
            {
                **public,
                "source_scenario_id": source_id,
                "selection_status": "outside_formal_scenario_inventory",
            }
        )
    resolution = [
        {
            "calibration_id": row["calibration_id"],
            "stimulus_sha256": row["stimulus_sha256"],
            **{f"consensus_{criterion}": "" for criterion in AUDIT_CRITERIA},
            "consensus_overall_accept": "",
            "calibration_discussion_note": "",
        }
        for row in sorted(public_rows, key=lambda item: item["calibration_id"])
    ]
    return {
        "private_manifest": sorted(private_manifest, key=lambda row: row["calibration_id"]),
        "annotator_A": _annotation_rows(public_rows, kind="scenario", annotator_id="A"),
        "annotator_B": _annotation_rows(public_rows, kind="scenario", annotator_id="B"),
        "resolution": resolution,
    }


def _unique_index(rows: Iterable[dict[str, Any]], fields: tuple[str, ...], label: str) -> dict[tuple[str, ...], dict[str, Any]]:
    output: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")).strip() for field in fields)
        if "" in key or key in output:
            raise ValueError(f"{label} keys must be nonempty and unique: {key}")
        output[key] = dict(row)
    return output


def _public_response_stimulus(audit_id: str, prompt: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    parsed = response.get("parsed_response", {})
    public = {
        "response_audit_id": audit_id,
        "protocol_version": RESPONSE_PROTOCOL,
        "scenario_text": _blind_scenario(prompt.get("scenario_text", prompt.get("prompt_text", ""))),
        "preserving_action": str(prompt.get("ethical_action", "")),
        "conflicting_action": str(prompt.get("shortcut_action", "")),
        "response_action": _response_action(response),
        "response_reason": str(parsed.get("reason", "")),
        "response_next_step": str(parsed.get("operational_next_step", "")),
    }
    return {**public, "stimulus_sha256": response_stimulus_sha256(public)}


def _select_response_complement(
    responses: list[dict[str, Any]],
    *,
    formal_response_keys: set[tuple[str, str]],
    expected_models: Sequence[str],
) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in responses:
        key = (str(row.get("model", "")), str(row.get("prompt_id", "")))
        if key not in formal_response_keys:
            by_model[key[0]].append(row)
    if set(by_model) != set(expected_models):
        raise ValueError("Response calibration model inventory does not match the expected target models")
    selected: list[dict[str, Any]] = []
    for model in expected_models:
        rows = by_model[model]
        by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_variant[str(row.get("variant", ""))].append(row)
        model_selected: list[dict[str, Any]] = []
        for variant in sorted(by_variant):
            ranked = sorted(by_variant[variant], key=lambda row: _rank(RESPONSE_SEED, model, variant, row["prompt_id"]))
            if ranked and len(model_selected) < 5:
                model_selected.append(ranked[0])
        used = {(str(row["model"]), str(row["prompt_id"])) for row in model_selected}
        remaining = sorted(
            (row for row in rows if (str(row["model"]), str(row["prompt_id"])) not in used),
            key=lambda row: _rank(RESPONSE_SEED, model, "fill", row["prompt_id"]),
        )
        model_selected.extend(remaining[: 5 - len(model_selected)])
        if len(model_selected) != 5:
            raise ValueError(f"Model {model!r} has fewer than five eligible calibration responses")
        selected.extend(model_selected)
    return selected


def build_response_calibration_pack(
    prompt_rows: Iterable[dict[str, Any]],
    response_rows: Iterable[dict[str, Any]],
    *,
    formal_response_keys: set[tuple[str, str]],
    expected_models: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    prompts = _unique_index(prompt_rows, ("prompt_id",), "Prompt")
    responses = list(_unique_index(response_rows, ("model", "prompt_id"), "Response").values())
    selected = _select_response_complement(
        responses,
        formal_response_keys=formal_response_keys,
        expected_models=expected_models,
    )
    public_rows: list[dict[str, Any]] = []
    private_manifest: list[dict[str, Any]] = []
    for response in selected:
        model, prompt_id = str(response["model"]), str(response["prompt_id"])
        prompt = prompts[(prompt_id,)]
        audit_id = f"CAL-HRA-{_rank(RESPONSE_SEED, model, prompt_id)[:12].upper()}"
        public = _public_response_stimulus(audit_id, prompt, response)
        public_rows.append(public)
        private_manifest.append(
            {
                **public,
                "model": model,
                "prompt_id": prompt_id,
                "base_scenario_id": str(response.get("base_scenario_id", prompt.get("base_scenario_id", ""))),
                "variant": str(response.get("variant", prompt.get("variant", ""))),
                "selection_status": "outside_formal_archived_response_inventory",
            }
        )
    if len(public_rows) != SAMPLE_SIZE:
        raise AssertionError("Archived-response calibration must contain exactly 20 items")
    resolution = [
        {
            "response_audit_id": row["response_audit_id"],
            "stimulus_sha256": row["stimulus_sha256"],
            "consensus_selected_action_label": "",
            "consensus_norm_deviation_0_to_3": "",
            "calibration_discussion_note": "",
        }
        for row in sorted(public_rows, key=lambda item: item["response_audit_id"])
    ]
    return {
        "private_manifest": sorted(private_manifest, key=lambda row: row["response_audit_id"]),
        "annotator_A": _annotation_rows(public_rows, kind="archived_response", annotator_id="A"),
        "annotator_B": _annotation_rows(public_rows, kind="archived_response", annotator_id="B"),
        "resolution": resolution,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty calibration CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_calibration_pack(
    output_dir: Path,
    pack: dict[str, list[dict[str, Any]]],
    *,
    calibration_kind: str,
    source_sha256: dict[str, str],
    source_relpath: dict[str, str] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(path.is_file() for path in output_dir.rglob("*")):
        raise RuntimeError(f"Calibration pack already exists and will not be overwritten: {output_dir}")
    if calibration_kind not in {"scenario", "archived_response"}:
        raise ValueError("calibration_kind must be scenario or archived_response")
    if len(pack.get("private_manifest", [])) != SAMPLE_SIZE:
        raise ValueError("Calibration pack must contain exactly 20 private manifest rows")
    if not source_sha256 or any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()) for value in source_sha256.values()):
        raise ValueError("Every source_sha256 value must be a 64-character hexadecimal digest")

    private_manifest = output_dir / "private" / "calibration_manifest.jsonl"
    resolution = output_dir / "private" / "calibration_resolution.csv"
    annotator_a = output_dir / "annotator_delivery" / "annotator_A" / "calibration.csv"
    annotator_b = output_dir / "annotator_delivery" / "annotator_B" / "calibration.csv"
    personnel = output_dir / "personnel_log_template.csv"
    readme = output_dir / "README.md"
    _write_jsonl(private_manifest, pack["private_manifest"])
    _write_csv(resolution, pack["resolution"])
    _write_csv(annotator_a, pack["annotator_A"])
    _write_csv(annotator_b, pack["annotator_B"])
    _write_csv(
        personnel,
        [
            {"pseudonym": "", "role": role, "training_date": "", "codebook_version": ""}
            for role in ("annotator_A", "annotator_B", "adjudicator")
        ],
    )
    readme.write_text(
        "# Human calibration pack\n\n"
        "Status: implementation complete; human calibration not yet run.\n\n"
        "These 20 training items are outside the formal analysis inventory. Give each annotator only their own delivery "
        "sheet and the applicable codebook. Resolve training disagreements in `private/calibration_resolution.csv`, freeze "
        "the codebook version, and ask Codex to run `scripts/finalize_human_calibration.py` on this directory before opening "
        "the formal annotation sheets. `pack_manifest.json` binds the blank issued pack; "
        "`calibration_completion_manifest.json` separately binds the completed A/B, resolution, and personnel files. "
        "Calibration decisions are excluded from reliability and outcome estimates. `personnel_log_template.csv` "
        "intentionally records only pseudonyms, roles, training dates, and codebook versions; use non-generic pseudonyms "
        "and do not add real identities to the anonymous pack.\n",
        encoding="utf-8",
    )
    artifacts = {
        "private_manifest": private_manifest,
        "resolution": resolution,
        "annotator_A": annotator_a,
        "annotator_B": annotator_b,
        "personnel_log_template": personnel,
        "readme": readme,
    }
    manifest = {
        "schema_version": "human_calibration_pack_v1",
        "status": "implementation_complete_human_run_pending",
        "calibration_kind": calibration_kind,
        "item_count": SAMPLE_SIZE,
        "analysis_policy": "training_only_excluded_from_reliability_and_effect_estimates",
        "identity_policy": "anonymous_pack_uses_pseudonym_role_training_date_codebook_version_only",
        "source_sha256": dict(sorted(source_sha256.items())),
        "source_relpath": dict(sorted((source_relpath or {}).items())),
        "artifact_sha256": {name: _sha256_file(path) for name, path in artifacts.items()},
    }
    (output_dir / "pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "private_manifest": output_dir / "private" / "calibration_manifest.jsonl",
        "resolution": output_dir / "private" / "calibration_resolution.csv",
        "annotator_A": output_dir / "annotator_delivery" / "annotator_A" / "calibration.csv",
        "annotator_B": output_dir / "annotator_delivery" / "annotator_B" / "calibration.csv",
        "personnel_log_template": output_dir / "personnel_log_template.csv",
        "readme": output_dir / "README.md",
    }


def validate_personnel_log(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    if not rows or tuple(rows[0]) != PERSONNEL_FIELDS:
        raise ValueError("Personnel log must contain only pseudonym, role, training_date, and codebook_version")
    by_role = {str(row["role"]).strip(): row for row in rows}
    expected_roles = {"annotator_A", "annotator_B", "adjudicator"}
    if len(rows) != 3 or set(by_role) != expected_roles:
        raise ValueError("Personnel log must contain exactly annotator_A, annotator_B, and adjudicator")
    if any(not str(row[field]).strip() for row in rows for field in PERSONNEL_FIELDS):
        raise ValueError("Completed personnel log fields cannot be blank")
    pseudonyms = {str(row["pseudonym"]).strip() for row in rows}
    codebooks = {str(row["codebook_version"]).strip() for row in rows}
    if len(pseudonyms) != 3:
        raise ValueError("Personnel log requires three distinct pseudonyms")
    if len(codebooks) != 1:
        raise ValueError("Personnel log requires one frozen codebook version")
    return by_role


def _validate_sources(manifest: dict[str, Any]) -> None:
    source_hashes = manifest.get("source_sha256", {})
    source_relpaths = manifest.get("source_relpath", {})
    if not source_relpaths:
        return
    if set(source_relpaths) != set(source_hashes):
        raise ValueError("Calibration source path and hash inventories do not match")
    root_resolved = ROOT.resolve()
    for name, relpath in source_relpaths.items():
        source = (ROOT / str(relpath)).resolve()
        if source != root_resolved and root_resolved not in source.parents:
            raise ValueError(f"Calibration source path escapes the project root: {name}")
        if not source.is_file() or _sha256_file(source) != source_hashes[name]:
            raise ValueError(f"Calibration source hash mismatch: {name}")


def _private_manifest(artifact_paths: dict[str, Path], kind: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(artifact_paths["private_manifest"])
    if len(rows) != SAMPLE_SIZE:
        raise ValueError("Calibration private manifest does not contain 20 items")
    for row in rows:
        expected = scenario_stimulus_sha256(row) if kind == "scenario" else response_stimulus_sha256(row)
        if str(row.get("stimulus_sha256", "")) != expected:
            raise ValueError("Calibration stimulus hash mismatch")
    return rows


def _validate_completed_content(
    artifact_paths: dict[str, Path],
    *,
    kind: str,
    private_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, str]], str]:
    personnel = validate_personnel_log(artifact_paths["personnel_log_template"])
    id_field = "calibration_id" if kind == "scenario" else "response_audit_id"
    public_fields = (
        ("calibration_id", "protocol_version", *AUDIT_CONTENT_FIELDS, "stimulus_sha256")
        if kind == "scenario"
        else (*PUBLIC_STIMULUS_FIELDS, "stimulus_sha256")
    )
    private_by_id = {str(row[id_field]): row for row in private_rows}
    sheets: dict[str, dict[str, dict[str, str]]] = {}
    for label in ("A", "B"):
        rows = _read_csv(artifact_paths[f"annotator_{label}"])
        indexed = {str(row.get(id_field, "")).strip(): row for row in rows}
        if len(rows) != SAMPLE_SIZE or set(indexed) != set(private_by_id):
            raise ValueError(f"Completed annotator {label} calibration inventory is incomplete")
        expected_pseudonym = str(personnel[f"annotator_{label}"]["pseudonym"]).strip()
        for audit_id, row in indexed.items():
            if str(row.get("annotator_id", "")).strip() != label:
                raise ValueError(f"Calibration annotator marker must remain {label}")
            if str(row.get("annotator_pseudonym", "")).strip() != expected_pseudonym:
                raise ValueError("Calibration sheet pseudonyms must match the personnel log")
            if any(str(row.get(field, "")) != str(private_by_id[audit_id].get(field, "")) for field in public_fields):
                raise ValueError(f"Completed calibration changed frozen stimulus content: {audit_id}")
            if kind == "scenario":
                if any(str(row.get(criterion, "")).strip() not in {"0", "1"} for criterion in AUDIT_CRITERIA):
                    raise ValueError("Completed scenario calibration criteria must be binary")
                if str(row.get("overall_accept", "")).strip() not in {"0", "1"}:
                    raise ValueError("Completed scenario calibration overall_accept must be binary")
            else:
                if str(row.get("selected_action_label", "")).strip() not in {"preserving", "conflicting", "other", "unclear"}:
                    raise ValueError("Completed response calibration action label is invalid")
                if str(row.get("norm_deviation_0_to_3", "")).strip() not in {"0", "1", "2", "3"}:
                    raise ValueError("Completed response calibration norm score is invalid")
            if str(row.get("confidence_1_to_5", "")).strip() not in {"1", "2", "3", "4", "5"}:
                raise ValueError("Completed calibration confidence must be 1 through 5")
        sheets[label] = indexed

    resolution_rows = _read_csv(artifact_paths["resolution"])
    resolution = {str(row.get(id_field, "")).strip(): row for row in resolution_rows}
    if len(resolution_rows) != SAMPLE_SIZE or set(resolution) != set(private_by_id):
        raise ValueError("Calibration resolution inventory is incomplete")
    for audit_id, row in resolution.items():
        if str(row.get("stimulus_sha256", "")) != str(private_by_id[audit_id]["stimulus_sha256"]):
            raise ValueError("Calibration resolution stimulus hash mismatch")
        if kind == "scenario":
            final_fields = [f"consensus_{criterion}" for criterion in AUDIT_CRITERIA] + ["consensus_overall_accept"]
            if any(str(row.get(field, "")).strip() not in {"0", "1"} for field in final_fields):
                raise ValueError("Scenario calibration consensus fields must be binary")
            disagreed = any(sheets["A"][audit_id][criterion] != sheets["B"][audit_id][criterion] for criterion in AUDIT_CRITERIA)
            disagreed = disagreed or sheets["A"][audit_id]["overall_accept"] != sheets["B"][audit_id]["overall_accept"]
        else:
            if str(row.get("consensus_selected_action_label", "")).strip() not in {"preserving", "conflicting", "other", "unclear"}:
                raise ValueError("Response calibration consensus action label is invalid")
            if str(row.get("consensus_norm_deviation_0_to_3", "")).strip() not in {"0", "1", "2", "3"}:
                raise ValueError("Response calibration consensus norm score is invalid")
            disagreed = sheets["A"][audit_id]["selected_action_label"] != sheets["B"][audit_id]["selected_action_label"]
            disagreed = disagreed or sheets["A"][audit_id]["norm_deviation_0_to_3"] != sheets["B"][audit_id]["norm_deviation_0_to_3"]
        if disagreed and not str(row.get("calibration_discussion_note", "")).strip():
            raise ValueError("Calibration disagreements require a discussion note")
    codebook = str(personnel["annotator_A"]["codebook_version"]).strip()
    return personnel, codebook


def finalize_calibration_pack(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "pack_manifest.json"
    completion_path = output_dir / "calibration_completion_manifest.json"
    if completion_path.exists():
        raise RuntimeError("Calibration completion manifest already exists and will not be overwritten")
    if not manifest_path.is_file():
        raise ValueError("Calibration issue manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = _artifact_paths(output_dir)
    for name in ("private_manifest", "readme"):
        if manifest.get("artifact_sha256", {}).get(name) != _sha256_file(artifact_paths[name]):
            raise ValueError(f"Immutable calibration artifact hash mismatch: {name}")
    _validate_sources(manifest)
    private_rows = _private_manifest(artifact_paths, str(manifest.get("calibration_kind")))
    personnel, codebook = _validate_completed_content(
        artifact_paths,
        kind=str(manifest.get("calibration_kind")),
        private_rows=private_rows,
    )
    mutable_names = ("resolution", "annotator_A", "annotator_B", "personnel_log_template")
    completion = {
        "schema_version": "human_calibration_completion_v1",
        "status": "calibration_complete",
        "pack_manifest_sha256": _sha256_file(manifest_path),
        "completed_artifact_sha256": {name: _sha256_file(artifact_paths[name]) for name in mutable_names},
        "codebook_version": codebook,
        "personnel_roles_sha256": _canonical_hash(
            "calibration-personnel-roles-v1",
            {role: row["pseudonym"] for role, row in sorted(personnel.items())},
        ),
    }
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return completion


def validate_calibration_pack(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "pack_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Calibration pack manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "human_calibration_pack_v1" or manifest.get("item_count") != SAMPLE_SIZE:
        raise ValueError("Calibration pack schema or item count is invalid")
    artifact_paths = _artifact_paths(output_dir)
    expected_hashes = manifest.get("artifact_sha256", {})
    completion_path = output_dir / "calibration_completion_manifest.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else None
    issue_bound_names = ("private_manifest", "readme") if completion else tuple(artifact_paths)
    for name in issue_bound_names:
        path = artifact_paths[name]
        if not path.is_file() or expected_hashes.get(name) != _sha256_file(path):
            raise ValueError(f"Calibration artifact hash mismatch: {name}")
    kind = manifest.get("calibration_kind")
    private_rows = _private_manifest(artifact_paths, str(kind))
    _validate_sources(manifest)
    if not completion:
        personnel_rows = _read_csv(artifact_paths["personnel_log_template"])
        if not personnel_rows or tuple(personnel_rows[0]) != PERSONNEL_FIELDS:
            raise ValueError("Personnel log template must contain only the anonymous identity-chain fields")
        return manifest
    if completion.get("schema_version") != "human_calibration_completion_v1":
        raise ValueError("Calibration completion manifest schema is invalid")
    if completion.get("pack_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError("Calibration completion is not bound to the issue manifest")
    for name, expected in completion.get("completed_artifact_sha256", {}).items():
        if name not in artifact_paths or _sha256_file(artifact_paths[name]) != expected:
            raise ValueError(f"Calibration completed artifact hash mismatch: {name}")
    _validate_completed_content(artifact_paths, kind=str(kind), private_rows=private_rows)
    return {**manifest, "status": "calibration_complete", "completion": completion}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-output", type=Path, default=SCENARIO_OUTPUT)
    parser.add_argument("--response-output", type=Path, default=RESPONSE_OUTPUT)
    args = parser.parse_args()

    scenario_rows = _read_jsonl(SCENARIO_SOURCE)
    formal_scenario_rows = _read_jsonl(SCENARIO_FORMAL_INVENTORY)
    formal_scenario_ids = {str(row["source_scenario_id"]) for row in formal_scenario_rows}
    scenario_pack = build_scenario_calibration_pack(scenario_rows, formal_ids=formal_scenario_ids)

    matrix_rows = _read_csv(MATRIX_STATUS)
    response_files = sorted({str(row["response_file"]) for row in matrix_rows})
    response_rows = [row for name in response_files for row in _read_jsonl(ROOT / "results" / name)]
    prompt_rows = _read_jsonl(PROMPT_SOURCE)
    formal_response_rows = _read_jsonl(RESPONSE_FORMAL_INVENTORY)
    formal_response_keys = {(str(row["model"]), str(row["prompt_id"])) for row in formal_response_rows}
    expected_models = tuple(sorted({str(row["model"]) for row in response_rows}))
    response_pack = build_response_calibration_pack(
        prompt_rows,
        response_rows,
        formal_response_keys=formal_response_keys,
        expected_models=expected_models,
    )

    write_calibration_pack(
        args.scenario_output,
        scenario_pack,
        calibration_kind="scenario",
        source_sha256={
            "scenario_source": _sha256_file(SCENARIO_SOURCE),
            "formal_scenario_inventory": _sha256_file(SCENARIO_FORMAL_INVENTORY),
        },
        source_relpath={
            "scenario_source": SCENARIO_SOURCE.relative_to(ROOT).as_posix(),
            "formal_scenario_inventory": SCENARIO_FORMAL_INVENTORY.relative_to(ROOT).as_posix(),
        },
    )
    response_sources = {
        "prompt_source": _sha256_file(PROMPT_SOURCE),
        "formal_response_inventory": _sha256_file(RESPONSE_FORMAL_INVENTORY),
        **{f"response_archive_{index + 1}": _sha256_file(ROOT / "results" / name) for index, name in enumerate(response_files)},
    }
    write_calibration_pack(
        args.response_output,
        response_pack,
        calibration_kind="archived_response",
        source_sha256=response_sources,
        source_relpath={
            "prompt_source": PROMPT_SOURCE.relative_to(ROOT).as_posix(),
            "formal_response_inventory": RESPONSE_FORMAL_INVENTORY.relative_to(ROOT).as_posix(),
            **{
                f"response_archive_{index + 1}": (ROOT / "results" / name).relative_to(ROOT).as_posix()
                for index, name in enumerate(response_files)
            },
        },
    )
    print(f"Wrote two separate {SAMPLE_SIZE}-item calibration packs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
