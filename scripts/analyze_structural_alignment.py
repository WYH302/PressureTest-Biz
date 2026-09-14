"""Analyze the benchmark subset aligned by the deterministic generator indices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "boundary_recovery_study" / "archived_boundary_analysis"
DEFAULT_CASES = DEFAULT_ANALYSIS / "archived_anchor_crossing_cases.csv"
DEFAULT_COHERENCE = ROOT / "results" / "scenario_coherence_audit" / "coherence_audit_merged.csv"
DEFAULT_HUMAN = DEFAULT_ANALYSIS / "archived_human_crossing_concordance_rows.csv"
DEFAULT_OUTPUT = DEFAULT_ANALYSIS
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_SCORE_MATRIX = DEFAULT_RESULTS / "four_by_five_crossed_matrix_submission_20260716" / "matrix_status.csv"

VARIANTS = ("V1_cognitive_load", "V2_resource_scarcity", "V3_combined_pressure")
CRITERIA = (
    "role_goal_fit",
    "constraint_action_fit",
    "stakeholder_harm_fit",
    "realism",
    "unambiguous_norm_conflict",
    "overall_pass",
)
EXPECTED_SCENARIOS = 1000
EXPECTED_MODELS = 4


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def scenario_structure(scenario_id: str) -> dict[str, Any]:
    match = re.fullmatch(r"biz_main_(\d{4})", str(scenario_id))
    if not match:
        raise ValueError(f"Noncanonical scenario id: {scenario_id!r}")
    n = int(match.group(1))
    if not 1 <= n <= EXPECTED_SCENARIOS:
        raise ValueError(f"Scenario id outside 1..{EXPECTED_SCENARIOS}: {scenario_id}")
    local = (n - 1) % 125
    role = local % 5
    goal = (local // 5) % 5
    resource = (local // 25) % 5
    constraint = (goal + resource) % 5
    action = (role + 2 * goal + resource) % 5
    return {
        "n": n,
        "local": local,
        "role_index": role,
        "goal_index": goal,
        "resource_index": resource,
        "constraint_index": constraint,
        "action_index": action,
        "is_aligned": int(constraint == action),
    }


def _int01(value: Any, label: str) -> int:
    text = str(value).strip()
    if text not in {"0", "1"}:
        raise ValueError(f"{label} must be 0/1, got {value!r}")
    return int(text)


def _validated_case_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed = [dict(row) for row in rows]
    if len(parsed) != EXPECTED_SCENARIOS * EXPECTED_MODELS * len(VARIANTS):
        raise ValueError(f"Expected 12000 archived cases, found {len(parsed)}")
    seen: set[tuple[str, str, str]] = set()
    for row in parsed:
        key = (str(row["model"]), str(row["base_scenario_id"]), str(row["variant"]))
        if key in seen or key[2] not in VARIANTS:
            raise ValueError(f"Duplicate or unexpected case key: {key}")
        seen.add(key)
        scenario_structure(key[1])
        _validate_case_flags(row)
    _validate_case_inventory(parsed, seen)
    return parsed


def _validate_case_flags(row: dict[str, Any]) -> None:
    baseline = str(row["baseline_label"])
    pressure = str(row["pressure_label"])
    if baseline not in {"ethical", "shortcut", "other"} or pressure not in {"ethical", "shortcut", "other"}:
        raise ValueError("Unexpected archived action label")
    ethical = _int01(row["baseline_ethical"], "baseline_ethical")
    crossed = _int01(row["crossed_to_shortcut"], "crossed_to_shortcut")
    shifted = _int01(row["shifted_to_other"], "shifted_to_other")
    reversed_ = _int01(row["reversed_to_ethical"], "reversed_to_ethical")
    expected = (
        int(baseline == "ethical"),
        int(baseline == "ethical" and pressure == "shortcut"),
        int(baseline == "ethical" and pressure == "other"),
        int(baseline == "shortcut" and pressure == "ethical"),
    )
    if (ethical, crossed, shifted, reversed_) != expected:
        raise ValueError(f"Inconsistent derived case flags for {row.get('base_scenario_id')}")


def _validate_case_inventory(rows: list[dict[str, Any]], seen: set[tuple[str, str, str]]) -> None:
    models = sorted({str(row["model"]) for row in rows})
    scenarios = sorted({str(row["base_scenario_id"]) for row in rows})
    expected_ids = [f"biz_main_{number:04d}" for number in range(1, EXPECTED_SCENARIOS + 1)]
    if len(models) != EXPECTED_MODELS or scenarios != expected_ids:
        raise ValueError("Archived model or scenario inventory is incomplete")
    expected = {(model, scenario, variant) for model in models for scenario in scenarios for variant in VARIANTS}
    if seen != expected:
        raise ValueError("Archived model-scenario-variant Cartesian product is incomplete")
    metadata: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        metadata[str(row["base_scenario_id"])].add((str(row["subdomain"]), str(row["action_family_id"])))
    if any(len(values) != 1 for values in metadata.values()):
        raise ValueError("Scenario metadata changes across archived rows")


def build_alignment_manifest(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed = _validated_case_rows(rows)
    metadata: dict[str, tuple[str, str]] = {}
    for row in parsed:
        metadata[str(row["base_scenario_id"])] = (str(row["subdomain"]), str(row["action_family_id"]))
    manifest = []
    for scenario_id in sorted(metadata):
        structure = scenario_structure(scenario_id)
        if structure["is_aligned"]:
            subdomain, family = metadata[scenario_id]
            manifest.append({"base_scenario_id": scenario_id, "subdomain": subdomain,
                             "action_family_id": family, **structure})
    _validate_manifest_balance(manifest)
    return manifest


def _validate_manifest_balance(rows: list[dict[str, Any]]) -> None:
    domains = Counter(str(row["subdomain"]) for row in rows)
    families = Counter(str(row["action_family_id"]) for row in rows)
    if len(rows) != 200 or len(domains) != 8 or set(domains.values()) != {25}:
        raise ValueError(f"Aligned manifest is not 8 domains x 25: {dict(domains)}")
    if len(families) != 40 or set(families.values()) != {5}:
        raise ValueError("Aligned manifest is not 40 action families x 5")


def _aligned_rows(rows: Iterable[dict[str, Any]], aligned_ids: set[str]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if str(row["base_scenario_id"]) in aligned_ids]


def summarize_crossings(rows: Iterable[dict[str, Any]], aligned_ids: set[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _aligned_rows(rows, aligned_ids):
        grouped[(str(row["model"]), str(row["variant"]))].append(row)
    output = []
    for (model, variant), group in sorted(grouped.items()):
        eligible = sum(_int01(row["baseline_ethical"], "baseline_ethical") for row in group)
        crossed = sum(_int01(row["crossed_to_shortcut"], "crossed_to_shortcut") for row in group)
        output.append({"model": model, "variant": variant, "n_scenarios": len(group),
                       "baseline_ethical": eligible, "crossed_to_shortcut": crossed,
                       "crossing_rate": crossed / eligible if eligible else None})
    if len(output) != EXPECTED_MODELS * len(VARIANTS) or {row["n_scenarios"] for row in output} != {200}:
        raise ValueError("Aligned crossing summary is incomplete")
    return output


def summarize_domain_concentration(rows: Iterable[dict[str, Any]], aligned_ids: set[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _aligned_rows(rows, aligned_ids):
        grouped[(str(row["model"]), str(row["variant"]), str(row["subdomain"]))].append(row)
    totals = Counter()
    for row in _aligned_rows(rows, aligned_ids):
        totals[(str(row["model"]), str(row["variant"]))] += _int01(row["crossed_to_shortcut"], "crossed")
    output = []
    for (model, variant, domain), group in sorted(grouped.items()):
        eligible = sum(_int01(row["baseline_ethical"], "baseline_ethical") for row in group)
        crossed = sum(_int01(row["crossed_to_shortcut"], "crossed") for row in group)
        total = totals[(model, variant)]
        output.append({"model": model, "variant": variant, "subdomain": domain,
                       "n_scenarios": len(group), "baseline_ethical": eligible,
                       "crossed_to_shortcut": crossed,
                       "crossing_rate": crossed / eligible if eligible else None,
                       "share_of_crossings": crossed / total if total else None})
    if len(output) != EXPECTED_MODELS * len(VARIANTS) * 8 or {row["n_scenarios"] for row in output} != {25}:
        raise ValueError("Aligned domain summary is incomplete")
    return output


def summarize_v3_overlap(rows: Iterable[dict[str, Any]], aligned_ids: set[str]) -> list[dict[str, Any]]:
    v3 = [row for row in _aligned_rows(rows, aligned_ids) if row["variant"] == "V3_combined_pressure"]
    models = sorted({str(row["model"]) for row in v3})
    by_model = {model: [row for row in v3 if str(row["model"]) == model] for model in models}
    output = []
    for left, right in combinations(models, 2):
        left_cross = _case_set(by_model[left], "crossed_to_shortcut")
        right_cross = _case_set(by_model[right], "crossed_to_shortcut")
        left_decidable = _decidable_set(by_model[left])
        right_decidable = _decidable_set(by_model[right])
        common = left_decidable & right_decidable
        common_left, common_right = left_cross & common, right_cross & common
        output.append(_overlap_row(left, right, left_cross, right_cross, common, common_left, common_right))
    return output


def _case_set(rows: list[dict[str, Any]], field: str) -> set[str]:
    return {str(row["base_scenario_id"]) for row in rows if _int01(row[field], field)}


def _decidable_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["base_scenario_id"]) for row in rows
            if _int01(row["baseline_ethical"], "baseline_ethical")
            and str(row["pressure_label"]) in {"ethical", "shortcut"}}


def _overlap_row(left: str, right: str, left_set: set[str], right_set: set[str],
                 common: set[str], common_left: set[str], common_right: set[str]) -> dict[str, Any]:
    intersection, union = len(left_set & right_set), len(left_set | right_set)
    common_intersection = len(common_left & common_right)
    common_union = len(common_left | common_right)
    return {"variant": "V3_combined_pressure", "model_a": left, "model_b": right,
            "crossings_a": len(left_set), "crossings_b": len(right_set),
            "intersection": intersection, "union": union,
            "jaccard": intersection / union if union else None,
            "jointly_decidable_n": len(common), "common_crossings_a": len(common_left),
            "common_crossings_b": len(common_right), "common_intersection": common_intersection,
            "common_union": common_union,
            "common_jaccard": common_intersection / common_union if common_union else None}


def summarize_coherence_strata(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        scenario_id = str(row["scenario_id"])
        if scenario_id in seen:
            raise ValueError(f"Duplicate coherence audit scenario: {scenario_id}")
        seen.add(scenario_id)
        _validate_audit_row(row)
        grouped["aligned" if scenario_structure(scenario_id)["is_aligned"] else "unaligned"].append(row)
    if set(grouped) != {"aligned", "unaligned"}:
        raise ValueError("Coherence audit must cover aligned and unaligned strata")
    output = []
    for stratum, group in sorted(grouped.items()):
        for metric in (*CRITERIA, "either_major"):
            field = f"both_{metric}" if metric in CRITERIA else metric
            numerator = sum(_int01(row[field], field) for row in group)
            output.append({"stratum": stratum, "metric": metric, "numerator": numerator,
                           "denominator": len(group), "rate": numerator / len(group)})
    return output


def _validate_audit_row(row: dict[str, Any]) -> None:
    for criterion in CRITERIA:
        a = _int01(row[f"rater_a_{criterion}"], f"rater_a_{criterion}")
        b = _int01(row[f"rater_b_{criterion}"], f"rater_b_{criterion}")
        both = _int01(row[f"both_{criterion}"], f"both_{criterion}")
        if both != int(bool(a) and bool(b)):
            raise ValueError(f"Inconsistent both_{criterion} for {row['scenario_id']}")
    severities = {str(row["rater_a_severity"]), str(row["rater_b_severity"])}
    if not severities <= {"none", "minor", "major"}:
        raise ValueError(f"Unexpected severity for {row['scenario_id']}")
    if _int01(row["either_major"], "either_major") != int("major" in severities):
        raise ValueError(f"Inconsistent either_major for {row['scenario_id']}")


def summarize_human_concordance(rows: Iterable[dict[str, Any]], aligned_ids: set[str],
                                case_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    case_map = _human_case_map(case_rows)
    filtered, seen, seen_keys = [], set(), set()
    for source in rows:
        row = dict(source)
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate human sample: {sample_id}")
        seen.add(sample_id)
        crossed = _int01(row["crossed_to_shortcut"], "crossed_to_shortcut")
        norm_ge2 = _int01(row["human_norm_ge2"], "human_norm_ge2")
        if norm_ge2 != int(float(row["human_norm_deviation"]) >= 2):
            raise ValueError(f"Inconsistent human threshold for {sample_id}")
        if str(row["variant"]) not in VARIANTS:
            raise ValueError(f"Unexpected human-audit variant for {sample_id}")
        key = (str(row["model"]), str(row["base_scenario_id"]), str(row["variant"]))
        if key in seen_keys or key not in case_map or crossed != case_map[key]:
            raise ValueError(f"Human/crossing case mismatch for {key}")
        seen_keys.add(key)
        if str(row["base_scenario_id"]) in aligned_ids:
            filtered.append({**row, "crossed_to_shortcut": crossed, "human_norm_ge2": norm_ge2})
    if not filtered or {str(row["variant"]) for row in filtered} != set(VARIANTS):
        raise ValueError("Aligned human sample must cover all three pressure variants")
    return _human_group_rows(filtered)


def _human_case_map(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    output = {}
    for row in rows:
        key = (str(row["model"]), str(row["base_scenario_id"]), str(row["variant"]))
        if key in output:
            raise ValueError(f"Duplicate crossing case for human join: {key}")
        output[key] = _int01(row["crossed_to_shortcut"], "crossed_to_shortcut")
    if not output:
        raise ValueError("No crossing cases supplied for human join")
    return output


def load_aligned_score_rows(matrix_path: Path, aligned_ids: set[str],
                            results_dir: Path = DEFAULT_RESULTS) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from analyze_boundary_crossings import _load_jsonl, _matrix_files

    _, judge_files = _matrix_files(matrix_path)
    raw = []
    for name in judge_files:
        rows = _load_jsonl(results_dir / name)
        if len(rows) != 4000:
            raise ValueError(f"Expected 4000 score rows in {name}, found {len(rows)}")
        raw.extend(rows)
    if len(raw) != 80000:
        raise ValueError(f"Expected 80000 raw score rows, found {len(raw)}")
    filtered = [row for row in raw if str(row.get("base_scenario_id")) in aligned_ids]
    _validate_aligned_scores(filtered, aligned_ids)
    return filtered, {"matrix_cells": len(judge_files), "raw_score_rows": len(raw),
                      "aligned_score_rows": len(filtered)}


def _validate_aligned_scores(rows: list[dict[str, Any]], aligned_ids: set[str]) -> None:
    models = {str(row.get("model")) for row in rows}
    judges = {str(row.get("judge_model")) for row in rows}
    variants = {str(row.get("variant")) for row in rows}
    expected_variants = {"V0_baseline", *VARIANTS}
    keys = {(str(row.get("model")), str(row.get("judge_model")),
             str(row.get("base_scenario_id")), str(row.get("variant"))) for row in rows}
    expected = {(model, judge, scenario, variant) for model in models for judge in judges
                for scenario in aligned_ids for variant in expected_variants}
    if len(models) != 4 or len(judges) != 5 or variants != expected_variants:
        raise ValueError("Aligned score model, judge, or variant inventory is incomplete")
    if len(rows) != 16000 or len(keys) != len(rows) or keys != expected:
        raise ValueError("Aligned score Cartesian product is not 4 x 5 x 200 x 4")


def _human_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    variants = ["ALL", *[variant for variant in VARIANTS if any(row["variant"] == variant for row in rows)]]
    for variant in variants:
        eligible = rows if variant == "ALL" else [row for row in rows if row["variant"] == variant]
        for crossed in (0, 1):
            group = [row for row in eligible if row["crossed_to_shortcut"] == crossed]
            count = sum(row["human_norm_ge2"] for row in group)
            output.append({"stratum": "aligned", "variant": variant,
                           "crossed_to_shortcut": crossed, "n": len(group),
                           "human_norm_ge2_count": count,
                           "human_norm_ge2_rate": count / len(group) if group else None})
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_analysis(*, cases_path: Path = DEFAULT_CASES, coherence_path: Path | None = DEFAULT_COHERENCE,
                 human_path: Path | None = DEFAULT_HUMAN, score_matrix: Path | None = DEFAULT_SCORE_MATRIX,
                 results_dir: Path = DEFAULT_RESULTS, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    cases = _validated_case_rows(read_csv_rows(cases_path))
    manifest = build_alignment_manifest(cases)
    aligned_ids = {str(row["base_scenario_id"]) for row in manifest}
    aligned_cases = _aligned_rows(cases, aligned_ids)
    if len(aligned_cases) != 2400:
        raise ValueError(f"Expected 2400 aligned crossing cases, found {len(aligned_cases)}")
    outputs = {
        "structural_alignment_manifest.csv": manifest,
        "structural_alignment_crossing_cases.csv": aligned_cases,
        "structural_alignment_crossing_summary.csv": summarize_crossings(cases, aligned_ids),
        "structural_alignment_domain_concentration.csv": summarize_domain_concentration(cases, aligned_ids),
        "structural_alignment_v3_model_overlap.csv": summarize_v3_overlap(cases, aligned_ids),
    }
    inputs = {"archived_cases": _sha256(cases_path)}
    _add_optional_outputs(outputs, inputs, coherence_path, human_path, aligned_ids, cases)
    score_integrity: dict[str, int] = {}
    if score_matrix is not None:
        from analyze_boundary_crossings import analyze_score_crossings

        score_rows, score_integrity = load_aligned_score_rows(score_matrix, aligned_ids, results_dir)
        outputs["structural_alignment_score_crossing_summary.csv"] = analyze_score_crossings(score_rows)
        inputs["score_matrix"] = _sha256(score_matrix)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        _write_csv(output_dir / name, rows)
    summary = _machine_summary(cases, manifest, outputs, inputs, score_integrity)
    with (output_dir / "structural_alignment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return summary


def _add_optional_outputs(outputs: dict[str, list[dict[str, Any]]], inputs: dict[str, str],
                          coherence_path: Path | None, human_path: Path | None,
                          aligned_ids: set[str], case_rows: list[dict[str, Any]]) -> None:
    if coherence_path is not None:
        audit = read_csv_rows(coherence_path)
        if len(audit) != 200:
            raise ValueError(f"Expected 200 coherence audit rows, found {len(audit)}")
        outputs["structural_alignment_coherence_strata.csv"] = summarize_coherence_strata(audit)
        inputs["coherence_audit"] = _sha256(coherence_path)
    if human_path is not None:
        human = read_csv_rows(human_path)
        outputs["structural_alignment_human_concordance_summary.csv"] = summarize_human_concordance(
            human, aligned_ids, case_rows
        )
        inputs["human_concordance"] = _sha256(human_path)


def _machine_summary(cases: list[dict[str, Any]], manifest: list[dict[str, Any]],
                     outputs: dict[str, list[dict[str, Any]]], inputs: dict[str, str],
                     score_integrity: dict[str, int]) -> dict[str, Any]:
    return {
        "analysis": "generator_structural_alignment_v1",
        "formula": {"local": "(n-1)%125", "role": "local%5", "goal": "(local//5)%5",
                    "resource": "(local//25)%5", "constraint_index": "(goal+resource)%5",
                    "action_index": "(role+2*goal+resource)%5", "is_aligned": "(role+goal)%5==0"},
        "integrity": {"n_case_rows": len(cases), "n_scenarios": EXPECTED_SCENARIOS,
                      "n_models": len({str(row['model']) for row in cases}),
                      "n_aligned_scenarios": len(manifest),
                      "domain_counts": dict(sorted(Counter(row["subdomain"] for row in manifest).items())),
                      "action_family_count": len({row["action_family_id"] for row in manifest}),
                      **score_integrity},
        "input_sha256": inputs,
        "output_row_counts": {name: len(rows) for name, rows in sorted(outputs.items())},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--coherence", type=Path, default=DEFAULT_COHERENCE)
    parser.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--score-matrix", type=Path, default=DEFAULT_SCORE_MATRIX)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_analysis(cases_path=args.cases, coherence_path=args.coherence,
                           human_path=args.human, score_matrix=args.score_matrix,
                           results_dir=args.results_dir, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
