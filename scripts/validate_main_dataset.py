"""Validate PressureTest-Biz main-study dataset files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from build_main_dataset import (
    DEFAULT_SCENARIOS_PER_SUBDOMAIN,
    MAIN_DETAILED_PROMPT_VERSION,
    MAIN_DATASET_DIR,
    MAIN_PROMPT_VERSION,
    RISK_TAGS,
    SUBDOMAIN_ORDER,
)
from validate_pilot_dataset import EXPECTED_VARIANTS, REQUIRED_RECORD_FIELDS, load_jsonl


REQUIRED_SCENARIO_FIELDS = {
    "scenario_id",
    "domain",
    "subdomain",
    "role",
    "goal",
    "resource_type",
    "stakeholders",
    "moral_constraints",
    "ethical_action",
    "shortcut_action",
    "harmed_party",
    "expected_risk_tags",
    "business_context",
    "case_detail",
    "generation_method",
    "template_cell",
}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    expected_scenarios_per_subdomain: int | None = None,
) -> ValidationResult:
    result = ValidationResult()
    scenario_ids = Counter()
    subdomain_counts = Counter()
    tag_counts = Counter()

    for index, scenario in enumerate(scenarios, start=1):
        missing = REQUIRED_SCENARIO_FIELDS - set(scenario)
        if missing:
            result.errors.append(f"scenario {index} missing fields: {', '.join(sorted(missing))}")
            continue

        scenario_ids[scenario["scenario_id"]] += 1
        subdomain_counts[scenario["subdomain"]] += 1

        if scenario["domain"] != "business":
            result.errors.append(f"{scenario['scenario_id']} has domain {scenario['domain']}, expected business")
        if scenario["subdomain"] not in SUBDOMAIN_ORDER:
            result.errors.append(f"{scenario['scenario_id']} has unknown subdomain {scenario['subdomain']}")
        if len(scenario["stakeholders"]) < 3:
            result.errors.append(f"{scenario['scenario_id']} has fewer than 3 stakeholders")
        if len(scenario["moral_constraints"]) < 2:
            result.errors.append(f"{scenario['scenario_id']} has fewer than 2 moral constraints")
        if len(scenario["expected_risk_tags"]) < 2:
            result.errors.append(f"{scenario['scenario_id']} has fewer than 2 expected risk tags")
        unknown_tags = set(scenario["expected_risk_tags"]) - RISK_TAGS
        if unknown_tags:
            result.errors.append(f"{scenario['scenario_id']} has unknown risk tags: {sorted(unknown_tags)}")
        for tag in scenario["expected_risk_tags"]:
            tag_counts[tag] += 1

    duplicate_ids = [item for item, count in scenario_ids.items() if count > 1]
    if duplicate_ids:
        result.errors.append(f"duplicate scenario_id values: {', '.join(sorted(duplicate_ids)[:10])}")

    if set(subdomain_counts) != set(SUBDOMAIN_ORDER):
        result.errors.append(f"subdomain set is incomplete: {sorted(subdomain_counts)}")

    if expected_scenarios_per_subdomain is not None:
        for subdomain in SUBDOMAIN_ORDER:
            actual = subdomain_counts[subdomain]
            if actual != expected_scenarios_per_subdomain:
                result.errors.append(
                    f"{subdomain} has {actual} scenarios, expected {expected_scenarios_per_subdomain}"
                )

    result.stats = {
        "scenarios": len(scenarios),
        "subdomain_counts": dict(subdomain_counts),
        "risk_tag_counts": dict(tag_counts),
        "warnings": len(result.warnings),
        "errors": len(result.errors),
    }
    return result


def validate_records(records: list[dict[str, Any]]) -> ValidationResult:
    result = ValidationResult()
    prompt_ids = Counter()
    prompt_texts = Counter()
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    variant_counts = Counter()
    subdomain_counts = Counter()
    prompt_version_counts = Counter()
    tag_counts = Counter()

    for index, record in enumerate(records, start=1):
        missing = REQUIRED_RECORD_FIELDS - set(record)
        if missing:
            result.errors.append(f"record {index} missing fields: {', '.join(sorted(missing))}")
            continue

        prompt_ids[record["prompt_id"]] += 1
        prompt_texts[record["prompt_text"]] += 1
        by_base[record["base_scenario_id"]].append(record)
        variant_counts[record["variant"]] += 1
        subdomain_counts[record["subdomain"]] += 1
        prompt_version_counts[record["prompt_version"]] += 1

        if record["prompt_version"] not in {MAIN_PROMPT_VERSION, MAIN_DETAILED_PROMPT_VERSION}:
            result.errors.append(f"{record['prompt_id']} has prompt_version {record['prompt_version']}")

        variant = record["variant"]
        if variant not in EXPECTED_VARIANTS:
            result.errors.append(f"{record['prompt_id']} has unknown variant {variant}")
        else:
            expected = EXPECTED_VARIANTS[variant]
            actual = (
                record["cognitive_load_level"],
                record["resource_scarcity_level"],
                record["pressure_level"],
            )
            if actual != expected:
                result.errors.append(f"{record['prompt_id']} has pressure levels {actual}, expected {expected}")

        if len(record["stakeholders"]) < 3:
            result.errors.append(f"{record['prompt_id']} has fewer than 3 stakeholders")
        if len(record["moral_constraints"]) < 2:
            result.errors.append(f"{record['prompt_id']} has fewer than 2 moral constraints")
        unknown_tags = set(record["expected_risk_tags"]) - RISK_TAGS
        if unknown_tags:
            result.errors.append(f"{record['prompt_id']} has unknown risk tags: {sorted(unknown_tags)}")
        for tag in record["expected_risk_tags"]:
            tag_counts[tag] += 1

        prompt_text = record["prompt_text"]
        if "Required output format" not in prompt_text:
            result.errors.append(f"{record['prompt_id']} prompt text lacks required output format")
        if "Action:" not in prompt_text:
            result.errors.append(f"{record['prompt_id']} prompt text lacks Action field")
        if "Operational next step:" not in prompt_text:
            result.errors.append(f"{record['prompt_id']} prompt text lacks Operational next step field")
        banned_phrases = [
            "Decision:",
            "Will you conceal key information?",
            "Will any stakeholder be knowingly harmed?",
            "Is there a less harmful alternative?",
            "norm-violating shortcut",
            "less harmful option",
        ]
        for phrase in banned_phrases:
            if phrase in prompt_text:
                result.errors.append(f"{record['prompt_id']} prompt text contains banned phrase {phrase!r}")
        if len(record["scenario_text"]) < 600:
            result.warnings.append(f"{record['prompt_id']} scenario text is short")

    duplicate_ids = [item for item, count in prompt_ids.items() if count > 1]
    if duplicate_ids:
        result.errors.append(f"duplicate prompt_id values: {', '.join(sorted(duplicate_ids)[:10])}")

    duplicate_texts = sum(1 for count in prompt_texts.values() if count > 1)
    if duplicate_texts:
        result.errors.append(f"{duplicate_texts} duplicate prompt_text values")

    for base_id, rows in sorted(by_base.items()):
        variants = {row["variant"] for row in rows}
        if variants != set(EXPECTED_VARIANTS):
            result.errors.append(f"{base_id} variant set is incomplete: {sorted(variants)}")
        if len(rows) != 4:
            result.errors.append(f"{base_id} has {len(rows)} records instead of 4 variants")

    result.stats = {
        "records": len(records),
        "base_scenarios": len(by_base),
        "variant_counts": dict(variant_counts),
        "subdomain_counts": dict(subdomain_counts),
        "prompt_version_counts": dict(prompt_version_counts),
        "risk_tag_counts": dict(tag_counts),
        "warnings": len(result.warnings),
        "errors": len(result.errors),
    }
    return result


def validate_files(
    scenarios_path: Path,
    prompts_path: Path,
    *,
    expected_scenarios_per_subdomain: int | None = DEFAULT_SCENARIOS_PER_SUBDOMAIN,
) -> ValidationResult:
    scenarios = load_jsonl(scenarios_path)
    records = load_jsonl(prompts_path)
    scenario_result = validate_scenarios(
        scenarios,
        expected_scenarios_per_subdomain=expected_scenarios_per_subdomain,
    )
    record_result = validate_records(records)

    result = ValidationResult()
    result.errors.extend(scenario_result.errors)
    result.errors.extend(record_result.errors)
    result.warnings.extend(scenario_result.warnings)
    result.warnings.extend(record_result.warnings)

    scenario_ids = {row["scenario_id"] for row in scenarios}
    record_base_ids = {row["base_scenario_id"] for row in records}
    missing_records = sorted(scenario_ids - record_base_ids)
    missing_scenarios = sorted(record_base_ids - scenario_ids)
    if missing_records:
        result.errors.append(f"scenarios without prompt records: {', '.join(missing_records[:10])}")
    if missing_scenarios:
        result.errors.append(f"prompt records without scenarios: {', '.join(missing_scenarios[:10])}")
    if len(records) != len(scenarios) * 4:
        result.errors.append(f"record count {len(records)} does not equal scenarios * 4 ({len(scenarios) * 4})")

    result.stats = {
        "scenarios": scenario_result.stats,
        "records": record_result.stats,
        "warnings": len(result.warnings),
        "errors": len(result.errors),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=MAIN_DATASET_DIR / "mother_scenarios_1000.jsonl")
    parser.add_argument("--prompts", type=Path, default=MAIN_DATASET_DIR / "main_prompts_v1_action_choice.jsonl")
    parser.add_argument("--expected-scenarios-per-subdomain", type=int, default=DEFAULT_SCENARIOS_PER_SUBDOMAIN)
    args = parser.parse_args()

    result = validate_files(
        args.scenarios,
        args.prompts,
        expected_scenarios_per_subdomain=args.expected_scenarios_per_subdomain,
    )
    print(json.dumps(result.stats, ensure_ascii=False, indent=2, sort_keys=True))
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings[:30]:
            print(f"- {warning}")
        if len(result.warnings) > 30:
            print(f"- ... {len(result.warnings) - 30} more warnings")
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"- {error}")
        return 1
    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
