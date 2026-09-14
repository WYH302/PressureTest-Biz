"""Validate PressureTest-Biz pilot dataset files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OWN_DATASET_DIR = ROOT / "own_dataset"

EXPECTED_VARIANTS = {
    "V0_baseline": (0, 0, 0),
    "V1_cognitive_load": (1, 0, 1),
    "V2_resource_scarcity": (0, 1, 1),
    "V3_combined_pressure": (1, 1, 2),
}

REQUIRED_RECORD_FIELDS = {
    "prompt_id",
    "prompt_version",
    "base_scenario_id",
    "domain",
    "subdomain",
    "role",
    "goal",
    "stakeholders",
    "resource_type",
    "cognitive_load_level",
    "resource_scarcity_level",
    "pressure_level",
    "pressure_condition",
    "variant",
    "moral_constraints",
    "ethical_action",
    "shortcut_action",
    "harmed_party",
    "expected_risk_tags",
    "scenario_text",
    "prompt_text",
}

RISK_TAGS = {
    "deception",
    "concealment",
    "manipulation",
    "norm_deviation",
    "unfair_sacrifice",
    "empathy_suppression",
}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def validate_records(records: list[dict[str, Any]]) -> ValidationResult:
    result = ValidationResult()
    prompt_ids = Counter()
    prompt_texts = Counter()
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    variant_counts = Counter()
    subdomain_counts = Counter()
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

        variant = record["variant"]
        if variant not in EXPECTED_VARIANTS:
            result.errors.append(f"{record['prompt_id']} has unknown variant {variant}")
        else:
            expected_load, expected_scarcity, expected_pressure = EXPECTED_VARIANTS[variant]
            actual = (
                record["cognitive_load_level"],
                record["resource_scarcity_level"],
                record["pressure_level"],
            )
            expected = (expected_load, expected_scarcity, expected_pressure)
            if actual != expected:
                result.errors.append(f"{record['prompt_id']} has pressure levels {actual}, expected {expected}")

        if len(record["stakeholders"]) < 3:
            result.errors.append(f"{record['prompt_id']} has fewer than 3 stakeholders")
        if len(record["moral_constraints"]) < 2:
            result.errors.append(f"{record['prompt_id']} has fewer than 2 moral constraints")
        if not set(record["expected_risk_tags"]).issubset(RISK_TAGS):
            result.errors.append(f"{record['prompt_id']} has unknown risk tags: {record['expected_risk_tags']}")
        for tag in record["expected_risk_tags"]:
            tag_counts[tag] += 1

        prompt_text = record["prompt_text"]
        if "Required output format" not in prompt_text or ("Decision:" not in prompt_text and "Action:" not in prompt_text):
            result.errors.append(f"{record['prompt_id']} prompt text lacks required output format")
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
        "risk_tag_counts": dict(tag_counts),
        "warnings": len(result.warnings),
        "errors": len(result.errors),
    }
    return result


def validate_file(path: Path) -> ValidationResult:
    return validate_records(load_jsonl(path))


def main() -> int:
    path = OWN_DATASET_DIR / "pilot_prompts.jsonl"
    result = validate_file(path)
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
