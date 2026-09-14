import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


class MainDatasetTests(unittest.TestCase):
    def test_main_scenarios_have_planned_size_balance_and_unique_ids(self):
        from build_main_dataset import build_main_scenarios

        scenarios = build_main_scenarios()

        self.assertEqual(len(scenarios), 1000)
        self.assertEqual(len({row["scenario_id"] for row in scenarios}), 1000)

        subdomains = Counter(row["subdomain"] for row in scenarios)
        self.assertEqual(len(subdomains), 8)
        self.assertTrue(all(count == 125 for count in subdomains.values()))

        required = {
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
        }
        for scenario in scenarios:
            self.assertTrue(required.issubset(scenario.keys()))
            self.assertEqual(scenario["domain"], "business")
            self.assertGreaterEqual(len(scenario["stakeholders"]), 3)
            self.assertGreaterEqual(len(scenario["moral_constraints"]), 2)
            self.assertGreaterEqual(len(scenario["expected_risk_tags"]), 2)

    def test_main_records_use_action_choice_and_four_variants(self):
        from build_main_dataset import MAIN_PROMPT_VERSION, build_main_records, build_main_scenarios
        from validate_main_dataset import validate_records

        scenarios = build_main_scenarios(scenarios_per_subdomain=2)
        records = build_main_records(scenarios)

        self.assertEqual(len(records), 64)
        self.assertTrue(all(row["prompt_version"] == MAIN_PROMPT_VERSION for row in records))

        by_scenario = defaultdict(list)
        for record in records:
            by_scenario[record["base_scenario_id"]].append(record)

        self.assertEqual(len(by_scenario), 16)
        for scenario_id, scenario_records in by_scenario.items():
            self.assertEqual(len(scenario_records), 4, scenario_id)
            self.assertEqual(
                {row["variant"] for row in scenario_records},
                {"V0_baseline", "V1_cognitive_load", "V2_resource_scarcity", "V3_combined_pressure"},
            )

        prompt_text = "\n".join(row["prompt_text"] for row in records)
        self.assertIn("Action:", prompt_text)
        self.assertIn("Operational next step:", prompt_text)
        self.assertNotIn("Decision:", prompt_text)
        self.assertNotIn("Will you conceal key information?", prompt_text)
        self.assertNotIn("Is there a less harmful alternative?", prompt_text)
        self.assertNotIn("norm-violating shortcut", prompt_text)
        self.assertNotIn("less harmful option", prompt_text)

        result = validate_records(records)
        self.assertEqual(result.errors, [])

    def test_default_main_record_counts_match_main_study_plan(self):
        from build_main_dataset import build_main_records, build_main_scenarios

        records = build_main_records(build_main_scenarios())

        self.assertEqual(len(records), 4000)
        self.assertEqual(Counter(row["variant"] for row in records), {
            "V0_baseline": 1000,
            "V1_cognitive_load": 1000,
            "V2_resource_scarcity": 1000,
            "V3_combined_pressure": 1000,
        })
        self.assertEqual(Counter(row["cognitive_load_level"] for row in records), {0: 2000, 1: 2000})
        self.assertEqual(Counter(row["resource_scarcity_level"] for row in records), {0: 2000, 1: 2000})

    def test_human_qc_sample_is_stratified_and_reviewable(self):
        from build_main_dataset import build_human_qc_sample, build_main_records, build_main_scenarios

        records = build_main_records(build_main_scenarios())
        sample = build_human_qc_sample(records, sample_size=300)

        self.assertEqual(len(sample), 300)
        self.assertEqual(len({row["prompt_id"] for row in sample}), 300)
        self.assertEqual({row["variant"] for row in sample}, {
            "V0_baseline",
            "V1_cognitive_load",
            "V2_resource_scarcity",
            "V3_combined_pressure",
        })
        self.assertEqual(len({row["subdomain"] for row in sample}), 8)
        self.assertTrue(all("human_quality_ok" in row and "human_notes" in row for row in sample))

    def test_smoke_prompt_sample_keeps_complete_variant_sets(self):
        from build_main_dataset import build_main_records, build_main_scenarios, build_smoke_prompt_sample
        from validate_main_dataset import validate_records

        records = build_main_records(build_main_scenarios())
        sample = build_smoke_prompt_sample(records, base_scenario_count=20)

        self.assertEqual(len(sample), 80)
        self.assertEqual(len({row["base_scenario_id"] for row in sample}), 20)
        self.assertEqual(len({row["subdomain"] for row in sample}), 8)

        by_scenario = defaultdict(list)
        for record in sample:
            by_scenario[record["base_scenario_id"]].append(record)

        for scenario_id, scenario_records in by_scenario.items():
            self.assertEqual(len(scenario_records), 4, scenario_id)
            self.assertEqual(
                {row["variant"] for row in scenario_records},
                {"V0_baseline", "V1_cognitive_load", "V2_resource_scarcity", "V3_combined_pressure"},
            )

        self.assertEqual(validate_records(sample).errors, [])

    def test_detailed_prompt_version_adds_operational_specificity(self):
        from build_main_dataset import (
            MAIN_DETAILED_PROMPT_VERSION,
            build_main_records,
            build_main_scenarios,
        )
        from validate_main_dataset import validate_records

        plain_records = build_main_records(build_main_scenarios(scenarios_per_subdomain=1))
        detailed_scenarios = build_main_scenarios(scenarios_per_subdomain=1, detailed=True)
        detailed_records = build_main_records(detailed_scenarios, prompt_version=MAIN_DETAILED_PROMPT_VERSION)

        self.assertEqual(len(detailed_records), 32)
        self.assertTrue(all(row["prompt_version"] == MAIN_DETAILED_PROMPT_VERSION for row in detailed_records))
        self.assertGreater(
            min(len(row["scenario_text"]) for row in detailed_records),
            min(len(row["scenario_text"]) for row in plain_records) + 250,
        )
        self.assertTrue(all("Time pressure detail:" in row["scenario_text"] for row in detailed_records))
        self.assertTrue(all("Evidence detail:" in row["scenario_text"] for row in detailed_records))
        self.assertTrue(all("Oversight detail:" in row["scenario_text"] for row in detailed_records))
        self.assertTrue(all("Reason: [one short sentence" in row["prompt_text"] for row in detailed_records))
        self.assertEqual(validate_records(detailed_records).errors, [])


if __name__ == "__main__":
    unittest.main()
