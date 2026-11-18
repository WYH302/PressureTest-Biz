"""Interactive offline explorer for multi-evaluator condition summaries."""
import csv
import hashlib
from pathlib import Path
from demo_ui import main

ROOT = Path(__file__).parent
SOURCE = "analysis/sources/cell_variant_summary.csv"
TARGETS = {"deepseek-chat", "gemini-3.1-flash-lite-preview", "qwen3-8b-q8_0", "llama-3.1-8b-instruct-q8_0"}

def load():
    source = ROOT / SOURCE
    with source.open(encoding="utf-8-sig", newline="") as handle:
        original = [r for r in csv.DictReader(handle) if r["target_id"] in TARGETS]
    fields = ["deception_mean", "manipulation_mean", "norm_deviation_mean", "empathy_suppression_mean", "unfair_sacrifice_mean"]
    rows = []
    for row in original:
        value = sum(float(row[key]) for key in fields) / 5
        if abs(value - float(row["behavior_risk_mean"])) > 1e-10:
            raise ValueError("The retained composite does not match its five components.")
        rows.append({"target":row["target_id"], "judge":row["judge_id"], "label":row["variant"], "value":value, "n":int(row["n"])})
    if len(rows) != 80:
        raise ValueError("Expected the retained 80-row, four-target release.")
    return {"rows":rows, "filters":["target", "judge"], "metric":"Composite behavioral risk score: arithmetic mean across the five direct-risk rubric dimensions.", "source":SOURCE, "sha256":hashlib.sha256(source.read_bytes()).hexdigest()}

if __name__ == "__main__":
    main("PressureTest-Biz", "Interactive offline viewer comparing V0-V3 operational regimes across evaluated models and rubric judge pipelines.", load)
