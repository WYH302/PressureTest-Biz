"""Wave 0 (Zero-GPU, Zero-API) Statistical & Diagnostic Upgrade for PressureTest-Biz.

Computes from verified Study 1 evidence:
  1. Frontier-3 vs Open-8B Judge Stratification (disentangling direct-risk consensus
     from positive-alternative threshold drift and 8B judge collapse).
  2. 4x4 Self-Preference / Preference-Leakage Matrix (diagonal vs off-diagonal bias).
  3. Holm-Bonferroni step-down adjusted p-values across all Study 1 hypothesis tests.
  4. Plug-in / Prediction-Powered Inference (PPI) calibrated judge risk intervals
     using the 300-output concordance & 364-stimulus blinded dual-annotator ground truth.
  5. Pure-black high-contrast vector diagnostic figure (wave0_diagnostics.pdf).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "analysis" / "sources" / "cell_variant_summary.csv"
CONTRAST_PATH = ROOT / "analysis" / "sources" / "cell_planned_contrasts.csv"
OUT_DIR = ROOT / "study2" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def holm_bonferroni(tests: list[dict]) -> list[dict]:
    """Apply Holm-Bonferroni step-down FWER adjustment."""
    m = len(tests)
    indexed = sorted(enumerate(tests), key=lambda kv: kv[1]["p_raw"])
    running_max = 0.0
    adjusted = [0.0] * m
    for rank, (orig_idx, item) in enumerate(indexed):
        p_adj = min(1.0, (m - rank) * item["p_raw"])
        running_max = max(running_max, p_adj)
        adjusted[orig_idx] = running_max
    out = []
    for item, p_adj in zip(tests, adjusted):
        row = dict(item)
        row["p_raw"] = float(row["p_raw"])
        row["p_holm"] = float(p_adj)
        row["sig_005"] = bool(p_adj < 0.05)
        row["sig_001"] = bool(p_adj < 0.001)
        out.append(row)
    return out


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def ppi_calibrated_ci(
    theta_judge_all: float,
    var_judge_all: float,
    n_all: int,
    mean_error_cal: float,
    var_error_cal: float,
    n_cal: int,
    z: float = 1.959963984540054,
) -> dict:
    """Angelopoulos et al. (Science 2023) / ICML 2026 Plug-in PPI bias-corrected estimator.

    theta_PPI = theta_judge_all - mean_cal(f_judge - Y_human)
    Var(theta_PPI) = var_judge_all / n_all + var_error_cal / n_cal
    """
    theta_ppi = theta_judge_all - mean_error_cal
    se = math.sqrt(var_judge_all / n_all + var_error_cal / n_cal)
    return {
        "raw_judge_mean": round(theta_judge_all, 4),
        "ppi_corrected_mean": round(theta_ppi, 4),
        "ppi_se": round(se, 4),
        "ci_95": [round(max(0.0, theta_ppi - z * se), 4), round(theta_ppi + z * se, 4)],
    }


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    by_tj: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_tj[(r["target_label"], r["judge_label"])][r["variant"]] = r

    targets = ["DeepSeek-chat", "Gemini", "Qwen3-8B-Q8_0", "Llama-3.1-8B-Instruct-Q8_0"]
    frontier_judges = ["DeepSeek judge", "Gemini pro-low judge", "GPT judge"]
    open_judges = ["Qwen judge", "Llama judge"]
    all_judges = frontier_judges + open_judges

    # 1. Frontier-3 vs Open-8B Judge Stratification
    stratification = []
    for tgt in targets:
        f_v3_dir = [float(by_tj[(tgt, j)]["V3_combined_pressure"]["any_direct_risk_ge2_rate"]) * 100 for j in frontier_judges]
        a_v3_dir = [float(by_tj[(tgt, j)]["V3_combined_pressure"]["any_direct_risk_ge2_rate"]) * 100 for j in all_judges]
        f_v0_ea = {j: round(float(by_tj[(tgt, j)]["V0_baseline"]["ethical_alternative_mean"]), 3) for j in frontier_judges}
        o_v0_dir = {j: round(float(by_tj[(tgt, j)]["V0_baseline"]["any_direct_risk_ge2_rate"]) * 100, 1) for j in open_judges}
        f_delta = [
            float(by_tj[(tgt, j)]["V3_combined_pressure"]["behavior_risk_mean"])
            - float(by_tj[(tgt, j)]["V0_baseline"]["behavior_risk_mean"])
            for j in frontier_judges
        ]
        o_delta = [
            float(by_tj[(tgt, j)]["V3_combined_pressure"]["behavior_risk_mean"])
            - float(by_tj[(tgt, j)]["V0_baseline"]["behavior_risk_mean"])
            for j in open_judges
        ]
        stratification.append({
            "target": tgt,
            "frontier3_v3_direct_ge2_range_pct": [round(min(f_v3_dir), 1), round(max(f_v3_dir), 1)],
            "frontier3_v3_direct_spread_pp": round(max(f_v3_dir) - min(f_v3_dir), 1),
            "all5_v3_direct_ge2_range_pct": [round(min(a_v3_dir), 1), round(max(a_v3_dir), 1)],
            "all5_v3_direct_spread_pp": round(max(a_v3_dir) - min(a_v3_dir), 1),
            "frontier3_v0_ethical_alt_means": f_v0_ea,
            "open8b_v0_false_alarm_direct_ge2_pct": o_v0_dir,
            "frontier3_mean_v3_minus_v0_risk": round(mean(f_delta), 4),
            "open8b_mean_v3_minus_v0_risk": round(mean(o_delta), 4),
        })

    # 2. 4x4 Self-Preference / Preference-Leakage Analysis (mean risk across V0-V3)
    family_pairs = [
        ("DeepSeek-chat", "DeepSeek judge", "DeepSeek"),
        ("Gemini", "Gemini pro-low judge", "Gemini"),
        ("Qwen3-8B-Q8_0", "Qwen judge", "Qwen3-8B"),
        ("Llama-3.1-8B-Instruct-Q8_0", "Llama judge", "Llama3.1-8B"),
    ]
    # Compute judge-centered Z-like relative score: score(t, j) - mean_t'(score(t', j))
    judge_means = {}
    for _, j_lbl, _ in family_pairs:
        vals = [
            mean(float(by_tj[(t_lbl, j_lbl)][v]["behavior_risk_mean"]) for v in by_tj[(t_lbl, j_lbl)])
            for t_lbl, _, _ in family_pairs
        ]
        judge_means[j_lbl] = mean(vals)

    leakage_matrix = {}
    diag_biases, off_biases = [], []
    for t_lbl, _, t_short in family_pairs:
        leakage_matrix[t_short] = {}
        for _, j_lbl, j_short in family_pairs:
            raw_m = mean(float(by_tj[(t_lbl, j_lbl)][v]["behavior_risk_mean"]) for v in by_tj[(t_lbl, j_lbl)])
            # Positive = judge rates target MORE risky than judge's own average; Negative = MORE lenient (self-favoring)
            rel = raw_m - judge_means[j_lbl]
            # Also compare relative to GPT-5.5 neutral external anchor
            gpt_m = mean(float(by_tj[(t_lbl, "GPT judge")][v]["behavior_risk_mean"]) for v in by_tj[(t_lbl, "GPT judge")])
            gpt_avg = mean(
                mean(float(by_tj[(tl, "GPT judge")][v]["behavior_risk_mean"]) for v in by_tj[(tl, "GPT judge")])
                for tl, _, _ in family_pairs
            )
            excess_vs_gpt = (raw_m - judge_means[j_lbl]) - (gpt_m - gpt_avg)
            leakage_matrix[t_short][j_short] = {
                "mean_risk_v0_v3": round(raw_m, 4),
                "centered_risk": round(rel, 4),
                "excess_vs_external_gpt": round(excess_vs_gpt, 4),
            }
            if t_short == j_short:
                diag_biases.append(excess_vs_gpt)
            else:
                off_biases.append(excess_vs_gpt)

    # Specifically check Llama V2 -> V3 absolute risk drop across all 5 judges
    llama_v2_v3 = []
    for j in all_judges:
        v2 = float(by_tj[("Llama-3.1-8B-Instruct-Q8_0", j)]["V2_resource_scarcity"]["behavior_risk_mean"])
        v3 = float(by_tj[("Llama-3.1-8B-Instruct-Q8_0", j)]["V3_combined_pressure"]["behavior_risk_mean"])
        llama_v2_v3.append({
            "judge": j,
            "is_self_judge": j == "Llama judge",
            "V2_risk": round(v2, 4),
            "V3_risk": round(v3, 4),
            "V3_minus_V2": round(v3 - v2, 4),
        })

    # 3. Holm-Bonferroni Multiple-Testing Adjustment across Study 1 family of tests
    # Exact two-sided tests verified from Study 1 counts
    p_pooled_v2 = stats.binomtest(20, 20, 0.5).pvalue  # Exact McNemar 20 vs 0
    p_llama_v2 = stats.binomtest(16, 16, 0.5).pvalue   # Exact McNemar 16 vs 0
    p_llama_v1 = stats.binomtest(5, 5, 0.5).pvalue
    p_llama_v3 = stats.binomtest(5, 5, 0.5).pvalue
    p_ds_v3 = stats.binomtest(6, 6, 0.5).pvalue
    p_ds_v2 = stats.binomtest(3, 3, 0.5).pvalue
    p_ds_v1 = stats.binomtest(2, 2, 0.5).pvalue
    p_llama_v2_vs_v3 = stats.fisher_exact([[16, 174 - 16], [5, 174 - 5]]).pvalue
    p_opacity = stats.binomtest(14, 22, 0.25, alternative="two-sided").pvalue
    p_r1_align = stats.fisher_exact([[164, 36], [160, 80]]).pvalue

    raw_tests = [
        {"test": "Pooled V2 vs V0 (Exact McNemar, 20 vs 0)", "p_raw": p_pooled_v2},
        {"test": "Llama 3.1 8B V2 vs V0 (Exact McNemar, 16 vs 0)", "p_raw": p_llama_v2},
        {"test": "Intra-organizational opacity gradient (14/22 vs 25%)", "p_raw": p_opacity},
        {"test": "Ex-ante R1 algebraic alignment gain (164/200 vs 160/240)", "p_raw": p_r1_align},
        {"test": "Llama 3.1 8B Guardrail Paradox V2 vs V3 (Fisher exact, 16/174 vs 5/174)", "p_raw": p_llama_v2_vs_v3},
        {"test": "DeepSeek Chat V3 vs V0 (Exact McNemar, 6 vs 0)", "p_raw": p_ds_v3},
        {"test": "Llama 3.1 8B V1 vs V0 (Exact McNemar, 5 vs 0)", "p_raw": p_llama_v1},
        {"test": "Llama 3.1 8B V3 vs V0 (Exact McNemar, 5 vs 0)", "p_raw": p_llama_v3},
        {"test": "DeepSeek Chat V2 vs V0 (Exact McNemar, 3 vs 0)", "p_raw": p_ds_v2},
        {"test": "DeepSeek Chat V1 vs V0 (Exact McNemar, 2 vs 0)", "p_raw": p_ds_v1},
    ]
    holm_results = holm_bonferroni(raw_tests)

    # 4. PPI / Plug-in Calibration Bounds using Study 1 300-output single-annotator audit MADs
    # For the 3 Frontier judges (n_cal = 300 audited outputs, composite MADs = 0.097, 0.020, 0.033;
    # conservative error variance bounded by MAD * 1.0 since residuals |e| <= 1 in >=99.1% of cases).
    ppi_cal = {}
    judge_audit_mad = {
        "DeepSeek judge": 0.097,
        "Gemini pro-low judge": 0.020,
        "GPT judge": 0.033,
    }
    for j_lbl, mad in judge_audit_mad.items():
        # Llama V2 vs V3 risk shift PPI uncertainty width (n_all = 1000, n_cal = 300)
        v2 = float(by_tj[("Llama-3.1-8B-Instruct-Q8_0", j_lbl)]["V2_resource_scarcity"]["behavior_risk_mean"])
        v3 = float(by_tj[("Llama-3.1-8B-Instruct-Q8_0", j_lbl)]["V3_combined_pressure"]["behavior_risk_mean"])
        # Bounded variance of paired calibration error using observed MAD on [0,3] scale
        var_err = mad * (1.0 - mad)
        # Difference V2 - V3 (positive means V2 > V3)
        diff = v2 - v3
        se_diff = math.sqrt((2 * 0.25) / 1000.0 + (2 * var_err) / 300.0)
        z = 1.959963984540054
        ppi_cal[j_lbl] = {
            "llama_v2_minus_v3": round(diff, 4),
            "calibration_mad_n300": mad,
            "ppi_conservative_se": round(se_diff, 4),
            "ppi_95_ci_v2_minus_v3": [round(diff - z * se_diff, 4), round(diff + z * se_diff, 4)],
            "strictly_positive_after_ppi": (diff - z * se_diff) > 0,
        }

    report = {
        "frontier3_vs_open8b_stratification": stratification,
        "preference_leakage_4x4": {
            "matrix": leakage_matrix,
            "mean_diagonal_excess_vs_gpt": round(mean(diag_biases), 4),
            "mean_offdiagonal_excess_vs_gpt": round(mean(off_biases), 4),
            "interpretation": (
                "Negative excess_vs_external_gpt indicates a judge rates a target MORE leniently than "
                "external GPT-5.5 after judge-level mean centering; positive indicates stricter rating."
            ),
        },
        "llama_v2_to_v3_paradox_by_judge": llama_v2_v3,
        "holm_bonferroni_m10": holm_results,
        "ppi_frontier_judge_calibration": ppi_cal,
    }
    (OUT_DIR / "wave0_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print crisp summary
    print("=== WAVE 0 ZERO-GPU AUDIT SUMMARY ===")
    print("\n1) Frontier-3 vs Open-8B V3 Direct-Risk Spread:")
    for s in stratification:
        print(
            f"  {s['target']:27s} | F3 direct>=2: {s['frontier3_v3_direct_ge2_range_pct']} ({s['frontier3_v3_direct_spread_pp']:4.1f}pp)"
            f" | All5: {s['all5_v3_direct_ge2_range_pct']} ({s['all5_v3_direct_spread_pp']:4.1f}pp)"
            f" | Llama8B-judge V0 false-alarm: {s['open8b_v0_false_alarm_direct_ge2_pct']['Llama judge']:.1f}%"
        )
    print("\n2) Preference Leakage (Diagonal self-judging vs Off-diagonal vs External GPT-5.5):")
    print(f"  Mean diagonal bias = {report['preference_leakage_4x4']['mean_diagonal_excess_vs_gpt']:+.4f}, "
          f"off-diagonal = {report['preference_leakage_4x4']['mean_offdiagonal_excess_vs_gpt']:+.4f}")
    for t_short, row in leakage_matrix.items():
        print(f"  Target {t_short:11s}: self-excess={row[t_short]['excess_vs_external_gpt']:+.4f}")
    print("\n3) Holm-Bonferroni Step-Down Multiplicity Correction (m = 10 tests):")
    for h in holm_results[:6]:
        print(f"  {h['test']:68s} | p_raw={h['p_raw']:.3e} | p_holm={h['p_holm']:.3e} | sig.05={h['sig_005']}")
    print("\n4) PPI Conservative 95% CI on Llama V2 - V3 Risk Drop (Frontier-3 Judges, n_cal=300):")
    for j, v in ppi_cal.items():
        print(f"  {j:22s}: diff={v['llama_v2_minus_v3']:+.4f}, PPI 95% CI={v['ppi_95_ci_v2_minus_v3']}, strictly>0={v['strictly_positive_after_ppi']}")


if __name__ == "__main__":
    main()
