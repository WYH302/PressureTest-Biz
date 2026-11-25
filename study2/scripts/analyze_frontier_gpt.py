"""Compute exact final statistics for Study 2 closed-weight frontier GPT models:
  - gpt-6.2-sol (reasoning.effort = medium): 640 8-arm + 160 matched explicit-meta EA
  - gpt-6.1-terra (reasoning.effort = high): 640 8-arm + 160 matched explicit-meta EA
Plus Stage 2 DeBERTa-v3-large-mnli Cascade audit metrics across all 5 models (2,480 generations).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from scipy.stats import wilcoxon

RES = Path(__file__).resolve().parents[1] / "outputs"

ARMS = [
    "M0_neutral_control",
    "ML_load_only",
    "MS_scarcity_only",
    "MH_leadership_only",
    "MD_detectability_only",
    "MLS_load_x_scarcity",
    "MB_bundled_pressure",
    "MR_constraint_regrounding",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]


def summarize_nli(path: Path) -> dict:
    rows = load_jsonl(path)
    n = len(rows)
    s1 = sum(1 for r in rows if r["stage1_exact"] in {"P", "C"})
    s2 = sum(1 for r in rows if r["stage2_cascade"] in {"P", "C"})
    dec = [r for r in rows if r["choice_action_consistent"] is not None]
    agr = sum(1 for r in dec if r["choice_action_consistent"])
    return {
        "n": n,
        "s1_n": s1,
        "s1_pct": round(100.0 * s1 / n, 2),
        "s2_n": s2,
        "s2_pct": round(100.0 * s2 / n, 2),
        "gain_pp": round(100.0 * (s2 - s1) / n, 2),
        "decisive_checked": len(dec),
        "agreed": agr,
        "agreement_pct": round(100.0 * agr / max(1, len(dec)), 2),
    }


def analyze_one(model_id: str, effort: str, f8: str, fea: str, fnli: str) -> dict:
    rows8 = load_jsonl(RES / f8)
    rows_ea = load_jsonl(RES / fea)
    by_arm_sid: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows8:
        by_arm_sid[r["condition_id"]][r["base_scenario_id"]] = r
    ea_by_arm_sid: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows_ea:
        ea_by_arm_sid[r["condition_id"]][r["base_scenario_id"]] = r

    arm_stats = {}
    for cid in ARMS:
        rs = list(by_arm_sid[cid].values())
        rt = [x["reasoning_tokens"] for x in rs]
        cr = sum(x["crossed"] for x in rs)
        nc = sum(x["non_committal"] for x in rs)
        arm_stats[cid] = {
            "n": len(rs),
            "crossed": cr,
            "crossing_pct": round(100.0 * cr / len(rs), 2),
            "non_committal": nc,
            "mean_reasoning_tokens": round(mean(rt), 2),
            "median_reasoning_tokens": round(median(rt), 1),
            "max_reasoning_tokens": max(rt),
        }

    paired_tests = {}
    for key, a, b in [
        ("MH_vs_M0", "MH_leadership_only", "M0_neutral_control"),
        ("MB_vs_M0", "MB_bundled_pressure", "M0_neutral_control"),
        ("MR_vs_MB", "MR_constraint_regrounding", "MB_bundled_pressure"),
        ("MR_vs_MH", "MR_constraint_regrounding", "MH_leadership_only"),
        ("MR_vs_M0", "MR_constraint_regrounding", "M0_neutral_control"),
    ]:
        sids = sorted(set(by_arm_sid[a]) & set(by_arm_sid[b]))
        xa = [by_arm_sid[a][s]["reasoning_tokens"] for s in sids]
        xb = [by_arm_sid[b][s]["reasoning_tokens"] for s in sids]
        _, p = wilcoxon(xa, xb)
        diff = mean(xa) - mean(xb)
        paired_tests[key] = {
            "n": len(sids),
            "mean_a": round(mean(xa), 2),
            "mean_b": round(mean(xb), 2),
            "diff_tokens": round(diff, 2),
            "rel_pct": round(100.0 * diff / mean(xb), 1),
            "wilcoxon_p": float(p),
        }

    ea_stats = {}
    for cid in ["MS_scarcity_only", "MB_bundled_pressure"]:
        sids = sorted(set(by_arm_sid[cid]) & set(ea_by_arm_sid[cid]))
        xc = [by_arm_sid[cid][s]["reasoning_tokens"] for s in sids]
        xm = [ea_by_arm_sid[cid][s]["reasoning_tokens"] for s in sids]
        cr_c = sum(by_arm_sid[cid][s]["crossed"] for s in sids)
        cr_m = sum(ea_by_arm_sid[cid][s]["crossed"] for s in sids)
        _, p = wilcoxon(xm, xc)
        ea_stats[cid] = {
            "n": len(sids),
            "clean_crossed": cr_c,
            "meta_crossed": cr_m,
            "clean_mean_reas_tok": round(mean(xc), 2),
            "meta_mean_reas_tok": round(mean(xm), 2),
            "diff_tokens": round(mean(xm) - mean(xc), 2),
            "rel_pct": round(100.0 * (mean(xm) - mean(xc)) / mean(xc), 1),
            "wilcoxon_p": float(p),
        }

    return {
        "model": model_id,
        "reasoning_effort": effort,
        "n_8arm": len(rows8),
        "n_ea_meta": len(rows_ea),
        "total_calls": len(rows8) + len(rows_ea),
        "total_crossed": sum(x["crossed"] for x in rows8) + sum(x["crossed"] for x in rows_ea),
        "est_usd_total": round(sum(x.get("est_usd", 0) for x in rows8) + sum(x.get("est_usd", 0) for x in rows_ea), 2),
        "arms": arm_stats,
        "reasoning_token_wilcoxon": paired_tests,
        "eval_awareness_matched80": ea_stats,
        "nli_cascade_640": summarize_nli(RES / fnli),
    }


def main() -> None:
    sol = analyze_one(
        "gpt-6.2-sol", "medium",
        "gpt62sol_medium_8arm_1920.jsonl",
        "gpt62sol_medium_ea_meta_160.jsonl",
        "nli_audited_gpt62sol_640.jsonl",
    )
    terra = analyze_one(
        "gpt-6.1-terra", "high",
        "gpt61terra_high_8arm_1920.jsonl",
        "gpt61terra_high_ea_meta_160.jsonl",
        "nli_audited_gpt61terra_640.jsonl",
    )
    # Pooled 5-model NLI Cascade (3 open + 2 frontier = 2,480 generations)
    nli_all = {
        "qwen2.5-3b": summarize_nli(RES / "nli_audited_gen_qwen2.5_3b_400.jsonl"),
        "qwen2.5-7b": summarize_nli(RES / "nli_audited_gen_qwen2.5_7b_400.jsonl"),
        "deepseek-r1-7b": summarize_nli(RES / "nli_audited_gen_deepseek_r1_7b_cot400.jsonl"),
        "gpt-6.2-sol": sol["nli_cascade_640"],
        "gpt-6.1-terra": terra["nli_cascade_640"],
    }
    tot_n = sum(v["n"] for v in nli_all.values())
    tot_s1 = sum(v["s1_n"] for v in nli_all.values())
    tot_s2 = sum(v["s2_n"] for v in nli_all.values())
    tot_dec = sum(v["decisive_checked"] for v in nli_all.values())
    tot_agr = sum(v["agreed"] for v in nli_all.values())
    nli_all["pooled_5model"] = {
        "n": tot_n,
        "s1_n": tot_s1,
        "s1_pct": round(100.0 * tot_s1 / tot_n, 2),
        "s2_n": tot_s2,
        "s2_pct": round(100.0 * tot_s2 / tot_n, 2),
        "gain_pp": round(100.0 * (tot_s2 - tot_s1) / tot_n, 2),
        "decisive_checked": tot_dec,
        "agreed": tot_agr,
        "agreement_pct": round(100.0 * tot_agr / tot_dec, 2),
    }

    summary = {
        "gpt-6.2-sol": sol,
        "gpt-6.1-terra": terra,
        "nli_cascade_5model_2480": nli_all,
    }
    out = RES / "frontier_gpt_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE:", out)
    print(json.dumps({
        "sol_total": sol["total_calls"], "sol_usd": sol["est_usd_total"], "sol_ea": sol["eval_awareness_matched80"],
        "terra_total": terra["total_calls"], "terra_usd": terra["est_usd_total"], "terra_ea": terra["eval_awareness_matched80"],
        "pooled_nli": nli_all["pooled_5model"],
    }, indent=2))


if __name__ == "__main__":
    main()
