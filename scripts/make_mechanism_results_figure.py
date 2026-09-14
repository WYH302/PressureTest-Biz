"""Plot the preregistered repeated-choice mechanism results from analysis CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "mechanism_decomposition_study" / "repeated_analysis"
DEFAULT_OUTPUT_STEM = (
    ROOT.parent.parent
    / "EACL2027论文投稿"
    / "ACL匿名投稿包_20260721"
    / "figures"
    / "fig_mechanism_repeated_choice"
)

CONDITIONS = (
    "M0_neutral_control",
    "ML_load_only",
    "MS_scarcity_only",
    "MH_leadership_only",
    "MD_detectability_only",
    "MLS_load_x_scarcity",
    "MB_bundled_pressure",
    "MR_constraint_regrounding",
)
CONDITION_LABELS = ("M0", "Load", "Scarcity", "Leadership", "Detect.", "Load×Scar.", "Bundle", "Reground")
COMPARISONS = (
    "ML_minus_M0",
    "MS_minus_M0",
    "MH_minus_M0",
    "MD_minus_M0",
    "MLS_minus_M0",
    "MB_minus_M0",
    "MR_minus_MB",
)
COMPARISON_LABELS = ("Load−M0", "Scar.−M0", "Leader−M0", "Detect.−M0", "L×S−M0", "Bundle−M0", "Regr.−Bundle")
MODELS = ("deepseek-chat", "llama-3.1-8b-instruct-q8_0")
MODEL_LABELS = {"deepseek-chat": "DeepSeek", "llama-3.1-8b-instruct-q8_0": "Llama"}
MODEL_COLORS = {"deepseek-chat": "#2B6CB0", "llama-3.1-8b-instruct-q8_0": "#D97706"}
MODEL_MARKERS = {"deepseek-chat": "o", "llama-3.1-8b-instruct-q8_0": "s"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], field: str, *, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source.name} has an invalid {field!r} value") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source.name} requires finite {field!r} values")
    return value


def _index_rows(
    rows: Iterable[dict[str, str]],
    *,
    key_field: str,
    expected_keys: tuple[str, ...],
    source: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        model = str(row.get("model", ""))
        key = str(row.get(key_field, ""))
        pair = (model, key)
        if model not in MODELS or key not in expected_keys or pair in indexed:
            raise ValueError(f"{source.name} has an invalid or duplicate row key: {pair}")
        indexed[pair] = row
    expected = {(model, key) for model in MODELS for key in expected_keys}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise ValueError(f"{source.name} is missing registered rows: {missing[:4]}")
    return indexed


def _bounded_interval(row: dict[str, str], *, value: str, source: Path) -> tuple[float, float, float]:
    estimate = _float(row, value, source=source)
    low = _float(row, "ci_low", source=source)
    high = _float(row, "ci_high", source=source)
    if not 0 <= low <= estimate <= high <= 1:
        raise ValueError(f"{source.name} contains an invalid probability interval")
    return estimate, low, high


def make_mechanism_results_figure(
    *,
    condition_rates_path: Path,
    paired_effects_path: Path,
    stability_path: Path,
    output_stem: Path,
) -> None:
    """Render vector and raster versions of the three-panel result figure."""

    condition_index = _index_rows(
        _read_csv(condition_rates_path),
        key_field="condition_id",
        expected_keys=CONDITIONS,
        source=condition_rates_path,
    )
    effect_index = _index_rows(
        _read_csv(paired_effects_path),
        key_field="comparison",
        expected_keys=COMPARISONS,
        source=paired_effects_path,
    )
    stability_index = _index_rows(
        _read_csv(stability_path),
        key_field="condition_id",
        expected_keys=CONDITIONS,
        source=stability_path,
    )

    condition_series: dict[str, tuple[list[float], list[float], list[float]]] = {}
    effect_series: dict[str, tuple[list[float], list[float], list[float]]] = {}
    stability_series: dict[str, tuple[list[float], list[float]]] = {}
    for model in MODELS:
        estimates, lows, highs = [], [], []
        for condition in CONDITIONS:
            estimate, low, high = _bounded_interval(
                condition_index[(model, condition)],
                value="shortcut_choice_probability",
                source=condition_rates_path,
            )
            estimates.append(100 * estimate)
            lows.append(100 * (estimate - low))
            highs.append(100 * (high - estimate))
        condition_series[model] = estimates, lows, highs

        effects, left_errors, right_errors = [], [], []
        for comparison in COMPARISONS:
            row = effect_index[(model, comparison)]
            estimate = _float(row, "risk_difference", source=paired_effects_path)
            low = _float(row, "ci_low", source=paired_effects_path)
            high = _float(row, "ci_high", source=paired_effects_path)
            if not -1 <= low <= estimate <= high <= 1:
                raise ValueError(f"{paired_effects_path.name} contains an invalid effect interval")
            effects.append(100 * estimate)
            left_errors.append(100 * (estimate - low))
            right_errors.append(100 * (high - estimate))
        effect_series[model] = effects, left_errors, right_errors

        agreements, unanimous_rates = [], []
        for condition in CONDITIONS:
            row = stability_index[(model, condition)]
            agreement = _float(row, "mean_pairwise_agreement", source=stability_path)
            unanimous = _float(row, "unanimous_rate", source=stability_path)
            if not 0 <= agreement <= 1 or not 0 <= unanimous <= 1:
                raise ValueError(f"{stability_path.name} contains a stability value outside [0, 1]")
            agreements.append(100 * agreement)
            unanimous_rates.append(100 * unanimous)
        stability_series[model] = agreements, unanimous_rates

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.45, 2.85), constrained_layout=True)
    x = np.arange(len(CONDITIONS), dtype=float)

    for model_index, model in enumerate(MODELS):
        estimates, lows, highs = condition_series[model]
        offset = (-0.08, 0.08)[model_index]
        axes[0].errorbar(
            x + offset,
            estimates,
            yerr=np.array([lows, highs]),
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=3.7,
            linestyle="none",
            capsize=2.0,
            label=MODEL_LABELS[model],
        )
    axes[0].set_title("A  Shortcut-choice probability", loc="left", fontweight="bold")
    axes[0].set_ylabel("Assignments (%)")
    axes[0].set_xticks(x, CONDITION_LABELS, rotation=35, ha="right")
    axes[0].grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.75)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    y = np.arange(len(COMPARISONS), dtype=float)
    axes[1].axvline(0, color="#6B7280", linewidth=0.8, linestyle="--", zorder=0)
    for model_index, model in enumerate(MODELS):
        estimates, left_errors, right_errors = effect_series[model]
        offset = (-0.10, 0.10)[model_index]
        axes[1].errorbar(
            estimates,
            y + offset,
            xerr=np.array([left_errors, right_errors]),
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            markersize=3.5,
            linewidth=0,
            elinewidth=1.2,
            capsize=2.0,
        )
    axes[1].set_title("B  Paired effects", loc="left", fontweight="bold")
    axes[1].set_xlabel("Risk difference (pp)")
    axes[1].set_yticks(y, COMPARISON_LABELS)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", color="#D1D5DB", linewidth=0.55, alpha=0.75)

    for model_index, model in enumerate(MODELS):
        agreements, unanimous_rates = stability_series[model]
        offset = (-0.08, 0.08)[model_index]
        axes[2].scatter(
            x + offset,
            agreements,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            s=15,
            zorder=3,
        )
        axes[2].scatter(
            x + offset,
            unanimous_rates,
            facecolors="none",
            edgecolors=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            s=18,
            linewidths=0.9,
            zorder=3,
        )
    axes[2].set_title("C  Replicate agreement", loc="left", fontweight="bold")
    axes[2].set_ylabel("Agreement (%)")
    axes[2].set_xticks(x, CONDITION_LABELS, rotation=35, ha="right")
    axes[2].set_ylim(0, 102)
    axes[2].grid(axis="y", color="#D1D5DB", linewidth=0.55, alpha=0.75)
    axes[2].text(
        0.01,
        0.03,
        "filled: pairwise mean\nopen: unanimous",
        transform=axes[2].transAxes,
        fontsize=6.4,
        color="#4B5563",
        va="bottom",
    )

    try:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-rates", type=Path, default=DEFAULT_ANALYSIS / "condition_rates.csv")
    parser.add_argument("--paired-effects", type=Path, default=DEFAULT_ANALYSIS / "paired_effects.csv")
    parser.add_argument("--stability", type=Path, default=DEFAULT_ANALYSIS / "stability_summary.csv")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()
    make_mechanism_results_figure(
        condition_rates_path=args.condition_rates,
        paired_effects_path=args.paired_effects,
        stability_path=args.stability,
        output_stem=args.output_stem,
    )
    print(f"Wrote {args.output_stem.with_suffix('.pdf')}")
    print(f"Wrote {args.output_stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
