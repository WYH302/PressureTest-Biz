"""Exact sensitivity analysis for paired binary boundary-crossing comparisons."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom, binomtest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "mechanism_decomposition_study" / "power"


@lru_cache(maxsize=None)
def _exact_rejection_region(discordant_pairs: int, alpha: float) -> tuple[int, ...]:
    if discordant_pairs == 0:
        return ()
    return tuple(
        successes
        for successes in range(discordant_pairs + 1)
        if binomtest(successes, discordant_pairs, p=0.5, alternative="two-sided").pvalue <= alpha
    )


def exact_mcnemar_power(
    *,
    n_scenarios: int,
    risk_difference: float,
    discordance_rate: float,
    alpha: float = 0.05,
) -> float:
    """Power of the exact conditional McNemar test under a paired multinomial model.

    ``discordance_rate`` is P(0->1)+P(1->0), while ``risk_difference``
    is P(0->1)-P(1->0).  Concordant-pair allocation does not affect the
    conditional McNemar statistic, so it need not be specified.
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0 <= discordance_rate <= 1:
        raise ValueError("discordance_rate must lie in [0, 1]")
    if abs(risk_difference) > discordance_rate + 1e-12:
        raise ValueError("The absolute risk difference cannot exceed total discordance")
    if discordance_rate == 0:
        return 0.0
    favorable_given_discordant = 0.5 + risk_difference / (2.0 * discordance_rate)
    power = 0.0
    for discordant_pairs in range(n_scenarios + 1):
        probability_d = binom.pmf(discordant_pairs, n_scenarios, discordance_rate)
        if probability_d == 0:
            continue
        rejection = _exact_rejection_region(discordant_pairs, alpha)
        if rejection:
            conditional = sum(
                binom.pmf(successes, discordant_pairs, favorable_given_discordant)
                for successes in rejection
            )
            power += probability_d * conditional
    return float(power)


def build_sensitivity_table(
    *,
    n_scenarios: int = 200,
    risk_differences: Sequence[float] = (0.02, 0.03, 0.05, 0.075, 0.10),
    discordance_rates: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for discordance in discordance_rates:
        for effect in risk_differences:
            if abs(effect) > discordance:
                continue
            rows.append(
                {
                    "n_scenarios": n_scenarios,
                    "independent_unit": "scenario",
                    "risk_difference": effect,
                    "discordance_rate": discordance,
                    "p_0_to_1": (discordance + effect) / 2.0,
                    "p_1_to_0": (discordance - effect) / 2.0,
                    "alpha_two_sided": alpha,
                    "exact_mcnemar_power": exact_mcnemar_power(
                        n_scenarios=n_scenarios,
                        risk_difference=effect,
                        discordance_rate=discordance,
                        alpha=alpha,
                    ),
                    "interpretation": "Design sensitivity, not observed/post-hoc power.",
                }
            )
    return rows


def build_sample_size_sensitivity_table(
    *,
    n_scenarios_values: Sequence[int] = (160, 200),
    risk_differences: Sequence[float] = (0.02, 0.03, 0.05, 0.075, 0.10),
    discordance_rates: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    if not n_scenarios_values or len(set(n_scenarios_values)) != len(n_scenarios_values):
        raise ValueError("n_scenarios_values must be nonempty and unique")
    return [
        row
        for n_scenarios in n_scenarios_values
        for row in build_sensitivity_table(
            n_scenarios=n_scenarios,
            risk_differences=risk_differences,
            discordance_rates=discordance_rates,
            alpha=alpha,
        )
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_power_figure(rows: Iterable[dict[str, Any]], output_stem: Path) -> None:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["n_scenarios"]), float(row["discordance_rate"]))
        grouped.setdefault(key, []).append(dict(row))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    discordances = sorted({key[1] for key in grouped})
    sample_sizes = sorted({key[0] for key in grouped})
    colors = dict(zip(discordances, ["#0072B2", "#009E73", "#E69F00", "#D55E00"]))
    fig, axes = plt.subplots(
        1,
        len(sample_sizes),
        figsize=(3.3 * len(sample_sizes), 2.7),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, n_scenarios in zip(axes[0], sample_sizes):
        for discordance in discordances:
            group = grouped.get((n_scenarios, discordance), [])
            if not group:
                continue
            ordered = sorted(group, key=lambda row: float(row["risk_difference"]))
            ax.plot(
                [100 * float(row["risk_difference"]) for row in ordered],
                [float(row["exact_mcnemar_power"]) for row in ordered],
                marker="o",
                color=colors[discordance],
                label=f"Discordance {100 * discordance:.0f}%",
            )
        ax.axhline(0.80, color="#555555", linewidth=1, linestyle="--", label="80% target")
        ax.set_title(f"N = {n_scenarios} scenarios", fontsize=9)
        ax.set_xlabel("Paired risk difference (percentage points)")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.18)
    axes[0][0].set_ylabel("Exact McNemar power")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.99, 0.5))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> int:
    effects = tuple(round(value / 1000, 3) for value in range(0, 101, 5))
    rows = build_sample_size_sensitivity_table(risk_differences=effects)
    _write_csv(OUTPUT_DIR / "paired_binary_power_sensitivity.csv", rows)
    make_power_figure(rows, OUTPUT_DIR / "fig_paired_binary_power_sensitivity")
    summary = {
        "status": "complete",
        "n_scenarios": [160, 200],
        "analysis": "Exact conditional McNemar design sensitivity",
        "independent_unit": "scenario",
        "replicate_note": "Repeated generations are clustered within scenario and do not multiply n_scenarios.",
        "scope_limit": "This is paired-binary endpoint sensitivity, not a formal power analysis of the three-repeat scenario-cluster bootstrap estimator.",
        "rows": len(rows),
        "not_observed_power": True,
    }
    (OUTPUT_DIR / "power_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
