"""Create the structurally aligned boundary-crossing figure."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "boundary_recovery_study" / "archived_boundary_analysis"
DEFAULT_FIGURE_DIR = ROOT / "boundary_recovery_study" / "figures"

MODEL_ORDER = (
    "deepseek-chat",
    "gemini-3.1-flash-lite-preview",
    "qwen3-8b-q8_0",
    "llama-3.1-8b-instruct-q8_0",
)
MODEL_LABELS = {
    "deepseek-chat": "DeepSeek",
    "gemini-3.1-flash-lite-preview": "Gemini",
    "qwen3-8b-q8_0": "Qwen",
    "llama-3.1-8b-instruct-q8_0": "Llama",
}
VARIANT_ORDER = (
    "V1_cognitive_load",
    "V2_resource_scarcity",
    "V3_combined_pressure",
)
VARIANT_LABELS = {
    "V1_cognitive_load": "Load (V1)",
    "V2_resource_scarcity": "Scarcity (V2)",
    "V3_combined_pressure": "Combined (V3)",
}
DOMAIN_LABELS = {
    "fundraising": "Fundraising",
    "layoffs_restructuring": "Layoffs",
    "customer_relations": "Customers",
    "compliance_audit": "Compliance",
    "public_relations": "PR",
    "supply_chain_crisis": "Supply",
    "performance_management": "Performance",
    "internal_politics": "Politics",
}


def build_overlap_matrix(summary: list[dict[str, str]], overlap: list[dict[str, str]]) -> np.ndarray:
    """Build the V3 matrix on each pair's jointly decidable scenario support."""
    crossing_totals = {
        (row["model"], row["variant"]): int(row["crossed_to_shortcut"])
        for row in summary
    }
    matrix = np.full((len(MODEL_ORDER), len(MODEL_ORDER)), np.nan)
    for index, model in enumerate(MODEL_ORDER):
        if crossing_totals.get((model, "V3_combined_pressure"), 0) > 0:
            matrix[index, index] = 1.0
    for row in overlap:
        if row["variant"] != "V3_combined_pressure":
            continue
        left = MODEL_ORDER.index(row["model_a"])
        right = MODEL_ORDER.index(row["model_b"])
        common_union = int(row.get("common_union") or 0)
        if common_union:
            value = float(row["common_jaccard"])
        else:
            raw_union = int(row.get("union") or 0)
            value = float(row["jaccard"]) if raw_union else np.nan
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix


def build_joint_n_matrix(overlap: list[dict[str, str]]) -> np.ndarray:
    """Return pairwise jointly decidable V3 support sizes for cell annotations."""
    matrix = np.zeros((len(MODEL_ORDER), len(MODEL_ORDER)), dtype=int)
    for row in overlap:
        if row["variant"] != "V3_combined_pressure":
            continue
        left = MODEL_ORDER.index(row["model_a"])
        right = MODEL_ORDER.index(row["model_b"])
        value = int(row.get("jointly_eligible_n") or row.get("jointly_decidable_n") or 0)
        matrix[left, right] = value
        matrix[right, left] = value
    return matrix
COLORS = ("#0072B2", "#009E73", "#E69F00", "#D55E00")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def build_figure(
    *,
    summary_path: Path,
    cases_path: Path,
    overlap_path: Path,
    output_pdf: Path,
    output_png: Path,
) -> None:
    _style()
    summary = _read_csv(summary_path)
    cases = _read_csv(cases_path)
    overlap = _read_csv(overlap_path)

    rates = {
        (row["model"], row["variant"]): 100 * float(row["crossing_rate"])
        for row in summary
    }
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.72),
        gridspec_kw={"width_ratios": [1.05, 1.08, 0.9], "wspace": 0.58},
    )

    ax = axes[0]
    x = np.arange(len(VARIANT_ORDER))
    width = 0.18
    for index, model in enumerate(MODEL_ORDER):
        values = [rates.get((model, variant), 0.0) for variant in VARIANT_ORDER]
        offset = (index - 1.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=COLORS[index],
            label=MODEL_LABELS[model],
            edgecolor="white",
            linewidth=0.4,
        )
        for bar, value in zip(bars, values):
            if value >= 0.8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.35,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.1,
                    rotation=90,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(["V1", "V2", "V3"])
    ax.set_ylabel("Boundary crossings (%)")
    ax.set_title("A  Structurally aligned crossings", loc="left")
    ax.set_ylim(0, max(13.2, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)

    focus = {
        ("deepseek-chat", "V3_combined_pressure"): ("DeepSeek V3", COLORS[0]),
        ("llama-3.1-8b-instruct-q8_0", "V2_resource_scarcity"): ("Llama V2", COLORS[3]),
    }
    domain_counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in cases:
        key = (row["model"], row["variant"])
        if key not in focus:
            continue
        cell = domain_counts[(row["model"], row["variant"], row["subdomain"])]
        cell[0] += int(row["baseline_ethical"])
        cell[1] += int(row["crossed_to_shortcut"])

    ax = axes[1]
    domains = list(DOMAIN_LABELS)
    y = np.arange(len(domains))
    for index, (key, (label, color)) in enumerate(focus.items()):
        values = []
        for domain in domains:
            eligible, crossed = domain_counts[(key[0], key[1], domain)]
            values.append(100 * crossed / eligible if eligible else 0.0)
        offset = (-0.11 if index == 0 else 0.11)
        ax.barh(y + offset, values, height=0.2, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([DOMAIN_LABELS[domain] for domain in domains])
    ax.invert_yaxis()
    ax.set_xlabel("Crossing rate (%)")
    ax.set_title("B  Crossing rates by conflict type", loc="left")
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)

    matrix = build_overlap_matrix(summary, overlap)
    joint_n = build_joint_n_matrix(overlap)

    ax = axes[2]
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    short = [MODEL_LABELS[model] for model in MODEL_ORDER]
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    for row in range(len(MODEL_ORDER)):
        for column in range(len(MODEL_ORDER)):
            value = matrix[row, column]
            if row == column:
                text = "—" if np.isnan(value) else f"{value:.2f}"
            else:
                value_text = "—" if np.isnan(value) else f"{value:.2f}"
                text = f"{value_text}\nn={joint_n[row, column]}"
            color = "white" if not np.isnan(value) and value > 0.55 else "#333333"
            ax.text(column, row, text, ha="center", va="center", fontsize=5.9, color=color)
    ax.set_title("C  V3 Jaccard (jointly decidable)", loc="left")
    cbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04)
    cbar.ax.tick_params(labelsize=6.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper left", bbox_to_anchor=(0.075, 0.995))
    fig.subplots_adjust(top=0.77, bottom=0.28, left=0.08, right=0.98)
    fig.text(
        0.08,
        0.025,
        "The panel contains 200 generator-aligned records; exact anchors define explicit transitions and Panel C uses jointly decidable records.",
        fontsize=6.8,
        color="#444444",
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    build_figure(
        summary_path=args.analysis_dir / "structural_alignment_crossing_summary.csv",
        cases_path=args.analysis_dir / "structural_alignment_crossing_cases.csv",
        overlap_path=args.analysis_dir / "structural_alignment_v3_model_overlap.csv",
        output_pdf=args.output_dir / "fig_structural_alignment_boundary_crossings.pdf",
        output_png=args.output_dir / "fig_structural_alignment_boundary_crossings.png",
    )
    print(args.output_dir / "fig_structural_alignment_boundary_crossings.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
