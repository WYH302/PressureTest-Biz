"""Create the reproducible full-width PressureTest-Biz study overview."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEM = (
    ROOT.parent.parent
    / "EACL2027论文投稿"
    / "ACL匿名投稿包_20260721"
    / "figures"
    / "fig_study_overview"
)


def _box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    dashed: bool = False,
    fontsize: float = 7.6,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.2)


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    color: str = "#4B5563",
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.2,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def make_study_overview_figure(output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.85, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "blue": ("#E8F1FA", "#0072B2"),
        "green": ("#E8F5EF", "#009E73"),
        "orange": ("#FFF3DF", "#E69F00"),
        "rose": ("#FBEAE4", "#D55E00"),
        "gray": ("#F3F4F6", "#6B7280"),
    }

    ax.text(0.13, 0.955, "Benchmark", ha="center", va="center", fontsize=9, fontweight="bold", color=colors["blue"][1])
    ax.text(0.43, 0.955, "Action-boundary framework", ha="center", va="center", fontsize=9, fontweight="bold", color=colors["green"][1])
    ax.text(0.70, 0.955, "Validation", ha="center", va="center", fontsize=9, fontweight="bold", color=colors["orange"][1])
    ax.text(0.90, 0.955, "RQ evidence", ha="center", va="center", fontsize=9, fontweight="bold", color=colors["rose"][1])

    _box(ax, (0.01, 0.63), 0.14, 0.20, "1,000 base records\n8 domains × 125", facecolor=colors["blue"][0], edgecolor=colors["blue"][1], fontsize=7.2)
    _box(ax, (0.17, 0.63), 0.15, 0.20, "Four matched cues\nV0–V3\n4,000 prompts", facecolor=colors["blue"][0], edgecolor=colors["blue"][1], fontsize=7.2)
    _box(ax, (0.34, 0.63), 0.15, 0.20, "Generator gate\n$c=a$\n200 aligned cases", facecolor=colors["green"][0], edgecolor=colors["green"][1], fontsize=7.2)
    _arrow(ax, (0.15, 0.73), (0.17, 0.73))
    _arrow(ax, (0.32, 0.73), (0.34, 0.73))

    _box(ax, (0.51, 0.63), 0.16, 0.20, "Four target LLMs\n16,000 archived\nresponses", facecolor=colors["green"][0], edgecolor=colors["green"][1], fontsize=7.1)
    _arrow(ax, (0.49, 0.73), (0.51, 0.73))

    _box(ax, (0.34, 0.29), 0.15, 0.20, "Exact-anchor\nhigh-specificity\narchived endpoint", facecolor=colors["green"][0], edgecolor=colors["green"][1], fontsize=7.0)
    _box(ax, (0.51, 0.29), 0.16, 0.20, "Direct choice\n32 accepted cases\n6 cues × 4 targets", facecolor=colors["green"][0], edgecolor=colors["green"][1], fontsize=6.8)
    _arrow(ax, (0.57, 0.63), (0.415, 0.49))
    _arrow(ax, (0.61, 0.63), (0.59, 0.49))

    _box(ax, (0.69, 0.63), 0.16, 0.20, "Five judge pipelines\n80,000 score records\nresponse-only", facecolor=colors["orange"][0], edgecolor=colors["orange"][1], fontsize=6.9)
    _arrow(ax, (0.67, 0.73), (0.69, 0.73))
    _box(ax, (0.69, 0.29), 0.16, 0.20, "Scenario audit\n440 reviewed\n324 accepted", facecolor=colors["orange"][0], edgecolor=colors["orange"][1], fontsize=6.8)
    _arrow(ax, (0.69, 0.39), (0.67, 0.39))

    _box(ax, (0.34, 0.06), 0.15, 0.14, "Response audit\n38/38 confirmed\n0/144 controls", facecolor=colors["green"][0], edgecolor=colors["green"][1], fontsize=6.3)
    _arrow(ax, (0.415, 0.29), (0.415, 0.20))
    _box(ax, (0.51, 0.06), 0.16, 0.14, "Forced-choice results\n766/768 valid\nDeepSeek +3.13 pp", facecolor=colors["green"][0], edgecolor=colors["green"][1], fontsize=6.3)
    _arrow(ax, (0.59, 0.29), (0.59, 0.20))
    _box(ax, (0.69, 0.06), 0.16, 0.14, "Evaluator reliability\n5 response-only judges\ncontext-aware humans", facecolor=colors["orange"][0], edgecolor=colors["orange"][1], fontsize=6.0)
    _arrow(ax, (0.77, 0.29), (0.77, 0.20))

    _box(ax, (0.87, 0.20), 0.12, 0.63, "RQ1  action\nfrequency\n\nRQ2  event\nconcentration\n\nRQ3  pipeline\nstability", facecolor=colors["rose"][0], edgecolor=colors["rose"][1], fontsize=6.35)
    _arrow(ax, (0.85, 0.73), (0.87, 0.73))
    _arrow(ax, (0.85, 0.39), (0.87, 0.39))

    ax.text(
        0.015,
        0.015,
        "All stages shown are released, hash-bound evidence or completed dual-human gates.",
        ha="left",
        va="bottom",
        fontsize=6.6,
        color="#4B5563",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_STEM)
    args = parser.parse_args(argv)
    make_study_overview_figure(args.output_stem)
    print(f"Wrote {args.output_stem.with_suffix('.pdf')} and {args.output_stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
