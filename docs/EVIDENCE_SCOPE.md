# Artifact Organization and Reproducibility Guide

This document summarizes the scripts, schemas, and aggregate tables included in the public `PressureTest-Biz` repository and what each component verifies offline.

| Module / Directory | Included Files | Offline Verification Scope |
| :--- | :--- | :--- |
| **Combinatorial Scenario Generator** | `scripts/build_main_dataset.py`, `scripts/validate_main_dataset.py`, `tests/test_main_dataset.py` | Deterministically synthesizes all 1,000 base business scenarios (`8 x 125` across 8 commercial domains) and 4,000 `V0`–`V3` condition prompts, validating schema invariants and prompt formatting. |
| **Structural Alignment Filter** | `scripts/analyze_structural_alignment.py`, `tests/test_release_offline.py` | Verifies the algebraic family-alignment rule (`c_idx == a_idx`, equivalent to `(r + g) % 5 == 0`) that extracts the balanced 200-scenario (`8 x 25`, 800-prompt) primary evaluation panel. |
| **Exact-Anchor Action Parser** | `scripts/analyze_boundary_crossings.py`, `tests/test_release_offline.py` | Implements deterministic lexical normalization and negation/quotation-aware exact-anchor classification (`classify_anchor_alignment`) alongside paired baseline-conditioned transition logic. |
| **Audit & Calibration Utilities** | `scripts/build_*_audit*.py`, `scripts/analyze_*_audit*.py`, `scripts/check_*.py` | Provides reusable review-sheet builders, inter-annotator agreement calculators (Cohen's kappa and quadratic-weighted kappa), and quality-gate utilities for custom annotation workflows. |
| **Multi-Judge Summary Tables** | `analysis/sources/cell_variant_summary.csv`, `analysis/sources/cell_planned_contrasts.csv`, `analysis/analyze_retained_bundle.py` | Verifies all 80 model–judge–regime summary rows, 20 cell profiles, 40 planned contrast equalities (`C_S` and `C_Delta`), and the exact hypergeometric zero-overlap reference probability (`0.9628`). |
| **Interactive Results Explorer** | `demo.py`, `demo_ui.py`, `demo.ipynb`, `Dockerfile.demo`, `tests/test_demo.py` | Builds a self-contained, zero-dependency local HTML viewer with SHA-256 provenance verification and CSV export. |

## Notes on Extending with Custom Model Runs

- All quick-start commands in `README.md` run entirely offline using the Python 3.10+ standard library without API keys or GPU dependencies.
- Optional scientific plotting and statistical packages for custom extensions are listed in `requirements-analysis.txt`.
- Sensitive credentials, private annotator worksheets, and local environment configurations are intentionally excluded to protect privacy and keep the repository lightweight.
