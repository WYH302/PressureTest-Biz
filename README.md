# PressureTest-Biz

**Counterfactual Scenario Generation and Paired Constraint-Adherence Evaluation for Language Models under Operational Stress**  
*面向运营压力下语言模型规范遵循评估的反事实场景生成与成对评测工具包*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WYH302/PressureTest-Biz/blob/main/demo.ipynb)
**[Quick Start](#quick-start-zero-api-calls)** · **[Interactive Viewer](#interactive-results-viewer)** · **[Repository Layout](#repository-structure)** · **[中文说明](#中文说明)**

![PressureTest-Biz Framework Overview](docs/framework_overview.png)

---

## Overview

**PressureTest-Biz** is an open-source research toolkit for studying whether language models uphold explicit operating constraints when workplace pressures intensify. Standard evaluations typically test models in calm, frictionless settings where rule compliance carries no operational cost. PressureTest-Biz instead pairs each calm-state baseline scenario (`V0`) with controlled counterfactual stress regimes—**cognitive load (`V1`)**, **resource scarcity (`V2`)**, and **combined operational pressure (`V3`)**—while holding the managerial role, business objective, explicit operating constraint, and candidate action pair (`A_P`, `A_C`) strictly invariant.

By conditioning within-case transitions on each model's own compliant baseline (`V0 -> V1/V2/V3`) and pairing a deterministic **exact-anchor action parser** (`P`, `C`, `U`) with multi-dimensional rubric contrasts (`C_S`, `C_Delta`), the toolkit helps isolate genuine pressure-induced recommendation reversals without relying on single-threshold LLM-as-a-judge heuristics.

### Core Design Highlights

| Component | Specification | Purpose |
| :--- | :--- | :--- |
| **Procedural Corpus** | `8` domains × `125` combinatorial blueprints = `1,000` base scenarios (`4,000` `V0`–`V3` prompts) | Mitigates static pretraining memorization and supports reproducible stress testing |
| **Aligned Primary Panel** | `200` structurally aligned scenarios (`8 × 25`, `800` matched prompts) via `(r + g) ≡ 0 (mod 5)` | Guarantees exact semantic pairing between operating constraints and candidate actions |
| **2 × 2 Stress Regimes** | `V0` (Calm Baseline), `V1` (Cognitive Load), `V2` (Resource Scarcity), `V3` (Combined Pressure) | Separates monotonic stress accumulation from non-monotonic single-stressor spikes |
| **Hybrid Evaluation** | Negation-aware exact-anchor parser (`P`, `C`, `U`) + `5` rubric evaluator contrast profiles | Combines zero-variance discrete commitment tracking with continuous risk diagnostics |

![Empirical Reversal Patterns Across Regimes and Domains](docs/results_overview.png)

---

## Quick Start (Zero API Calls)

All core verification scripts use **only the Python 3.10+ standard library**—no GPU, model weights, or API keys are required.

### 1. Run Unit Tests & Verify Multi-Evaluator Contrasts

```bash
python -m unittest discover -s tests -v
python analysis/analyze_retained_bundle.py
```

- `unittest` validates the deterministic `8 × 125` scenario generator, prompt structure, algebraic panel alignment (`200` aligned scenarios), exact-anchor parser edge cases, and offline HTML viewer.
- `analysis/analyze_retained_bundle.py` recomputes and verifies all `80` model–judge–regime summary rows, `20` cell profiles, `40` planned contrast equalities (`C_S`, `C_Delta`), and the exact hypergeometric overlap reference, writing outputs to `analysis/reproduced/`.

### 2. Generate & Validate the Full Scenario Corpus Locally

```bash
python scripts/build_main_dataset.py
python scripts/validate_main_dataset.py
```

This deterministically generates all `1,000` base scenario records and `4,000` rendered `V0`–`V3` prompts under `main_dataset/`, together with blank audit templates and structural validation checks (use `--help` to inspect CLI options and prompt variants).

---

## Interactive Results Viewer

You can explore the multi-evaluator condition profiles (`V0`–`V3`) interactively in any local browser:

```bash
python demo.py
```

- Generates a self-contained offline HTML explorer (`demo_output/index.html`) with zero external web requests, verifies the SHA-256 digest of `analysis/sources/cell_variant_summary.csv`, and supports interactive filtering and CSV export.
- On a headless server, run `python demo.py --no-open` (or open [`demo.ipynb`](demo.ipynb) in Jupyter / Colab).

![Interactive Results Explorer](docs/demo-preview.png)

<details>
<summary><b>Optional: Run Viewer via Docker</b></summary>

```bash
docker build -f Dockerfile.demo -t pressuretest-results .
docker run --rm -p 127.0.0.1:7860:7860 pressuretest-results
```

Then visit `http://localhost:7860` (watch a quick preview in [`docs/demo.webm`](docs/demo.webm)).
</details>

---

## Repository Structure

| Path | Description |
| :--- | :--- |
| `scripts/build_main_dataset.py`, `validate_main_dataset.py` | Deterministic `1,000`-scenario / `4,000`-prompt generator and structural validator |
| `scripts/analyze_boundary_crossings.py` | Negation- and quotation-aware exact-anchor parser (`classify_anchor_alignment`) and paired transition utilities |
| `scripts/analyze_structural_alignment.py` | Algebraic verification of the `200`-scenario structurally aligned subset and domain/overlap helpers |
| `scripts/*audit*.py`, `scripts/check_*.py` | Reusable human-audit pack builders, inter-annotator reliability calculators, and quality gates |
| `analysis/sources/` & `analysis/analyze_retained_bundle.py` | Multi-evaluator condition summaries (`cell_variant_summary.csv`), planned contrasts (`cell_planned_contrasts.csv`), and standalone verifier |
| `demo.py`, `demo_ui.py`, `demo.ipynb` | Zero-dependency interactive browser explorer and Colab notebook |
| `tests/` | Automated unit tests for scenario generation, parser logic, alignment math, and viewer safety |
| `docs/EVIDENCE_SCOPE.md` | Concise module-by-module guide to included scripts and verification scope |

Optional dependencies for custom plotting and extended statistical analyses can be installed via:

```bash
python -m pip install -r requirements-analysis.txt
```

---

## 中文说明

### 1. 项目简介

**PressureTest-Biz** 是一个围绕“商业运营压力下语言模型行动建议与规范遵循”整理的小型开源研究工具包。在常规无摩擦基线（`V0`）测试中，模型往往能够轻松给出守规建议；但在真实业务辅助场景中，决策常伴随时间紧迫、信息嘈杂、预算或人力吃紧等现实约束。本项目通过程序化组合蓝图构造商业两难场景，在保持角色、目标、明确规范约束与候选行动（守规路径 `A_P` 与违规捷径 `A_C`）严格不变的前提下，成对施加四种运营条件：

- **`V0`（平稳基线）**：时间可控、规则清晰、具备充分沟通空间；
- **`V1`（认知负载）**：多头规则并行、无关信息干扰、1 小时内给出简短决策；
- **`V2`（资源稀缺）**：现金流、时间、人力或谈判筹码严重紧缺；
- **`V3`（复合压力）**：认知负载与资源稀缺叠加高层施压、低可追责性与短期捷径激励。

通过将结构化三段输出（`Action` / `Reason` / `Operational next step`）上的**确定性精确锚点解析器**（Exact-Anchor Parser）与 5 套多维量表评估对比相结合，辅助识别模型从基线守规（`P`）转向压力下违规捷径（`C`）的成对行动反转（`P -> C`）。

### 2. 快速体验（仅需 Python 3.10+ 标准库，无需联网或 API Key）

```bash
# 1) 运行全部离线单元测试（生成平衡性、代数对齐面板、否定/引号感知解析器与看板测试）
python -m unittest discover -s tests -v

# 2) 复核 80 行多评估器条件均值、20 个模型-评估器单元及 40 项计划对比恒等式
python analysis/analyze_retained_bundle.py

# 3) 在本地生成并校验 1,000 个基础商业场景与 4,000 条 V0-V3 提示词
python scripts/build_main_dataset.py
python scripts/validate_main_dataset.py

# 4) 生成并打开本地离线交互式结果看板
python demo.py
```

### 3. 交流与说明

- 本仓库将研究中可复用的场景生成、精确锚点解析、审校模板构建及汇总核验脚本整理公开，供感兴趣的同学与同行参考复现。项目仍有诸多值得改进与拓展之处（例如多轮智能体博弈、跨语言商业治理场景及开放式改写语义识别等），欢迎通过 GitHub Issues 提出宝贵意见与建议。
- **隐私保护说明**：为保护个人及合作者隐私，公开仓库仅保留维护者 GitHub 账号（[@WYH302](https://github.com/WYH302)），不包含任何个人隐私信息、API 密钥或本地环境配置。
