# MoE contextual bandits

Synthetic clustered contextual bandits with mixture-of-experts training, LinUCB / softmax / ε-greedy policies, and experiment tooling for grids and report bundles.

## Directory overview

| Path | Purpose |
|------|---------|
| `moe_bandit/` | Library code: synthetic data, expert training, joint MoE (`train_joint_moe`), policies, bandit `runner`, and `experiments/` (main grid, joint \(d\) sweep, **report packet**). |
| `notebooks/` | Jupyter workflows (single runs, joint MoE phase 1, phase 2). |
| `tests/` | Pytest suite (including a small smoke test for the report packet). |
| `main.py` | CLI for `run_main_grid` (general experiment grid). |
| `joint_d_sweep.py` | Thin CLI around `run_joint_d_sweep`. |
| `generate_best_arm_acc_tables.py` | Helper to build tables from artifacts. |
| `outputs/` | Generated experiment outputs. New runs land here by default; the directory is gitignored, but a **frozen report packet** is committed under `outputs/report_packet/` (see below). |

## Final figures and raw results (in-repo)

The bundled report packet from the paper-style run lives here:

**[`outputs/report_packet/20260426_192145/`](outputs/report_packet/20260426_192145/)**

| Subfolder / files | Contents |
|-------------------|----------|
| `plots/main/` | Main-line figures (e.g. regret vs \(d\), α sweep, LinUCB vs softmax comparisons, regret curves). |
| `plots/motivation/` | Motivation block figures (reward heatmaps, ε RMSE comparison). |
| `plots/supplementary/` | Supplementary plots (e.g. router diagnostics, regret curves). |
| `raw/` | Per-configuration compressed arrays: `.../seed_data.npz` (contexts, rewards, oracle arms, per-policy trajectories). |
| `diagnostics/` | Extra diagnostics (e.g. regret-growth fits, best-arm accuracy tables) generated around that run. |
| `manifest.json`, `results_rows.csv`, `results_rows.jsonl`, `artifacts.json`, `*.jsonl` | Run manifest, aggregated metrics, and full `artifacts` payload for curves and metadata. |

## Report ↔ code (short map)

Pointers from the written methodology / results to this repo:

| Topic | Where it lives |
|-------|----------------|
| **Gaussian cluster data generation** | `moe_bandit/data.py` (`generate_synthetic_data`), cluster means in `_build_cluster_means`. |
| **K, d grid, horizon T, train/bandit seeds** | Defaults in `moe_bandit/experiments/report_packet.py` (`ReportPacketMotivationSettings`, `ReportPacketMainSettings`). |
| **Independent vs joint experts; ridge RMSE motivation** | Same file: motivation block (`train_experts` vs `train_joint_moe`), `_linear_diagnostics`. Implementations: `moe_bandit/experts.py`, `moe_bandit/train_joint_moe.py`. |
| **Joint MoE loss (CE + load balance), epochs / early stopping** | `moe_bandit/train_joint_moe.py`; hyperparameters mirrored in `ReportPacketSettings`. |
| **Reward table (clipped log-prob), regret definition** | Expert outputs → `moe_bandit/experts.py` (`expert_reward_matrix`); bandit loop and regret → `moe_bandit/runner.py` (`run_bandit`, `RunResult`). Wired in `report_packet._evaluate_and_store`. |
| **Policies** (uniform, ε-greedy, LinUCB, online/offline softmax, cluster router) | `moe_bandit/policies/`; orchestration and α sweep in `report_packet.py`. |
| **Main d sweep, aggregated metrics, plots** | Produced by `run_report_packet` → see committed bundle under `outputs/report_packet/20260426_192145/` (`plots/main/`, `results_rows.csv`, `artifacts.json`). |
| **Best-arm accuracy tables** | `generate_best_arm_acc_tables.py`; TeX under `outputs/report_packet/20260426_192145/diagnostics/best_arm_accuracy/` for that run. |
| **Regret growth (R(T) ≈ c T^β)** | `diagnose_regret_growth.py`; outputs for the frozen packet in `outputs/report_packet/20260426_192145/diagnostics/regret_growth/`. |

## Report packet

The **report packet** is a single reproducible folder of tables, JSON, per-run raw arrays, and plots produced by `run_report_packet` in `moe_bandit/experiments/report_packet.py`.

### What it does

1. **Creates an output directory** (by default under `outputs/report_packet/` with a `YYYYMMDD_HHMMSS` subfolder unless you disable timestamps).
2. **Writes `manifest.json`** with creation time, Python version, optional git `HEAD`, and the full `ReportPacketSettings` snapshot.
3. **Runs two experiment blocks** on synthetic data (`generate_synthetic_data`), z-scoring bandit contexts using training-set statistics:
   - **Motivation**: fixed dimension \(d\) (from `ReportPacketMotivationSettings`), for each seed compares **independent** per-cluster experts vs **joint** MoE (`train_joint_moe`), then runs the same policy suite and stores diagnostics.
   - **Main**: sweeps **\(d\)** across `d_values` × `seeds`, trains joint MoE experts each time, evaluates policies, and records LinUCB α behavior where configured.
4. **Aggregates results** into `results_rows.csv` / `results_rows.jsonl`, `approx_error_by_regime.jsonl` (ε diagnostics), `joint_training_stats.jsonl`, and `artifacts.json` (includes per-policy cumulative regret curves and other run metadata).
5. **Saves raw arrays** under `raw/<block>_<regime>_d=<d>_seed=<seed_idx>/seed_data.npz` (contexts, rewards, oracle arms, per-policy trajectories, etc.).
6. **Renders figures** under `plots/` (motivation heatmaps, regret vs \(d\), α sweep, diagnostics, etc.).

Defaults (dimensions, seeds, horizons, training epochs) live on `ReportPacketSettings` and its nested `motivation` / `main` dataclasses in that module; edit there or pass a custom `ReportPacketSettings` when calling from Python.

### How to recreate it

From the repo root, install dependencies and run the module:

```bash
uv sync
uv run python -m moe_bandit.experiments.report_packet
```

Optional flags:

- `--output-dir PATH` — base directory (default: `outputs/report_packet`).
- `--no-timestamp` — write directly into `output-dir` instead of a new timestamped subdirectory.

**Programmatic run** (e.g. smaller settings for a quick check):

```python
from pathlib import Path
from moe_bandit.experiments import ReportPacketSettings, run_report_packet

run_report_packet(Path("outputs/my_packet"), settings=ReportPacketSettings(), timestamped=False)
```

The smoke test in `tests/test_report_packet.py` uses tiny grids and is a good reference for minimal settings.

### GPU acceleration (NVIDIA CUDA vs Apple MPS)

Neural parts (per-cluster experts, joint MoE, offline softmax router) use `default_torch_device()` in `moe_bandit/torch_device.py`. **Default is CPU** for stability; GPUs are opt-in via environment variables.

| Hardware | Set before running |
|----------|-------------------|
| **NVIDIA GPU** | Install a **CUDA-enabled** PyTorch build for your driver/CUDA stack ([pytorch.org](https://pytorch.org/get-started/locally/)), then set `MOE_BANDIT_USE_CUDA=1`. Optional: `CUDA_VISIBLE_DEVICES=0` to pin one GPU on multi-GPU machines. |
| **Apple Silicon (MPS)** | The committed report packet was produced with Metal acceleration: `MOE_BANDIT_USE_MPS=1`. Requires a PyTorch build with MPS support. |

If both `MOE_BANDIT_USE_CUDA=1` and `MOE_BANDIT_USE_MPS=1` are set and both backends are available, **CUDA is chosen**. Numbers can differ slightly across devices (CUDA vs MPS vs CPU); compare runs on the same stack when checking reproducibility.

Example (NVIDIA, report packet):

```bash
export MOE_BANDIT_USE_CUDA=1
uv run python -m moe_bandit.experiments.report_packet
```

**Note:** Full default settings train neural components and sweep many \((d, \text{seed})\) pairs; expect long runtimes and GPU/CPU load comparable to a full experiment batch.

A fresh run writes a new timestamped sibling next to the committed packet (e.g. `outputs/report_packet/YYYYMMDD_HHMMSS/`); compare against `20260426_192145` if you need to verify reproducibility.
