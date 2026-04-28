# Report Packet Spec

## Goal

Produce a self-contained `outputs/report_packet/` directory containing every
artifact (data, plot, metric, log) needed to write the report and reproduce
its results. Single command, deterministic, all paths relative to the packet
root.

## Scope and constraints

This packet supports a 10-minute presentation and a written report. It is
**joint-training-centric**. Independent-regime artifacts appear only as
motivational evidence (one comparison block at one reference config), not as
a co-equal experimental axis.

**Critical constraint that drives the entire plot design:**

Contamination has no mechanism to operate through under joint training. Joint
training pools all data across clusters; the `contamination` parameter is a
per-expert-training-data-partition knob that only exists in the independent
regime. **Plots must reflect this.** Specifically:

- Joint regime: never plot regret as a function of contamination. The values
  would be identical across contam by construction, and presenting them on a
  shared (contam × sep) axis with the independent regime misleadingly
  suggests "joint is robust to contam" when it is in fact inert to it.
- Independent regime (used only in the motivation block): contamination is a
  legitimate axis but is not the headline. Present it as a misspecification
  diagnostic.
- The two regimes should use different plot types. Independent: heatmap or
  faceted line plot over (contam × sep). Joint: line plot over d (or sep, but
  *never* over contam).

If a single figure compares both regimes, the joint side must be a single
slice (one contam value) or a single line, not a heatmap with flat rows.

## Directory structure

```
outputs/report_packet/<timestamp>/
├── manifest.json
├── results_rows.csv
├── results_rows.jsonl                  # same content, jsonl for streaming
├── artifacts.json                      # curves, diagnostics
├── joint_training_stats.jsonl          # one row per trained joint MoE
├── approx_error_by_regime.jsonl        # independent vs joint epsilon
├── plots/
│   ├── motivation/
│   │   ├── reward_heatmap_independent.png
│   │   ├── reward_heatmap_joint.png
│   │   ├── epsilon_rmse_comparison.png
│   │   └── joint_training_diagnostics.png
│   ├── main/
│   │   ├── regret_vs_d.png             # main d-sweep figure
│   │   ├── regret_curves_per_d.png     # T trajectories, faceted by d
│   │   ├── alpha_sweep.png
│   │   ├── linucb_minus_online_softmax_vs_d.png
│   │   └── linucb_minus_offline_softmax_vs_d.png
│   └── supplementary/
│       └── per_seed_regret_vs_d.png    # show seed variance
├── raw/
│   ├── independent_d=4_sep=1.5_contam=0.05/
│   │   ├── seed_<s>.npz                # full data + rewards + per-policy traces
│   │   ├── chosen_arms_<policy>_seed<s>.npy
│   │   ├── rewards_<policy>_seed<s>.npy
│   │   └── regret_curve_<policy>_seed<s>.npy
│   ├── joint_d=4_sep=1.5/
│   │   └── ...
│   └── joint_d=<d>_sep=1.5/
│       └── ...
└── logs/
    ├── run.log
    └── train.log
```

## Two experimental blocks

### Block A: Motivation (independent vs joint at reference config)

**Purpose:** justify "why joint" by showing independent training produces a
spike-plus-plateau reward surface with large RMSE epsilon, while joint
training produces a smoother surface with smaller RMSE epsilon. This is the
3-slide motivation for the talk and the methodology section of the report.

**Config:**
- `d=4, K=4, cluster_sep=1.5, contamination=0.05, n_train=8000, T=10000`
- `seeds=[0, 1, 2]`
- Both regimes (independent at this single contam, joint at this single config)
- Epsilon-greedy uses the existing decaying schedule
  `epsilon_t = min(1, c / t)` with `c=50.0`.

**Outputs to save:**
- Reward matrix per regime (averaged over seeds, also one example seed for
  heatmap visualization).
- Per-arm offline ridge fit `theta_a, intercept_a, ridge_lambda=1.0`.
- Per-arm RMSE and max-residual epsilon, mean and std across seeds.
- Joint training diagnostics from `train_joint_moe.JointTrainingStats`:
  CE loss curve, val acc curve, per-epoch gate means, collapse warning,
  final gate means.
- Full raw `.npz` bundle per `(regime, seed)` with:
  `X_train`, `y_train`, `cluster_train`, `X_bandit`, `y_bandit`,
  `cluster_bandit`, `R_raw`, `R_scaled`, `oracle_arm`, and per-policy
  `chosen_arm`, `reward`, `oracle_reward`, `regret`, `cumulative_regret`.
  This is what makes plot changes possible without rerunning experiments.

**Plots:**

1. `reward_heatmap_independent.png` and `reward_heatmap_joint.png`: 4×4
   heatmaps showing mean reward `R[cluster_id, arm]` for one example seed.
   Same color range across both panels for visual comparability. Title
   includes ε_RMSE.

2. `epsilon_rmse_comparison.png`: bar chart, two bars (independent, joint),
   y-axis = RMSE epsilon, error bars = std across seeds. Annotate the
   percent reduction.

3. `joint_training_diagnostics.png`: 2×2 panel. (a) CE loss vs epoch.
   (b) val accuracy vs epoch. (c) gate means vs epoch (one line per expert).
   (d) bar chart of final gate means with collapse threshold (0.05) marked.

### Block B: Main result (joint d-sweep)

**Purpose:** the headline empirical result. How does bandit routing scale
with context dimension under realistic (joint-trained) experts?

**Config:**
- `K=4, cluster_sep=1.5, n_train=8000, T=10000`
- `d ∈ [2, 4, 8, 16, 32, 64]`
- `seeds=[0, 1, 2, 3, 4]`
- Joint regime only

**LinUCB alpha sweep:**
- For each d, run LinUCB with `alpha ∈ [0.5, 1.0, 2.0, 4.0, 8.0]`.
- Pick best alpha per d based on mean cumulative regret.
- Report best alpha per d in a table.
- Use best alpha per d for the headline `regret_vs_d` plot.

**Other policies:**
- `uniform`, `epsilon_greedy` with decaying
  `epsilon_t = min(1, c / t)` and `c=50.0`, `online_softmax_best_arm`,
  `softmax_best_arm`, `oracle`.

**Online softmax router baseline:**
- Include `online_softmax_best_arm` as the fair learned bandit comparator.
- It uses a stochastic softmax policy trained online from observed rewards only
  via policy-gradient updates. It should be presented beside LinUCB,
  epsilon-greedy, and uniform as an online method.
- Its role is to test whether a flexible learned online router can match or
  beat LinUCB under the same bandit-feedback constraint.

**Offline supervised softmax router reference:**
- Use Setup 2: train the softmax router on the same context distribution used
  for expert training, not on the bandit evaluation stream.
- Concretely, after freezing the experts, compute
  `R_router_train = expert_reward_matrix(experts, X_train, y_train)` on the
  expert-training contexts, train `softmax_best_arm` using labels
  `argmax_a R_router_train[t, a]`, then evaluate that frozen router on
  `X_bandit`.
- This is a full-information offline learned-router reference: it gets
  best-arm labels during offline training, but it does not see the evaluation
  stream. Do not present it as an online bandit method or as an upper bound.
- Do not use Setup 1, where softmax is trained on `X_bandit` and evaluated on
  the same `X_bandit`; that leaks evaluation contexts into the supervised
  baseline.

**Outputs to save:**
- One row per (d, seed, policy) in `results_rows.csv` with columns:
  `d, seed_idx, policy, alpha, final_cum_regret, avg_regret,
  chosen_arm_mean_reward, best_arm_acc`.
- Per-run cumulative regret curves (T points each), saved in
  `artifacts.json` indexed by (d, seed, policy).
- Per-run chosen arms, oracle arms, rewards (T points each) saved as
  `.npy` in `raw/`.
- Joint training diagnostics per d (one trained MoE per d) in
  `joint_training_stats.jsonl`.
- Full raw `.npz` bundle per `(d, seed)` with:
  `X_train`, `y_train`, `cluster_train`, `X_bandit`, `y_bandit`,
  `cluster_bandit`, `R_raw`, `R_scaled`, `R_router_train_raw`,
  `R_router_train_scaled`, `oracle_arm`, and per-policy `chosen_arm`,
  `reward`, `oracle_reward`, `regret`, `cumulative_regret`. The `.npz`
  should be enough to remake regret plots, reward heatmaps, oracle-gap
  histograms, cluster-conditioned diagnostics, and policy trace plots
  without rerunning training/evaluation.

**Plots:**

1. `regret_vs_d.png`: x-axis d (log scale, 2^k ticks), y-axis mean final
   cumulative regret. One line per policy. **Shaded bands = std across
   seeds.** This is the headline figure for the talk.

2. `regret_curves_per_d.png`: 2×3 grid (one panel per d), each showing
   cumulative regret vs t for all policies. Mean across seeds, no shading
   (clarity over completeness for this plot).

3. `alpha_sweep.png`: x-axis alpha (log scale), y-axis final regret. One
   line per d. Annotate the chosen alpha per d with a marker.

4. `linucb_minus_online_softmax_vs_d.png`: x-axis d, y-axis
   (LinUCB regret − online softmax regret). Negative = LinUCB wins. With
   shaded std band. This is the fair online-method comparison; the expected
   pattern is near-zero gaps with sign changes across d.

5. `linucb_minus_offline_softmax_vs_d.png`: x-axis d, y-axis
   (LinUCB regret − offline supervised softmax regret). Negative = LinUCB
   wins. With shaded std band. This isolates the claim that offline best-arm
   classification does not solve routing in this setting.

6. `per_seed_regret_vs_d.png` (supplementary): same as #1 but with one line
   per seed per policy (no aggregation), to show variance directly.

## What `report_packet.py` should do

```
def main(out_dir: Path, settings: ReportPacketSettings) -> None:
    # 1. Setup
    out_dir = out_dir / timestamp()
    write_manifest(out_dir, git_hash(), settings, datetime.now())

    # 2. Block A: motivation
    motivation_results = run_motivation_block(settings.motivation, out_dir)
    save_motivation_plots(motivation_results, out_dir / "plots/motivation")

    # 3. Block B: main d-sweep
    main_results = run_main_block(settings.main, out_dir)
    save_main_plots(main_results, out_dir / "plots/main")

    # 4. Tidy outputs
    write_results_csv(motivation_results + main_results, out_dir)
    write_artifacts_json(motivation_results + main_results, out_dir)

    print(f"Report packet written to {out_dir}")
```

It should reuse:
- `train_joint_moe` for joint training (return the full `JointTrainingStats`,
  don't discard).
- `train_experts` for independent training in Block A only.
- `grid_runner.run_bandit` (or its analog) for executing policies on a fixed
  reward matrix.
- The existing offline ridge ε computation.

It should not:
- Reimplement bandit policies inline.
- Run the full (contam × sep) grid that exists in `grid_runner.py` — the
  motivation block is one (contam, sep) cell, not a grid.

## Settings dataclass

```python
@dataclass
class ReportPacketMotivationSettings:
    d: int = 4
    K: int = 4
    cluster_sep: float = 1.5
    contamination: float = 0.05
    n_train: int = 8000
    T: int = 10000
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    epsilon_greedy_c: float = 50.0
    linucb_alpha: float = 1.0  # for motivation only; main block sweeps

@dataclass
class ReportPacketMainSettings:
    K: int = 4
    cluster_sep: float = 1.5
    n_train: int = 8000
    T: int = 10000
    d_values: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32, 64])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    alpha_values: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])
    epsilon_greedy_c: float = 50.0
    online_softmax_lr: float = 1e-2
    online_softmax_temperature: float = 1.0
    online_softmax_baseline_momentum: float = 0.95
    select_alpha_by: Literal["mean_regret", "median_regret"] = "mean_regret"

@dataclass
class ReportPacketSettings:
    motivation: ReportPacketMotivationSettings = field(default_factory=ReportPacketMotivationSettings)
    main: ReportPacketMainSettings = field(default_factory=ReportPacketMainSettings)
```

## Pre-flight cleanup before running the packet

These are blockers; fix before generating any report artifacts.

1. **Resolve cluster mean generation.** The report outline says
   `ortho_group`; the code currently uses random unit vectors. Pick one and
   commit. Recommendation: use `scipy.stats.ortho_group.rvs(d)[:K]` for
   `d >= K`, fall back to random unit vectors for `d < K`. Document in the
   report. Update `_build_cluster_means` accordingly.

2. **Update `tests/test_data.py`.** The basis-vector assumption is dead.
   Tests should now check: (a) cluster means have norm `cluster_sep *
   cluster_std`, (b) at `d >= K`, pairwise distances are exactly
   `sqrt(2) * cluster_sep * cluster_std` (the orthogonal property), (c) at
   `d < K`, fall back behavior is exercised but with looser tolerance.

3. **Fix `tests/test_joint_d_sweep.py`.** Current expectation of 4 policies
   is stale. Update to expect uniform, eps-greedy, oracle, softmax, plus
   one LinUCB row per alpha in the sweep.

4. **Investigate `tests/test_joint_moe_phase1.py` val acc threshold.** It
   missed by 0.005 (0.645 vs 0.65). Either (a) the threshold was wrong,
   (b) training is slightly under-converged at the test config, or (c) the
   `ortho_group` switch will fix it. Run once with the new geometry and see
   if it passes; if not, lower the threshold to 0.62 with a comment.

5. **Fix offline softmax baseline leakage.** The current runners use Setup 1:
   `softmax_router` is trained on `X_bandit` with labels from the same
   evaluation reward matrix `R`, then evaluated on that same stream. That is
   information leakage. For the packet, use Setup 2: train the router on
   `X_train` with labels from the frozen experts' rewards on `X_train`, then
   evaluate on `X_bandit`. This keeps offline softmax as a full-information
   learned-router reference without letting it memorize the evaluation
   contexts.

## What to NOT include in the packet

- The original (contam × sep) grid results. Stale, joint-regime is inert to
  contam, and the report doesn't use them.
- Independent-regime d-sweep. Out of scope; the report's d-sweep claim is
  about joint training as the realistic regime.
- Any plot that puts joint regret on a contamination axis. Mechanically
  meaningless and visually misleading.
- Per-policy hyperparameter grid searches beyond the LinUCB alpha sweep.
  Epsilon-greedy uses the fixed decaying schedule parameter `c=50.0`, and the
  online/offline softmax settings are fixed at sensible defaults; document
  these choices rather than tuning.

## Reproducibility checklist

- [ ] All random seeds set explicitly in settings.
- [ ] `manifest.json` includes git commit, full settings dict, timestamp,
      Python version, and the exact command used.
- [ ] Each plot script reads from `results_rows.csv` and `artifacts.json`,
      not from in-memory state. (So plots can be regenerated post-hoc.)
- [ ] One smoke test that runs the packet at reduced settings (d_values=[2,
      4], seeds=[0], T=500) in <2 minutes, to verify the pipeline before
      the full run.
- [ ] Full run completes in <30 minutes on CPU (estimate, validate before
      committing the smoke test).

## Plot prompt template for Cursor

When asking Cursor to generate each plot, prompt it explicitly:

> "Plot [name] using `results_rows.csv` and (if needed) `artifacts.json`.
> Joint regime should be on a [d / sep] axis, never on contamination.
> Independent regime appears only in the motivation block. Use [matplotlib
> / seaborn]. Save to `plots/[block]/[name].png` at 150 dpi. Include
> readable axis labels, legend, and title. For aggregate plots, show
> mean across seeds with std as a shaded band."

Don't let it auto-pick "compare regimes on a heatmap" — that's where the
contamination-invariance trap lives.