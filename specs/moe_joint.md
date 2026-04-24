# Cursor Spec: Joint-Training Ablation for MoE Bandit Routing

## Goal

Add a second expert-training regime to the existing pipeline: instead of training each
expert independently on its own cluster, train all experts jointly with a softmax router
(standard MoE training). Freeze the experts after joint training.

This work is intentionally split into two phases:

- **Phase 1: validate joint training in isolation.** Before touching any bandit-routing
  code, prove that joint training converges, avoids expert collapse, and produces
  reasonable frozen experts on a single reference configuration.
- **Phase 2: compare downstream bandit behavior.** Only after Phase 1 passes, run the
  existing bandit evaluation unchanged and compare LinUCB regret between the two regimes
  on the existing (contamination, cluster_sep) grid.

The hypothesis being tested: joint training produces smoother per-expert reward surfaces
with smaller linear approximation error ε, so LinUCB should perform relatively better
(closer to softmax router) in the misspecified regime (low contamination, low cluster_sep).

## Non-goals

- Do not change the bandit environment, reward definition, oracle computation, or any policy
  code (uniform, epsilon-greedy, LinUCB, softmax router baseline, oracle). These are held
  constant across both training regimes.
- Do not change the synthetic data generator. Both regimes train on identical data.
- Do not introduce top-k or sparse routing. Use soft (dense) routing during training, i.e.
  weighted sum of all K expert outputs. This is the right choice for small K=4 and keeps
  gradient flow clean.
- Do not add expert capacity, token dropping, or expert parallelism. Irrelevant at this scale.

## What to build

### Phase 1. Joint-training validation only

The objective of this phase is to answer one question cleanly: **did joint MoE training
itself go okay?** Do not work on bandit-routing comparisons until this phase passes.

### 1. New module: `train_joint_moe.py`

A drop-in replacement for the current independent-expert-training function. Same inputs
(training data, K, d, expert architecture config), same output shape (K frozen expert
networks that take x and return class logits).

For Phase 1, however, the implementation should expose **training diagnostics in addition
to the frozen experts**. Returning only the experts is not enough for a validation-first
workflow because we need programmatic access to convergence and utilization metrics before
moving downstream.

**Architecture.**
- K expert MLPs, each with architecture identical to the current per-cluster experts:
  `d -> 64 -> 64 -> K_classes` with ReLU. Do not change this — capacity must match.
- One router: a single linear layer `d -> K` followed by softmax. Keep the router small
  and simple. No hidden layer. Justification: the true optimal router is (approximately)
  linear in x given how clusters are placed, so linear router capacity is sufficient and
  avoids confounding "smoother experts" with "fancier router."
- Optionally: support a small MLP router (`d -> 16 -> K`) behind a config flag, but
  default to linear. We may want this as a sanity check later.

**Forward pass.**
```
logits_per_expert = [expert_i(x) for i in range(K)]           # list of (B, K_classes)
stacked = stack(logits_per_expert, dim=1)                     # (B, K, K_classes)
gate_logits = router(x)                                       # (B, K)
gate_weights = softmax(gate_logits, dim=-1)                   # (B, K)
mixed_logits = einsum("bk,bkc->bc", gate_weights, stacked)    # (B, K_classes)
```
Compute cross-entropy on `mixed_logits` vs. true labels. This is soft routing with a
weighted sum of expert logits. Mixing logits (not probabilities) is the standard choice
and what all the MoE literature does.

**Loss.**
```
L_total = L_CE + alpha * L_load
```

Load balancing loss (Switch Transformer formulation, Fedus et al. 2022):
```
f_i = mean over batch of gate_weights[:, i]     # fraction of routing weight to expert i
P_i = f_i                                       # for soft routing, same quantity
L_load = K * sum_i (f_i * P_i)                  # minimized when all f_i = 1/K
```
Note: in the Switch Transformer paper f and P are distinct (f is the hard top-1 assignment
fraction, P is the mean soft probability). Because we use soft routing and don't take argmax,
both reduce to the same mean gate weight, so L_load simplifies to `K * sum_i f_i^2` which
is minimized at f_i = 1/K. That's fine — it's the correct soft-routing analog.

Use `alpha = 0.01` as the default (standard value from Switch Transformer paper, also used
in Joint MoE Scaling Laws 2025). Expose it as a config knob. Do not tune it yet.

**Training loop.**
- Optimizer: Adam, lr=1e-3 (same as current independent-expert training).
- Epochs: 30 (same as current).
- Batch size: 64 (same as current).
- Train on the FULL pooled dataset from all clusters, not per-cluster subsets. This is the
  key difference: the router sees everything and learns to dispatch.
- Use the same training data the independent experts currently see, pooled. So
  `n_train_total = n_train_per_cluster * K` samples. Contamination still applies to how
  the base data is generated (it determines the marginal distribution of each cluster's
  data); joint training then just sees the union.

**Monitoring (log these each epoch, print at end, and return in a diagnostics object).**
- Total loss, CE loss, load loss separately.
- Per-expert mean gate weight over the epoch (should converge to something non-degenerate,
  ideally ~1/K with some specialization variance but not collapse).
- Training accuracy.
- If any expert's mean gate weight drops below 0.05 by epoch 5, that's a collapse warning
  — log it prominently but do not auto-restart. We want to see if it happens.
- Return these histories in a small structured object, e.g. `JointTrainingStats`, so a
  notebook/test can assert on them directly instead of relying on console output.

**Output.**
For eventual pipeline compatibility, the primary artifact is still a list of K frozen
expert `nn.Module`s, ready to plug into the existing reward matrix construction.

For Phase 1 validation, also return or expose:
- epoch-wise loss/accuracy histories,
- epoch-wise per-expert gate means,
- final pooled accuracy on the training split,
- a collapse warning flag.

Do NOT thread the training-time router into the bandit pipeline. We still discard it once
training/diagnostics are complete. Post-hoc, the bandit replaces the router's job, and
the softmax router baseline in the existing pipeline gets retrained on the co-trained
experts later in Phase 2 (see section 3).

### 2. Config flag wiring

In the top-level experiment config (wherever `build_experts(...)` is currently called),
add a knob:
```
expert_training_regime: Literal["independent", "joint"] = "independent"
```

Route it to either the existing `train_independent_experts(...)` or the new
`train_joint_moe(...)`. Everything downstream (reward matrix construction, bandit
running, regret computation) stays identical.

**Phase boundary:** do not wire this into the full grid runner until Phase 1 validation
has succeeded on the reference configuration below.

### 3. Softmax router baseline — retrain, don't reuse

**Important detail.** The existing softmax-router baseline is trained on the independent
experts. When running the joint-training regime, retrain the baseline softmax router on
the co-trained frozen experts using the same training procedure as before (supervised
classification of which arm has max reward per context). Otherwise the softmax baseline
is comparing against a different set of experts than LinUCB and the comparison is unfair.

Concretely: the "softmax router" baseline in the current pipeline should be trained
against whichever expert set is active. This probably means threading the experts through
to the baseline training step, which it likely already does. Just make sure it uses the
current regime's experts.

This is **Phase 2 only**. Do not work on this until the Phase 1 validation checklist has
passed.

### 4. Experiment runner

Add a single top-level script / notebook cell that:
1. For each (contamination, cluster_sep) cell in the existing grid:
   - Generates data once (same seed as current runs).
   - Trains experts in BOTH regimes: independent (baseline) and joint (new).
   - Builds a reward matrix for each regime.
   - Runs all policies (uniform, eps-greedy, LinUCB, softmax, oracle) on BOTH reward
     matrices.
2. Logs results with an extra `expert_regime` field.
3. Saves to the same JSONL format as the existing grid output, plus the new field.

Keep seeds identical across regimes so differences are attributable to training procedure
only, not data variance.

This is **Phase 2 only**.

### 5. Diagnostic: measure ε directly

Add a diagnostic that, for each (config, expert_regime), fits the best offline ridge
regressor per arm:
```
theta_i = argmin_theta sum_t (r_i(x_t) - <theta, x_t>)^2 + lambda * ||theta||^2
epsilon_i = max_t |r_i(x_t) - <theta_i, x_t>|
```
Report mean and max ε across arms for each cell. This is the direct measurement of
linear approximation error the hypothesis predicts should drop under joint training.

Save these as a separate JSONL (`approx_error_by_regime.jsonl`) indexed by
(contamination, cluster_sep, expert_regime, arm, seed).

For workflow purposes, implement the single-config version of this diagnostic first and
use it as a gate before running the full grid.

## Phase 1 validation target

Before any routing/bandit work, validate joint training on a single reference config:

- `contamination = 0.05`
- `cluster_sep = 1.5`
- `seed = 0` (or the repo's equivalent explicit seed bundle)

Use a train/holdout split so "joint training went okay" is not judged only from the
training set. The current codebase often inspects training behavior directly, but this
phase should include a small pooled validation/test check.

Phase 1 is considered successful only if all of the following hold:

1. Cross-entropy decreases materially over training.
2. No expert collapses: by the end of training, all 4 experts have clearly non-degenerate
   gate weight (target: each > 0.10).
3. Pooled holdout classification accuracy is reasonable (target: > 70%).
4. The frozen experts produce a sane reward matrix with visible specialization structure.
5. The single-config linear-approximation diagnostic is directionally promising
   (`epsilon_joint < epsilon_indep`) or, if not, the failure mode is understood before
   moving on.

If any of these checks fail, stop and debug joint training before touching the bandit
runner, softmax baseline retraining, or full-grid execution.

## Expected results (what success looks like)

If the hypothesis is correct:

1. **ε drops substantially under joint training**, especially in the low-contamination,
   low-cluster_sep cells where the independent-training ε was largest. Predicted effect
   size: the spike-plus-plateau gap of ~2.5 nats should compress to something much smaller.

2. **LinUCB regret drops under joint training** in those same cells, and the gap between
   LinUCB and the softmax router narrows or closes.

3. **Reward surface plots** along a line from mu_j to mu_i for each (i, j) pair show
   smooth monotonic transitions under joint training, vs. cliff-plateau under independent.

If the hypothesis is wrong or partial:
- If ε doesn't drop, either the joint training didn't smooth things (check for expert
  collapse via per-expert gate weights) or the reward surface is intrinsically nonlinear
  in raw x even after smoothing. Both are informative negative results.
- If ε drops but LinUCB doesn't improve proportionally, that's interesting and suggests
  something else is bottlenecking LinUCB (exploration, feature normalization).

## Implementation order

### Phase 1

1. Build `train_joint_moe.py` with load-balancing loss, expert utilization logging, and
   a returned diagnostics object/history.
2. Add one focused notebook cell or test that runs the reference config
   `(contamination=0.05, cluster_sep=1.5, seed=0)`.
3. Verify:
   - Training converges (CE loss drops).
   - All 4 experts get non-degenerate gate weight (>10% each by end of training).
   - Classification accuracy on pooled holdout data is reasonable (>70%).
   - Reward matrix / expert specialization looks sane.
4. Run the ε diagnostic on the single config for both regimes. Confirm `epsilon_joint <
   epsilon_indep`, or stop and debug if the result is ambiguous or negative.

### Phase 2

5. Wire the `expert_training_regime` flag through the pipeline.
6. Verify the softmax baseline retrains on the active expert set.
7. Run the full grid with matched seeds across regimes.

## Things to watch out for

- **Expert collapse.** If one expert wins everything, load loss should push back, but
  check the per-expert gate weights after training. With alpha=0.01 and soft routing,
  collapse is unlikely at this scale but possible. If it happens, try alpha=0.1.
- **Overspecialization via load loss.** Conversely, if alpha is too high, the router is
  forced to be uniform and experts don't specialize. If the final per-expert weights are
  all very close to 1/K AND classification accuracy is low, alpha is too aggressive.
  Drop it by 10x.
- **The router used during training is NOT the softmax baseline used during bandit eval.**
  The training-time router is discarded (it only exists to shape the experts during training).
  The softmax baseline is retrained separately as a supervised best-arm classifier on the
  frozen co-trained experts, same as the existing pipeline does for independent experts.
- **Validation blindness.** Do not declare Phase 1 successful from train-set metrics alone.
  Include a pooled holdout check so we know joint training generalizes at least reasonably
  before evaluating routing behavior.
- **Mixed logits vs mixed probabilities.** Mix logits, not probabilities. This is the
  standard choice and matches the reference implementations in the MoE literature.
- **Seed discipline.** Use the same data-generation seed across regimes. Use a separate
  seed for expert training so the two regimes don't share initialization. Record both
  in the output.

## Deliverables

### Phase 1 deliverables

1. `train_joint_moe.py` with the joint training function, docstring, and diagnostics
   return structure.
2. One focused notebook cell or test that validates the single reference config before
   any downstream bandit work.
3. A single-config ε comparison for independent vs joint training, used as a go/no-go
   gate for Phase 2.

### Phase 2 deliverables

4. Updated experiment runner with `expert_training_regime` flag.
5. Updated softmax baseline that uses whichever experts are active.
6. `approx_error_by_regime.jsonl` containing ε diagnostics.
7. Updated grid results JSONL with `expert_regime` field.
8. A single plot: 2x1 panel of the existing "softmax − LinUCB" heatmap, one for each
   regime. If the hypothesis holds, the joint-training panel should be much less blue
   in the bottom-left corner.