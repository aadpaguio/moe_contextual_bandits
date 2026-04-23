# DS 592 Project — Implementation Plan

**Project:** Contextual Bandit Routing for Mixture-of-Experts
**Role:** Implementation & experiments (Arnald). Theory handled separately (Aidan).
**Deadline:** Report April 30, presentation May 1.

---

## Empirical narrative (locked after Week 1 debugging)

The original proposal framed the project as "implement LinUCB for MoE routing and compare to baselines." Week 1 empirical work surfaced a stronger story:

**Raw-feature disjoint LinUCB consistently fails on this environment**, not due to any implementation bug but because the true reward surface (spike-at-cluster-center, plateau elsewhere) is poorly approximated by linear-in-raw-$x$ models. This was confirmed by extensive diagnostics:
- Runner, reward-matrix alignment, and ε-greedy all verified correct.
- $\alpha$ sweep, forced exploration, reward rescaling, and context standardization all fail to recover correct routing.
- Confusion matrix reveals a stable *permutation equilibrium*: each arm confidently routes a wrong cluster, with predicted means like `[0.496, 0.720, 0.597, 0.581]` for cluster-0 contexts (true values `[0.994, 0.638, 0.638, 0.638]`).

This is the misspecification regime of Lattimore, Szepesvári & Weisz (2020) Theorem E.1 — observed empirically on a small, controlled problem.

**The revised empirical story has four pieces:**
1. **Headline figure.** Regret curves at the locked config: Uniform, ε-greedy, raw-LinUCB, RBF-LinUCB. Raw-LinUCB underperforms baselines; RBF-LinUCB dominates them.
2. **Contamination sweep.** Regret-at-$T$ vs contamination ∈ {0.05, 0.1, 0.2, 0.3, 0.5}. Raw-LinUCB improves as contamination rises; RBF-LinUCB is flat.
3. **Cluster-separation sweep (secondary).** Same plot along the `cluster_sep` axis.
4. **Approximation-error measurement.** For each environment config, compute the best-offline-linear-fit RMSE per arm. Plot regret vs this error. Shows the empirical relationship between misspecification and regret has the shape Theorem E.1 predicts.

**What this is and is not:**
- *Is:* raw-feature disjoint LinUCB is misspecified for this environment; the observed regret is consistent with misspecification dominating performance.
- *Is not:* LinUCB is fundamentally broken; raw features are always wrong; contamination literally *is* the theorem's $\varepsilon$.

Contamination and cluster_sep are *proxies* for misspecification severity, validated against directly-measured approximation error (item 4 above).

---

## 0. Design decisions (locked)

- **Context** = raw input embedding $x_t \in \mathbb{R}^d$.
- **Reward model.** Per-arm linear in the chosen feature map: $\mathbb{E}[r(\phi(x), i)] \approx \langle \theta_i, \phi(x) \rangle$. Two feature maps studied:
  - **Raw:** $\phi_\text{raw}(x) = [x; 1]$ (bias-augmented identity). Expected to be *misspecified* for the spike-plus-plateau reward surface. This is the misspecification baseline.
  - **RBF:** $\phi_\text{rbf}(x) = [\exp(-\gamma \|x - \mu_1\|^2), \ldots, \exp(-\gamma \|x - \mu_K\|^2); 1] \in \mathbb{R}^{K+1}$ with $\gamma = 0.5$ (≈ $1/(2\sigma^2)$) and cluster means $\mu_i$ reused from `_build_cluster_means`. Expected to be *well-specified* because the reward is essentially linear in cluster membership.
- **LinUCB variant** = disjoint (per-arm ridge regressions). Equivalent to course LinUCB with stacked feature map; implementing as $K$ independent regressions is the efficient way to exploit the block-diagonal Gram matrix.
- **Bias augmentation** (raw only). Cluster means are at $1.5\sigma \cdot e_i$ — contexts are not zero-centered, so omitting the intercept would bake cluster-conditional means into misspecification. RBF features already include a bias term; no separate augmentation needed. Tell Aidan the effective dimension is $d+1$ for raw and $K+1$ for RBF.
- **Linear algebra: `solve` only, never `inv`.** Both $\hat\theta_i = V_i^{-1} b_i$ and the width $\phi^\top V_i^{-1} \phi$ are computed via `np.linalg.solve`, which broadcasts over leading dimensions. All $K$ systems solve in one batched call. No Sherman-Morrison, no Cholesky caching.
- **Forced initial exploration.** Round-robin for `forced_explore_per_arm * K` steps (default 20 per arm, so 80 total at $K=4$). Standard LinUCB preprocessing. Too little warmup (<= 2 per arm) can cause lock-in artifacts; 20 per arm is safe and cheap at $T=5000$.
- **Exploration coefficient $\alpha$.** Tuned constant. Default $\alpha = 1.0$. Sweep $\{0.1, 0.25, 0.5, 1.0, 2.0\}$ on 10 seeds at short horizon; pick best for full-horizon main figures. The theoretical $\alpha_t$ from Abbasi-Yadkori et al. 2011 is available as a supplementary "LinUCB-theory" variant (noting sub-Gaussian assumption is approximate). Main results use tuned constant.
- **Regularization $\lambda = 1.0$.** Default; expose but don't sweep.
- **Tie-breaking in argmax.** Random choice among arms within $10^{-12}$ of the UCB max.
- **Reward rescaling.** Rescale clipped log-probabilities from $[\log\epsilon, 0]$ to $[0, 1]$ before feeding to LinUCB via $\tilde r = (r - \log\epsilon) / (-\log\epsilon)$. Makes $\alpha$ calibration match the regime Li et al. (2010) used. The underlying learning problem is unchanged; only the numerical scale of $\alpha$ vs the reward values is.
- **Context standardization.** Z-score each coordinate of the bandit stream $X$ (fit $\mu, \sigma$ on $X$, apply `(X - μ) / σ`) before running bandits. Keeps `cluster_sep` geometry intact while putting LinUCB's width calibration in a typical regime.
- **Oracle definition.** Primary oracle is `R.argmax(axis=1)` — best realized expert for each sample. In the writeup, also report the in-expectation oracle $a^\star(x_t) = \arg\max_i \mathbb{E}[r \mid x_t]$ (approximated by averaging labels at fixed $x_t$) and note the cumulative-regret difference in an appendix.
- **Reward clipping** = clip expert predicted probabilities to $[\epsilon, 1-\epsilon]$ with $\epsilon = 10^{-3}$ before cross-entropy, so reward is bounded in $[\log \epsilon, 0]$ and trivially sub-Gaussian. Document in writeup.
- **Default config (locked after expert-training sanity checks):**
  - `K = 4`, `d = 4` (effective dim $d+1 = 5$ for raw LinUCB, $K+1 = 5$ for RBF)
  - `cluster_sep = 1.5` (sigma units — cluster means at $1.5 \cdot \sigma \cdot e_i$, pairwise mean distance $\sqrt{2} \cdot 1.5 \approx 2.12\sigma$)
  - `cluster_std = 1.0`
  - `contamination = 0.05` (default for main figure; sweep value in Week 2)
  - `n_train_per_cluster ≈ 2000` (verified stable at 10000)
  - Expert MLP: `d -> 64 -> 64 -> K`, Adam lr 1e-3, 30 epochs, batch 64
  - RBF bandwidth: $\gamma = 0.5$
  - Observed oracle gap (unscaled rewards): mean 1.85, median 1.71, p10/p90 = 0.56 / 3.32.
- **Misspecification knobs (primary: contamination; secondary: cluster_sep).**
  - `contamination` — how much each expert knows about other clusters. Primary sweep axis.
  - `cluster_sep` — geometric overlap of cluster supports. Secondary axis. Sweep if compute permits; otherwise leave at 1.5.
  - Both are *proxies* for misspecification severity. Validate against directly-measured approximation error (see Experiments §5).

---

## 1. Synthetic data generator — DETAILED

This is the foundation. Everything else plugs into it. Keep it deterministic under a seed.

### Function signature

```python
def generate_synthetic_data(
    n_samples: int,
    K: int,                    # number of clusters / experts
    d: int,                    # ambient dimension
    cluster_sep: float,        # controls overlap (e.g., distance between cluster means in sigma units)
    cluster_std: float = 1.0,  # within-cluster std
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X: (n_samples, d) — context vectors
        y: (n_samples,)   — class labels in [0, K)
        cluster_id: (n_samples,) — which cluster each point was sampled from
                                   (= y in the simplest setup, but keep separate in case
                                   we decouple labels from clusters later)
    """
```

### Implementation notes

- **Cluster means.** Place $K$ cluster centers on an orthogonal (or near-orthogonal) configuration in $\mathbb{R}^d$ so that `cluster_sep` has a clean geometric interpretation. Simplest: centers at `cluster_sep * e_i` for the first $K$ standard basis vectors (requires $d \ge K$). Alternatively, sample random unit vectors and rescale — less clean but works for $d < K$.
- **Sampling.** For each point, draw cluster index uniformly in $[K]$, then $x \sim \mathcal{N}(\mu_{\text{cluster}}, \sigma^2 I)$.
- **Labels.** In the simplest setup, `y == cluster_id` and each expert is a classifier for its own cluster's points. Later you may want to test a regime where labels depend on cluster in a non-trivial way (e.g., $K$ classes but each expert sees all classes within its cluster region) — keeping `cluster_id` separate from `y` in the API makes this easy.
- **Determinism.** Use a local `np.random.default_rng(seed)` instance, not global state. Every experiment config gets a seed; log it in output filenames.
- **Sanity check.** After generating, plot the first 2 dims of `X` colored by `cluster_id` for $K=4, d=2$. You should see four visibly separated Gaussian blobs. If clusters overlap too much the oracle gap shrinks and all algorithms look the same — keep `cluster_sep` large enough (e.g., $\ge 3$) in the main experiments.

### Misspecification knob

The proposal's $\varepsilon$ in Theorem E.1 of Lattimore et al. measures how badly $\mathbb{E}[r(x, i)]$ fails to be linear in $x$. You want to *control* this empirically. Options, in order of increasing complexity:

1. **Cluster overlap.** Increasing `cluster_std` (or decreasing `cluster_sep`) makes the boundary between competence regions fuzzier, and the best-linear-fit error grows. Cheap knob, already in the API.
2. **Nonlinear contamination.** After training experts, measure the true reward surface $\mathbb{E}[r(x, i)]$ empirically (by averaging over many points near $x$) and compare to the best linear fit $\hat\theta_i^\top x$. The residual norm *is* your empirical $\varepsilon$. Report it alongside regret curves.
3. **Explicit nonlinearity.** Add a known nonlinear term to the reward, e.g., $r(x, i) = \langle \theta_i, x\rangle + \alpha \cdot \|x - \mu_i\|^2 + \text{noise}$, and sweep $\alpha$. Cleanest for the theoretical validation but departs from the "real MoE" story.

Recommend starting with (1), then (2) for the actual $\varepsilon$ estimate in the report.

### Deliverables

- `data.py` with `generate_synthetic_data(...)`.
- A notebook cell that plots cluster structure for a small case.
- A small unit test: fixed seed, fixed config, check means/stds/label distribution match expected values.

---

## 2. Expert networks — DETAILED

### Design decisions (lock these before coding)

- **One expert per cluster.** Expert $i$ is trained *only* on points where `cluster_id == i`. No overlap, no weighted mixtures. Each expert becomes a $K$-way classifier that's competent on its own cluster and essentially uninformed on others. This gives the cleanest oracle story (the best expert for context $x$ is the one whose cluster $x$ came from) and makes the report narrative simple.
- **Architecture.** `d -> 64 -> 64 -> K` MLP, ReLU activations, linear output head. Apply softmax only when computing probabilities for the reward; keep logits as the model output internally so cross-entropy is numerically stable.
- **Training data scale.** Each expert needs its own training set, separate from the bandit stream. Target ~2000 points per cluster (so ~$2000 K$ total training points). Use a distinct seed from the bandit stream seed so the bandit data is genuinely held out.
- **Training config.** Adam, lr 1e-3, ~30 epochs, batch size 64. No scheduler, no early stopping — this is auxiliary infrastructure, not the research contribution. If an expert doesn't converge to near-100% accuracy on its own cluster at this config, bump epochs before anything fancier.
- **Freezing.** After training, set `requires_grad=False` on all params and call `.eval()`. Inference path wraps everything in `torch.no_grad()`.
- **Reward clipping.** Clip softmax probabilities to $[\epsilon, 1-\epsilon]$ with $\epsilon = 10^{-3}$ before cross-entropy (locked in §0). So reward is $r(x, i) = \log \tilde p_i(y \mid x)$ where $\tilde p$ is the clipped probability, bounded in $[\log \epsilon, 0] = [-6.9, 0]$.

### Function signatures

```python
class Expert(nn.Module):
    """Small MLP: d -> 64 -> 64 -> K, linear output (logits)."""
    def __init__(self, d: int, K: int, hidden: int = 64): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # returns logits (batch, K)
        ...

def train_experts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cluster_id_train: np.ndarray,
    K: int,
    d: int,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
) -> list[Expert]:
    """Train K experts, one per cluster. Returns list of frozen experts in .eval() mode."""
    ...

def expert_reward_matrix(
    experts: list[Expert],
    X: np.ndarray,
    y: np.ndarray,
    clip_eps: float = 1e-3,
) -> np.ndarray:
    """
    Returns:
        R: (n_samples, K) array where R[t, i] = log(clipped p_i(y_t | x_t)).
           This is the full reward matrix. Bandit feedback reads R[t, a_t];
           oracle reads argmax over axis 1.
    """
    ...
```

### Implementation notes

- **Why return the full reward matrix upfront.** We precompute $R \in \mathbb{R}^{T \times K}$ once and reuse it across every policy and every seed in a sweep. The bandit feedback loop just does `R[t, a_t]` — no further expert inference. The oracle is `R.max(axis=1)`. This matters for speed: expert inference is the most expensive per-step operation by far, and caching the matrix makes the bandit simulations effectively free. Costs $O(TK)$ memory which is nothing.
- **GPU vs CPU.** Training on CPU is fine at this scale (2 hidden layers, 64 units, ~8K training points). Keep everything CPU to avoid a mixed-device headache; the bandit loop is pure NumPy anyway.
- **Determinism.** Set `torch.manual_seed(seed)` and `np.random.seed(seed)` before training. Expert init and minibatch order both depend on this.

### Sanity checks (run these in `01_expert_training.ipynb`)

- **Per-expert accuracy on own cluster.** For expert $i$, accuracy on `X[cluster_id == i]` should be > 95%. If not, bump epochs or check for a bug.
- **Per-expert accuracy on other clusters.** Expert $i$ on `X[cluster_id == j != i]` should be near chance (1/K). If it's high, your clusters aren't well-separated or training leaked across clusters somehow.
- **Reward matrix inspection.** Plot a heatmap of mean $R[t, i]$ grouped by `cluster_id[t]`: rows are true cluster, columns are expert. Should be bright on the diagonal (expert $i$ gives high reward to its own cluster's points) and dim off-diagonal. This is the clearest visualization that the oracle has a meaningful gap.
- **Oracle gap distribution.** Histogram of `R.max(axis=1) - R.mean(axis=1)` (average suboptimality per step). If this is near zero for a lot of points, the bandit problem is too easy (all experts roughly equal) and all policies will look the same.

### Deliverables

- `experts.py` with `Expert`, `train_experts`, `expert_reward_matrix`.
- `01_expert_training.ipynb` that trains experts on a default config and runs all four sanity checks above.
- Unit test: train experts on a tiny config ($K=2, d=2$, 200 training points, 5 epochs), assert own-cluster accuracy > 80% on the training set. Fast smoke test; not a correctness proof.

---

## 3. Bandit feedback loop — DETAILED

This is the driver that every policy runs through. Keep it policy-agnostic: the runner shouldn't know whether it's executing uniform-random, ε-greedy, LinUCB, or anything else. That's how you get one tested piece of infrastructure that everything downstream plugs into.

### Design decisions (lock these before coding)

- **Precomputed reward matrix.** The runner does NOT call experts during simulation. It takes the full $R \in \mathbb{R}^{T \times K}$ matrix as input (produced once by `expert_reward_matrix`) and indexes into it: `r_t = R[t, a_t]`. Oracle comes for free as `R.max(axis=1)` / `R.argmax(axis=1)`. This is the single biggest speedup in the project — policy sweeps become pure NumPy.
- **Policy interface.** Every policy is a class with exactly two methods: `select(x_t) -> int` and `update(x_t, a_t, r_t) -> None`. No other methods, no hidden state the runner needs to know about. This is a `typing.Protocol`, not an abstract base class — keeps it lightweight and doesn't force inheritance.
- **Regret accounting.** Report three quantities per step: the reward the policy actually got, the reward the oracle would have gotten, and instantaneous regret `oracle_reward - policy_reward`. Cumulative regret is computed post-hoc by the analysis code, not the runner, so the runner output is a clean raw log.
- **Randomness.** Runner takes a seed for the *policy's* internal randomness (TS sampling, ε-greedy coin flips). The bandit stream itself (`X`, `R`) is already deterministic from the data/expert seeds. Keep these seeds separately sourced so you can average over policy randomness with a fixed stream, or over streams with a fixed policy, as needed.
- **Single run, many seeds.** `run_bandit` returns results from *one* run. Averaging across seeds happens in a wrapper (`run_seeds`), not inside the runner. Keeps the inner function testable and the outer one trivial.

### Function signatures

```python
from typing import Protocol

class Policy(Protocol):
    """Policies implement exactly these two methods."""
    def select(self, x_t: np.ndarray) -> int: ...
    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None: ...


@dataclass
class RunResult:
    chosen_arm: np.ndarray    # (T,) int64
    oracle_arm: np.ndarray    # (T,) int64
    reward: np.ndarray        # (T,) float64 — reward actually received
    oracle_reward: np.ndarray # (T,) float64 — reward the oracle would have gotten
    regret: np.ndarray        # (T,) float64 — oracle_reward - reward, >= 0

    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.regret)


def run_bandit(
    policy: Policy,
    X: np.ndarray,   # (T, d) context stream
    R: np.ndarray,   # (T, K) reward matrix (precomputed from experts)
    seed: int = 0,
) -> RunResult:
    """
    Run `policy` on the (X, R) stream for T = len(X) steps. Policy-agnostic.
    """
    ...


def run_seeds(
    policy_factory: Callable[[int], Policy],
    X: np.ndarray,
    R: np.ndarray,
    n_seeds: int,
    base_seed: int = 0,
) -> list[RunResult]:
    """
    Run the same policy factory n_seeds times with different seeds.
    policy_factory(seed) returns a fresh Policy instance — necessary because
    policies carry state across .update() calls and must be reset per run.
    """
    ...
```

### Implementation notes

- **`run_bandit` inner loop.** Straightforward:
  ```python
  T = len(X)
  chosen = np.zeros(T, dtype=np.int64)
  rewards = np.zeros(T, dtype=np.float64)
  for t in range(T):
      x_t = X[t]
      a_t = policy.select(x_t)
      r_t = R[t, a_t]
      policy.update(x_t, a_t, r_t)
      chosen[t] = a_t
      rewards[t] = r_t
  ```
  Oracle quantities are computed outside the loop from `R` directly (vectorized). No per-step oracle calls.
- **Shape contract.** `X.shape == (T, d)`, `R.shape == (T, K)`, `len(X) == len(R)`. Validate at the top of `run_bandit` with a clear error message — this will save debugging time when you eventually swap in a different reward matrix size.
- **No early stopping, no inner-loop logging.** The runner runs all $T$ steps. If you want checkpoints, you compute them post-hoc from `RunResult`.
- **Why `policy_factory` not `policy` in `run_seeds`.** If `run_seeds` took a pre-constructed `policy` object and called it repeatedly, the policy's internal state (posterior estimates, Gram matrix, etc.) would carry over across runs and pollute the averages. The factory pattern forces a fresh policy per seed. Also lets you thread per-seed configuration into the policy constructor cleanly.
- **Performance.** For $T = 10^4$ and $K = 4$, pure-Python inner loop with NumPy ops inside `select`/`update` runs in seconds. Don't `@njit` yet. If you later hit a wall at $T = 10^5$ or $10^6$, the bottleneck will be inside LinUCB's `select` (per-arm matrix inverse) not the runner, so Numba-ing the runner wouldn't help anyway.
- **Determinism.** The runner itself is deterministic given `X`, `R`, and the policy. Any randomness lives inside the policy (seeded at construction via `policy_factory(seed)`). This is the cleanest separation.

### Uniform-random policy (write this first)

Before LinUCB or anything else, write a trivial uniform-random policy and run it through the loop. This is the sanity check that the runner itself works.

```python
class UniformRandomPolicy:
    def __init__(self, K: int, seed: int = 0) -> None:
        self.K = K
        self.rng = np.random.default_rng(seed)

    def select(self, x_t: np.ndarray) -> int:
        return int(self.rng.integers(0, self.K))

    def update(self, x_t: np.ndarray, a_t: int, r_t: float) -> None:
        pass
```

### Sanity checks (run these in `02_single_run.ipynb`)

- **Uniform-random regret is linear.** Plot cumulative regret for `UniformRandomPolicy`. Should be a nearly-straight line with slope $\approx$ mean oracle gap (1.85 for the locked config). If it's sublinear, something is wrong with the runner (likely: oracle and reward computed from different indices).
- **Uniform-random reward matches expected.** Mean reward per step should equal `R.mean()` (i.e., average over all (t, i) entries), because uniform-random picks every arm with probability 1/K. Useful arithmetic check.
- **Oracle reward matches `R.max(axis=1)`.** Plot `result.oracle_reward.sum()` vs `R.max(axis=1).sum()` — should be identical.
- **Regret non-negative.** `result.regret.min() >= 0` must hold. If ever negative, you've mixed up `oracle_reward` and `reward` somewhere.
- **Cross-seed variance.** Run `UniformRandomPolicy` with 10 different seeds, plot all cumulative regret curves on one axis. Should be visually close — slope is the same (determined by `R`), only the per-step arm choices differ slightly.

### Deliverables

- `runner.py` with `Policy` protocol, `RunResult` dataclass, `run_bandit`, `run_seeds`.
- `policies/random.py` with `UniformRandomPolicy`.
- `02_single_run.ipynb` running all five sanity checks above on the locked default config.
- Unit test: build a toy `R` matrix where arm 0 always has reward 1 and other arms reward 0; run `UniformRandomPolicy` for 1000 steps; assert cumulative regret grows roughly linearly at rate $(K-1)/K$.

---

## 4. Policies (in implementation order) — skeleton

1. **Uniform random** — sanity check for the driver. ✅
2. **ε-greedy** — confirm update loop works. Schedule: $\varepsilon_t = \min(1, c/t)$, $c = 50$. ✅
3. **LinUCB-raw (disjoint, bias-augmented)** — misspecification baseline. Per-arm $V_i, b_i$ at dimension $d+1$. `solve` only, forced warmup $20K$ steps, random tie-breaking. ✅
4. **LinUCB-RBF (disjoint, RBF features)** — well-specified comparison. Same code path as LinUCB-raw but feature map $\phi_\text{rbf}(x) = [\exp(-\gamma\|x - \mu_i\|^2)]_i \cup \{1\}$ at dimension $K+1$. Cluster means $\mu_i$ and bandwidth $\gamma$ passed at construction.
5. **Linear Thompson Sampling (raw features)** — reuses LinUCB-raw infrastructure; sample $\tilde\theta_i \sim \mathcal{N}(\hat\theta_i, \beta V_i^{-1})$ instead of computing UCB. Included for completeness; will also show misspecification failure.
6. **Softmax router baseline** — small router network trained on the same data with supervised loss, run in eval mode. Upper-bound baseline (has access to labels during training).

**Implementation note for LinUCB-RBF.** Cleanest design is to parameterize `LinUCBPolicy` with a feature-map callable `phi: (x: np.ndarray) -> np.ndarray`. Default is bias-augmented identity (raw). Pass `phi=RBFFeatureMap(centers=cluster_means, gamma=0.5)` to get the RBF variant. No separate policy class needed; it's the same algorithm at a different feature dimension.

---

## 5. Experiments — DETAILED

Four experiments, in order. Each produces one figure for the report.

### Experiment 1: Headline regret curves (locked config)

**Configs.** Default (`K=4, d=4, cluster_sep=1.5, contamination=0.05`). $T = 5000$.

**Policies.** Uniform, ε-greedy, LinUCB-raw, LinUCB-RBF. (Linear TS and softmax baseline optional; add if time permits.)

**Method.** 30 seeds per policy, fixed data/expert seeds. Plot mean cumulative regret with shaded ±1 std band.

**Expected outcome.** LinUCB-raw underperforms ε-greedy (this is the misspecification finding). LinUCB-RBF dominates everything else with near-sublinear regret. Headline figure for the report.

### Experiment 2: Contamination sweep (primary $\varepsilon$-proxy axis)

**Configs.** `contamination ∈ {0.05, 0.1, 0.2, 0.3, 0.5}`, all else default. Retrain experts for each value.

**Policies.** Uniform, ε-greedy, LinUCB-raw, LinUCB-RBF. 10 seeds per cell.

**Method.** For each contamination level, compute final cumulative regret (mean over seeds). Plot: x-axis = contamination, y-axis = final regret, one line per policy.

**Expected outcome.**
- LinUCB-raw regret *decreases* as contamination increases (misspecification shrinks, linear fit gets better).
- LinUCB-RBF regret is roughly flat (feature map already well-specified; contamination mostly affects oracle gap, not model fit).
- ε-greedy regret behavior depends on whether best-marginal-arm changes as contamination varies — note this in writeup.

### Experiment 3: Cluster-separation sweep (secondary axis, if time permits)

**Configs.** `cluster_sep ∈ {1.0, 1.5, 2.0, 2.5, 3.0}`, all else default.

**Method.** Same as Experiment 2.

**Expected outcome.** Both LinUCB variants improve with higher `cluster_sep` (more geometric separation), but raw-LinUCB benefits less because its fundamental issue is feature misspecification, not context geometry.

### Experiment 4: Approximation-error measurement (empirical $\varepsilon$)

**Goal.** Validate that the misspecification-proxy knobs (contamination, cluster_sep) actually correspond to measurable approximation error.

**Method.** For each environment config used in Experiments 2 and 3:
1. Generate a large held-out dataset ($n = 10^4$) from the same data-generator seed family.
2. Compute the true reward matrix $R^\text{holdout}$.
3. For each arm $i$, fit the best offline ridge regressor $\hat\theta_i^\text{offline} = \arg\min_\theta \sum_t (R^\text{holdout}_{t,i} - \theta^\top \phi(x_t))^2 + \lambda \|\theta\|^2$ using the *same feature map* as the online policy.
4. Compute approximation error $\hat\varepsilon_i = \sqrt{\text{mean}_t (R^\text{holdout}_{t,i} - \hat\theta_i^{\text{offline},\top} \phi(x_t))^2}$, take max over arms.

**Plot.** x-axis = $\max_i \hat\varepsilon_i$, y-axis = LinUCB-raw final regret. One point per config in Experiments 2 & 3. Overlay the theoretical shape $c_1 \sqrt{T \log T} + c_2 \varepsilon T \sqrt{\log T}$ fit by least squares. If the fit is reasonable, that's a direct empirical validation of Theorem E.1's functional form.

### Sanity plots (not main figures, but include in appendix)

- Confusion matrix (rows = true cluster, cols = arm pulled) per policy per config. LinUCB-raw should show the permutation pattern; LinUCB-RBF should be diagonal.
- Per-arm pull counts over time. LinUCB-raw arms should plateau early; LinUCB-RBF arms should track cluster frequencies.
- Mean/bonus/UCB diagnostic at typical cluster contexts, post-training.

### Run count

- Experiment 1: 30 seeds × 4 policies × 1 config = 120 runs.
- Experiment 2: 10 seeds × 4 policies × 5 contamination levels = 200 runs. Plus 5 expert-training jobs.
- Experiment 3: same = 200 runs. Plus 5 expert-training jobs.
- Experiment 4: deterministic; no seeds needed. ~10 offline-fit jobs total.

Total: ~520 bandit runs, each ≤5 seconds at $T=5000$. Under an hour on a laptop. Tractable.

---

## 6. Repository layout

```
moe_bandit/
  data.py              # §1
  experts.py           # §2
  runner.py            # §3
  features.py          # feature maps: RawFeatureMap, RBFFeatureMap
  policies/
    __init__.py
    random.py          # UniformRandomPolicy
    epsilon_greedy.py  # EpsilonGreedyPolicy
    linucb.py          # LinUCBPolicy (takes feature map at construction)
    thompson.py        # LinearTSPolicy
    softmax.py         # SoftmaxRouterPolicy
  experiments/
    configs.py
    exp1_headline.py       # §5 Exp 1
    exp2_contamination.py  # §5 Exp 2
    exp3_separation.py     # §5 Exp 3
    exp4_approx_error.py   # §5 Exp 4
  plots/
    make_figures.py
  tests/
    test_data.py
    test_experts.py
    test_runner.py
    test_linucb.py
    test_features.py
  notebooks/
    00_data_sanity.ipynb
    01_expert_training.ipynb
    02_single_run.ipynb
    03_contamination_sweep.ipynb
    04_approx_error.ipynb
```

---

## 7. Week-by-week (revised)

- **Week 1 (done):** data generator, expert training, feedback loop, uniform + ε-greedy + LinUCB-raw. Extensive debugging revealed misspecification as the dominant failure mode; reframed empirical story around it.
- **Week 2:** Implement `features.py` (RawFeatureMap, RBFFeatureMap) and refactor LinUCBPolicy to take a feature map. Build LinUCB-RBF. Run Experiment 1 (headline). Start Experiment 2 (contamination sweep).
- **Week 3:** Finish Experiments 2 and 3. Run Experiment 4 (approximation error measurement). Generate final figures. Write report.

---

## 8. Theoretical caveats / notes for the writeup

- **Claim precision.** The finding is *"raw-feature disjoint LinUCB is misspecified for this environment, and the observed regret is consistent with misspecification dominating performance."* Not *"LinUCB fails fundamentally."* The RBF variant working on the same environment is the evidence that the feature map, not the algorithm, is the issue.
- **Contamination is a proxy, not $\varepsilon$.** Experiment 4 closes this gap by measuring actual approximation error directly. In the writeup: contamination/cluster_sep are environment knobs; $\hat\varepsilon$ is the measured misspecification; the regret-vs-$\hat\varepsilon$ relationship is what maps onto Theorem E.1.
- **Parameter norm.** The "replace $d$ by $d+1$" (raw) or "$K+1$" (RBF) accounting is simple. But the stacked parameter norm picks up a $\sqrt{K}$ when translating from per-arm to joint bound. Sync with Aidan.
- **Sub-Gaussian reward.** Resolved via probability clipping plus rescaling to $[0, 1]$. Mention in writeup.
- **In-distribution vs extrapolation.** A key insight worth spelling out: LinUCB's UCB mechanism requires the linear model to give reasonable predictions on contexts the arm hasn't pulled. With spike-plus-plateau rewards, the best in-distribution fit (using the arm's visited contexts) extrapolates arbitrarily poorly to unvisited regions. Forced exploration helps but doesn't solve this — the issue is representational, not informational.