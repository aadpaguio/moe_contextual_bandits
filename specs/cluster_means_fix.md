# Fix: cluster means should occupy R^d, not a fixed low-d subspace

## The problem

`_build_cluster_means` currently places cluster centers along standard basis
vectors when `d >= K`:

```python
if d >= K:
    means = np.zeros((K, d), dtype=np.float64)
    means[np.arange(K), np.arange(K)] = separation_scale
    return means
```

For K=4 and d=32 this produces:

```
mu_0 = [s, 0, 0, 0, 0, ..., 0]   # nonzero only in dim 0
mu_1 = [0, s, 0, 0, 0, ..., 0]   # nonzero only in dim 1
mu_2 = [0, 0, s, 0, 0, ..., 0]
mu_3 = [0, 0, 0, s, 0, ..., 0]
# dims 4..31 are zero for every cluster mean
```

where `s = cluster_sep * cluster_std`. Noise is then added as full-dimensional
isotropic Gaussian `N(0, cluster_std^2 · I_d)`, so samples carry meaningful
signal in dims 0..K-1 and pure noise in dims K..d-1.

**Intrinsic cluster structure is always 4-dimensional regardless of `d`.** The
extra dimensions are noise the bandit has to learn to ignore.

## Why this breaks a `d` sweep

The motivation for varying `d` is to test whether the misspecification story
("linear approximation error is large under independent training, smaller under
joint training") holds in high-dimensional settings — which is the realistic
MoE regime.

Under the current generator, varying `d` doesn't vary the intrinsic geometry of
the problem. It only varies how many nuisance dimensions the bandit must
regress over.

Consequences:
- Linear approximation error `ε` should not change much with `d`. The best
  linear classifier always lives in the same 4-d subspace; extra dimensions
  contribute zero signal and a linear fit can ignore them (in expectation).
- LinUCB regret may grow with `d` due to sample complexity (more coefficients
  to estimate), but not due to genuine misspecification.
- The writeup claim "joint training mitigates misspecification at higher d" is
  not actually tested, because the misspecification (spike-plus-plateau
  expert reward geometry) is unchanged.

This is a **sample complexity** experiment, not a **misspecification at high d**
experiment. Those are different questions.

## The fix

Place cluster means in generic positions in R^d using all `d` coordinates:

```python
def _build_cluster_means(
    K: int,
    d: int,
    cluster_sep: float,
    cluster_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build K cluster centers as random unit directions in R^d, scaled by
    cluster_sep * cluster_std.

    Every coordinate contributes to separation. Pairwise distances between
    centers are approximately sqrt(2) * cluster_sep * cluster_std in
    expectation (random unit vectors are nearly orthogonal in high d).
    """
    separation_scale = cluster_sep * cluster_std
    raw = rng.normal(size=(K, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = raw / np.clip(norms, 1e-12, None)
    return separation_scale * unit
```

Each cluster mean is a random unit vector in R^d scaled to norm
`cluster_sep * cluster_std`. All `d` coordinates carry signal.

## Caveats introduced by the fix

### Pairwise distances become random

Under the old generator, every pair of centers was exactly
`cluster_sep · cluster_std · √2` apart (since means were along orthogonal
basis vectors).

Under the fix, pairwise distances have variance:

- At high `d`: random unit vectors are nearly orthogonal, so pairwise
  distances concentrate around `sqrt(2) · cluster_sep · cluster_std`.
- At low `d` (e.g. d=4 with K=4): directions are not orthogonal and pairwise
  distances vary noticeably.

Document this in the writeup: `cluster_sep` now controls distance from origin,
not exact pairwise separation.

### Old results at d=4 are not directly comparable

Your previous (contam × sep) grid was run with the old generator. If you
re-run at d=4 using the fixed generator, the geometry is slightly different
(random directions instead of standard basis). For the d-sweep this is fine
since you're comparing across `d` values *all using the fixed generator*.

If you want your d=4 cell in the new sweep to match the old runs exactly,
keep the old generator as a fallback for `d >= K` and only switch to random
directions for `d > K`. This is more work and introduces an inconsistency
across the `d` axis — not recommended.

Cleaner: re-run everything (including d=4) with the fixed generator.

### Optional strengthening: orthonormal directions at d >= K

If you want strict orthogonality whenever possible, use
`scipy.stats.ortho_group.rvs(d)[:K]` to sample K orthonormal rows from a
random orthogonal matrix. This gives exact orthogonality at `d >= K` and
preserves the fixed pairwise distance property. Fall back to random unit
vectors for `d < K`.

```python
from scipy.stats import ortho_group

def _build_cluster_means(K, d, cluster_sep, cluster_std, rng):
    separation_scale = cluster_sep * cluster_std
    if d >= K:
        # Sample a random orthogonal matrix, take first K rows as orthonormal
        # directions. This preserves sqrt(2) * cluster_sep pairwise distances
        # while using all d coordinates (the basis is rotated).
        Q = ortho_group.rvs(dim=d, random_state=rng)
        unit = Q[:K]
    else:
        raw = rng.normal(size=(K, d))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        unit = raw / np.clip(norms, 1e-12, None)
    return separation_scale * unit
```

This is the cleanest option: centers are at identical distances from origin,
pairwise distances are identical at `d >= K`, and every coordinate contributes
to separation. Recommended.

## What changes downstream

Only `_build_cluster_means` needs updating. Everything else in the pipeline
(independent and joint expert training, reward matrix construction, bandit
policies, regret computation) operates on `(X, y)` pairs and doesn't depend on
how the means were placed.

Update the docstring of `generate_synthetic_data` to remove the "along
orthogonal directions" phrasing, which is no longer accurate (or strictly
accurate only at `d >= K` if using the `ortho_group` variant).

## Sanity check after the fix

Before running the full d-sweep, verify:

1. At d=4 with the fix, joint-vs-independent results are qualitatively similar
   to the previous d=4 results (directions are random but the geometry is
   comparable). Small quantitative differences are expected and fine.
2. At d=32, linear approximation error `ε` under independent training is
   similar to or smaller than at d=4. If `ε` drops substantially with d, the
   misspecification story is largely a low-d phenomenon and the writeup needs
   to qualify its claims.
3. Joint training's RMSE `ε` is still lower than independent's at each `d`.
   If they converge at high d, joint training's advantage is dimensionality-
   dependent.

These are the load-bearing checks for whether your joint-training claim
generalizes to realistic MoE dimensionalities.