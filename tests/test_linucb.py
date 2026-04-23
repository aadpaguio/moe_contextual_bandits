import numpy as np

from moe_bandit.policies import LinUCBPolicy


def test_shapes_and_init():
    policy = LinUCBPolicy(K=4, d=4, alpha=1.0, lambda_reg=1.0, seed=0)
    assert policy.V.shape == (4, 5, 5)
    assert policy.b.shape == (4, 5)
    assert np.allclose(policy.V[0], np.eye(5))
    assert int(policy.pull_counts.sum()) == 0


def test_select_returns_valid_arm():
    policy = LinUCBPolicy(K=4, d=4, seed=0)
    x = np.random.default_rng(0).normal(size=4)
    arm = policy.select(x)
    assert isinstance(arm, int)
    assert 0 <= arm < 4


def test_update_changes_state():
    policy = LinUCBPolicy(K=4, d=4, seed=0)
    x = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float64)

    V0_before = policy.V[0].copy()
    b0_before = policy.b[0].copy()
    V1_before = policy.V[1].copy()
    b1_before = policy.b[1].copy()

    policy.update(x_t=x, a_t=0, r_t=1.0)

    assert not np.allclose(policy.V[0], V0_before)
    assert not np.allclose(policy.b[0], b0_before)
    assert np.allclose(policy.V[1], V1_before)
    assert np.allclose(policy.b[1], b1_before)


def test_pull_counts_track_arms():
    policy = LinUCBPolicy(K=4, d=4, seed=0)
    x = np.zeros(4, dtype=np.float64)
    policy.update(x_t=x, a_t=0, r_t=1.0)
    policy.update(x_t=x, a_t=0, r_t=0.5)
    policy.update(x_t=x, a_t=2, r_t=0.0)
    assert np.array_equal(policy.pull_counts, np.array([2, 0, 1, 0], dtype=np.int64))


def test_raises_on_wrong_context_dim():
    policy = LinUCBPolicy(K=4, d=4, seed=0)
    try:
        policy.select(np.zeros(3, dtype=np.float64))
        raise AssertionError("Expected ValueError for wrong context dimension.")
    except ValueError:
        pass


def test_learns_best_arm_in_easy_case():
    rng = np.random.default_rng(0)
    policy = LinUCBPolicy(K=2, d=2, alpha=1.0, lambda_reg=1.0, seed=0)
    T = 500
    for _ in range(T):
        x_t = rng.normal(size=2)
        a_t = policy.select(x_t)
        noise = rng.normal(scale=0.05)
        rewards = np.array([1.0 + noise, 0.0 + noise], dtype=np.float64)
        policy.update(x_t=x_t, a_t=a_t, r_t=float(rewards[a_t]))
    assert policy.pull_counts[0] / T > 0.8


def test_raises_on_negative_forced_explore_per_arm():
    try:
        LinUCBPolicy(K=4, d=4, forced_explore_per_arm=-1)
        raise AssertionError("Expected ValueError for negative forced_explore_per_arm.")
    except ValueError:
        pass
