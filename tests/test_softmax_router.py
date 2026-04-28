import numpy as np

from moe_bandit.policies import (
    OnlineSoftmaxPolicy,
    UniformRandomPolicy,
    train_cluster_label_router,
    train_softmax_router,
)
from moe_bandit.runner import run_bandit


def test_training_returns_usable_policy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    R = rng.normal(size=(200, 3))
    policy = train_softmax_router(X, R, epochs=10, batch_size=32, seed=0)
    arm = policy.select(X[0])
    assert isinstance(arm, int)
    assert 0 <= arm < 3


def test_update_is_noop():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    R = rng.normal(size=(50, 2))
    policy = train_softmax_router(X, R, epochs=5, batch_size=16, seed=1)
    policy.update(X[0], 0, 0.0)


def test_select_raises_on_wrong_shape():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 3))
    R = rng.normal(size=(80, 2))
    policy = train_softmax_router(X, R, epochs=5, batch_size=16, seed=2)
    try:
        policy.select(np.zeros(2))
        raise AssertionError("Expected ValueError for mismatched x_t shape.")
    except ValueError:
        pass


def test_router_beats_random_on_toy_problem():
    rng = np.random.default_rng(3)
    T = 600
    d = 4
    K = 3
    X = rng.normal(size=(T, d))
    # Oracle arm depends on sign of x0 for clear signal.
    best = (X[:, 0] > 0).astype(np.int64)
    R = np.full((T, K), 0.0, dtype=np.float64)
    R[np.arange(T), best] = 1.0

    router = train_softmax_router(X, R, epochs=25, batch_size=64, seed=3)
    random_policy = UniformRandomPolicy(K=K, seed=3)

    res_router = run_bandit(router, X, R, seed=3)
    res_random = run_bandit(random_policy, X, R, seed=3)

    assert float(res_router.reward.mean()) > float(res_random.reward.mean()) + 0.2


def test_cluster_label_router_learns_labels():
    rng = np.random.default_rng(4)
    T = 500
    d = 3
    K = 2
    X = rng.normal(size=(T, d))
    y = (X[:, 0] > 0).astype(np.int64)

    router = train_cluster_label_router(X, y, K=K, epochs=25, batch_size=64, seed=4)
    preds = np.asarray([router.select(x) for x in X])

    assert float((preds == y).mean()) > 0.85


def test_online_softmax_policy_updates_from_bandit_feedback():
    rng = np.random.default_rng(5)
    T = 300
    d = 2
    K = 2
    X = rng.normal(size=(T, d))
    R = np.zeros((T, K), dtype=np.float64)
    best = (X[:, 0] > 0).astype(np.int64)
    R[np.arange(T), best] = 1.0

    policy = OnlineSoftmaxPolicy(d=d, K=K, lr=0.05, temperature=1.0, seed=5)
    result = run_bandit(policy, X, R, seed=5)

    assert result.reward.shape == (T,)
    assert np.isfinite(result.reward).all()
    assert policy.t == T
