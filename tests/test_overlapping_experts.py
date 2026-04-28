from __future__ import annotations

import numpy as np

from moe_bandit.data import generate_synthetic_data
from moe_bandit.experts import expert_reward_matrix
from moe_bandit.train_overlapping_experts import (
    build_cyclic_mixture_weights,
    train_overlapping_experts,
)


def test_build_cyclic_mixture_weights_deterministic():
    K = 4
    W1 = build_cyclic_mixture_weights(K=K, overlap_strength=0.5)
    W2 = build_cyclic_mixture_weights(K=K, overlap_strength=0.5)
    assert W1.shape == (K, K)
    assert np.allclose(W1, W2)
    assert np.all(W1 >= 0.0)
    assert np.allclose(W1.sum(axis=1), 1.0)


def test_train_overlapping_experts_smoke():
    K = 3
    d = 4
    X, y, cluster_id = generate_synthetic_data(
        n_samples=300,
        K=K,
        d=d,
        cluster_sep=2.0,
        cluster_std=1.0,
        seed=42,
    )
    experts, stats = train_overlapping_experts(
        X_train=X,
        y_train=y,
        cluster_id_train=cluster_id,
        K=K,
        d=d,
        overlap_strength=0.6,
        epochs=3,
        batch_size=32,
        seed=1,
        weight_decay=1e-4,
        label_smoothing=0.05,
    )
    assert len(experts) == K
    for expert in experts:
        assert not expert.training
        assert all(not p.requires_grad for p in expert.parameters())
    assert stats.mixture_weights.shape == (K, K)
    assert stats.sampled_cluster_proportions.shape == (K, K)
    assert stats.own_cluster_accuracy.shape == (K,)
    assert stats.cross_cluster_accuracy.shape == (K,)

    R = expert_reward_matrix(experts=experts, X=X, y=y, clip_eps=1e-3)
    assert R.shape == (len(X), K)
    assert np.isfinite(R).all()

    R_prob = expert_reward_matrix(
        experts=experts,
        X=X,
        y=y,
        clip_eps=1e-3,
        reward_type="prob",
        temperature=2.0,
    )
    assert R_prob.shape == (len(X), K)
    assert np.isfinite(R_prob).all()
    assert np.min(R_prob) >= 0.0 - 1e-9
    assert np.max(R_prob) <= 1.0 + 1e-9
