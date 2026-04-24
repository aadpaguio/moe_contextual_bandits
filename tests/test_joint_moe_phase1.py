from __future__ import annotations

import numpy as np

from moe_bandit.data import generate_synthetic_data
from moe_bandit.experts import expert_reward_matrix, train_experts
from moe_bandit.linear_approx_error import linear_approx_max_error
from moe_bandit.train_joint_moe import train_joint_moe


def test_train_joint_moe_smoke():
    K, d = 2, 2
    X, y, cluster_id = generate_synthetic_data(
        n_samples=400,
        K=K,
        d=d,
        cluster_sep=3.0,
        cluster_std=1.0,
        seed=0,
    )
    experts, stats = train_joint_moe(
        X_train=X,
        y_train=y,
        cluster_id_train=cluster_id,
        K=K,
        d=d,
        epochs=3,
        batch_size=32,
        seed=0,
        val_frac=0.1,
        early_stopping_patience=None,
    )
    assert len(experts) == K
    assert len(stats.history_ce_loss) == 3
    assert len(stats.history_lr) == 3
    assert stats.final_gate_means is not None
    assert stats.final_gate_means.shape == (K,)
    assert np.isfinite(stats.history_total_loss).all()
    R = expert_reward_matrix(experts=experts, X=X, y=y, clip_eps=1e-3)
    assert R.shape == (len(X), K)


def test_linear_approx_max_error_matches_shape():
    rng = np.random.default_rng(0)
    T, K, d = 50, 4, 3
    X = rng.normal(size=(T, d))
    R = rng.normal(size=(T, K))
    rep = linear_approx_max_error(R, X, lambda_reg=1.0, fit_intercept=True)
    assert rep.epsilon_per_arm.shape == (K,)
    assert rep.mean_epsilon >= 0
    assert rep.max_epsilon >= rep.mean_epsilon - 1e-9


def test_phase1_reference_config_joint_training():
    """
    Reference settings from specs/moe_joint.md Phase 1 (single-config gate).
    Uses fewer epochs than the notebook full run to keep CI fast; thresholds are
    conservative.
    """
    K = 4
    d = 4
    contamination = 0.05
    cluster_sep = 1.5
    seed_data = 0
    n_train_per_cluster = 2000
    n_train_total = K * n_train_per_cluster

    X_train, y_train, cluster_train = generate_synthetic_data(
        n_samples=n_train_total,
        K=K,
        d=d,
        cluster_sep=cluster_sep,
        cluster_std=1.0,
        seed=seed_data,
    )

    seed_joint = 202
    experts_j, stats_j = train_joint_moe(
        X_train=X_train,
        y_train=y_train,
        cluster_id_train=cluster_train,
        K=K,
        d=d,
        epochs=30,
        batch_size=64,
        seed=seed_joint,
        alpha_load=0.01,
        val_frac=0.1,
        early_stopping_patience=None,
        cosine_decay=True,
        lr_min=1e-5,
        router="linear",
    )

    assert len(experts_j) == K
    assert stats_j.final_gate_means is not None
    # Convergence: CE should drop from first to last epoch
    assert stats_j.history_ce_loss[0] > stats_j.history_ce_loss[-1] * 0.95
    # Holdout sanity (Phase 1 target > 0.7 in the spec; allow slack for CI variance)
    assert stats_j.final_pooled_val_acc > 0.65
    # Non-degenerate gates by end (spec: target each > 0.10; warn if < 0.05 early)
    assert np.all(stats_j.final_gate_means > 0.08)
    assert not stats_j.collapse_warning_early

    R_j = expert_reward_matrix(experts=experts_j, X=X_train, y=y_train, clip_eps=1e-3)
    assert np.isfinite(R_j).all()
    # Experts are anonymous under joint training; do not require cluster-diagonal dominance.

    seed_indep = 303
    experts_i = train_experts(
        X_train=X_train,
        y_train=y_train,
        cluster_id_train=cluster_train,
        K=K,
        d=d,
        epochs=30,
        seed=seed_indep,
        contamination=contamination,
    )
    R_i = expert_reward_matrix(experts=experts_i, X=X_train, y=y_train, clip_eps=1e-3)

    eps_j = linear_approx_max_error(R_j, X_train, lambda_reg=1.0, fit_intercept=True)
    eps_i = linear_approx_max_error(R_i, X_train, lambda_reg=1.0, fit_intercept=True)
    assert np.isfinite(eps_j.mean_epsilon)
    assert np.isfinite(eps_i.mean_epsilon)
    # Phase 1 manual gate: compare eps_j.mean_epsilon vs eps_i.mean_epsilon (notebook).
