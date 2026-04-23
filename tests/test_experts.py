import numpy as np
import torch

from moe_bandit.data import generate_synthetic_data
from moe_bandit.experts import expert_reward_matrix, train_experts


def _accuracy(logits: torch.Tensor, y_true: np.ndarray) -> float:
    preds = logits.argmax(dim=1).cpu().numpy()
    return float((preds == y_true).mean())


def test_tiny_expert_training_smoke():
    K = 2
    d = 2
    X, y, cluster_id = generate_synthetic_data(
        n_samples=200,
        K=K,
        d=d,
        cluster_sep=3.0,
        cluster_std=1.0,
        seed=0,
    )
    experts = train_experts(
        X_train=X,
        y_train=y,
        cluster_id_train=cluster_id,
        K=K,
        d=d,
        epochs=5,
        lr=1e-3,
        batch_size=32,
        seed=0,
    )
    assert len(experts) == K

    for i, expert in enumerate(experts):
        assert not expert.training
        assert all(not p.requires_grad for p in expert.parameters())
        mask = cluster_id == i
        x_i = torch.as_tensor(X[mask], dtype=torch.float32)
        with torch.no_grad():
            logits = expert(x_i)
        acc = _accuracy(logits, y[mask])
        assert acc > 0.80

    R = expert_reward_matrix(experts=experts, X=X, y=y, clip_eps=1e-3)
    assert R.shape == (len(X), K)
    assert np.isfinite(R).all()
    assert np.max(R) <= 0.0 + 1e-9
    assert np.min(R) >= np.log(1e-3) - 1e-9
