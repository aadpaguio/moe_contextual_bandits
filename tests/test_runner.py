import numpy as np

from moe_bandit.policies import UniformRandomPolicy
from moe_bandit.runner import run_bandit


def test_uniform_random_linear_regret_toy_matrix():
    T = 1000
    K = 4
    d = 3
    X = np.zeros((T, d), dtype=np.float64)
    R = np.zeros((T, K), dtype=np.float64)
    R[:, 0] = 1.0  # oracle arm always 0

    policy = UniformRandomPolicy(K=K, seed=0)
    result = run_bandit(policy=policy, X=X, R=R, seed=0)

    assert result.reward.shape == (T,)
    assert result.regret.shape == (T,)
    assert np.all(result.regret >= -1e-12)

    expected_rate = (K - 1) / K
    observed_rate = result.cumulative_regret()[-1] / T
    assert abs(observed_rate - expected_rate) < 0.07
