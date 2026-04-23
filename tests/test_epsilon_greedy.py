import numpy as np

from moe_bandit.policies import EpsilonGreedyPolicy
from moe_bandit.runner import run_bandit


def test_epsilon_greedy_learns_best_context_free_arm():
    T = 2000
    K = 3
    d = 2
    X = np.zeros((T, d), dtype=np.float64)
    R = np.zeros((T, K), dtype=np.float64)
    R[:, 0] = 1.0
    R[:, 1] = 0.4
    R[:, 2] = 0.0

    policy = EpsilonGreedyPolicy(K=K, c=50.0, seed=123)
    result = run_bandit(policy=policy, X=X, R=R, seed=123)

    # Should strongly prefer the best arm by the end.
    tail = result.chosen_arm[-500:]
    frac_best = float((tail == 0).mean())
    assert frac_best > 0.80

    # Should beat uniform random on average reward in this stationary setup.
    uniform_expected = float(R.mean())
    assert float(result.reward.mean()) > uniform_expected + 0.2
