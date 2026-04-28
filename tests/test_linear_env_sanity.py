from __future__ import annotations

from moe_bandit.experiments.linear_env_sanity import (
    LinearEnvSettings,
    run_linear_env_sanity,
)


def test_linear_env_sanity_runs(tmp_path):
    rows = run_linear_env_sanity(
        output_dir=tmp_path,
        settings=LinearEnvSettings(K=3, d=4, T=400, seed=0),
    )
    assert len(rows) == 4
    assert (tmp_path / "linear_env_sanity.csv").exists()
    policy_names = {r.policy for r in rows}
    assert {"uniform", "epsilon_greedy", "linucb_raw", "linucb_rbf"} <= policy_names
