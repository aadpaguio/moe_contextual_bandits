from __future__ import annotations

from moe_bandit.experiments.joint_d_sweep import JointDSweepSettings, run_joint_d_sweep


def test_joint_d_sweep_small_runs(tmp_path):
    settings = JointDSweepSettings(
        K=3,
        T=120,
        n_train_per_cluster=80,
        d_values=(2, 4),
        seed_bundles=((11, 21, 31, 41),),
        softmax_epochs=5,
        softmax_hidden_dim=16,
        softmax_batch_size=32,
        joint_moe_max_epochs=5,
        joint_moe_early_stopping_patience=None,
    )
    rows, artifacts = run_joint_d_sweep(output_dir=tmp_path, settings=settings)

    expected_policies = {
        "uniform",
        "epsilon_greedy",
        "softmax_router",
        "oracle",
        "linucb_raw",
        "linucb_raw_alpha_0.5",
        "linucb_raw_alpha_1",
        "linucb_raw_alpha_2",
        "linucb_raw_alpha_4",
        "linucb_raw_alpha_8",
    }
    assert len(rows) == 20  # 2 d values x 1 seed x 10 policy rows
    assert {r.policy for r in rows} == expected_policies
    assert all(
        r.linucb_alpha is not None
        for r in rows
        if r.policy == "linucb_raw" or r.policy.startswith("linucb_raw_alpha_")
    )
    assert "d_2_seed_0" in artifacts["runs"]
    assert "d_4_seed_0" in artifacts["runs"]
    assert (tmp_path / "results_rows.csv").exists()
    assert (tmp_path / "results_rows.jsonl").exists()
    assert (tmp_path / "d_sweep_summary.csv").exists()
    assert (tmp_path / "artifacts.json").exists()
    assert (tmp_path / "regret_curves_by_d.png").exists()
    assert (tmp_path / "d_sweep_summary_metrics.png").exists()
