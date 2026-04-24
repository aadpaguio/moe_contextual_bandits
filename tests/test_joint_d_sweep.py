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

    assert len(rows) == 8  # 2 d values x 1 seed x 4 policies
    assert {r.policy for r in rows} == {"linucb_raw", "epsilon_greedy", "softmax_router", "oracle"}
    assert "d_2_seed_0" in artifacts["runs"]
    assert "d_4_seed_0" in artifacts["runs"]
    assert (tmp_path / "results_rows.csv").exists()
    assert (tmp_path / "results_rows.jsonl").exists()
    assert (tmp_path / "d_sweep_summary.csv").exists()
    assert (tmp_path / "artifacts.json").exists()
    assert (tmp_path / "regret_curves_by_d.png").exists()
    assert (tmp_path / "d_sweep_summary_metrics.png").exists()
