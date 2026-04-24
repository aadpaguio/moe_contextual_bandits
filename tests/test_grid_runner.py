from __future__ import annotations

from moe_bandit.experiments.grid_runner import FixedSettings, run_main_grid


def test_main_grid_small_runs(tmp_path):
    settings = FixedSettings(
        K=3,
        d=4,
        T=120,
        n_train_per_cluster=80,
        softmax_epochs=5,
        softmax_hidden_dim=16,
        softmax_batch_size=32,
    )
    rows, artifacts = run_main_grid(
        output_dir=tmp_path,
        settings=settings,
        contaminations=[0.10],
        cluster_seps=[1.5],
        seed_bundles=[(11, 21, 31, 41)],
    )

    assert len(rows) == 5  # one grid cell, one seed bundle, five policies
    assert all(r.expert_regime == "independent" for r in rows)
    assert "contam_0.10_sep_1.5_seed_0__independent" in artifacts["runs"]
    assert (tmp_path / "results_rows.csv").exists()
    assert (tmp_path / "results_rows.jsonl").exists()
    assert (tmp_path / "artifacts.json").exists()


def test_main_grid_independent_and_joint(tmp_path):
    settings = FixedSettings(
        K=3,
        d=4,
        T=120,
        n_train_per_cluster=80,
        softmax_epochs=5,
        softmax_hidden_dim=16,
        softmax_batch_size=32,
        joint_moe_max_epochs=5,
        joint_moe_early_stopping_patience=None,
    )
    rows, artifacts = run_main_grid(
        output_dir=tmp_path,
        settings=settings,
        contaminations=[0.10],
        cluster_seps=[1.5],
        seed_bundles=[(11, 21, 31, 41)],
        expert_training_regimes=("independent", "joint"),
        write_approx_error_jsonl=True,
    )

    assert len(rows) == 10
    assert {r.expert_regime for r in rows} == {"independent", "joint"}
    assert "contam_0.10_sep_1.5_seed_0__independent" in artifacts["runs"]
    assert "contam_0.10_sep_1.5_seed_0__joint" in artifacts["runs"]
    assert (tmp_path / "approx_error_by_regime.jsonl").exists()
