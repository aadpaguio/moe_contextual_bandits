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
    assert "contam_0.10_sep_1.5_seed_0" in artifacts["runs"]
    assert (tmp_path / "results_rows.csv").exists()
    assert (tmp_path / "results_rows.jsonl").exists()
    assert (tmp_path / "artifacts.json").exists()
