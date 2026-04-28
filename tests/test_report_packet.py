from __future__ import annotations

import numpy as np

from moe_bandit.experiments.report_packet import (
    ReportPacketMainSettings,
    ReportPacketMotivationSettings,
    ReportPacketSettings,
    run_report_packet,
)


def test_report_packet_smoke(tmp_path):
    settings = ReportPacketSettings(
        motivation=ReportPacketMotivationSettings(
            K=2,
            d=2,
            n_train=80,
            T=60,
            seeds=(0,),
        ),
        main=ReportPacketMainSettings(
            K=2,
            n_train=80,
            T=60,
            d_values=(2,),
            seeds=(0,),
            alpha_values=(0.5, 1.0),
        ),
        softmax_epochs=2,
        softmax_hidden_dim=8,
        softmax_batch_size=16,
        joint_moe_max_epochs=2,
        joint_moe_early_stopping_patience=None,
        joint_moe_batch_size=16,
        forced_explore_per_arm=2,
    )

    packet_dir = run_report_packet(tmp_path, settings=settings, timestamped=False)

    assert (packet_dir / "manifest.json").exists()
    assert (packet_dir / "results_rows.csv").exists()
    assert (packet_dir / "results_rows.jsonl").exists()
    assert (packet_dir / "artifacts.json").exists()
    assert (packet_dir / "approx_error_by_regime.jsonl").exists()
    assert (packet_dir / "joint_training_stats.jsonl").exists()
    assert (packet_dir / "plots" / "main" / "regret_vs_d.png").exists()

    raw_files = sorted((packet_dir / "raw").glob("**/seed_data.npz"))
    assert raw_files
    payload = np.load(raw_files[0])
    for key in [
        "X_train",
        "y_train",
        "cluster_train",
        "X_bandit",
        "y_bandit",
        "cluster_bandit",
        "R_raw",
        "R_scaled",
        "oracle_arm",
    ]:
        assert key in payload
