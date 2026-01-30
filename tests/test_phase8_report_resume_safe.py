from pathlib import Path

from scripts.phase8_spatial_harness_v1 import _write_report


def test_report_resume_safe(tmp_path: Path) -> None:
    report_path = tmp_path / "PHASE8_SPATIAL_REPORT.md"
    agg_rows = [
        {
            "case": "null_full",
            "seed": "1",
            "status": "OK",
            "windows_completed": "10",
            "accept_mean_last_m": "0.01",
            "ep_mean_last_m": "1e-4",
            "ep_ci_half_last_m": "2e-4",
            "ep_slope_last_m": "-1e-5",
            "k_drive_mean_last_m": "1e-3",
            "k_drive_slope_last_m": "2e-4",
            "mismatch_mean_last_m": "0.9",
            "mismatch_slope_last_m": "-1e-3",
            "strobe_l2_mean_last_m": "0.0",
            "strobe_l2_slope_last_m": "0.0",
        }
    ]
    _write_report(agg_rows, report_path)
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").startswith("# Phase 8 spatial harness v1")
