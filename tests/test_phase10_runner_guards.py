from pathlib import Path

from scripts.phase10_clockwork_fabric_search_v1 import _match_cycle_weights, _write_report


def test_match_control_cycle_weights() -> None:
    kw = {"a": 2.0, "b": 0.0, "c": 0.5}
    cycle = ["a", "c"]
    matched = _match_cycle_weights(kw, cycle)
    assert matched["a"] == 1.0
    assert matched["c"] == 1.0
    assert matched["b"] == 0.0


def test_report_write_resume_safe(tmp_path: Path) -> None:
    rows = [
        {
            "case": "protocol_p3_on",
            "seed": "1",
            "status": "PASS",
            "control_best": "0.01",
            "protocol_best": "0.02",
            "delta": "0.01",
            "best_key": "k_axis_bias",
            "best_metric": "travel",
        }
    ]
    report_path = tmp_path / "PHASE10_CLOCKWORK_REPORT.md"
    _write_report(rows, report_path, "cmd")
    assert report_path.exists()
