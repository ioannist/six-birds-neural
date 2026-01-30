import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase9_hazard_attention_highways_v1 import _compute_realloc_scores, _hazard_metrics


def test_realloc_score_smoke() -> None:
    baseline = {
        "mismatch_region": [0.5] * 10,
        "k_contrast": [0.1] * 10,
        "w_contrast": [0.1] * 10,
        "k_patch": [0.2] * 10,
        "w_patch": [0.2] * 10,
        "k_delta_focus": [0.0] * 10,
        "w_delta_focus": [0.0] * 10,
        "k_focus": [0.0] * 10,
        "w_focus": [0.0] * 10,
        "k_axis_bias_focus": [0.0] * 10,
    }
    hazard = {
        "mismatch_region": [0.5, 0.5, 0.5, 0.5, 0.8, 0.9, 0.8, 0.6, 0.55, 0.52],
        "k_contrast": [0.1] * 10,
        "w_contrast": [0.1] * 10,
        "k_patch": [0.2] * 10,
        "w_patch": [0.2] * 10,
        "k_delta_focus": [0.0, 0.0, 0.0, 0.0, 0.08, 0.1, 0.08, 0.02, 0.01, 0.0],
        "w_delta_focus": [0.0, 0.0, 0.0, 0.0, 0.05, 0.06, 0.05, 0.01, 0.01, 0.0],
        "k_focus": [0.0, 0.0, 0.0, 0.0, 0.07, 0.09, 0.07, 0.02, 0.01, 0.0],
        "w_focus": [0.0, 0.0, 0.0, 0.0, 0.04, 0.05, 0.04, 0.01, 0.01, 0.0],
        "k_axis_bias_focus": [0.0, 0.0, 0.0, 0.0, 0.06, 0.08, 0.06, 0.02, 0.01, 0.0],
    }
    scores = _compute_realloc_scores(hazard, baseline, hazard_start=5, hazard_duration=3, total_windows=10)
    assert scores["best"] > 0.0

    h_metrics = _hazard_metrics(hazard["mismatch_region"], 5, 3, 10)
    assert h_metrics["spike"] > 0.0
