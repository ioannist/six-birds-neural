import pytest

from scripts.phase14_motif_semantics_v1 import (
    _validate_bins,
    _validate_features,
    _validate_hazard_schedule,
)


def test_phase14_runner_guards_hazard_schedule():
    _validate_hazard_schedule(1, 1, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 1, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(1, 0, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(5, 2, 5)


def test_phase14_runner_guards_bins_and_features():
    _validate_bins(2, 2)
    with pytest.raises(ValueError):
        _validate_bins(1, 2)
    with pytest.raises(ValueError):
        _validate_bins(2, 1)
    _validate_features(["k_axis_bias", "k_entropy"])
    with pytest.raises(ValueError):
        _validate_features(["k_axis_bias", "unknown"])
