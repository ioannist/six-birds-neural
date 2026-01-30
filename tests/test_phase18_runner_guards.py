import pytest

from scripts.phase18_motif_dictionary_semantics_v1 import (
    _parse_motif_features,
    _validate_bins,
    _validate_hazard_schedule,
)


def test_phase18_hazard_schedule_guards() -> None:
    _validate_hazard_schedule(1, 2, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 2, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(2, 0, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(4, 3, 5)


def test_phase18_motif_feature_guard() -> None:
    feats = _parse_motif_features("k_axis_bias,k_entropy")
    assert feats == ["k_axis_bias", "k_entropy"]
    with pytest.raises(ValueError):
        _parse_motif_features("k_axis_bias,unknown")


def test_phase18_bins_guard() -> None:
    _validate_bins(2, 2)
    with pytest.raises(ValueError):
        _validate_bins(1, 2)
    with pytest.raises(ValueError):
        _validate_bins(2, 1)
