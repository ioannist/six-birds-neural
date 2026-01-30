import pytest

from scripts.phase12_motif_proto_syntax_v1 import (
    _parse_keys,
    _validate_bins,
    _validate_features,
    _validate_hazard_schedule,
)


def test_validate_hazard_schedule() -> None:
    _validate_hazard_schedule(1, 1, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(0, 1, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(1, 0, 5)
    with pytest.raises(ValueError):
        _validate_hazard_schedule(5, 2, 5)


def test_validate_features() -> None:
    feats = _parse_keys("k_axis_bias,k_entropy")
    _validate_features(feats)
    with pytest.raises(ValueError):
        _validate_features(["bad_feature"])


def test_validate_bins() -> None:
    _validate_bins(2, 2)
    with pytest.raises(ValueError):
        _validate_bins(1, 2)
