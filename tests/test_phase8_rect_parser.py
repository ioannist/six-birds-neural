import pytest

from ratchet_gpu.interventions import parse_rect


def test_parse_rect_valid() -> None:
    mask, flat_idx = parse_rect("1:3,2:4", (6, 6))
    assert mask.shape == (6, 6)
    assert flat_idx.numel() == 4


def test_parse_rect_invalid() -> None:
    with pytest.raises(ValueError):
        parse_rect("3:1,0:2", (6, 6))
