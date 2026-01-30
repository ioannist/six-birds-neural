from scripts.phase13_pattern_memory_setpoints_v1 import _injury_apply_between_windows


def test_injury_apply_between_windows() -> None:
    assert _injury_apply_between_windows(8, 1) == [7]
    assert _injury_apply_between_windows(8, 3) == [7, 8, 9]
    assert _injury_apply_between_windows(1, 2) == [0, 1]
