from scripts.phase18_motif_dictionary_semantics_v1 import _hazard_active_for_window


def test_hazard_active_semantics() -> None:
    start = 6
    duration = 3
    active = [i for i in range(1, 11) if _hazard_active_for_window(i, start, duration)]
    assert active == [6, 7, 8]


def test_hazard_active_zero_duration() -> None:
    assert _hazard_active_for_window(5, 2, 0) is False
