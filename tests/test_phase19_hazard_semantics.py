from scripts.phase19_motif_phrase_semantics_v1 import (
    _hazard_active_for_window,
    _hazard_applied_windows,
)


def test_phase19_hazard_start_is_first_applied() -> None:
    applied = _hazard_applied_windows(
        window_offset=0,
        max_windows=6,
        hazard_start=3,
        hazard_duration=2,
        hazard_refresh_each_window=True,
    )
    assert applied[0] == 3
    assert 3 in applied


def test_phase19_hazard_applied_matches_active_when_refreshing() -> None:
    hazard_start = 2
    hazard_duration = 3
    max_windows = 6
    applied = set(
        _hazard_applied_windows(
            window_offset=0,
            max_windows=max_windows,
            hazard_start=hazard_start,
            hazard_duration=hazard_duration,
            hazard_refresh_each_window=True,
        )
    )
    active = {
        idx
        for idx in range(1, max_windows + 1)
        if _hazard_active_for_window(idx, hazard_start, hazard_duration)
    }
    assert applied == active


def test_phase19_hazard_applied_single_when_not_refreshing() -> None:
    applied = _hazard_applied_windows(
        window_offset=0,
        max_windows=6,
        hazard_start=2,
        hazard_duration=3,
        hazard_refresh_each_window=False,
    )
    assert applied == [2]
