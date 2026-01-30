import pytest


def test_phase11_runner_guards_importable():
    import scripts.phase11_motif_token_discovery_v1 as runner  # noqa: F401


def test_phase11_hazard_schedule_guard():
    import scripts.phase11_motif_token_discovery_v1 as runner

    with pytest.raises(ValueError):
        runner._validate_hazard_schedule(0, 1, 10)
    with pytest.raises(ValueError):
        runner._validate_hazard_schedule(1, 0, 10)
    with pytest.raises(ValueError):
        runner._validate_hazard_schedule(9, 3, 10)
