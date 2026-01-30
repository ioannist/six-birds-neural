from scripts.phase13_pattern_memory_setpoints_v1 import _pass_suppression


def test_pass_suppression_true() -> None:
    assert _pass_suppression(
        spike_c=0.01,
        spike_d=0.04,
        damage_c=0.0,
        damage_d=0.02,
        suppression_frac_max=0.5,
        damage_max=0.0,
        damage_adv_min=0.005,
    )


def test_pass_suppression_false() -> None:
    assert not _pass_suppression(
        spike_c=0.03,
        spike_d=0.04,
        damage_c=0.02,
        damage_d=0.01,
        suppression_frac_max=0.5,
        damage_max=0.0,
        damage_adv_min=0.005,
    )
