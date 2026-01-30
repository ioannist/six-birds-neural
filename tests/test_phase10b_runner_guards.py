import pytest

from scripts.phase10b_excitable_state_upgrade_v1 import _validate_args


def _base_args() -> object:
    return type(
        "Args",
        (),
        {
            "max_windows": 10,
            "last_m": 5,
            "excited_frac_min": 0.01,
            "excited_frac_max": 0.50,
        },
    )()


def test_validate_args_accepts_valid() -> None:
    _validate_args(_base_args())


def test_validate_args_rejects_bad_windows() -> None:
    args = _base_args()
    args.max_windows = 0
    with pytest.raises(ValueError):
        _validate_args(args)

    args = _base_args()
    args.last_m = 11
    with pytest.raises(ValueError):
        _validate_args(args)


def test_validate_args_rejects_excited_bounds() -> None:
    args = _base_args()
    args.excited_frac_min = -0.1
    with pytest.raises(ValueError):
        _validate_args(args)

    args = _base_args()
    args.excited_frac_max = 1.1
    with pytest.raises(ValueError):
        _validate_args(args)

    args = _base_args()
    args.excited_frac_min = 0.6
    args.excited_frac_max = 0.5
    with pytest.raises(ValueError):
        _validate_args(args)
