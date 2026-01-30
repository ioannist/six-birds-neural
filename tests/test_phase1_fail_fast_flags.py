import importlib.util
import pathlib
import sys

import types


def _load_module():
    path = pathlib.Path(__file__).parent.parent / "scripts" / "phase1_null_screen_v4.py"
    spec = importlib.util.spec_from_file_location("phase1_screen_v4", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase1_screen_v4"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def test_fail_fast_mean_flag(monkeypatch, tmp_path):
    mod = _load_module()
    events = []

    def fake_run_sim(
        params,
        seed,
        steps,
        report_every,
        device,
        report_callback,
        stop_callback,
    ):
        # two windows with identical rates -> finite CI
        for i in range(2):
            step = (i + 1) * report_every
            total = (i + 1) * 10.0
            ledger = {
                "ep_total_exact": total,
                "window_steps": 10,
                "window_proposals": 10,
                "window_accepted": 1,
                "window_accept_frac": 0.1,
            }
            report_callback(None, step, ledger, 0.1)
            events.append(("report", i))
            if stop_callback(None, step, ledger, 0.1):
                break

    monkeypatch.setattr(mod, "run_sim", fake_run_sim)

    params = mod.Params(
        shape=(2, 2),
        layers=2,
        beta=0.1,
        J=1.0,
        eta=0.0,
        eta_drive=0.0,
        l_s=0,
        l_w=1,
        l_k=1,
        B_w=0,
        B_k=0,
        radius_w=1,
        radius_k=0,
        stencil_policy_w="unit",
        stencil_policy_k="unit",
        kernel_weights={"spin_flip_color0": 1.0},
        report_every=1,
        device="cpu",
    )
    out_path = tmp_path / "log.jsonl"
    res_fast = mod._run_early_stop(
        params=params,
        seed=1,
        burn_in_steps=0,
        window_steps=1,
        min_windows=2,
        max_windows=5,
        last_m=2,
        mean_thresh=0.01,
        ci_thresh=1.0,
        fail_fast_mean=True,
        fail_fast_ci=False,
        out_path=out_path,
        progress_label=None,
        progress_handle=None,
        run_start_time=mod.time.monotonic(),
        max_seconds_per_run=10.0,
    )
    assert res_fast["status"] == "FAIL_MEAN_EARLY"

    res_no_fast = mod._run_early_stop(
        params=params,
        seed=1,
        burn_in_steps=0,
        window_steps=1,
        min_windows=2,
        max_windows=5,
        last_m=2,
        mean_thresh=0.01,
        ci_thresh=1.0,
        fail_fast_mean=False,
        fail_fast_ci=False,
        out_path=out_path,
        progress_label=None,
        progress_handle=None,
        run_start_time=mod.time.monotonic(),
        max_seconds_per_run=10.0,
    )
    assert res_no_fast["status"] != "FAIL_MEAN_EARLY"
