import numpy as np
import torch

from ratchet_gpu.interventions import (
    apply_k_redistribute_radial_inward_in_ring,
    apply_k_redistribute_radial_random_in_ring,
    check_k_invariants,
)
from ratchet_gpu.params import Params
from ratchet_gpu.semantics import hazard_center, ring_masks_from_rect
from ratchet_gpu.state import State


def test_k_radial_redistribute_preserves_budget() -> None:
    params = Params(shape=(8, 8), layers=3, B_k=3, radius_k=2, l_k=3)
    state = State.initialize(params, seed=1)
    region, ring, _ = ring_masks_from_rect("2:4,2:4", params.shape, 1)
    ring_idx = torch.as_tensor(np.flatnonzero(ring), dtype=torch.long)
    center = hazard_center("2:4,2:4", params.shape)

    sum_before = int(state.K.sum().item())
    apply_k_redistribute_radial_inward_in_ring(
        state, params, ring_idx, center=center, interfaces="all", strength=1.0
    )
    ok, msg = check_k_invariants(state, params)
    assert ok, msg
    sum_after = int(state.K.sum().item())
    assert sum_before == sum_after

    apply_k_redistribute_radial_random_in_ring(state, params, ring_idx, interfaces="all")
    ok, msg = check_k_invariants(state, params)
    assert ok, msg
