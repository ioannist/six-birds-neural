import itertools

import pytest
import torch

from ratchet_gpu.lattice import (
    DEFAULT_STENCIL_POLICY,
    Lattice,
    gather_neighbors,
    generate_stencil,
)


def test_index_coord_roundtrip():
    lattice = Lattice((3, 4, 5))
    indices = torch.arange(lattice.N, dtype=torch.long)
    coords = lattice.index_to_coord(indices)
    back = lattice.coord_to_index(coords)
    assert torch.equal(indices, back)

    coord_samples = torch.tensor(
        [
            [-1, 0, 0],
            [3, 4, 5],
            [0, -2, 7],
        ],
        dtype=torch.long,
    )
    wrapped = lattice.wrap_coord(coord_samples)
    expected = coord_samples % torch.tensor(lattice.shape, dtype=torch.long)
    assert torch.equal(wrapped, expected)

    lattice_1d = Lattice((7,))
    indices_1d = torch.arange(lattice_1d.N, dtype=torch.long)
    coords_1d = lattice_1d.index_to_coord(indices_1d)
    back_1d = lattice_1d.coord_to_index(coords_1d)
    assert torch.equal(indices_1d, back_1d)


def test_stencil_closure_symmetry():
    d = 4
    radius = 3
    offsets = generate_stencil(
        d=d, policy=DEFAULT_STENCIL_POLICY, radius=radius, bipartite=True
    )
    offsets_set = {tuple(r.tolist()) for r in offsets}

    assert (0, 0, 0, 0) not in offsets_set

    for r in offsets_set:
        assert tuple(-v for v in r) in offsets_set
        assert sum(r) % 2 != 0

        for perm in itertools.permutations(range(d)):
            permuted = tuple(r[i] for i in perm)
            assert permuted in offsets_set

        for signs in itertools.product([-1, 1], repeat=d):
            flipped = tuple(signs[i] * r[i] for i in range(d))
            assert flipped in offsets_set


def test_stencil_even_parity():
    offsets = generate_stencil(d=3, policy="l1_ball_even", radius=2, bipartite=False)
    offsets_set = {tuple(r.tolist()) for r in offsets}

    assert (0, 0, 0) not in offsets_set
    for r in offsets_set:
        assert sum(r) % 2 == 0
        assert tuple(-v for v in r) in offsets_set
        for perm in itertools.permutations(range(3)):
            permuted = tuple(r[i] for i in perm)
            assert permuted in offsets_set
        for signs in itertools.product([-1, 1], repeat=3):
            flipped = tuple(signs[i] * r[i] for i in range(3))
            assert flipped in offsets_set


def test_empty_stencil_radius_zero():
    lattice = Lattice((2, 2))
    offsets = generate_stencil(
        d=lattice.d, policy=DEFAULT_STENCIL_POLICY, radius=0, bipartite=True
    )
    assert offsets.shape == (0, lattice.d)

    x = torch.arange(lattice.N, dtype=torch.long)
    gathered = gather_neighbors(x, lattice, offsets)
    assert gathered.shape == (lattice.N, 0)


def test_bipartite_warning_on_odd_shape():
    with pytest.warns(RuntimeWarning):
        generate_stencil(
            d=2,
            policy=DEFAULT_STENCIL_POLICY,
            radius=1,
            bipartite=True,
            shape=(3, 4),
        )


def _slow_gather(x, lattice, offsets):
    result = []
    offsets_list = offsets.tolist()
    for i in range(lattice.N):
        coord = lattice.index_to_coord(i)
        row = []
        for off in offsets_list:
            nbr = tuple(
                (coord[k] + off[k]) % lattice.shape[k] for k in range(lattice.d)
            )
            idx = lattice.coord_to_index(nbr)
            row.append(x[idx])
        result.append(torch.stack(row, dim=0))
    return torch.stack(result, dim=0)


def test_gather_neighbors_cpu():
    lattice = Lattice((2, 3))
    offsets = torch.tensor([[0, 1], [1, 0], [-1, -1]], dtype=torch.long)
    x = torch.arange(lattice.N * 2, dtype=torch.long).reshape(lattice.N, 2)

    gathered = gather_neighbors(x, lattice, offsets)
    expected = _slow_gather(x, lattice, offsets)

    assert torch.equal(gathered, expected)


@pytest.mark.cuda
def test_gather_neighbors_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    lattice = Lattice((2, 3))
    offsets = torch.tensor([[0, 1], [1, 0], [-1, -1]], dtype=torch.long)
    x = torch.arange(lattice.N * 2, dtype=torch.long).reshape(lattice.N, 2)

    gathered = gather_neighbors(x.cuda(), lattice, offsets.cuda())
    expected = _slow_gather(x, lattice, offsets).to(gathered.device)

    assert torch.equal(gathered, expected)
