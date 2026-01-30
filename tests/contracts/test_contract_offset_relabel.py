import torch
import pytest

from ratchet_gpu.lattice import Lattice, gather_neighbors


@pytest.mark.contract
def test_contract_offset_relabel_invariance():
    shape = (4, 5)
    lattice = Lattice(shape)
    d = lattice.d
    radius = 2

    grid = range(-radius, radius + 1)
    offsets = []
    for vec in torch.cartesian_prod(*[torch.tensor(list(grid)) for _ in range(d)]):
        vec = vec.tolist()
        if all(v == 0 for v in vec):
            continue
        if sum(abs(v) for v in vec) > radius:
            continue
        offsets.append(vec)

    R_K = torch.tensor(sorted({tuple(v) for v in offsets}), dtype=torch.long)
    K_K = R_K.shape[0]
    B = 3

    gen = torch.Generator().manual_seed(123)
    sigma_lower = torch.randint(0, 2, (lattice.N,), generator=gen, dtype=torch.float32)
    sigma_lower = sigma_lower * 2 - 1

    token_indices = torch.randint(0, K_K, (lattice.N, B), generator=gen)
    K = torch.zeros((lattice.N, K_K), dtype=torch.float32)
    K.scatter_add_(1, token_indices, torch.ones_like(token_indices, dtype=torch.float32))

    neighbors = gather_neighbors(sigma_lower, lattice, R_K)
    pred1 = (K * neighbors).sum(dim=-1) / float(B)

    perm = torch.randperm(K_K, generator=gen)
    R2 = R_K[perm]
    K2 = K[:, perm]
    neighbors2 = gather_neighbors(sigma_lower, lattice, R2)
    pred2 = (K2 * neighbors2).sum(dim=-1) / float(B)

    assert torch.allclose(pred1, pred2, atol=1e-6)
