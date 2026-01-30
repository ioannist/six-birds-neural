import torch

from ratchet_gpu.setpoints import block_mask


def test_block_mask_region_selection() -> None:
    mask = torch.zeros((8, 8), dtype=torch.bool)
    mask[:4, :4] = True
    coarse = block_mask(mask, block=4)
    assert coarse.shape == (2, 2)
    assert coarse[0, 0].item() is True
    assert coarse[0, 1].item() is False
    assert coarse[1, 0].item() is False
    assert coarse[1, 1].item() is False
