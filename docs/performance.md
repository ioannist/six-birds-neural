# Performance notes (future work)

This implementation is correctness-first. Several patterns will limit GPU scaling and should be revisited in a performance-focused step.

## Known bottlenecks

- Single-site updates with `.item()` extraction cause CPU<->GPU synchronization each step.
- Python control flow per site/kernel prevents effective GPU parallelism.
- `gather_neighbors` relies on Python loops and `torch.roll` per offset, which is expensive for large stencils.

## Suggested refactors

- Batch updates by color class (vectorized masks) instead of single-site moves.
- Keep RNG and acceptance on-device; avoid `.item()` in hot loops.
- Precompute neighbor index tables and use tensor gather/scatter updates.
- Apply accept/reject masks in bulk.
- Consider custom CUDA kernels once the algorithm is stable.
