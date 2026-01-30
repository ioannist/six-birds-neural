# ratchet-gpu

Minimal CUDA scaffolding for a deterministic GPU smoke test and CLI harness.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Run

```bash
./build/bin/ratchet_gpu_cli --n 1048576 --steps 10 --seed 1 --add 1
```

## Python lattice (step 1)

```bash
pip install -e .[dev]
```

```python
import torch

from ratchet_gpu import DEFAULT_STENCIL_POLICY, Lattice, gather_neighbors, generate_stencil

lattice = Lattice((3, 4, 5))
offsets = generate_stencil(d=lattice.d, policy=DEFAULT_STENCIL_POLICY, radius=3, bipartite=True)
x = torch.arange(lattice.N)
neighbors = gather_neighbors(x, lattice, offsets)
```
