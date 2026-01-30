# ratchet-gpu

Minimal CUDA scaffolding for a deterministic GPU smoke test and CLI harness.

## Paper reference (Zenodo)

This repository is referenced by the paper archived at:

https://zenodo.org/records/18420406

## How this repo connects to the paper

This repo is the neural/meta-layer substrate used in the paper’s canonical
theory-package instantiation (microstate, lenses/observables, definability,
completion/packaging rule, audit). It supplies the concrete state space,
stroboscopic diagnostics, hazard response under matched baselines, and the
refined-lens predicate families (motif inventories, proto-syntax shifts, and
intervention-conditioned decoding statistics with shift-null controls).

Scope and limitations (paper-consistent):
- Protocol holonomy (P3) is reported as route-dependence diagnostics; arrow-of-time
  claims require a clean audit/drive channel (P6) separated from a calibrated null.
- Audit quantities reported here are proxies, not full path-space KL audits.
- Idempotence defects of the completion/packaging operator are not measured.
- “Novelty/extension” is lens-relative and not claimed as unbounded open-ended evolution.

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
