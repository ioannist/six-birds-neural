# Six Birds: Neural Substrate

This repository contains the **neural/meta-layer substrate** for the paper:

> **To Wake a Stone with Six Birds: A Life is A Theory**
>
> Archived at: https://zenodo.org/records/18420406

This paper is the life-focused instantiation of the emergence calculus introduced in *Six Birds: Foundations of Emergence Calculus*. It demonstrates how the canonical theory-package view (microstate, lens/observables, definability, completion/packaging rule, and audit) can be instantiated in working substrates, and what life-like phenomena are observed in those instantiations.

## What this repository provides

The neural/meta-layer substrate implements:

- **Budgeted token-mediated coupling** for resource-constrained computation
- **Stroboscopic diagnostics** for observing system dynamics
- **Hazard response under matched baselines** for controlled experiments
- **Refined-lens predicate families**: motif inventories, proto-syntax shifts, and intervention-conditioned decoding statistics with shift-null controls

See also: [six-birds-particle](https://github.com/anthropics/six-birds-particle) for the particle-based substrate.

## Scope and limitations

The paper is explicit about what it does and does not establish:

- Protocol holonomy (P3) is reported as route-dependence diagnostics; arrow-of-time claims require a clean audit/drive channel (P6) separated from a calibrated null
- Reported audit quantities are proxies, not full path-space KL audits
- Idempotence defects of the completion/packaging operator are not measured
- "Novelty/extension" is lens-relative and not claimed as unbounded open-ended evolution

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
