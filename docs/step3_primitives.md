# Step 3 primitives (implementation notes)

This document describes the Step-3 primitive kernels, protocol scheduling, and EP diagnostics.

## Kernels implemented

Single-site reversible proposals (torch-only, CPU/CUDA):

- `SpinFlipColor(c)`: flip `sigma` at a random site of color `c`.
- `NFlip`: flip `n` at a random site.
- `SStep`: propose `s -> s + delta` with delta in {±1} at a random site.
- `WLocalExchange`: move one token between two offsets at a random site.
- `KLocalExchange`: move one token between two offsets at a random upper-layer site.
- `WNeighborExchange`: move one token between two sites along a random axis and sign (`delta=±e_axis`).

All proposals are symmetric by construction; infeasible proposals are self-loops.

CUDA note: `WNeighborExchange` uses parity-disjoint batch updates to avoid write collisions; CPU uses single-pair updates.

## Energy and acceptance

Null energy:

```
E(Z) = E_W + E_bar + E_inter
```

Metropolis acceptance (P6 off):

```
a = min(1, exp(-beta * DeltaE))
```

P6 work coupling (if enabled, only on moves that change E_inter):

```
W6 = -eta_drive * DeltaE_inter
DeltaE_eff = DeltaE - W6
```

Per accepted move, micro-EP increments by:

```
Delta_ep_micro = -beta * DeltaE_eff
```

## P3 protocol schedule

When `p3_on=1`, the deterministic cycle is:

1. `SpinFlipColor(0)`
2. `WLocalExchange`
3. `SStep`
4. `KP5Exchange`
5. `KLocalExchange`
6. `NFlip`
7. `SpinFlipColor(1)`
8. `WNeighborExchange`
9. `KNeighborTrade`

Cycle repeats; stroboscopic EP is computed at cycle boundaries.

## Diagnostics

- `ep_micro_rate` (windowed, proposal-normalized): tracks microstep entropy production. In null (P3 off, P6 off) it should be near 0.
- `ep_strobe_rate`: coarse-grained stroboscopic EP from transition counts between binned macrostates; used to detect P3 pumping.

## Implementation notes / deviations from kernel spec

- W budget is enforced globally: `sum_{ell,i,r} W[ell,i,r] == B_w`.
- K budget is enforced as a per-site constant: `sum_r K[ell,i,r] == B_k`.
- `R_K` excludes the zero offset by default (even parity is still enforced).
