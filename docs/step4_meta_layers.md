# Step 4 meta-layers and cross-layer operator substrate

This document describes the Step-4 changes that make meta-layers explicit and ensure cross-layer operators are part of the evolving state.

## State tensors and shapes

For L meta layers and spatial size N:

- `sigma`: shape `(L, N)`, values in {-1, +1}
- `n`: shape `(L, N)`, values in {-1, +1}
- `s`: shape `(L, N)`, values in {0..l_s}
- `W`: shape `(L, N, K_W)`, values in {0..l_w}
- `K_cross` (alias of `K`): shape `(L-1, N, K_K)` for interfaces (layer-1 -> layer)

Invariants:

- `sum_{ell,i,r} W[ell,i,r] == B_w` (global)
- `sum_r K_cross[ell-1,i,r] == B_k` for each interface site
- bounds: `0 <= W <= l_w`, `0 <= K_cross <= l_k`

## Cross-layer prediction and mismatch

For each upper layer `ell >= 1`:

```
pred(ell,i) = sum_r (K_cross[ell-1,i,r] / B_k) * sigma(ell-1, i+r)
```

Mismatch per site:

```
mismatch(ell,i) = (sigma(ell,i) - pred(ell,i))^2
```

Summary metric:

```
mismatchMean = mean_{ell=1..L-1,i} mismatch(ell,i)
```

## Energy and drive work

Conservative coupling energy:

```
E_inter = (eta / 2) * sum_{ell=1..L-1,i} mismatch(ell,i)
```

Drive-only mismatch potential (not part of E):

```
Phi_drive = (1/2) * sum_{ell=1..L-1,i} mismatch(ell,i)
```

P6 work for operator updates uses:

```
W6 = -eta_drive * Delta Phi_drive
```

Acceptance remains:

```
a = min(1, exp(-beta * (DeltaE - W6)))
```

## K_cross update kernels

- P1: `k_local_exchange` (within-site exchange between offsets)
- P2: `k_neighbor_trade` (swap/trade between neighboring sites)
- P5: `k_p5_exchange` (a P5-tagged within-site exchange)

All proposals are symmetric; infeasible proposals are self-loops.
