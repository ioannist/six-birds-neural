# Null regime (Step 2)

This document describes the Step-2 null-regime implementation aligned with `docs/kernel-spec.md`.

## State variables

Implemented state (per site u = (x, \ell)):

- `sigma_u \in {-1,+1}`
- `n_u \in {-1,+1}`
- `s_u \in {0..l_s}`
- `K_{u,r} \in {0..l_k}` for \ell >= 1 and r \in R_K

Stencils:

- `R_W`: odd-parity offsets (sum(r) mod 2 == 1)
- `R_K`: even-parity offsets (sum(r) mod 2 == 0)

Implementation note: `R_K` excludes the zero offset by default (even parity still holds). If you need the zero offset for inter-layer edges, enable it via `include_zero=True` in `generate_stencil`.

Step-2 implementation note: `B_k` is enforced as a per-site constant budget, so for every upper-layer site u, `sum_r K_{u,r} == B_k`. This is a restricted special case of the kernel spec's per-interface budget.

## Energy terms

The null regime uses a single scalar energy:

```
E(Z) = E_bar(Z) + E_inter(Z)
```

Barrier (P5 as energy barrier only):

```
E_bar(Z) = kappa_T * sum_u s_u * (1 - sigma_u n_u) / 2
```

Inter-layer mismatch (conservative coupling):

```
E_inter(Z) = (eta / 2) * sum_{ell=1..L-1} sum_x (sigma_(x,ell) - sigma_hat_(x,ell))^2
sigma_hat_u = (1 / B_k) * sum_{r in R_K} K_{u,r} * sigma_(x+r, ell-1)
```

P5 affects dynamics only through the barrier energy (no proposal gating).

## Acceptance and epExact

Proposals are symmetric. In null regime (P6 off), Metropolis acceptance is:

```
alpha = min(1, exp(-beta * DeltaE))
```

Per accepted move, epExact increments by:

```
Delta_epExact = -beta * DeltaE
```

Rejected moves contribute 0.
