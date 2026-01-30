Below is a **GPU‑exact, parallel, kernel‑level spec** for a *P1–P6‑only* multi‑layer “neural‑tissue” Markov substrate, built around your two hard decisions:

* **P₂ constrains P₁ via hard conserved budgets (token economy)** — no extra penalty functional needed.
* **P₅ couples back only as energy barriers** — never by proposal gating/rate modulation.

It is written so that:

* **Null regime** = `P3=OFF` and `P6=OFF` is **exactly reversible** (detailed balance) and supports `epExact ≈ 0` windowed.
* **P3** introduces noncommuting protocol (can create pumping without any extra objective).
* **P6** introduces antisymmetric work terms (EP) and can maintain “codes/clocks” by paying dissipation.

---

# A. Frozen mathematical spec (state, energy, kernels, schedule)

## A1) Discrete space and layers

* Spatial lattice: (d)-dim torus with side length (G) in each dimension.

  * Coordinates: (x \in \mathbb{Z}_G^d).
  * Flatten index: (i=i(x)\in{0,\dots,N-1}), (N=G^d).
* Layers: (\ell \in {0,\dots,L-1}).
* Site in the full stack: (u=(x,\ell)) with flatten id
  [
  s(u) = \ell N + i(x)\in{0,\dots,LN-1}.
  ]

### Bipartite color (for exact parallel null)

Define color
[
c(u)=\big(\underbrace{\sum_{k=1}^d x_k}_{\text{spatial parity}} + \ell\big)\bmod 2 \in {0,1}.
]

We choose interaction stencils so **every interaction edge connects opposite colors**. This is the backbone that makes the GPU‑parallel null regime exact.

---

## A2) Fixed symmetric stencils

We use two stencils:

### Within‑layer stencil (\mathcal{R}_W)

A fixed set of offsets (r\in\mathbb{Z}^d) such that:

* symmetry: (r\in\mathcal{R}_W \Rightarrow -r\in\mathcal{R}_W),
* **odd parity**: (\sum_k r_k \equiv 1 \pmod 2) (so within‑layer edges cross colors),
* size: (|\mathcal{R}_W|=K_W).

### Inter‑layer stencil (\mathcal{R}_K)

A fixed set of offsets (r\in\mathbb{Z}^d) such that:

* symmetry: (r\in\mathcal{R}_K \Rightarrow -r\in\mathcal{R}_K),
* **even parity**: (\sum_k r_k \equiv 0 \pmod 2) (so cross‑layer edges cross colors),
* size: (|\mathcal{R}_K|=K_K).

Parity is a first‑class constraint: (\mathcal{R}_W) uses odd parity and (\mathcal{R}_K) uses even parity. “Bipartite” is a property that follows for odd‑parity stencils on even‑sized tori, not a separate stencil type.

> This is the clean “no‑bias envelope”: you pick (\mathcal{R}_W,\mathcal{R}_K) by symmetry and capacity only. No feature detection, no semantics.

---

## A3) State variables (all bounded carriers)

For each site (u=(x,\ell)):

### Micro “symbol / activity”

[
\sigma_u \in {-1,+1}.
]

### Template / regime bit (P4)

[
n_u \in {-1,+1}.
]

### Closure / barrier (P5)

[
s_u \in {0,1,\dots, \ell_S}.
]

### Optional local analog field (P2‑like analog)

[
a_u \in {-\ell_A,\dots,+\ell_A}.
]
(You can omit (a) if you want the leanest spec; everything below still works.)

### Within‑layer coupling tokens (P1/P2 economy substrate)

For each (r\in\mathcal{R}*W):
[
W*{u,r}\in{0,1,\dots,\ell_W}.
]
Interpretation: discrete “synaptic capacity tokens” from (u) to neighbor (u+r) in the same layer.

**Global hard budget (economy)**
[
\sum_{u}\sum_{r\in\mathcal{R}*W} W*{u,r} = B_W \quad \text{(conserved)}.
]

Implementation note (Step 3): the current code enforces this global budget directly.

### Cross‑layer operator tokens (meta coupling substrate)

Only for (\ell\ge 1), for each (r\in\mathcal{R}*K):
[
K*{u,r}\in{0,1,\dots,\ell_K}.
]

**Per‑interface hard budget**
For each upper layer (\ell\ge 1):
[
\sum_{x}\sum_{r\in\mathcal{R}*K} K*{(x,\ell),r} = B_{K,\ell}\quad\text{(conserved)}.
]

Implementation note (Step 3): the current code uses a per-site constant budget
`sum_r K_{u,r} = B_k` as a restricted special case.

Define the local token sums
[
B^K_u:=\sum_{r\in\mathcal{R}*K}K*{u,r}.
]
(So (B^K_u) can vary spatially; only the interface total is conserved.)

---

## A4) Energy (single scalar (E(Z)) for null)

Let (Z=(\sigma,n,s,a,W,K)). Energy is:

### Within‑layer alignment energy (directed, but still a state function)

[
E_W(Z) = -J\sum_{u}\sum_{r\in\mathcal{R}*W} W*{u,r};\sigma_u,\sigma_{u+r}.
]

### Closure/template barrier energy (P5 as barrier)

[
E_{\text{bar}}(Z)= \kappa_T \sum_{u} s_u;\frac{1-\sigma_u n_u}{2}.
]
If (\sigma_u=n_u), the term is 0; if mismatch, energy rises proportional to closure (s_u).

### Optional local field energy

[
E_a(Z)= \frac{\lambda_A}{2}\sum_u a_u^2;-;h\sum_u a_u \sigma_u.
]

### Inter‑layer conservative coupling (optional, toggle via (\eta))

For (\ell\ge 1), define linear prediction
[
\widehat{\sigma}*u ;=;
\begin{cases}
\frac{1}{B^K_u}\sum*{r\in\mathcal{R}*K}K*{u,r};\sigma_{(x+r,\ell-1)} & \text{if }B^K_u>0,[6pt]
0 & \text{if }B^K_u=0,
\end{cases}
]
and mismatch energy
[
E_{\text{inter}}(Z)=\frac{\eta}{2}\sum_{\ell=1}^{L-1}\sum_{x}\Big(\sigma_{(x,\ell)}-\widehat{\sigma}_{(x,\ell)}\Big)^2.
]

Total:
[
E(Z)=E_W(Z)+E_{\text{bar}}(Z)+E_a(Z)+E_{\text{inter}}(Z).
]

> Null regime uses **exactly this** scalar (E). No other “objective”.

---

## A5) Move set (P1–P6 as Markov kernels)

All updates are local, bounded, and reversible in null.

### Generic acceptance rule (Metropolis + P6 work)

For any proposed move (Z\to Z'),
[
\alpha(Z\to Z')=\min{1,\exp[-\beta(\Delta E - W_{6})]},
]
where (\Delta E=E(Z')-E(Z)), and (W_6) is an antisymmetric work term:
[
W_6(Z'\to Z)=-W_6(Z\to Z').
]

Null: (P6=OFF \Rightarrow W_6=0).

EP bookkeeping (per accepted move):
[
\Delta \sigma_{\text{EP}} = -\beta(\Delta E - W_6).
]
(Proposal ratios are 0 because proposals are symmetric by construction.)

Implementation note (Step 4): the drive work term for operator updates uses a
separate mismatch potential \Phi_drive, so W6 can act even when eta=0.

---

## A6) Concrete kernels per primitive

### X‑kernel (microstate): **spin flip** (\sigma_u\mapsto -\sigma_u)

* proposal is symmetric (flip).
* uses red/black color to enable parallel independence.

### P1: **within‑site token rearrangement**

* acts on (W_{u,\cdot}) and/or (K_{u,\cdot}) by token exchange inside a site:
  [
  W_{u,r_1}!\downarrow,;W_{u,r_2}!\uparrow
  \quad\text{or}\quad
  K_{u,r_1}!\downarrow,;K_{u,r_2}!\uparrow.
  ]
* preserves local token sum (so P2 economy does not need to intervene).

### P2: **between‑site token exchange (economy diffusion)**

* conserves global budgets but redistributes “capacity” spatially:
  [
  W_{u,r_u}!\uparrow,; W_{v,r_v}!\downarrow
  \quad\text{(same layer, }v=u+\delta\text{ small)}
  ]
  and similarly for (K) on each interface layer (\ell\ge 1).

This is the literal “hard conserved economy” and it *constrains what P1 can do locally*.

### P4: **template flip** (n_u\mapsto -n_u)

* local reversible bit flip.
* creates long‑lived identity when coupled to (s_u) via (E_{\text{bar}}).

(Optional: attach a separate global clock variable here; see GPU section.)

### P5: **closure update** (s_u\mapsto s_u\pm 1)

* local reversible integer walk in ([0,\ell_S]),
* couples back only through (E_{\text{bar}}) (barrier), not gating.

### P3: **protocol**

* when OFF: random mixture of reversible kernels ⇒ null reversible.
* when ON: deterministic noncommuting sequence of kernels ⇒ pumping / currents even with P6 off.

### P6: **work**

* antisymmetric additions (W_6) on selected kernels.
* minimal, bias‑free choice for “maintenance”:

  * for moves that change mismatch energy (E_{\text{inter}}),
    [
    W_6 = -\eta_{\text{drive}};\Delta E_{\text{inter}}.
    ]
    This exactly matches your earlier `etaDrive` logic: it **actively maintains** cross‑layer codes by paying EP, without adding a new optimizer.

---

# B. Kernel‑level spec (exact per‑kernel I/O, proposals, ΔE)

Below I write each kernel exactly as a transition kernel.

Notation:

* (u=(x,\ell)).
* neighbor in same layer: (u+r=(x+r,\ell)).
* lower layer neighbor: ((x+r,\ell-1)).

## B1) X kernel: SpinFlip(color c)

**Domain**: all sites (u) with (c(u)=c).

**Proposal per site**:
[
\sigma_u'=-\sigma_u.
]

**Local ΔE computation** (exact)

1. Within‑layer directed interaction terms involving (\sigma_u):

Outgoing contribution change:
[
\Delta E^{\text{out}}*W(u)=2J,\sigma_u \sum*{r\in\mathcal{R}*W} W*{u,r},\sigma_{u+r}.
]

Incoming (because energy is directed; neighbor’s outgoing terms depend on (\sigma_u)):

For each (r\in\mathcal{R}*W), the neighbor (v=u-r) has an outgoing term (W*{v,r}\sigma_v\sigma_u). Flipping (\sigma_u) changes it by:
[
\Delta E^{\text{in}}*W(u)=2J,\sigma_u \sum*{r\in\mathcal{R}*W} W*{u-r,r},\sigma_{u-r}.
]

So
[
\Delta E_W(u)=\Delta E^{\text{out}}_W(u)+\Delta E^{\text{in}}_W(u).
]

2. Barrier term:
   [
   \Delta E_{\text{bar}}(u)= \kappa_T s_u,\sigma_u n_u.
   ]

3. Optional (a) coupling:
   [
   \Delta E_a(u)=2h,a_u,\sigma_u.
   ]

4. Conservative interlayer mismatch:

* If (\ell\ge 1): self mismatch changes by
  [
  \Delta E_{\text{inter,self}}(u)=2\eta,\sigma_u,\widehat{\sigma}_u.
  ]
* If (\ell\le L-2): this spin participates in predictions of upper sites (w=(x-r,\ell+1)) for every (r\in\mathcal{R}_K). Exactly:

  * For each (r\in\mathcal{R}_K), define upper (w=(x-r,\ell+1)).
  * Its prediction changes by (\Delta \widehat{\sigma}*w = \frac{-2\sigma_u}{B^K_w}K*{w,r}).
  * So its mismatch energy changes by
    [
    \Delta E_{\text{inter,upper}}(w)=\frac{\eta}{2}\Big[(\sigma_w-(\widehat{\sigma}_w+\Delta \widehat{\sigma}_w))^2-(\sigma_w-\widehat{\sigma}_w)^2\Big].
    ]
    Sum all such (w).

Total:
[
\Delta E = \Delta E_W(u)+\Delta E_{\text{bar}}(u)+\Delta E_a(u)+\Delta E_{\text{inter,self}}(u)+\sum_{w}\Delta E_{\text{inter,upper}}(w).
]

**Work term (W_6)**:

* If `P6=OFF`: (W_6=0).
* If `P6=ON`: minimal maintenance choice
  [
  W_6 = -\eta_{\text{drive}},\Delta E_{\text{inter}}^{(\text{affected terms})},
  ]
  where (\Delta E_{\text{inter}}^{(\text{affected terms})}) is exactly the part of (\Delta E) coming from (E_{\text{inter}}). (This guarantees antisymmetry.)

**Acceptance**: (\alpha=\min(1,e^{-\beta(\Delta E-W_6)})).

**Outputs**:

* If accepted: (\sigma_u\leftarrow -\sigma_u), EP accumulators update by (-\beta(\Delta E-W_6)).

---

## B2) P4 kernel: NFlip

**Domain**: all sites (u) (fully parallel; local only).

Proposal:
[
n_u'=-n_u.
]

ΔE: only barrier term changes:
[
\Delta E_{\text{bar}}(u)= \kappa_T s_u,\sigma_u n_u.
]
(no other term depends on (n_u)).

Work term: usually 0 (unless you want P6 to bias identity changes).

Acceptance as above.

---

## B3) P5 kernel: SStep

**Domain**: all sites (u).

Proposal:

* pick (\delta\in{+1,-1}) with prob 1/2.
* propose (s_u'=\text{clip}(s_u+\delta,0,\ell_S)).
  (If clipped equals old value, treat as null proposal.)

ΔE from barrier term:
[
\Delta E_{\text{bar}}(u)=\kappa_T,\delta\cdot\frac{1-\sigma_u n_u}{2}.
]

If you include a stiffness cost (E_s=\frac{\lambda_S}{2}\sum s_u^2), then add
[
\Delta E_s(u)=\frac{\lambda_S}{2}\big((s_u+\delta)^2-s_u^2\big).
]
(If you want the strictest “no extra penalty” reading, set (\lambda_S=0). The bound (\ell_S) still keeps it finite.)

Work term: 0 by default.

---

## B4) P1 kernel: WLocalExchange

**Domain**: all sites (u) (parallel, independent).

Proposal:

* choose (r_1\neq r_2) uniformly from (\mathcal{R}_W).
* if (W_{u,r_1}>0) and (W_{u,r_2}<\ell_W):
  [
  W_{u,r_1}'=W_{u,r_1}-1,\quad W_{u,r_2}'=W_{u,r_2}+1.
  ]
  else reject.

ΔE affects only (E_W) outgoing terms at (u):
[
\Delta E_W(u)= -J\Big[(\sigma_u\sigma_{u+r_2})-(\sigma_u\sigma_{u+r_1})\Big].
]
(If you include convex costs on (W), add them here; but you don’t need them for the pure budget economy.)

Work term (optional, P6):

* “resource bias” version:
  [
  W_6 = \mu_u \cdot \Big[(\sigma_u\sigma_{u+r_2})-(\sigma_u\sigma_{u+r_1})\Big]
  ]
  with (\mu_u) a symmetric zero‑mean field (hashed from (u)).
* “maintenance” version: usually none for WLocal.

---

## B5) P2 kernel: WNeighborExchange (economy diffusion)

This is the **hard‑budget, locality‑respecting “economy”** move.

**Domain**: disjoint pairs ((u,v)) in the same layer with (v=u+\delta) for a small (\delta) (e.g. (\delta=e_0)). Pairing is chosen so no site appears twice in the same launch (details in GPU section).

Proposal per pair:

* choose (r_u\in\mathcal{R}_W) and (r_v\in\mathcal{R}_W) uniformly.
* if (W_{v,r_v}>0) and (W_{u,r_u}<\ell_W):
  [
  W_{u,r_u}'=W_{u,r_u}+1,\quad W_{v,r_v}'=W_{v,r_v}-1.
  ]
  else reject.

This conserves (\sum_{u,r} W_{u,r}) exactly and moves budget locally.

ΔE involves outgoing energy at (u) and at (v):
[
\Delta E_W = -J\Big[\sigma_u\sigma_{u+r_u}-\sigma_v\sigma_{v+r_v}\Big].
]

Work term: optional P6 bias, same antisymmetric form.

---

## B6) P1/P2 for K tokens (cross‑layer operators)

Identical structure to WLocal/WNeighbor but:

* domain: only sites (u) with (\ell\ge 1),
* tokens: (K_{u,r}),
* ΔE computed from mismatch energy at site (u) only:
  [
  E_{\text{inter}}(u)=\frac{\eta}{2}(\sigma_u-\widehat{\sigma}_u)^2.
  ]

Local rearrangement (K_{u,r_1}\downarrow, K_{u,r_2}\uparrow) changes (\widehat{\sigma}*u) by
[
\Delta \widehat{\sigma}*u=
\frac{1}{B^K_u}\Big[\sigma*{(x+r_2,\ell-1)}-\sigma*{(x+r_1,\ell-1)}\Big]
\quad\text{(when }B^K_u \text{ constant; otherwise adjust exactly)}.
]

Then
[
\Delta E_{\text{inter}}(u)=\frac{\eta}{2}\Big[(\sigma_u-(\widehat{\sigma}_u+\Delta\widehat{\sigma}_u))^2-(\sigma_u-\widehat{\sigma}_u)^2\Big].
]

**P6 maintenance work (the clean version):**
[
W_6=-\eta_{\text{drive}};\Delta E_{\text{inter}}(u).
]
This is exactly your “drive‑only code maintenance costs EP” mechanism, now fully operator‑lifted.

---

## B7) P3 protocol schedule

### Null mode (P3 OFF)

Each microstep draws one kernel (m) from fixed probabilities (p(m)) (state‑independent):

* SpinFlip(color chosen uniformly)
* NFlip
* SStep
* WLocalExchange
* WNeighborExchange(direction chosen uniformly)
* KLocalExchange
* KNeighborExchange(direction chosen uniformly)

Because this is a **convex mixture** of reversible kernels (when P6 off), the null regime is reversible.

### Protocol mode (P3 ON)

Use a fixed periodic sequence, e.g. length (T):

[
\text{Cycle} = [
\text{SpinFlip}(c=0),;
\text{WLocal},;
\text{SStep},;
\text{NFlip},;
\text{KLocal},;
\text{WNeighbor}(+\hat e_0),;
\text{SpinFlip}(c=1),;
\text{KNeighbor}(+\hat e_1)
].
]

Any noncommuting sequence is acceptable; this is just a concrete frozen choice.

---

# C. GPU‑exact parallel version of the same spec

This section is *the same math*, but now expressed as **exact GPU kernels**, with conflict‑free parallelism and deterministic RNG.

## C1) Data layout (device arrays)

Let (S = LN) sites.

Recommended storage types:

* (\sigma): `int8` (values (\pm 1))
* (n): `int8` ((\pm 1))
* (s): `uint8`
* (a): `int16` (if used)
* (W): `uint8` length (S\times K_W)
* (K): `uint8` length ((L-1)N\times K_K) (store only for upper layers)

Flattening:

* site id: `sid = layer*N + i`.
* for W: `W[sid*K_W + k]`.
* for K: store per upper-layer site `uid = (layer-1)*N + i` and `K[uid*K_K + k]`.

Store stencils:

* `R_W[k][d]` and `R_K[k][d]` in constant memory.
* Also store “linear stride” array `stride[dim]`.

Wrap rule:

* torus: `(x + dx) mod G`.
* for GPU speed, choose `G = 2^m` so mod is `& (G-1)`.

## C2) Deterministic RNG per thread

Use a counter‑based RNG:

* Key: `seed`.
* Counter: `(global_step, kernel_id, thread_global_id, subcall_idx)`.

This guarantees:

* deterministic results independent of block scheduling,
* no stateful RNG arrays needed.

## C3) Kernel launch granularity

Each microstep = exactly **one kernel launch** (or a small fixed list for P3 cycle).

When `P3 OFF` (null mode):

* draw `kernel_id` on CPU (or a 1‑thread GPU kernel) using a single RNG stream,
* launch the corresponding kernel.

When `P3 ON`:

* launch kernels in the predetermined fixed order.

## C4) Exact parallel kernels

### (1) SpinFlipColor kernel

**Launch**: 1 thread per site `sid` with `color(sid)==c`.

Compute `color(sid)` from `(x, layer)` parity:

* if you do `G=2^m`, parity of coordinate sum can be computed from bit extraction of `i`; or simplest: precompute `parity[i]` array of size N and use `parity[i]^ (layer&1)`.

Each thread:

1. Read local `sigma, n, s, a`.

2. Compute:

   * outgoing sum over `k=0..K_W-1`:

     * neighbor sid_out = `(layer*N + idx(x + R_W[k]))`
     * `sum_out += W[sid,k] * sigma[neighbor]`
   * incoming sum:

     * neighbor sid_in = `(layer*N + idx(x - R_W[k]))`
     * `sum_in += W[sid_in,k] * sigma[neighbor sid_in]`

3. ΔE_W = `2*J*sigma_u*(sum_out + sum_in)`.

4. ΔE_bar = `kappa_T * s_u * sigma_u * n_u`.

5. ΔE_a = `2*h*a_u*sigma_u` if used.

6. If conservative interlayer coupling enabled (`eta>0`):

   * compute pred at self if `layer>0`:

     * loop k over `R_K`, read `K[uid,k]` and sigma at lower neighbors
   * compute ΔE_inter_self
   * compute contributions to upper sites if `layer < L-1` by looping `r` and accessing those upper `w` (expensive but exact).

7. ΔE = sum.

8. Work term:

   * if `P6 OFF`: 0.
   * if `P6 ON`: set `W6 = -etaDrive * ΔE_inter_terms` (maintenance) OR 0 if you want P6 only on K updates.

9. Accept:

   * generate `u ~ U(0,1)`,
   * accept if `log(u) < -beta*(ΔE - W6)`.

10. If accepted:

    * write `sigma[sid] = -sigma[sid]`.

11. EP accounting:

    * each thread computes `ep = -beta*(ΔE-W6)` if accepted else 0.
    * block reduce in shared memory, atomicAdd to global `epExactTotal` and `epByMove[SPIN]`.

> This is GPU‑exact and preserves null reversibility because updates are on one color class at a time and all interactions cross colors.

---

### (2) NFlip kernel (P4)

**Launch**: 1 thread per site `sid` (all layers).

Thread:

* propose `n'=-n`.
* ΔE = `kappa_T*s*sigma*n`.
* accept with Metropolis (and optional work if you want).
* if accepted, flip `n`.

No conflicts, fully parallel.

---

### (3) SStep kernel (P5)

**Launch**: 1 thread per site.

Thread:

* sample δ ∈ {+1,-1}.
* propose `s' = clamp(s+δ, 0, lS)`.
* compute ΔE_bar = `kappa_T*δ*(1 - sigma*n)/2`.
* plus ΔE_s if you include `λ_S`.
* accept.
* update `s` if accepted.

No conflicts.

---

### (4) WLocalExchange kernel (P1)

**Launch**: 1 thread per site.

Thread:

* choose k1 != k2 uniformly from `[0,K_W)`.
* if `W[sid,k1]>0 && W[sid,k2]<lW`:

  * ΔE = `-J*(sigma_u*sigma[u+r2] - sigma_u*sigma[u+r1])`.
  * plus any optional convex W cost if you add it.
  * accept with optional work.
  * apply token transfer.

No conflicts.

---

### (5) WNeighborExchange kernel (P2 economy diffusion)

We must avoid a node being updated twice in one launch.

**Pairing choice (conflict‑free):**
Pick one spatial dimension `dim` and set `delta = +e_dim` (unit step).
Pair sites by a “stripe” criterion:

* for each site, compute `x_dim`.
* only threads where `x_dim % 2 == phase` act as “left endpoints”.
* pair `u = (x,layer)` with `v=(x+e_dim, layer)`.

This partitions each layer into disjoint pairs.

**Launch**: 1 thread per pair endpoint `(sid_left)`.

Thread:

* determine `sid_u` and `sid_v`.
* choose `ku, kv` uniformly in `[0,K_W)`.
* if `W[v,kv]>0 && W[u,ku]<lW`:

  * propose `W[u,ku]++`, `W[v,kv]--`.
  * ΔE = `-J*(sigma_u*sigma[u+ru] - sigma_v*sigma[v+rv])`.
  * accept and apply.
* EP accumulate as above.

This exactly conserves global W budget, and is strictly local.

---

### (6) KLocalExchange and KNeighborExchange (meta coupling substrate)

Identical to WLocal/WNeighbor, but run only for `layer>=1` and using `K` arrays and ΔE from mismatch energy at that site.

KLocalExchange:

* 1 thread per upper site.
* choose k1,k2 in `[0,K_K)`, exchange one token.
* recompute `pred` for that site and ΔE_inter.
* accept with optional work:

  * maintenance: `W6 = -etaDrive * ΔE_inter`.

KNeighborExchange:

* disjoint pair schedule within the same upper layer (same pairing logic as WNeighborExchange).

---

## C5) How P3 and P6 map to GPU control flow

### P3 OFF (null mixing)

At each microstep:

1. Sample `kernel_id` from a fixed categorical distribution over enabled kernels.
2. If kernel requires extra choices, sample them too:

   * SpinFlip: sample `color`.
   * NeighborExchange: sample `dim` and `phase`.
3. Launch exactly that one kernel.

Because this is a convex mixture of reversible kernels (when P6 off), null is reversible.

### P3 ON (protocol)

At each protocol tick:

* run a fixed predeclared sequence of kernel launches, e.g.:

```
SpinFlipColor(c=0)
WLocalExchange
SStep
NFlip
KLocalExchange
WNeighborExchange(dim=0, phase=0)
SpinFlipColor(c=1)
KNeighborExchange(dim=1, phase=1)
...
(repeat)
```

Noncommutativity is now explicit and measurable as pumped currents / loop areas even with P6 off.

### P6 ON (work)

Inside each kernel, for accepted moves, include the chosen antisymmetric work rule:

* simplest and most principled:

  * only K moves (and optionally σ moves) get
    [
    W_6 = -\eta_{\text{drive}}\Delta E_{\text{inter}}.
    ]
* you may also add independent clock work later, but that’s optional and should be a separate move type.

---

# D. What this guarantees (so you can audit “no baked‑in bias”)

## D1) Null regime exactness

When `P3=OFF` and `P6=OFF`:

* Every launched kernel is reversible w.r.t. (\pi(Z)\propto e^{-\beta E(Z)}) on the budget manifold because:

  * proposals are symmetric,
  * acceptance is Metropolis with the single scalar energy,
  * parallelization is done on disjoint independent sets (colors or disjoint pairs).

So:

* stationary measure exists and is exactly (\pi),
* no systematic EP: `epExact → 0` windowed.

## D2) Economy/weakness are implemented “the same way”

* **Economy**: finite hard budgets (B_W), (B_{K,\ell}) and caps (\ell_W,\ell_K).
* **Weakness‑like**: no site can create infinite leverage because tokens are bounded and conserved; no coupling can exceed caps; and economy diffusion is local.

No external optimizer, no “distance penalty,” no feature detector.

---

# E. Minimal “frozen knobs” you should literally hard‑declare

To avoid drift/ambiguity later, freeze these as constants in the spec:

* (d), (G) (and recommend (G=2^m) for GPU wrap)
* (L)
* stencils (\mathcal{R}_W,\mathcal{R}_K) explicitly listed
* caps (\ell_W,\ell_K,\ell_S,\ell_A)
* budgets (B_W) and (B_{K,\ell})
* physical parameters (J,\kappa_T,\beta) and optional (h,\lambda_A,\eta)
* drive parameters (\eta_{\text{drive}})
* P3 protocol cycle list (kernel sequence)

---

If you want one more layer of “GPU‑exactness”, the next thing to freeze would be **the precise distribution (p(m))** used in P3‑OFF (null mixing), because that’s part of the Markov kernel. But what I wrote above is already a fully specified family once you pick those probabilities.
