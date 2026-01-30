# Canonical theory package (paper.tex terms) for ratchet-neural

This document defines the canonical finite theory package tuple

((Z, f, Sigma_f, E, A))

using the terms from `paper.tex`. Repo-specific terms remain only as aliases.

Canonical aliases used below (non-exhaustive):
- "snapshot" or "diagnostics" = lens output f_snap
- "strobe" or "stroboscopic" = lens output f_strobe and its coarse transition audit
- "epExact" / "ep_micro" = A_EP_proxy (accepted-move EP proxy)
- "mismatch" / "k_axis_bias" / "k_entropy" / "k_radial_focus" = meta-layer lenses (f_meta)

## Explicit tuple block (canonical)

Z (microstate space):
- Z is the finite set of all discrete states with bounded tensors
  (sigma, n, s, W, K) on a fixed lattice and stencil structure.
  Each element z in Z is a specific assignment of those tensors that satisfies
  the invariants and budgets enforced by `State.check_invariants()`.
  (State definition/invariants: `ratchet_gpu/state.py#L12`, `ratchet_gpu/state.py#L58`.)

f (lens family) and Sigma_f (definability):
- f := {f_snap, f_strobe, f_meta}, where each lens is a function from Z (or Z x L
  for diagnostic ledger L) to a finite observation space X.
- Sigma_f is the partition of Z induced by equality of lens outputs.
- Pushforward Q_f for any lens f is:
  (Q_f mu)(x) := sum_{z: f(z)=x} mu(z).
  (Canonical definition applied to lenses below.)

E (completion/packaging endomap):
- Let P be the Markov kernel induced by `run_sim` (kernel mixture when P3 off,
  deterministic protocol cycle when P3 on). (`ratchet_gpu/sim.py#L57`.)
- For any lens f in {f_snap, f_strobe, f_meta} and lag tau:
  E_{tau,f}(mu) := U_f( Q_f( mu P^tau ) ).
- U_f is an explicit prototype lift defined below (Section 4).

A (audit functional):
- A_EP_proxy: accepted-move EP proxy from `_metropolis_accept` + `EPTracker`.
- A_strobe: coarse-grained strobe current/symgap/EP from `StrobeTracker`.
- Other metrics (mismatch, axis-bias focus, etc.) are *proxies* and are not
  audits unless explicitly stated.

---

## 1) Microstate space (Z)

Canonical Z is the discrete finite microstate of the simulator, defined by:
- sigma: per-layer spin/activity field (Ising or excitable) (`State.sigma`).
- n: per-layer template bit (`State.n`).
- s: per-layer barrier/closure integer (`State.s`).
- W: within-layer token counts (`State.W`).
- K (aka K_cross): cross-layer token counts (`State.K` / `State.K_cross`).
- Fixed lattice shape and stencils R_W, R_K.

These are explicitly defined and checked here:
- State tensors and aliases: `ratchet_gpu/state.py#L12`.
- Bounds and budget invariants: `ratchet_gpu/state.py#L58`.
- Initialization with stencils and budgets: `ratchet_gpu/state.py#L92`.
- Parameter bounds and toggles: `ratchet_gpu/params.py#L25`.

Raw simulator vs canonical Z:
- Raw simulator state includes Params, Lattice, stencils (R_W, R_K), and
  color indices. These are *fixed environment structure*, not dynamic
  microstate components. Canonical Z is the finite set of admissible
  (sigma, n, s, W, K) tensors under those fixed parameters.

---

## 2) Lens family (f: Z -> X) and definability (Sigma_f)

We use three canonical lenses that are used throughout the repo.

### 2.1 f_snap (diagnostics/snapshot lens)

Alias: `compute_snapshot` (repo "snapshot" diagnostics).

Definition:
- f_snap maps (state, ep_ledger, accepted_frac) to a finite dictionary of
  diagnostics (EP rates, strobe stats, mismatch, K-proxies, etc.).
- In practice it is a lens on an *extended* microstate Z_diag := Z x L_ep
  where L_ep is the ledger accumulated by `run_sim`.

Observation space X_snap:
- Finite dictionary of keys/values such as `ep_rate_exact_window`,
  `ep_rate_by_kernel_proposal_window`, `strobe_current_l2_window`,
  `mismatch_abs_mean`, `k_entropy_mean`, etc.

Code pointers:
- Snapshot construction: `ratchet_gpu/diagnostics.py#L203`.
- EP windowing and ledger fields: `ratchet_gpu/diagnostics.py#L12` and
  `ratchet_gpu/diagnostics.py#L65`.

Sigma_f for f_snap:
- Partition of Z_diag by equality of all snapshot keys.
- Q_f uses the canonical pushforward:
  (Q_f mu)(x) := sum_{z: f(z)=x} mu(z).

### 2.2 f_strobe (stroboscopic coarse lens)

Alias: strobe signature / strobe bins.

Definition:
- f_strobe maps z in Z to a coarse bin tuple based on sigma and s (and
  optionally W anisotropy), using signature-specific rules.
- Strobe bins are recorded in `StrobeTracker.record_state` and used to build
  coarse transition counts (stroboscopic currents).

Observation space X_strobe:
- Finite tuple of integers (bin ids), e.g.
  - signature "mag_s": (b0, b1, b2)
  - signature "mag_stag": (q(m0), q(ms0), q(m1), q(ms1))
  - signature "mag_wmass": (b0, b1, b2)

Code pointers:
- Strobe tracker + transitions: `ratchet_gpu/ep.py#L48`.
- Coarse bin definition: `ratchet_gpu/ep.py#L206`.
- Strobe recording cadence: `ratchet_gpu/sim.py#L84` and `ratchet_gpu/sim.py#L149`.

Sigma_f for f_strobe:
- Partition of Z by equality of the strobe bin tuple.
- Q_f is the standard pushforward defined above.

### 2.3 f_meta (meta-layer / operator lens)

Alias: spatial maps / meta-layer features (k_axis_bias, k_entropy, mismatch).

Definition:
- f_meta maps z in Z to spatial grids or aggregated summaries derived from
  K, W, and sigma, including k_axis_bias, k_entropy, mismatch, etc.

Observation space X_meta:
- Finite grids (after binning) or scalar summaries (after region-reduction).

Code pointers:
- Spatial map definitions: `ratchet_gpu/spatial.py#L29` through
  `ratchet_gpu/spatial.py#L173` (k_axis_bias, k_entropy, k_radial_focus,
  mismatch, sigma, w_*).

Sigma_f for f_meta:
- Partition of Z by equality of the chosen binned map outputs.
- Q_f is the standard pushforward defined above.

---

## 3) Description space (V)

Canonical description space is the simplex over microstates:
- V := Delta(Z).

In practice, the repo uses empirical summaries rather than full Delta(Z):
- Windowed EP rates and per-kernel rates from `compute_snapshot`.
- Strobe transition counts and currents from `StrobeTracker`.
- Spatial map windows and region summaries from `compute_spatial_maps`.

Code pointers:
- Snapshot value set: `ratchet_gpu/diagnostics.py#L203`.
- Strobe transitions/currents: `ratchet_gpu/ep.py#L48`.
- Spatial maps: `ratchet_gpu/spatial.py#L146`.

---

## 4) Completion/packaging endomap (E)

We instantiate E using the dynamics-induced template from `paper.tex`:

E_{tau,f}(mu) := U_f( Q_f( mu P^tau ) ).

Here P is the Markov kernel induced by `run_sim`:
- P3 OFF: a fixed mixture of kernels with weights (`kernel_weights`).
- P3 ON: a deterministic protocol cycle, time-inhomogeneous unless protocol
  phase is included in the state.
(Code pointers: `ratchet_gpu/sim.py#L57`, `ratchet_gpu/sim.py#L106`,
`ratchet_gpu/sim.py#L125`, `ratchet_gpu/sim.py#L43`.)
Note: the literal power (mu P^tau) is valid only in the time-homogeneous case
(P3 OFF, or P3 ON with phase included in Z). Otherwise, use the time-ordered
product along the cycle: mu P_t P_{t+1} ... P_{t+tau-1}.

### Explicit U_f (prototype lift) for f_strobe

We define U_f explicitly for the strobe lens f_strobe. This is a *canonical
construction* (not computed in the repo) that makes E_{tau,f} well-defined.

Fix a parameter set theta (e.g., from a preset such as
`scripts/params/phase3_p3_best_eta1.0_magstag.json`) and a fixed seed s0.
Let z0 := State.initialize(theta, seed=s0).

For any strobe bin x:
- If signature is "mag_s" and x = (b0, b1, b2):
  1) Convert bins to target means m0, m1, s_bar via bin centers used in
     `_coarse_bin` (same low/high ranges). (`ratchet_gpu/ep.py#L214`.)
  2) Define sigma for each layer as a deterministic pattern with the required
     mean: set the first n_plus sites to +1 and the rest to -1 where
     n_plus := round((m + 1) * N / 2).
  3) Set s to a constant integer round(s_bar) at every site.
  4) Keep W and K from z0 (they already satisfy budgets).

- If signature is "mag_stag" and x = (q(m0), q(ms0), q(m1), q(ms1)):
  1) Convert q-bins back to target sums m and ms via the same scale used
     in `_coarse_bin` (q * scale). (`ratchet_gpu/ep.py#L223`.)
  2) For each layer, solve for sums on color classes:
     S0 = (m + ms)/2, S1 = (m - ms)/2, where S0 is the sum on color-0 sites
     and S1 on color-1 sites. (`ratchet_gpu/ep.py#L226`.)
  3) For each color class with size N0/N1, set the first
     n_plus := round((S + N)/2) sites to +1 and the rest to -1.
  4) Set s and keep W, K as above.

This yields a deterministic prototype z_x for every strobe bin x. Then:
- U_f(delta_x) := delta_{z_x}.
- U_f(nu) := sum_x nu(x) delta_{z_x}.

Idempotence defect:
- The repo does **not** measure idempotence defect of E (no direct E(E(mu))
  diagnostics are computed). This is a conceptual packaging map only.

---

## 5) Audit functional (A)

### A_EP_proxy (accepted-move EP proxy)

Alias: epExact / ep_micro.

Definition (code-level):
- For each accepted move, EP increment is
  Delta_ep = -beta * (DeltaE - W6), where W6 = -eta_drive * DeltaPhi_drive
  when P6 is enabled.
- Rejected moves contribute 0.

Code pointers:
- EP increment rule: `ratchet_gpu/kernels.py#L27`.
- Drive work W6 via delta_phi: `ratchet_gpu/energy.py#L117` and
  `ratchet_gpu/energy.py#L151`.
- EP tracking (accepted-only): `ratchet_gpu/ep.py#L12`.
- Ledger fields exposed to snapshots: `ratchet_gpu/sim.py#L201` and
  `ratchet_gpu/diagnostics.py#L12`.

Limitations (explicit):
- This is **not** the full path-space KL audit Sigma_T(rho) from the
  framework; proposal ratios are assumed symmetric and ignored.
- EP is only accrued on accepted moves, which is a proxy for true
  entropy production in the discrete-time chain.

### A_strobe (coarse stroboscopic audit)

Alias: strobe currents / strobe EP.

Definition (code-level):
- Coarse transition counts between strobe bins define currents, symgap, and
  stroboscopic EP proxy.

Code pointers:
- Transition counts and currents: `ratchet_gpu/ep.py#L60` through
  `ratchet_gpu/ep.py#L203`.
- Reported snapshot keys: `ratchet_gpu/sim.py#L171` and
  `ratchet_gpu/diagnostics.py#L239`.

Limitations:
- These are coarse observational proxies. They are **not** guaranteed to be
  monotone under coarse-graining and do not constitute a formal KL audit.

### Other proxies (not audits)

Examples:
- mismatch_abs_mean, k_axis_bias_focus, k_entropy_*.
- These track structural effects but are not directionality audits without
  additional justification.

Monotonicity note:
- KL-type audits are monotone under coarse-graining (data processing), but
  the proxy metrics used here (strobe L2, symgap, mismatch drop, etc.) do not
  have formal monotonicity guarantees.

---

## 6) Null regime definition (A_NULL alignment)

Canonical null regime (framework):
- P3 OFF and P6 OFF.
- Symmetric proposals with Metropolis acceptance by a conservative energy
  imply detailed balance and near-zero EP.

Repo definition of null (operational):
- Minimal null toggle set from Phase 1:
  p3_on=0, p6_on=0, eta=0, eta_drive=0, B_k=0, radius_k=0, l_s=0
  (`docs/EXPERIMENTS.md#Phase-1` section).
- General null in code: `run_null` sets p3_on=False, p6_on=False and forces
  a reversible kernel mix. (`ratchet_gpu/sim.py#L362`.)

Framework assumptions in null:
- A_AUT (autonomy):
  - Violated if p3_on=True without including protocol phase in state, because
    `run_sim` selects kernels by step index. (`ratchet_gpu/sim.py#L125`.)
  - For p3_on=False, the kernel mixture is time-homogeneous and A_AUT is OK.
- A_REV (microreversibility):
  - In null (p3_off, p6_off), kernels are symmetric and accept by
    exp(-beta DeltaE), yielding detailed balance for the Gibbs measure.
    (`ratchet_gpu/kernels.py#L27`, `ratchet_gpu/energy.py#L49`.)
- A_NULL (zero affinity):
  - Operationally: eta_drive=0 and p6_on=False so W6=0 in the accept rule;
    expected EP is near 0 in equilibrium.

Null evidence (reports/tests):
- Phase 1 reports: `PHASE1_NULL_QUICKSELECT_REPORT_v1.md`,
  `PHASE1_NULL_SCALEUP_24x24_REPORT_v1.md`.
- Contracts/tests:
  - `tests/contracts/test_contract_null_ep.py#L7`.
  - `tests/test_state.py#L79` (CPU null EP near zero).

---

## 7) P1–P6 mapping (paper.tex-consistent)

P1 (operator rewrites / kernels):
- Kernel implementations and selection:
  `ratchet_gpu/kernels.py#L48`, `ratchet_gpu/sim.py#L27`.

P2 (gating/constraints / budgets):
- Global and per-site budgets enforced in state invariants:
  `ratchet_gpu/state.py#L82`.

P3 (protocol holonomy / external schedule):
- Deterministic cycle and step-index scheduling:
  `ratchet_gpu/sim.py#L43`, `ratchet_gpu/sim.py#L125`.
- This is **external protocol** unless protocol phase is explicitly added to Z.
  Therefore P3 metrics alone are not arrow-of-time certificates.

P5 (packaging/closure):
- Barrier energy term E_bar: `ratchet_gpu/energy.py#L71`.
- P5 kernel (alias): `k_p5_exchange` is a tagged exchange
  (`ratchet_gpu/kernels.py#L271`), implemented via `k_local_exchange`
  (`ratchet_gpu/kernels.py#L212`).
- Conceptual packaging endomap E_{tau,f} defined in Section 4.

P6 (audit/accounting / drive):
- Drive work term W6 in acceptance rule: `ratchet_gpu/kernels.py#L27`.
- DeltaPhi for K-updates: `ratchet_gpu/energy.py#L117` and
  `ratchet_gpu/energy.py#L151`.
- EP ledger exposure: `ratchet_gpu/sim.py#L201`.

Critical rule:
- P3-only effects (protocol scheduling) are *not* arrow-of-time evidence unless
  paired with a valid audit A (e.g., A_EP_proxy) and protocol/clock-state handling
  is explicit (A_AUT).

---

## 8) Artifact pointers per claim bundle (Phase 1–20)

Each phase below lists scripts/presets, reports, tests, and metric keys with
code pointers to where those keys are computed.

### Phase 1 - Null baseline calibration

Scripts/presets:
- `scripts/phase1_null_calibration.py` (main driver).
- Preset: `scripts/params/phase1_null_balanced.json`.

Reports:
- `PHASE1_NULL_QUICKSELECT_REPORT_v1.md`.
- `PHASE1_NULL_SCALEUP_24x24_REPORT_v1.md`.
- `docs/EXPERIMENTS.md#Phase-1` section.

Tests/contracts:
- `tests/contracts/test_contract_null_ep.py#L7`.
- `tests/test_state.py#L79`.

Metric keys (code):
- `ep_rate_exact_window`, `acceptedFrac` from snapshots
  (`ratchet_gpu/diagnostics.py#L203`).
- `w_zero_frac`, `w_entropy_mean` from phase1 W metrics
  (`scripts/phase1_null_calibration.py#L37`).

### Phase 2 - Separability with per-kernel EP normalization (v6)

Scripts/presets:
- `scripts/phase2_separability_v6.py`.
- Plan alias: `scripts/phase3_p6_drive_v1.py` (Phase-3 plan numbering).
- Preset: `scripts/params/phase2_drive_k_balanced_v6.json`.

Reports:
- `docs/EXPERIMENTS.md#Phase-2` section (results recorded in narrative).
- Script can emit `PHASE2_V5_REPORT.md` for v5 (not committed).

Tests/contracts:
- `tests/contracts/test_contract_phase2_drive_k_separability.py#L16`.

Metric keys (code):
- `ep_rate_by_kernel_proposal_window` used for k-drive rate
  (`tests/contracts/test_contract_phase2_drive_k_separability.py#L26`,
   `ratchet_gpu/diagnostics.py#L65`).
- `ep_rate_exact_window`, `mismatch_abs_mean` from snapshots
  (`ratchet_gpu/diagnostics.py#L203`).

### Phase 3 - P3 protocol effect (diff vs matched control)

Scripts/presets:
- `scripts/phase3_p3_pumping_v4.py` (primary).
- Preset: `scripts/params/phase3_p3_best_eta1.0_magstag.json`.

Reports:
- `docs/EXPERIMENTS.md#Phase-3` section.
- Script output: `PHASE3_P3_PUMPING_REPORT.md` (in run directory).

Tests/contracts:
- `tests/contracts/test_contract_phase3_p3_protocol_changes_strobe_currents.py#L59`.
- Non-degenerate strobe signature:
  `tests/contracts/test_contract_phase3_strobe_signature_nondegenerate.py#L16`.

Metric keys (code):
- `strobe_current_l2_window`, `strobe_symgap_window`,
  `strobe_unique_states_window`, `strobe_bidirectional_edges_window`
  (`ratchet_gpu/sim.py#L171`, `ratchet_gpu/diagnostics.py#L239`).
- `window_accept_frac` from EP ledger (`ratchet_gpu/sim.py#L201`).

### Phase 4 - Plan alias of Phase 3

Scripts/presets:
- `scripts/phase4_p3_pumping_v1.py` (alias of phase3 script).

Reports/tests/metrics:
- Same as Phase 3 (Phase-4 plan alias only).

### Phase 7 - Meta-layer sanity

Scripts/presets:
- `scripts/phase7_meta_layer_sanity_v1.py`.
- Preset: `scripts/params/phase5_p3p6_combo_balanced_v1.json`.

Reports:
- `docs/EXPERIMENTS.md#Phase-7` section.

Tests/contracts:
- `tests/test_meta_layers.py` (drive-only mismatch + EP checks).

Metric keys (code):
- `ep_rate_exact_window`, `mismatch_abs_mean`, `strobe_transitions_window`,
  `strobe_unique_states_window`, `strobe_bidirectional_edges_window` from
  snapshots (`scripts/phase7_meta_layer_sanity_v1.py#L226`).
- `k_drive_ep_window` from `ep_rate_by_kernel_proposal_window`
  (`scripts/phase7_meta_layer_sanity_v1.py#L109`).

### Phase 9 - Hazard response + attention highways

Scripts/presets:
- `scripts/phase9_hazard_attention_highways_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-9` section.

Tests/contracts:
- `tests/contracts/test_contract_phase9p5_paired_attention_highway.py`.

Metric keys (code):
- Raw CSV header includes:
  `mismatch_abs_mean`, `mismatch_region`, `w_mass_region`, `k_entropy_region`,
  `k_axis_bias_abs_focus`, etc.
  (`scripts/phase9_hazard_attention_highways_v1.py#L258`).
- Spatial maps used: `ratchet_gpu/spatial.py#L146`.

### Phase 9.5 - Paired baseline vs hazard

Scripts/presets:
- `scripts/phase9p5_paired_hazard_baseline_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-9.5` section.

Metric keys (code):
- Raw CSV header includes `mismatch_region`, `k_axis_bias_abs_focus_if0`, etc.
  (`scripts/phase9p5_paired_hazard_baseline_v1.py#L223`).

### Phase 10 - Clockwork fabric search

Scripts/presets:
- `scripts/phase10_clockwork_fabric_search_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-10` section.

Metric keys (code):
- Raw CSV header includes `ep_rate`, `accept_window`, `mismatch_abs_mean`,
  `fabric_score_*` (`scripts/phase10_clockwork_fabric_search_v1.py#L218`).

### Phase 10b - Excitable-state upgrade

Scripts/presets:
- `scripts/phase10b_excitable_state_upgrade_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-10b` section.

Metric keys (code):
- Raw CSV header includes `accept_window`, `excited_frac`, `ep_rate`
  (`scripts/phase10b_excitable_state_upgrade_v1.py#L137`).

### Phase 11 - Motif/token discovery

Scripts/presets:
- `scripts/phase11_motif_token_discovery_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-11` section.

Metric keys (code):
- Raw CSV header includes `motif_entropy`, `topN_coverage`,
  `mismatch_region`, `mismatch_outside`
  (`scripts/phase11_motif_token_discovery_v1.py#L429`).

### Phase 12 - Motif proto-syntax

Scripts/presets:
- `scripts/phase12_motif_proto_syntax_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-12` section.

Metric keys (code):
- Raw CSV header includes `motif_entropy`, `topN_coverage`, `motif_count`,
  `mismatch_region`, `mismatch_outside`
  (`scripts/phase12_motif_proto_syntax_v1.py#L464`).

### Phase 13 - Pattern memory and setpoints

Scripts/presets:
- `scripts/phase13_pattern_memory_setpoints_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-13` section.

Metric keys (code):
- Raw CSV header includes `mismatch_region`, `mismatch_outside`,
  `dist_target_sigma_coarse`, `corr_target_sigma_coarse`
  (`scripts/phase13_pattern_memory_setpoints_v1.py#L235`).

### Phase 14 - Motif semantics I (predictive meaning)

Scripts/presets:
- `scripts/phase14_motif_semantics_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-14` section.

Metric keys (code):
- Raw CSV header includes `motif_entropy`, `topN_coverage`,
  `mismatch_region`, `mismatch_outside`
  (`scripts/phase14_motif_semantics_v1.py#L546`).

### Phase 15 - Motif semantics as routing intent

Scripts/presets:
- `scripts/phase15_motif_semantics_routing_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-15` section.

Metric keys (code):
- Raw CSV fields include `alignment_score`, `k_axis_bias_region`,
  `k_entropy_region`, `mismatch_region`
  (`scripts/phase15_motif_semantics_routing_v1.py#L203`).

### Phase 16 - Causal motif semantics (injection/ablation)

Scripts/presets:
- `scripts/phase16_causal_motif_semantics_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-16` section.

Metric keys (code):
- Summary metrics include `focus_delta`, `spike`, `recovery`, `p_effect`
  (`scripts/phase16_causal_motif_semantics_v1.py#L489`).

### Phase 17 - Directional motif semantics (radial inward focus)

Scripts/presets:
- `scripts/phase17_directional_motif_semantics_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-17` section.

Metric keys (code):
- Raw CSV fields include `radial_focus_ring`, `mismatch_region`,
  `acceptedFracWindow` (`scripts/phase17_directional_motif_semantics_v1.py#L655`).
- Shift-null p-values computed by `radial_focus_shift_null`
  (`ratchet_gpu/semantics.py#L504`).

### Phase 18 - Motif dictionary semantics

Scripts/presets:
- `scripts/phase18_motif_dictionary_semantics_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-18` section.

Metric keys (code):
- Raw CSV fields include `mismatch_region`, `mismatch_outside`,
  `acceptedFracWindow` (`scripts/phase18_motif_dictionary_semantics_v1.py#L340`).

### Phase 19 - Motif phrase semantics

Scripts/presets:
- `scripts/phase19_motif_phrase_semantics_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-19` section.

Metric keys (code):
- Raw CSV fields include `token`, `k_axis_bias_focus`, `radial_ring_mean`,
  `motif_entropy` (`scripts/phase19_motif_phrase_semantics_v1.py#L427`).
- Shift-null correlations via `shift_null_corr`
  (`ratchet_gpu/semantics.py#L540`).

### Phase 20 - Phrase decoding + compositionality

Scripts/presets:
- `scripts/phase20_phrase_decode_proto_syntax_v1.py`.

Reports:
- `docs/EXPERIMENTS.md#Phase-20` section.

Metric keys (code):
- Aggregate metrics include `token_acc`, `token_p`, `jsd_out_in`
  (`scripts/phase20_phrase_decode_proto_syntax_v1.py#L705`).
- Raw CSV fields include `token`, `token_score`, `token_pred`
  (`scripts/phase20_phrase_decode_proto_syntax_v1.py#L722`).
- Shift-null p-values for accuracy via
  `ratchet_gpu/semantics.py#L471`.
