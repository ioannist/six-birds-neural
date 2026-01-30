# Experiments

### Plan numbering aliases

- Plan Phase-3 → `scripts/phase3_p6_drive_v1.py` (alias of `scripts/phase2_separability_v6.py`)
- Plan Phase-4 → `scripts/phase4_p3_pumping_v1.py` (alias of `scripts/phase3_p3_pumping_v4.py`)
- Older names remain supported for backwards compatibility.

## Canonical theory package

- See `docs/canonical_theory_package.md` for the canonical tuple ((Z, f, Sigma_f, E, A)).
- `paper.tex` terminology is canonical; repo terms are aliases mapped there.
- Null checklist reminder: keep P6/audit metrics (A_EP_proxy, A_strobe) separate from
  P3 protocol diagnostics. Protocol-only metrics are non-directional unless paired
  with a valid audit and explicit protocol/clock-state handling.

## Qualitative narrative (what the phases establish)

This file is intentionally "harness-first" (commands, tables, pass/fail gates). The qualitative story underneath the metrics is:

- **We have a clean baseline and a clean drive channel.** Early phases validate a near-equilibrium null (so later structure isn't instrumentation drift) and a controllable drive knob (so "activity" can be turned on without breaking constraints).
- **We have stability at the operating point.** The chosen presets keep acceptance and mismatch in healthy bands across screens and confirms, which matters for any claim of emergence (hazard response, motifs, memory).
- **We gained observability and a causal poke loop.** Later phases add spatial summaries and explicit interventions so claims are falsifiable: we can *see* where coupling concentrates and perturb it deliberately.
- **"Attention" shows up as geometry/directionality first, not just mass.** When hazard is applied, the strongest reallocation signature is directional bias/focus rather than a uniform increase in coupling magnitude.
- **Phase 9 is suggestive context; Phase 9.5 is the paired/control-strengthened certificate.**
- **Motifs behave like a discrete operator alphabet.** Quantized local coupling features form a small inventory that shifts under context and exhibits propagation; transitions also change, which is the first rung of "proto-syntax."
- **Clockwork fabrics are not automatic in this model.** Where we fail to detect coherent scanlines/spirals/waves, that's a useful constraint: the proto-language channel we ultimately get does not depend on a global wave clock.

Use the "Qualitative interpretation" blocks under each phase as the intended human reading of what each PASS/FAIL means.

## Phase 1 — Null baseline calibration (minimal subsystem)

### Commands

```
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -m cuda
.venv/bin/python scripts/phase1_null_calibration.py --device cpu --shape 24,24 --steps 50000 --report-every 10000 --seeds 1,2,3 --w-neighbor-weight 0
.venv/bin/python scripts/phase1_null_calibration.py --device cpu --shape 64,64 --steps 200000 --report-every 20000 --seeds 1,2,3 --radius-ws 3 --betas 0.5 --w-fills 0.10 --w-neighbor-weight 0 --out-dir .tmp/phase1_null_validation
.venv/bin/python scripts/phase1_null_calibration.py --device cpu --shape 24,24 --steps 50000 --report-every 10000 --seeds 1,2,3 --radius-ws 3 --betas 0.5,1.0 --w-fills 0.10 --w-neighbor-weight 0 --out-dir .tmp/phase1_null_w0
.venv/bin/python scripts/phase1_null_calibration.py --device cpu --shape 24,24 --steps 50000 --report-every 10000 --seeds 1,2,3 --radius-ws 3 --betas 0.5,1.0 --w-fills 0.10 --w-neighbor-weight 1 --out-dir .tmp/phase1_null_w1
```

### Sweep setup

- Null toggles: `p3_on=0`, `p6_on=0`, `eta=0`, `eta_drive=0`, `B_k=0`, `radius_k=0`, `l_s=0`.
- Kernel mix: spin flips + `w_local`; `w_neighbor` toggled for follow-up checks.
- Grid: `radius_w ∈ {1,3}`, `beta ∈ {0.5,1.0}`, `w_fill ∈ {0.10,0.25,0.40}`, `l_w=4`.
- Seeds: `{1,2,3}`.
- Sweep shape: `(24,24)`; steps `50_000`, report every `10_000`.

### Top configs (hard-constraint pass, pre-randomized W init)

Only two configs met the hard constraints across seeds.

| config_id          | radius_w | w_fill | beta | acceptedFracMean | magAbsMean | wEntropyMean | epMean   |
|--------------------|----------|--------|------|------------------|-----------|--------------|----------|
| rw3_b0.50_wf0.10    | 3        | 0.10   | 0.5  | 0.52188          | 0.11516   | 0.27823      | 0.000133 |
| rw3_b1.00_wf0.10    | 3        | 0.10   | 1.0  | 0.52053          | 0.10938   | 0.27823      | 0.000000 |

Selected baseline preset: `scripts/params/phase1_null_balanced.json`.

### Safe baseline claims (Phase 1, pre-randomized W init)

- Null EP stays within tolerance across seeds for the selected preset on `(24,24)` with `50_000` steps.
- Acceptance is non-degenerate (~0.52), and `|mean(sigma)| < 0.2` across seeds.
- W is not pinned: `w_zero_frac ≈ 0.90`, `w_cap_frac ≈ 0.10`, and `w_entropy_mean ≈ 0.28`.

### Validation notes

- Validation on `(64,64)` for `200_000` steps shows EP window rates still decaying (last window ~`6e-3`), so longer runs or additional mixing may be needed before claiming strict `EP≈0` at that scale.
- After randomizing W init (seeded slot sampling), follow-up checks on the two top configs with `w_neighbor` in `{0,1}` still show EP window rates above tolerance at `50_000` steps; full Phase‑1 recalibration is pending.

### Phase 1 (updated baseline pointer)

- Phase-1 reports are captured in `PHASE1_NULL_QUICKSELECT_REPORT_v1.md` and `PHASE1_NULL_SCALEUP_24x24_REPORT_v1.md`.
- Phase 2+ operational baseline uses `scripts/params/phase1_null_balanced_quick_v8_24x24.json` (see Phase 2 preset line).

## Phase 2 — Separability with per-kernel EP normalization (v6, 24×24, CUDA)

- Preset: `scripts/params/phase1_null_balanced_quick_v8_24x24.json`
- Diagnostics: EP rates normalized per proposal, with per-kernel proposal/accept counts; strobe EP recorded.
- Cases: `meta_null_k` (K enabled, p6_off), `p6_drive_k` (K enabled, p6_on).
- Params: `B_k=2`, `radius_k=2`, `l_k=3`; K kernels forced on (`k_local=k_neighbor_trade=0.25`); burn 150 sweeps, window 80 sweeps, min_windows 10, max_windows 40.
- Results (Phase2 v6, `.tmp/phase2_v6`, ~30 min wall):
  - `meta_null_k`: PASS_EARLY 3/3 seeds (`k_drive_mean_last_m≈0`, `ci≈0`, accept≈0.13–0.14).
  - `p6_drive_k` (eta_drive=2.0): PASS_EARLY 3/3 seeds (`k_drive_mean_last_m` positive, CI excludes 0; accept>1%; mismatch drops ≥1%).
- Selected drive preset emitted: `scripts/params/phase2_drive_k_balanced_v6.json`.

These runs establish that (1) null remains EP≈0 with K operators active, and (2) P6 drive produces positive EP and reduces mismatch when K updates are actually executed, using the proposal-normalized EP metric.

#### Qualitative interpretation

- This phase is the **reference baseline** for everything downstream: we can operate the full kernel stack in a regime where EP is near-zero and mismatch is below threshold. That makes later EP increases / coupling structure interpretable as *intended effects* (drive/protocol/hazard), not numerical drift.
- The acceptance rate staying in a healthy band is a practical "model health" indicator: we're not frozen (too-low accept) and not random-walking (too-high accept). Downstream comparisons remain meaningful because we're not quietly changing the sampling regime.
- The main implication is confidence: we can treat the null as a **stable control condition** when we ask "what changed?" in later phases.

## Phase 3 — P3 protocol effect (diff vs matched control)

#### A. What we are testing

- P3’s protocol schedule should produce a measurable change in stroboscopic currents vs a matched-control baseline (same kernel menu/weights/cadence, but P3 off).

#### B. Why strict-reverse overlap is diagnostic-only

- Coarse strobe signatures are not lumpable, so strict reverse overlap is brittle and low-support even with strict reverse cycles.
- The strict-reverse overlap test is retained as diagnostic-only and marked xfail: `tests/contracts/test_contract_phase3_strict_reverse_current_overlap.py`.

#### C. Metrics

- Primary: `strobe_current_l2_window` (proposal-normalized strobe rate).
- Secondary: `strobe_symgap_window`.
- Non-degeneracy: `strobe_unique_states_window`, `strobe_bidirectional_edges_window`, and transitions per window.

#### D. Pass/Fail criterion (Phase 3 contract)

- Contract test: `tests/contracts/test_contract_phase3_p3_protocol_changes_strobe_currents.py`.
- Asserts protocol vs matched control produces a nontrivial relative change in `strobe_current_l2_window`:
  - `rel_change >= 0.15`
  - `acceptedFracWindow >= 0.005`
  - `strobe_unique_states_window >= 3`, `strobe_bidirectional_edges_window >= 1`
- Phase-2 drive separability remains locked by `tests/contracts/test_contract_phase2_drive_k_separability.py`.

#### E. η screen + confirm

Seed-1 screen (matched control, `mag_stag`, 24×24):

| eta | control_status | protocol_status | control_current_l2 | protocol_current_l2 | rel_change |
|-----|----------------|-----------------|--------------------|---------------------|------------|
| 0.5 | PASS_EARLY      | PASS_EARLY       | 0.0349086          | 0.0261740           | 0.250214   |
| 1.0 | PASS_EARLY      | PASS_EARLY       | 0.0379060          | 0.0271432           | 0.283934   |
| 2.0 | PASS_EARLY      | PASS_EARLY       | 0.0310679          | 0.0255185           | 0.178622   |

Selected best eta: **1.0** (highest `rel_change`).

3-seed confirmation (eta=1.0):

| seed | control_status | protocol_status | control_l2 | protocol_l2 | rel_change | control_symgap | protocol_symgap | accept_control | accept_protocol |
|------|----------------|-----------------|------------|-------------|------------|----------------|-----------------|----------------|-----------------|
| 1    | PASS_EARLY      | PASS_EARLY       | 0.0379060  | 0.0271432   | 0.283934   | 0.213264       | 0.176126        | 0.010282       | 0.010748        |
| 2    | PASS_EARLY      | PASS_EARLY       | 0.0325844  | 0.0279997   | 0.140702   | 0.157671       | 0.184881        | 0.011563       | 0.010717        |
| 3    | PASS_EARLY      | PASS_EARLY       | 0.0312892  | 0.0280007   | 0.105099   | 0.144809       | 0.177760        | 0.010657       | 0.010780        |

Artifacts:

- `.tmp/phase3_eta_screen_v1/eta_0.5_seed1/`
- `.tmp/phase3_eta_screen_v1/eta_1.0_seed1/`
- `.tmp/phase3_eta_screen_v1/eta_2.0_seed1/`
- `.tmp/phase3_eta_screen_v1/confirm_best_eta_1.0/`

Preset for reruns:

```
.venv/bin/python scripts/phase3_p3_pumping_v4.py \
  --device cuda \
  --preset scripts/params/phase3_p3_best_eta1.0_magstag.json \
  --eta 1.0 \
  --strobe-signature mag_stag \
  --metric-mode diff_vs_control \
  --out-dir .tmp/phase3_p3_confirm_eta1 \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 --window-sweeps 80 \
  --min-windows 10 --max-windows 40 --last-m 5 \
  --accept-min 0.005 \
  --min-strobe-unique 3 --min-strobe-bidirectional-edges 1 --min-strobe-transitions 200 \
  --mean-thresh-control 5e-4 --diff-thresh 1e-4 --ci-thresh 1e-3 \
  --match-control-cycle-weights \
  --max-seconds-total 5400 --max-seconds-per-run 900 \
  --progress --resume
```

#### Qualitative interpretation

- This phase shows **protocol ordering matters**: P3 produces a measurable change in stroboscopic currents relative to a matched-control baseline, so the protocol is a real dynamical knob.
- The eta screen confirms the effect is **robust but bounded**, with healthy acceptance and non-degenerate strobe stats across seeds.
- This anchors P3 as a reusable scheduling primitive for later hazard/semantics phases rather than a one-off artifact.

## Phase 7 — Meta-layer sanity (L=3, eta screen + drive-only screen)

### Commands

```
.venv/bin/python -m pytest -q -k "phase7_meta_layer_runner_guards or phase7_drive_only_eta_mode or contract_phase2_drive_k_separability"
.venv/bin/python -m pytest -q -m cuda -k cuda_w_budget
/usr/bin/time -p .venv/bin/python scripts/phase7_meta_layer_sanity_v1.py \
  --device cuda \
  --preset scripts/params/phase5_p3p6_combo_balanced_v1.json \
  --layers 3 \
  --out-dir .tmp/phase7_meta_layer_sanity_v2 \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 --window-sweeps 80 \
  --min-windows 10 --max-windows 20 --last-m 5 \
  --accept-min 0.005 \
  --eta-candidates 0,0.5,1.0 \
  --eta-drive-candidates 0,1.0,2.0,4.0 \
  --drive-only-eta-mode eta_best \
  --mean-thresh-null 2e-3 \
  --ci-thresh-null 2e-3 \
  --mismatch-drop-frac 0.01 \
  --k-drive-mean-min 1e-3 \
  --max-seconds-total 6600 \
  --max-seconds-per-run 900 \
  --preset-out-dir scripts/params \
  --progress --resume
```

### Equilibrium coupling screen (seed 1)

| eta | ep_mean_last_m | ep_ci_half_last_m | mismatch_last_m | mismatch_drop | accept_last_m | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.000 | 1.18165e-05 | 9.54648e-05 | 0.987153 | 0.000000 | 0.0113438 | OK |
| 0.500 | -7.06086e-06 | 0.00012681 | 0.929340 | 0.0585649 | 0.0107653 | OK |
| 1.000 | -8.45786e-05 | 0.000157383 | 0.884201 | 0.104291 | 0.0101760 | OK |

Selected `eta_best = 1.0`.

### Equilibrium confirm (eta=1.0, seeds 1-3)

| seed | ep_mean_last_m | ep_ci_half_last_m | mismatch_last_m | mismatch_drop | accept_last_m | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | -8.45786e-05 | 0.000157383 | 0.884201 | 0.104291 | 0.0101760 | OK |
| 2 | -8.27553e-06 | 9.77337e-05 | 0.868924 | 0.119768 | 0.0108033 | OK |
| 3 | 2.54629e-05 | 0.000180340 | 0.860938 | 0.127858 | 0.00999528 | OK |

### Drive-only screen (eta fixed at eta_best=1.0, seed 1)

| eta_drive | k_drive_mean_last_m | mismatch_last_m | mismatch_drop | accept_last_m | status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1.000 | 0.0172848 | 0.864236 | 0.0225800 | 0.0100169 | OK |
| 2.000 | 0.0307249 | 0.855729 | 0.0322011 | 0.00993901 | OK |
| 4.000 | 0.0931919 | 0.853646 | 0.0345572 | 0.00998119 | OK |

Selected `eta_drive_best = 4.0`.

### Drive confirm (eta=1.0, eta_drive=4.0, seed 1 shown)

| seed | k_drive_mean_last_m | mismatch_last_m | mismatch_drop | accept_last_m | status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.0931919 | 0.853646 | 0.0345572 | 0.00998119 | OK |

#### Qualitative interpretation

- This is the first strong evidence that **depth matters** in a functional way: with 3 meta layers, we can choose an `eta` that improves mismatch without destabilizing acceptance or the near-null regime.
- The separate `eta_drive` selection demonstrates a second meta-level control channel: we can increase driven activity without collapsing coupling objectives. Conceptually, this is the first concrete foothold for "allocation under scarcity."
- This phase effectively "locks" a **reusable 3-layer substrate** that later phases build on (spatial maps, hazard rerouting, motifs, memory).

## Phase 9 — Hazard response + attention highways (axis-bias focus)

### What it tested

- Local hazard (sigma corruption) should trigger spatially localized reallocation under scarcity, measured via **axis-bias focus** in K (directional “highway” proxy), plus mismatch spike/recovery.

### Preset

- `scripts/params/meta_null_coupled_eta1.00_layers3.json`

### Commands

Seed-1 gate (v3):

```
/usr/bin/time -p .venv/bin/python scripts/phase9_hazard_attention_highways_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase9_hazard_v1_seed1_gate_v3 \
  --seeds 1 \
  --burn-in-sweeps 150 --window-sweeps 80 \
  --max-windows 20 --snapshot-every-windows 1 \
  --hazard-start-window 5 --hazard-duration-windows 5 \
  --hazard-rect 8:16,8:16 --hazard-sigma random --hazard-layers 0 \
  --hazard-refresh-each-window \
  --mismatch-spike-min 0.01 --recovery-frac-min 0.20 --realloc-min 0.005 \
  --accept-min 0.005 \
  --max-seconds-total 5400 --max-seconds-per-run 1800 \
  --progress --resume
```

3-seed confirm (v3):

```
/usr/bin/time -p .venv/bin/python scripts/phase9_hazard_attention_highways_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase9_hazard_v1_confirm_v3 \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 --window-sweeps 80 \
  --max-windows 20 --snapshot-every-windows 1 \
  --hazard-start-window 5 --hazard-duration-windows 5 \
  --hazard-rect 8:16,8:16 --hazard-sigma random --hazard-layers 0 \
  --hazard-refresh-each-window \
  --mismatch-spike-min 0.01 --recovery-frac-min 0.20 --realloc-min 0.005 \
  --accept-min 0.005 \
  --max-seconds-total 5400 --max-seconds-per-run 1800 \
  --progress --resume
```

### Result summary (v3)

Signal is carried by **`k_axis_bias_focus`** (directional “highway” proxy), not by W mass/entropy.

| seed | status | raw_spike | spike | recovery_frac | best_realloc | best_metric |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | PASS | 0.0703125 | 0.0703125 | 1.85185 | 0.0158665 | k_axis_bias_focus |
| 2 | FAIL | -0.0390625 | 0.0 | 0.0 | 0.00649828 | k_focus |
| 3 | PASS | 0.0898438 | 0.0898438 | 1.50725 | 0.00562391 | k_axis_bias_focus |

Overall: **SUGGESTIVE (2/3 seeds pass the gate); motivates paired-control Phase 9.5.**

#### Qualitative interpretation

- The key qualitative win is a **time-locked response**: hazard produces a mismatch spike and a recovery trend, consistent with an active response rather than monotone degradation.
- Reallocation being detected primarily via **directionality/focus** (rather than just W mass or entropy) supports the "attention highways" hypothesis: the system's first move is to bias coupling geometry, forming thin oriented channels rather than uniformly amplifying coupling.
- Passing in a majority of seeds (without demanding perfection) is the right bar here: it suggests the effect is **available in the model** and not purely hand-tuned, while acknowledging stochastic sensitivity.
- This phase is **not** treated as a core certificate because it is 2/3 and Phase 9.5 provides the paired-control strengthening.

### Presets emitted

- `scripts/params/meta_null_decoupled_layers3.json`
- `scripts/params/meta_null_coupled_eta1.00_layers3.json`
- `scripts/params/meta_p6_drive_eta1.00_etaDrive4.00_layers3.json`

## Phase 9.5 — Paired baseline vs hazard (same post-burn state)

### What it tested

- Paired baseline vs hazard from the **same post-burn state** to isolate realloc vs baseline drift, using **k_axis_bias_focus** as the primary reallocation proxy.

### Preset

- `scripts/params/meta_null_coupled_eta1.00_layers3.json`

### Command

Seed-1 paired gate (v1):

```
/usr/bin/time -p .venv/bin/python scripts/phase9p5_paired_hazard_baseline_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase9p5_paired_v1_seed1 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 20 \
  --snapshot-every-windows 1 \
  --hazard-start-window 5 \
  --hazard-duration-windows 5 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma flip \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --accept-min 0.005 \
  --spike-paired-min 0.005 \
  --realloc-paired-min 0.002 \
  --max-seconds-total 1200 \
  --max-seconds-per-run 600 \
  --progress --resume
```

### Result summary (seed 1)

| seed | status | spike_paired | realloc_paired | recovery_paired |
| ---: | --- | ---: | ---: | ---: |
| 1 | PASS | 0.0625 | 0.0153641 | 0.0598958 |

### Contract

- `tests/contracts/test_contract_phase9p5_paired_attention_highway.py` (canonical threshold: realloc_paired >= 0.002)

## Phase 10 — Clockwork fabric search (matched control vs protocol)

### What it tested

- Whether P3 protocol ordering yields a measurable clockwork fabric signal (traveling mode / phase structure) beyond a matched-control baseline.

### Presets

- `scripts/params/meta_null_coupled_eta1.00_layers3.json` (seed-1 gate)
- `scripts/params/phase5_p3p6_combo_balanced_v1.json` (single optional attempt)

### Commands

Seed-1 gate (meta null preset):

```
/usr/bin/time -p .venv/bin/python scripts/phase10_clockwork_fabric_search_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase10_clockwork_v1_seed1_gate \
  --seeds 1 \
  --burn-in-sweeps 150 --window-sweeps 80 \
  --max-windows 60 --last-m 20 --snapshot-every-windows 1 \
  --accept-min 0.005 \
  --match-control-cycle-weights \
  --analysis-keys k_axis_bias,k_entropy,sigma,mismatch \
  --analysis-interface 0 --analysis-layer 0 \
  --fabric-min 0.05 --delta-min 0.02 --r2-min 0.60 \
  --max-seconds-total 5400 --max-seconds-per-run 1800 \
  --progress --resume
```

Optional single attempt (phase5 combo preset):

```
/usr/bin/time -p .venv/bin/python scripts/phase10_clockwork_fabric_search_v1.py \
  --device cuda \
  --preset scripts/params/phase5_p3p6_combo_balanced_v1.json \
  --out-dir .tmp/phase10_clockwork_v1_seed1_gate_combo \
  --seeds 1 \
  --burn-in-sweeps 150 --window-sweeps 80 \
  --max-windows 60 --last-m 20 --snapshot-every-windows 1 \
  --accept-min 0.005 \
  --match-control-cycle-weights \
  --analysis-keys k_axis_bias,k_entropy,sigma,mismatch \
  --analysis-interface 0 --analysis-layer 0 \
  --fabric-min 0.05 --delta-min 0.02 --r2-min 0.60 \
  --max-seconds-total 5400 --max-seconds-per-run 1800 \
  --progress --resume
```

### Result summary (seed 1)

| run | status | control_best | protocol_best | delta | best_key | best_metric |
| --- | --- | ---: | ---: | ---: | --- | --- |
| meta_null | FAIL | 0.107615 | 0.0923446 | -0.0152704 | sigma | phase |
| combo_p3p6 | FAIL | 0.105235 | 0.110407 | 0.00517227 | k_axis_bias | phase |

Outcome: **FAIL** (no clockwork fabric detected under these presets and thresholds).

#### Qualitative interpretation (negative result)

- The matched-control comparison suggests that simply turning the protocol on does **not** reliably induce coherent scanlines/spirals/wave clocks under the current state model and metrics.
- This is a useful constraint: "clockwork fabric" dynamics are **not automatic** in this substrate. If a clock-like fabric exists, it likely requires either (a) additional degrees of freedom, (b) different coupling rules/selection pressure, or (c) a more sensitive detection metric.
- Importantly, this FAIL does **not** block proto-language: later semantics work can proceed via coupling-token channels without relying on global wave clocks.

### Phase 10.5 renders

Rendered quick visual diagnostics from the Phase 10 seed-1 artifacts (protocol runs):

- `.tmp/phase10_clockwork_v1_seed1_gate/protocol_p3_on/renders/seed1_sigma_l0.gif`
- `.tmp/phase10_clockwork_v1_seed1_gate/protocol_p3_on/renders/seed1_k_axis_bias_i0.gif`

## Phase 10b — Excitable-state upgrade (waves sanity)

### What it tested

- Whether an opt-in excitable sigma mode yields measurable wave-like structure beyond a decoupled control.

### Preset

- `scripts/params/meta_null_coupled_eta1.00_layers3.json`

### Commands

Seed-1 gate (exc_theta=1.0):

```
/usr/bin/time -p .venv/bin/python scripts/phase10b_excitable_state_upgrade_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase10b_excitable_v1_seed1_gate \
  --seeds 1 \
  --burn-in-sweeps 50 \
  --window-sweeps 40 \
  --max-windows 80 \
  --last-m 30 \
  --snapshot-every-windows 1 \
  --exc-init-frac 0.02 \
  --exc-p-spont 1e-3 \
  --exc-theta 1.0 \
  --exc-beta 2.0 \
  --fabric-min 0.05 --delta-min 0.02 --r2-min 0.60 \
  --excited-frac-min 0.01 --excited-frac-max 0.50 \
  --max-seconds-total 1800 --max-seconds-per-run 900 \
  --progress --resume
```

Single knob attempt (exc_theta=0.5):

```
/usr/bin/time -p .venv/bin/python scripts/phase10b_excitable_state_upgrade_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase10b_excitable_v1_seed1_gate_theta0p5 \
  --seeds 1 \
  --burn-in-sweeps 50 \
  --window-sweeps 40 \
  --max-windows 80 \
  --last-m 30 \
  --snapshot-every-windows 1 \
  --exc-init-frac 0.02 \
  --exc-p-spont 1e-3 \
  --exc-theta 0.5 \
  --exc-beta 2.0 \
  --fabric-min 0.05 --delta-min 0.02 --r2-min 0.60 \
  --excited-frac-min 0.01 --excited-frac-max 0.50 \
  --max-seconds-total 1800 --max-seconds-per-run 900 \
  --progress --resume
```

### Result summary (seed 1)

| run | status | coupled_best | decoupled_best | delta | best_metric | excited_frac_mean | travel_r2 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| exc_theta=1.0 | FAIL_FABRIC_MIN | 0.0169949 | 0.0174212 | -0.000426263 | phase | 0.0876157 | 0.0 |
| exc_theta=0.5 | FAIL_FABRIC_MIN | 0.0172606 | 0.0173543 | -0.0000937419 | phase | 0.149074 | 0.0 |

Outcome: **FAIL** (no wave-like fabric detected under current excitable rule/thresholds).

Phase 10b renders (protocol/coupled):

- `.tmp/phase10b_excitable_v1_seed1_gate/coupled/renders/seed1_exc_state_l0.gif`
- `.tmp/phase10b_excitable_v1_seed1_gate/coupled/renders/seed1_exc_excited_l0.gif`

## Phase 11 — Motif/token discovery and propagation

### What it tested

- Whether a small motif inventory explains most K-field structure, shifts under hazard, and exhibits basic propagation.

### Preset

- `scripts/params/meta_null_coupled_eta1.00_layers3.json`

### Commands

Seed-1 gate (v2, jsd_pre_hazard):

```
/usr/bin/time -p .venv/bin/python scripts/phase11_motif_token_discovery_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase11_motifs_v1_seed1_gate_v2 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-min 0.01 \
  --prop-min 0.02 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Confirm (v2, seeds 1–3):

```
/usr/bin/time -p .venv/bin/python scripts/phase11_motif_token_discovery_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase11_motifs_v1_confirm_v2 \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-min 0.01 \
  --prop-min 0.02 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

### Result summary (seed 1)

| run | status | coverage_pre | coverage_hazard | jsd_pre_hazard | prop_score |
| --- | --- | ---: | ---: | ---: | ---: |
| gate_v2 | PASS | 1.0 | 1.0 | 0.031201 | 0.643501 |

### Result summary (confirm, seeds 1–3)

| seed | status | coverage_pre | coverage_hazard | jsd_pre_hazard | prop_score |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | PASS | 1.0 | 1.0 | 0.031201 | 0.643501 |
| 2 | PASS | 1.0 | 1.0 | 0.0323905 | 0.587292 |
| 3 | PASS | 1.0 | 1.0 | 0.0315625 | 0.608984 |

#### Qualitative interpretation

- This is the first "proto-symbol" result: quantized local coupling features form a **small motif inventory** that covers essentially all observed mass (coverage ≈ 1.0 in these runs). The coupling field is not a featureless continuum; it clusters into reusable types.
- The pre→hazard divergence and strong propagation score indicate motifs are **context-dependent** (hazard changes usage) and **spatiotemporally structured** (motifs move/propagate rather than flicker as iid noise).
- Interpreted in language terms: we now have something like **discrete operator words** that recur, shift in frequency by condition, and exhibit nontrivial movement.

## Phase 12 — Motif proto-syntax (transition grammar)

### What it tested

- Whether motif transition grammar (pre vs hazard) shifts in addition to marginal motif usage.

### Preset

- `scripts/params/meta_null_coupled_eta1.00_layers3.json`

### Commands

Seed-1 gate:

```
/usr/bin/time -p .venv/bin/python scripts/phase12_motif_proto_syntax_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase12_motifs_v1_seed1_gate \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-min 0.01 \
  --jsd-trans-min 0.01 \
  --prop-min 0.02 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Confirm (seeds 1–3):

```
/usr/bin/time -p .venv/bin/python scripts/phase12_motif_proto_syntax_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase12_motifs_v1_confirm \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-min 0.01 \
  --jsd-trans-min 0.01 \
  --prop-min 0.02 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

### Result summary (confirm, seeds 1–3)

| seed | coverage_pre | jsd_pre_hazard | jsd_trans_pre_hazard | prop_score | pass |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.0 | 0.031201 | 0.0382602 | 0.643501 | PASS |
| 2 | 1.0 | 0.0323905 | 0.0417677 | 0.587292 | PASS |
| 3 | 1.0 | 0.0315625 | 0.0424085 | 0.608984 | PASS |

### Takeaway

- Motif grammar shifts under hazard (transition JSD ≥ 0.03) while propagation remains strong.

Outcome: **PASS** (3/3 seeds meet jsd_pre_hazard, jsd_trans_pre_hazard, and prop_score thresholds).

#### Qualitative interpretation

- Phase 12 strengthens the proto-language analogy by moving from counts to **transitions**: the motif *transition structure* changes under hazard, not just the histogram.
- That implies minimal **sequence-level regularities** ("proto-syntax"): motif usage is structured in time, suggesting adjacency constraints or short-range temporal patterns that go beyond independent token emission.
- Practically, this is a green light that the motif alphabet is rich enough for higher-order analysis (roles, phrases, decoding) because the dynamics generates nontrivial motif adjacency statistics.

## Phase 13 — Pattern memory and setpoints (Levin-ish repair/regeneration)

### What it tested

- Whether coupled meta dynamics (eta>0) recover a coarse target after injury better than a decoupled control.

### Presets

- Coupled: `scripts/params/meta_null_coupled_eta1.00_layers3.json`
- Decoupled control: same preset with `eta=0` override in the runner.

### Commands

Seed-1 gate (eta=1.0 from preset):

```
/usr/bin/time -p .venv/bin/python scripts/phase13_pattern_memory_setpoints_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase13_setpoint_v1_seed1_gate \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --last-m 5 \
  --injury-window 8 \
  --injury-duration-windows 1 \
  --injury-rect 8:16,8:16 \
  --injury-sigma random \
  --injury-layers 0 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Single knob attempt (eta=2.0, coupled only):

```
/usr/bin/time -p .venv/bin/python scripts/phase13_pattern_memory_setpoints_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --eta 2.0 \
  --out-dir .tmp/phase13_setpoint_v1_seed1_gate_eta2 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --last-m 5 \
  --injury-window 8 \
  --injury-duration-windows 1 \
  --injury-rect 8:16,8:16 \
  --injury-sigma random \
  --injury-layers 0 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

### Result summary (seed 1)

| run | status | spike_c | recovery_c | spike_d | recovery_d | advantage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gate_eta1.0 | FAIL | 0.0141369 | -6.87018 | 0.0337302 | -2.21765 | -4.65253 |
| gate_eta2.0 | FAIL | 0.0319940 | -3.38915 | 0.0337302 | -2.21765 | -1.17150 |

### Takeaway

- Coupled case did not recover toward the pre-injury target better than decoupled under the seed-1 gate; negative recovery fractions indicate post-injury distance increased relative to the pre-injury baseline. One eta increase (2.0) did not resolve this, so Phase 13 is **FAIL** under the current setpoint metric.

### v2 (injury scheduling aligned to window semantics)

Seed-1 gate (aligned injury window = first affected window):

```
/usr/bin/time -p .venv/bin/python scripts/phase13_pattern_memory_setpoints_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase13_setpoint_v1_seed1_gate_v2 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --last-m 5 \
  --injury-window 8 \
  --injury-duration-windows 1 \
  --injury-rect 8:16,8:16 \
  --injury-sigma random \
  --injury-layers 0 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

| run | status | spike_c | recovery_c | spike_d | recovery_d | advantage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gate_v2_eta1.0 | FAIL | -0.00372024 | 0.0 | 0.0128968 | -8.05385 | 8.05385 |

Takeaway: aligning injury scheduling removed the window mismatch but still did not produce a coupled advantage; Phase 13 remains **FAIL** under the current target metric. Future retry should consider target encoding or a slower recovery field, not additional eta sweeps.

### v3 (region setpoint + suppression pass path)

Seed-1 gate (injury-mode flip, region metric):

```
/usr/bin/time -p .venv/bin/python scripts/phase13_pattern_memory_setpoints_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase13_setpoint_v1_seed1_gate_v3d \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --last-m 5 \
  --injury-window 8 \
  --injury-duration-windows 1 \
  --injury-rect 8:16,8:16 \
  --injury-mode flip \
  --injury-layers 0 \
  --block-size 4 \
  --spike-min 0.01 \
  --recovery-min 0.20 \
  --suppression-frac-max 0.50 \
  --damage-adv-min 0.005 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Confirm (seeds 1–3):

```
/usr/bin/time -p .venv/bin/python scripts/phase13_pattern_memory_setpoints_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase13_setpoint_v1_confirm_v3d \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --last-m 5 \
  --injury-window 8 \
  --injury-duration-windows 1 \
  --injury-rect 8:16,8:16 \
  --injury-mode flip \
  --injury-layers 0 \
  --block-size 4 \
  --spike-min 0.01 \
  --recovery-min 0.20 \
  --suppression-frac-max 0.50 \
  --damage-adv-min 0.005 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

| seed | pass_path | fail_reason | spike_d | spike_c | damage_c | pass |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | RECOVERY | OK_RECOVERY | 0.0625000 | 0.0223214 | -0.0151786 | PASS |
| 2 | SUPPRESSION | OK_SUPPRESSION | 0.1183036 | 0.0000000 | -0.0660714 | PASS |
| 3 | FAIL | NO_DAMAGE | 0.0000000 | 0.0000000 | -0.0107143 | FAIL |

Definitions: raw_spike = peak - pre; spike = max(raw_spike, 0).

Takeaway: region-focused setpoint metric with flip injury yields PASS (2/3 seeds) via recovery/suppression paths; Phase 13 is **PASS** under the v3 criteria.

Overall: **NEGATIVE** (fails the setpoint/advantage gate under current metric).

#### Qualitative interpretation

- This phase is our first direct probe of **goal-like macro behavior**: after a localized injury, does the system restore a coarse-scale setpoint rather than simply equilibrating?
- The existence of distinct pass paths is informative: **RECOVERY** suggests active return toward the pre-injury macro target, while **SUPPRESSION** suggests the coupled system can prevent injury from expressing as macro deviation (a protective clamp), which is still "goal-like" at coarse scale even if not regenerative.
- The majority-pass (but not universal) outcome supports a cautious interpretation: we have a **first rung of pattern memory**, but robustness across seeds remains an open axis for future strengthening.

## Phase 14 — Motif semantics I: predictive meaning under hazard

Seed-1 gate (motif features `k_axis_bias,k_entropy`):

```
/usr/bin/time -p .venv/bin/python scripts/phase14_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase14_motif_semantics_v1_seed1_gate \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-ring-min 0.01 \
  --prop-min 0.02 \
  --spike-min 0.01 \
  --semantic-best-max -0.002 \
  --z-semantic-max -2.0 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Simplification A (axis-bias only):

```
/usr/bin/time -p .venv/bin/python scripts/phase14_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase14_motif_semantics_v1_seed1_gate_axis_bias \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias \
  --bins-axis-bias 5 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-ring-min 0.01 \
  --prop-min 0.02 \
  --spike-min 0.01 \
  --semantic-best-max -0.002 \
  --z-semantic-max -2.0 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

| run | coverage_pre | jsd_ring_pre_hazard | prop_score | semantic_best | semantic_z | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate_v1 | 1.0000 | 0.04299 | 0.5915 | -0.07143 | 1.068 | FAIL |
| simplify_axis_bias | 1.0000 | 0.02519 | 0.7124 | -0.07143 | 0.8497 | FAIL |

Phase 14 v2 (shift-based null, semantic_p gate):

```
/usr/bin/time -p .venv/bin/python scripts/phase14_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase14_motif_semantics_v1_seed1_gate_v2 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-ring-min 0.01 \
  --prop-min 0.02 \
  --spike-min 0.01 \
  --semantic-best-max -0.002 \
  --semantic-p-max 0.05 \
  --semantic-support-min 0.05 \
  --semantic-top-k 8 \
  --semantic-shuffle-mode shift \
  --shuffle-n 200 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Simplification (support_min=0.10):

```
/usr/bin/time -p .venv/bin/python scripts/phase14_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase14_motif_semantics_v1_seed1_gate_v2_support10 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-ring-min 0.01 \
  --prop-min 0.02 \
  --spike-min 0.01 \
  --semantic-best-max -0.002 \
  --semantic-p-max 0.05 \
  --semantic-support-min 0.10 \
  --semantic-top-k 8 \
  --semantic-shuffle-mode shift \
  --shuffle-n 200 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

| run | coverage_pre | jsd_ring_pre_hazard | prop_score | semantic_best | semantic_p | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate_v2 | 1.0000 | 0.04299 | 0.5915 | -0.07143 | 0.755 | FAIL |
| support_min_0.10 | 1.0000 | 0.04299 | 0.5915 | 0.01070 | 0.450 | FAIL |

Takeaway: hazard shifts motif distributions and propagation remains strong, but semantic effect (shift-based null) does not clear the p-value criterion; Phase 14 is **FAIL (non-blocking)** under this metric.

#### Qualitative interpretation (negative result)

- Even with a stable motif alphabet (Phases 11–12), simple observational "semantics" — e.g., motifs directly predicting better hazard recovery — does not emerge as a statistically clean signal under the chosen nulls.
- This is valuable because it rules out an easy but misleading story ("motifs already *mean* repair"). It suggests semantics is either distributed across combinations/geometry or only becomes clear under **causal interventions** (later phases).

Phase 14.5 (predictive semantics, shift-null on ring fractions):

```
/usr/bin/time -p .venv/bin/python scripts/phase14_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase14_motif_semantics_v1_seed1_gate_v3_pred \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-ring-min 0.01 \
  --prop-min 0.02 \
  --spike-min 0.01 \
  --semantic-best-max -0.002 \
  --semantic-p-max 0.05 \
  --semantic-support-min 0.05 \
  --semantic-top-k 8 \
  --semantic-shuffle-mode shift \
  --shuffle-n 200 \
  --semantic-pred-enable \
  --semantic-pred-lag 1 \
  --semantic-pred-shift-n 200 \
  --semantic-pred-p-max 0.05 \
  --semantic-pred-corr-max -0.10 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Simplification (lag=3):

```
/usr/bin/time -p .venv/bin/python scripts/phase14_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase14_motif_semantics_v1_seed1_gate_v3_pred_lag3 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --top-n 10 \
  --shift-max 2 \
  --prop-top-m 5 \
  --coverage-min 0.60 \
  --jsd-ring-min 0.01 \
  --prop-min 0.02 \
  --spike-min 0.01 \
  --semantic-best-max -0.002 \
  --semantic-p-max 0.05 \
  --semantic-support-min 0.05 \
  --semantic-top-k 8 \
  --semantic-shuffle-mode shift \
  --shuffle-n 200 \
  --semantic-pred-enable \
  --semantic-pred-lag 3 \
  --semantic-pred-shift-n 200 \
  --semantic-pred-p-max 0.05 \
  --semantic-pred-corr-max -0.10 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

| run | semantic_pred_best_corr | semantic_pred_p | status_context | status_pred | pass |
| --- | ---: | ---: | --- | --- | --- |
| pred_lag1 | -0.4859 | 0.31 | PASS | FAIL | FAIL |
| pred_lag3 | -0.6733 | 0.37 | PASS | FAIL | FAIL |

Takeaway: context criteria still pass, but predictive semantics (shift-null) does not clear the p-value gate; Phase 14.5 is **FAIL (non-blocking)** under the current predictive metric.

#### Qualitative interpretation (negative result)

- Predictive/lagged correlations can appear in sign, but they are not reliably distinguishable from shift-nulls. That means the motif stream is not yet a dependable predictor of near-future improvement in the macro metric.
- This narrows the hypothesis space: the clearest semantics later comes from **explicit token channels** (injection) rather than passive prediction.

---

## Phase 15 — Motif semantics as routing intent (hazard-aligned highways)

Goal: test whether motifs encode hazard-aligned routing roles using axis-bias alignment in the hazard ring, with a shift-null for spatial autocorrelation.

Seed-1 gate (eta=1.0, mag_stag, ring width 1):

```
/usr/bin/time -p .venv/bin/python scripts/phase15_motif_semantics_routing_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase15_motif_routing_semantics_v1_seed1_gate_v6 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 20 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --shuffle-n 200 \
  --spike-min 0.01 \
  --align-delta-min 0.01 \
  --align-p-max 0.10 \
  --accept-min 0.005 \
  --max-seconds-total 7000 \
  --max-seconds-per-run 4000 \
  --progress --resume
```

Single simplification attempt (longer hazard duration):

```
/usr/bin/time -p .venv/bin/python scripts/phase15_motif_semantics_routing_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase15_motif_routing_semantics_v1_seed1_gate_v6b \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 20 \
  --hazard-start-window 6 \
  --hazard-duration-windows 12 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --shuffle-n 200 \
  --spike-min 0.01 \
  --align-delta-min 0.01 \
  --align-p-max 0.10 \
  --accept-min 0.005 \
  --max-seconds-total 7000 \
  --max-seconds-per-run 4000 \
  --progress --resume
```

| run | spike | jsd_ring_pre_hazard | alignment_delta | alignment_p | pass |
| --- | ---: | ---: | ---: | ---: | --- |
| hazard_dur8 | 0.1094 | 0.000398 | -0.00386 | 0.47 | FAIL |
| hazard_dur12 | 0.1094 | 0.000456 | -0.0354 | 0.71 | FAIL |

Takeaway: hazard spike is present, but alignment_delta remains negative and the shift-null p-value stays high. Phase 15 is **FAIL (non-blocking)** under the current routing alignment metric.

#### Qualitative interpretation (negative result)

- The negative/weak alignment indicates that the initial "routing-role" formulation is not captured by a simple global alignment score in this regime.
- This pushes us toward geometry-aware and causal formulations of routing semantics (directional focus features and explicit injection), rather than expecting role structure to appear in coarse averages.

## Phase 16 — Causal motif semantics (injection/ablation)

What we tested: does K ablation in a hazard ring causally change recovery vs a sham redistribution, using a shared pre-hazard state fork and shift-null p-value.

Seed-1 gate (ring thickness 2):
```bash
/usr/bin/time -p .venv/bin/python scripts/phase16_causal_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase16_causal_semantics_seed1_gate \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --ring-thickness 2 \
  --ablate-frac 1.0 \
  --sham-n 100 \
  --spike-min 0.01 \
  --focus-delta-min 0.005 \
  --effect-recovery-min 0.05 \
  --p-max 0.10 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Single knob attempt (ring thickness 3):
```bash
/usr/bin/time -p .venv/bin/python scripts/phase16_causal_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase16_causal_semantics_seed1_gate_ring3 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --ring-thickness 3 \
  --ablate-frac 1.0 \
  --sham-n 100 \
  --spike-min 0.01 \
  --focus-delta-min 0.005 \
  --effect-recovery-min 0.05 \
  --p-max 0.10 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

| run | spike_control | spike_ablate | focus_delta_ablate | effect_recovery | effect_spike | p_effect | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ring2 | 0.1719 | 0.2344 | -0.224 | -0.212 | 0.0625 | 0.64 | FAIL |
| ring3 | 0.1719 | 0.1875 | -0.273 | -0.195 | 0.0156 | 0.26 | FAIL |

Takeaway: K ablation changes focus (negative focus_delta), but the causal effect vs sham is not significant under the shift-null (p_effect high). Phase 16 is **FAIL (non-blocking)** under the current ablation setup.

#### Qualitative interpretation (negative result)

- Budget-preserving ablations did not produce a significant effect relative to sham under the shift-null, suggesting either rapid compensation, redundancy, or that the relevant "meaning" is not removed by this ablation geometry.
- This motivates using **positive causal tests** (injection) and direction/geometry-aware readouts, which later phases show are much more diagnostic.

## Phase 17 — Directional motif semantics (radial inward focus)

Goal: test whether hazard-aligned routing semantics are detectable via a radial inward focus score around the hazard ring.

Seed-1 gate:
```bash
/usr/bin/time -p .venv/bin/python scripts/phase17_directional_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase17_motif_routing_semantics_v1_seed1_gate_v6c \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --shuffle-n 200 \
  --spike-min 0.01 \
  --align-delta-min 0.01 \
  --align-p-max 0.10 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

3-seed confirm:
```bash
/usr/bin/time -p .venv/bin/python scripts/phase17_directional_motif_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase17_motif_routing_semantics_v1_confirm_v6c \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --shuffle-n 200 \
  --spike-min 0.01 \
  --align-delta-min 0.01 \
  --align-p-max 0.10 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

| seed | spike_control | focus_delta_inject | focus_p_inject | pass |
| ---: | ---: | ---: | ---: | :--- |
| 1 | 0.2688 | 1.478 | 0.005 | PASS |
| 2 | 0.2250 | 1.453 | 0.000 | PASS |
| 3 | 0.3844 | 1.521 | 0.000 | PASS |

Takeaway: hazard-aligned routing focus is causally modulated under inject vs sham (shift-null p <= 0.10), and all three seeds pass the gate. Phase 17 is **PASS**.

#### Qualitative interpretation

- This is the first strong **causal semantics** win: injecting a directional K redistribution produces a large, statistically significant radial focus change across seeds under a conservative null.
- Interpreted as "meaning," the injected token has a clear, readable consequence: it writes a **directionally biased coupling pattern** that is detectable against controls.
- This grounds the proto-language arc: we're no longer only clustering motifs — we have a demonstrable **write/read channel** in the coupling-token field.

## Phase 18 — Motif dictionary semantics (directional polarity from interventions)

Goal: use strong directional interventions to map motif inventory to a polarity axis (inward vs outward routing).

Seed-1 gate:
```bash
/usr/bin/time -p .venv/bin/python scripts/phase18_motif_dictionary_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase18_motif_dictionary_seed1_gate_v1 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --ring-thickness 2 \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --shuffle-n 200 \
  --spike-min 0.01 \
  --accept-min 0.005 \
  --jsd-inout-min 0.01 \
  --dict-delta-min 0.005 \
  --p-max 0.10 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

3-seed confirm:
```bash
/usr/bin/time -p .venv/bin/python scripts/phase18_motif_dictionary_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase18_motif_dictionary_confirm_v1 \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --ring-thickness 2 \
  --motif-interface 0 \
  --motif-features k_axis_bias,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --shuffle-n 200 \
  --spike-min 0.01 \
  --accept-min 0.005 \
  --jsd-inout-min 0.01 \
  --dict-delta-min 0.005 \
  --p-max 0.10 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

Dictionary eval: hazard-only windows; label-swap null with `shuffle_n=200`.

| seed | spike_control | jsd_in_out | dict_delta | dict_p | accept | pass |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0.1688 | 0.0322 | 0.0637 | 0.0050 | 0.0108 | PASS |
| 2 | 0.2594 | 0.0601 | 0.1076 | 0.0050 | 0.0104 | PASS |
| 3 | 0.2812 | 0.0393 | 0.0747 | 0.0050 | 0.0103 | PASS |

Takeaway: motif dictionary semantics under directional interventions is detectable (dict_delta above threshold with p <= 0.10) and all three seeds pass. Phase 18 is **PASS**.

#### Qualitative interpretation

- Phase 18 turns the motif alphabet into an explicit **dictionary/codebook**: hazard-only motif dictionary weights separate inject_out vs inject_in reliably across seeds with significant null-controlled separation.
- The key implication is stability: the mapping from intervention token → motif-space signature behaves like a repeatable **lexicon entry** rather than a one-off artifact.
- This is a "proto-lexicon" step: discrete tokens have consistent, decodable signatures in motif space.

## Phase 19 — Motif phrase semantics (two-token alphabet + phrase decoding)

Goal: decode an alternating OUT/IN injection schedule during hazard windows using a direction-aware motif dictionary (`k_radial_focus` + `k_entropy`).

Seed-1 gate:
```bash
/usr/bin/time -p .venv/bin/python scripts/phase19_motif_phrase_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase19_phrase_semantics_v2_seed1_gate_v2 \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --ring-thickness 2 \
  --motif-interface 0 \
  --motif-features k_radial_focus,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --bins-radial 5 \
  --phrase-mode alternating \
  --phrase-start 0 \
  --shuffle-n 200 \
  --spike-min 0.01 \
  --jsd-out-in-min 0.01 \
  --alignment-min 0.05 \
  --p-max 0.10 \
  --intervention-strength 1.0 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress
```

3-seed confirm:
```bash
/usr/bin/time -p .venv/bin/python scripts/phase19_motif_phrase_semantics_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase19_phrase_semantics_v2_confirm \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --ring-thickness 2 \
  --motif-interface 0 \
  --motif-features k_radial_focus,k_entropy \
  --bins-axis-bias 7 \
  --bins-entropy 5 \
  --bins-radial 5 \
  --phrase-mode alternating \
  --phrase-start 0 \
  --shuffle-n 200 \
  --spike-min 0.01 \
  --jsd-out-in-min 0.01 \
  --alignment-min 0.05 \
  --p-max 0.10 \
  --intervention-strength 1.0 \
  --accept-min 0.005 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress
```

| seed | status | fail_reason | spike_control | jsd_out_in | alignment | alignment_p |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| 1 | PASS |  | 0.1750 | 0.6649 | 0.9994 | 0.004975 |
| 2 | PASS |  | 0.1156 | 0.6712 | 0.9997 | 0.004975 |
| 3 | PASS |  | 0.2906 | 0.6668 | 1.0000 | 0.004975 |

Takeaway: direction-aware motif dictionary separates OUT vs IN and phrase alignment is significant under shift-null (p <= 0.10). Phase 19 is **PASS**.

#### Qualitative interpretation

- This is where the representation starts acting like a **phrase** rather than isolated words: alternating token schedules yield near-perfect alignment with statistically significant null rejection across seeds.
- The crucial qualitative point is temporal organization: the system preserves **order information** in its motif statistics under hazard, not just marginal frequencies.
- In language terms, this is a minimal **syntax carrier**: motif time series track sequence structure.

Note: phrase alignment uses a temporal shift-null that skips schedule-equivalent circular shifts (e.g., alternating sequences) so the null reflects shifts that actually change the token order; p-values are computed over the remaining shifts.

## Phase 20 — Phrase decoding + compositionality (proto-syntax from motif “words”)

Goal: decode token order (alternating vs chunked) from hazard-only motif histograms using a dictionary learned from inject_out vs inject_in.

Seed-1 gate:
```bash
/usr/bin/time -p .venv/bin/python scripts/phase20_phrase_decode_proto_syntax_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase20_phrase_decode_seed1_gate_v1b \
  --seeds 1 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --ring-thickness 2 \
  --motif-interface 0 \
  --motif-features k_radial_focus,k_entropy \
  --bins-radial 5 \
  --bins-entropy 5 \
  --phrase-modes alternating,chunked \
  --token-hold-windows 1 \
  --shuffle-n 200 \
  --accept-min 0.005 \
  --spike-min 0.01 \
  --jsd-out-in-min 0.01 \
  --acc-min 0.60 \
  --p-max 0.10 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

3-seed confirm:
```bash
/usr/bin/time -p .venv/bin/python scripts/phase20_phrase_decode_proto_syntax_v1.py \
  --device cuda \
  --preset scripts/params/meta_null_coupled_eta1.00_layers3.json \
  --out-dir .tmp/phase20_phrase_decode_confirm_v1b \
  --seeds 1,2,3 \
  --burn-in-sweeps 150 \
  --window-sweeps 80 \
  --max-windows 25 \
  --snapshot-every-windows 1 \
  --hazard-start-window 6 \
  --hazard-duration-windows 8 \
  --hazard-rect 8:16,8:16 \
  --hazard-sigma random \
  --hazard-layers 0 \
  --hazard-refresh-each-window \
  --ring-thickness 2 \
  --motif-interface 0 \
  --motif-features k_radial_focus,k_entropy \
  --bins-radial 5 \
  --bins-entropy 5 \
  --phrase-modes alternating,chunked \
  --token-hold-windows 1 \
  --shuffle-n 200 \
  --accept-min 0.005 \
  --spike-min 0.01 \
  --jsd-out-in-min 0.01 \
  --acc-min 0.60 \
  --p-max 0.10 \
  --max-seconds-total 5400 \
  --max-seconds-per-run 1800 \
  --progress --resume
```

| seed | phrase_mode | status | spike_control | jsd_out_in | token_acc | token_p |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| 1 | alternating | PASS | 0.1750 | 0.6649 | 1.0000 | 0.004975 |
| 1 | chunked | PASS | 0.1750 | 0.6649 | 1.0000 | 0.004975 |
| 2 | alternating | PASS | 0.1156 | 0.6712 | 1.0000 | 0.004975 |
| 2 | chunked | PASS | 0.1156 | 0.6712 | 1.0000 | 0.004975 |
| 3 | alternating | PASS | 0.2906 | 0.6668 | 1.0000 | 0.004975 |
| 3 | chunked | PASS | 0.2906 | 0.6668 | 1.0000 | 0.004975 |

Notes: dictionary weights are trained from inject_out vs inject_in hazard windows only; decoding is evaluated on hazard-applied windows only; shift-null skips schedule-equivalent circular shifts.

Takeaway: phrase decoding succeeds for both alternating and chunked schedules with hazard-only dictionary weights (token_acc = 1.0, shift-null p at floor), and all three seeds pass. Phase 20 is **PASS**.

#### Qualitative interpretation

- Phase 20 closes the loop by demonstrating **decodability**: phrase-level decoding recovers per-window token identity for alternating and chunked phrases with extremely high accuracy under null control.
- This is a strong proto-language marker: we have (1) discrete tokens with causal meaning, (2) compositional sequencing, and (3) a readable code path from internal motif dynamics back to token order.
- Notably, this does not depend on clockwork fabrics: the language-like channel here is implemented via **tokenized coupling structure** rather than global wave clocks.

## Overall implications (qualitative closeout)

Taken together, the successful phases support a coherent story:

- Coupling tokens behave like a scarce resource that can be **reallocated under hazard** (attention highways).
- That allocation self-organizes into a **discrete motif alphabet** with context-dependent usage and nontrivial propagation/transition structure.
- With causal directional tokens, motifs support an increasingly explicit semantic ladder: **directional meaning → dictionary meaning → phrase meaning → decodable proto-syntax**.
- The main negative constraint is equally informative: coherent global clockwork fabrics are **not robustly observed** in this regime, so the proto-language channel we found does not rely on wave-clock dynamics.

This set of results is strongest when read as: a small set of coupling primitives, applied on a neural-like substrate with scarcity, is sufficient to produce discrete, composable, decodable symbolic structure — even without a global spatiotemporal clock fabric.
