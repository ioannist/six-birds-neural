# PHASE1_NULL_SCALEUP_24x24_REPORT_v1

## Runtime summary

- total_seconds: 1526.215
- seed 1: seconds_used=515.398
- seed 2: seconds_used=505.115
- seed 3: seconds_used=505.702

## Per-seed results

| seed | windows_used | mean_ep | ci_half | acceptedFracWindow | best_ci_half | status |
|---|---|---|---|---|---|---|
| 1 | 3 | -5.787037037037037e-05 | 0.000183411887063107 | 0.5377777777777778 | 0.000183411887063107 | PASS_EARLY |
| 2 | 3 | 2.3148148148148147e-05 | 0.00018341188706310702 | 0.5403645833333334 | 0.00018341188706310702 | PASS_EARLY |
| 3 | 3 | 2.3148148148148147e-05 | 0.00027729270894974066 | 0.5425520833333334 | 0.00027729270894974066 | PASS_EARLY |

## Aggregate

- mean(mean_ep): -3.8580246913580265e-06
- std(mean_ep): 3.819249589742155e-05
- mean(ci_half): 0.00021470549435865155
- max(ci_half): 0.00027729270894974066
- mean(acceptedFracWindow): 0.5402314814814815

## Interpretation for next step

- All 3 seeds PASS_EARLY; proceed to a small 24x24 screen (2–4 configs) or a 200/100 run with resume.

## Progress verification

- validate.csv lines: 4
- validate_progress.csv non-empty: True
- seed jsonl exists: True
