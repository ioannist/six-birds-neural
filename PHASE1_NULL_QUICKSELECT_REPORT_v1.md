# PHASE1_NULL_QUICKSELECT_REPORT_v1

Date: 2025-12-28T10:15:14

## Runtime accounting

- screen stage seconds: 1059.8
- validate stage seconds: 529.4

## Screen results summary (8 runs)

| config_id | pass | windows_used | mean_ep | ci_half | acceptedFracWindow | status | seconds_used |
|---|---|---|---|---|---|---|---|
| rw3_b0.25_J0.50_wf0.05_wn0.25 | true | 3 | 0.00013671874999999999 | 0.001136912208187319 | 0.59578125 | PASS_EARLY | 114.334 |
| rw3_b0.25_J0.50_wf0.05_wn1.00 | true | 3 | -5.208333333333334e-05 | 0.0006144020676711068 | 0.513515625 | PASS_EARLY | 112.686 |
| rw3_b0.25_J1.00_wf0.05_wn0.25 | true | 3 | -6.510416666666667e-05 | 0.00014823786317071528 | 0.5125 | PASS_EARLY | 109.464 |
| rw3_b0.25_J1.00_wf0.05_wn1.00 | true | 4 | -9.114583333333332e-05 | 0.0018033914393326025 | 0.429296875 | PASS_EARLY | 137.94 |
| rw3_b0.50_J0.50_wf0.05_wn0.25 | true | 3 | -6.510416666666667e-05 | 0.00014823786317071528 | 0.5125 | PASS_EARLY | 108.328 |
| rw3_b0.50_J0.50_wf0.05_wn1.00 | true | 4 | -9.114583333333332e-05 | 0.0018033914393326025 | 0.429296875 | PASS_EARLY | 143.14 |
| rw3_b0.50_J1.00_wf0.05_wn0.25 | true | 7 | 0.0002864583333333334 | 0.001376983053260253 | 0.385234375 | PASS_EARLY | 213.588 |
| rw3_b0.50_J1.00_wf0.05_wn1.00 | true | 3 | 0.0005989583333333333 | 0.0014567447916666665 | 0.333359375 | PASS_EARLY | 115.952 |

## Selected winner

- config_id: rw3_b0.25_J1.00_wf0.05_wn0.25
- beta: 0.25
- J: 1.0
- w_fill: 410 / (l_w * layers * N * K_W)
- w_neighbor_weight: 0.25
- sorting keys: pass_rate desc, ci_half_mean asc, windows_used_mean asc

## Validation results (3 seeds)

| seed | pass | windows_used | mean_ep | ci_half | acceptedFracWindow | status | seconds_used |
|---|---|---|---|---|---|---|---|
| 1 | true | 3 | -8.138020833333332e-06 | 0.0005503495554735828 | 0.50302734375 | PASS_EARLY | 185.873 |
| 2 | true | 3 | 0.0 | 0.0013163192251627832 | 0.50810546875 | PASS_EARLY | 170.797 |
| 3 | true | 3 | 7.32421875e-05 | 0.0002643793747963056 | 0.503369140625 | PASS_EARLY | 170.462 |

## Pipeline checklist

- screen_raw.csv non-empty: True
- screen_progress.csv non-empty: True
- preset json written: True
- validate.csv non-empty: True
