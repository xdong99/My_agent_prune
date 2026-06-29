# Surrogate-Guided High-IG Task Generation PoC

## Question

Can a differentiable surrogate trained on EKI/KL graph-parameter IG guide task-spec optimization toward higher verified real IG?

## Setup

- Graph source: `fresh`
- Initial random specs scored: 150
- Initial -> optimized pairs verified: 20
- Output directory: `outputs\surrogate_ig_training_150_soft_numeric_agents3_gpt35_optimized_poc\run_20260602_213844`
- Training-label nonzero fraction: 0.8600
- Training-label mean IG: 2.426549
- Training-label max IG: 26.228912

## Surrogate Metrics

- Train/test split: 120 / 30
- Final train loss: 0.014823
- Final test loss: 0.6122679114341736
- Train predicted-vs-real log1p IG correlation: 0.9984
- Test predicted-vs-real log1p IG correlation: 0.4535
- Test MAE log1p IG: 0.5745
- Test R2 log1p IG: 0.0975

## Real EKI/KL Verification

- Mean real IG before: 2.578888
- Mean real IG after: 9.312434
- Mean delta IG: 6.733546
- Fraction positive delta: 0.7000

## Optimized vs Random Generated Specs

- Optimized mean verified IG: 9.312434
- Random mean verified IG: 2.314036
- Mean difference: 6.998398
- Mean-difference bootstrap 95% CI: [2.244416, 11.947481]
- Optimized nonzero IG fraction: 0.8500
- Random nonzero IG fraction: 0.8500
- Fraction paired optimized > random: 0.7000
- Bootstrap P(mean optimized > mean random): 0.9988

## Pair-Level Results

| Pair | Initial Family | Optimized Family | Real IG Before | Real IG After | Delta |
|---:|---|---|---:|---:|---:|
| 0 | lcm_meeting | lcm_meeting | 0.000000 | 0.000000 | 0.000000 |
| 1 | lcm_meeting | lcm_meeting | 2.712767 | 2.204534 | -0.508233 |
| 2 | defective_rate | defective_rate | 0.293952 | 17.563585 | 17.269633 |
| 3 | lcm_meeting | lcm_meeting | 0.000000 | 3.615049 | 3.615049 |
| 4 | lcm_meeting | lcm_meeting | 0.000000 | 0.000000 | 0.000000 |
| 5 | capacity_ceiling | lcm_meeting | 2.006496 | 3.464750 | 1.458254 |
| 6 | capacity_ceiling | lcm_meeting | 0.934623 | 0.557604 | -0.377019 |
| 7 | lcm_meeting | lcm_meeting | 1.268652 | 4.814927 | 3.546275 |
| 8 | lcm_meeting | lcm_meeting | 1.391077 | 3.202657 | 1.811580 |
| 9 | overlap_count | defective_rate | 0.000000 | 13.898125 | 13.898125 |
| 10 | lcm_meeting | lcm_meeting | 2.750303 | 29.600739 | 26.850435 |
| 11 | average_inverse | lcm_meeting | 0.914150 | 15.018658 | 14.104507 |
| 12 | percent_base | lcm_meeting | 3.406404 | 28.653297 | 25.246893 |
| 13 | percent_base | defective_rate | 5.002140 | 0.786251 | -4.215889 |
| 14 | lcm_meeting | lcm_meeting | 0.000000 | 12.821262 | 12.821262 |
| 15 | lcm_meeting | lcm_meeting | 4.305796 | 8.346210 | 4.040415 |
| 16 | lcm_meeting | lcm_meeting | 11.929798 | 0.000000 | -11.929798 |
| 17 | lcm_meeting | lcm_meeting | 13.197601 | 27.897232 | 14.699631 |
| 18 | defective_rate | lcm_meeting | 1.463992 | 13.385727 | 11.921735 |
| 19 | capacity_ceiling | defective_rate | 0.000000 | 0.418074 | 0.418074 |

## Token Accounting

- Initial generation tokens: 0
- Initial IG scoring tokens: 0
- Verification generation tokens: 42638
- Verification IG scoring tokens: 1624463
- Total run tokens: 1667101

## Interpretation

The verified real EKI/KL results support the PoC claim: surrogate-guided specs achieved higher verified IG than random generated specs.
This result is only about graph-parameter IG. It does not claim final graph accuracy improves.
