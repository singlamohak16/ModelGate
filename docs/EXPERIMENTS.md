# Experiments

## Phase 1 — Baselines measured on 2026-09-15

Source and exact fingerprints are recorded in [DATASET.md](DATASET.md). This run
uses the complete pinned IBM CSV with seed 42, 5,634 training rows and 1,409
reference rows. There are 374 positive reference labels and 1,035 negatives.
Both pipelines fit the same 19 inputs and produce 46 transformed features.

Environment: Python 3.12.14, Windows 11, NumPy 2.5.3, pandas 2.3.3,
scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0. Tested dependency versions are
also recorded in `constraints-python312.txt`.

| Reference metric | Logistic Regression | Random Forest |
| --- | ---: | ---: |
| Decision threshold | 0.50 | 0.50 |
| Precision | 0.63517915 | 0.65248227 |
| Recall | 0.52139037 | 0.49197861 |
| F1 | 0.57268722 | 0.56097561 |
| PR-AUC (trapezoidal) | 0.65965537 | 0.63981720 |
| Average precision | 0.66038218 | 0.64058756 |
| Predicted-positive rate | 0.21788502 | 0.20014194 |
| True negatives | 923 | 937 |
| False positives | 112 | 98 |
| False negatives | 179 | 190 |
| True positives | 195 | 184 |

The confusion matrix order is `[[TN, FP], [FN, TP]]`: rows are actual labels and
columns are predicted labels, both ordered `[0, 1]`. The decision rule is
`probability_of_class_1 >= 0.50`.

At this threshold, Logistic Regression recovered 195 of 374 churn cases;
Random Forest recovered 184. The forest had fewer false positives and slightly
higher precision. These measurements do not establish statistical superiority
or business benefit. Both baselines are retained; no threshold was optimized.

## Reproducibility evidence

The preparation command generated the raw CSV, two prepared CSVs, `audit.json`,
and `manifest.json` under ignored `data/`. Training ran twice into ignored
`models/baseline/` and `models/repeat/`.

- Recomputing the split from the raw CSV reproduced both partitions' ID order
  and membership exactly.
- Both runs produced identical metric dictionaries for each classifier.
- Both runs' reference probability arrays matched with `rtol=0, atol=1e-12`.
- Each saved pipeline was reloaded and its probabilities compared with the
  original in-memory pipeline at the same tolerance.
- Neither baseline emitted training warnings, including convergence warnings.
- Local JSON summaries contain full parameters, data provenance, artifact hashes,
  actual timestamps, package versions, and Git state.

The runs occurred before a Phase 1 commit: the recorded parent is
`e71b48b5ce58bc1ec9b79d3626dd6ba8d0b4c2fb` with `dirty=true`. After committing the
phase, a new run can record the exact Phase 1 commit. No later commit or
reproducibility result is claimed here.

## Metrics and limitations

F1 is the harmonic mean of precision and recall. PR-AUC is calculated from
probability rankings using trapezoidal integration. Average precision uses
recall-increment weighting; it is deliberately a separate JSON field. Reference
prevalence is about 26.54%, which supplies context when interpreting PR metrics.

There is one random reference split, no parameter search, no confidence intervals,
and no temporal evaluation. Repeatedly changing settings based on this reference
set would introduce selection bias. Calibration, segment behavior, drift, and
release thresholds have not been evaluated. Dataset-specific sample results
cannot substantiate production ROI or fairness claims.

The four controlled defective scenarios remain Phase 7 work. No contamination,
leakage, drift, or segment-degradation experiment is claimed complete here.

## Phase 2 — Real-data check demonstration on 2026-09-17

Ran `scripts/demo_data_checks.py` on the unchanged prepared Phase 1 split using
the default `DataCheckPolicy`: 5% maximum missingness, zero permitted duplicate
or ID-overlap fraction, non-strict extra columns, and the documented exact
target-like heuristic. The JSON contains 136 individual check results:
135 PASS, 1 WARNING, 0 FAIL. All 136 were evaluated.

| Measurement | Train | Reference |
| --- | ---: | ---: |
| Rows | 5,634 | 1,409 |
| Missing TotalCharges | 10 | 1 |
| Missing TotalCharges percentage | 0.177494% | 0.070972% |
| Exact duplicate rows after first occurrence | 0 | 0 |
| Duplicate complete IDs after first occurrence | 0 | 0 |

ID overlap was zero. Predictor overlap identified 10 matching reference rows
(10/1,409 = 0.709723%), using all 19 predictors but excluding IDs and target.
This produced the one WARNING. A separate pandas merge against deduplicated
training predictor rows independently produced the same count of 10. No input
data was changed in response to this warning.

Different IDs with identical predictor values do not prove either contamination
or independence of the underlying people. The appropriate result is bounded
evidence and an interpretation limit. No target-like two-value relationship was
identified among the checked features; more complex leakage was not tested.

The generated JSON stays local under ignored `reports/generated/`. This is a
library demonstration, not a full audit report or release verdict. The focused
synthetic unit tests are not the four complete controlled scenarios of Phase 7.

## Phase 3 — Model validation on 2026-09-17

Loaded the trusted original Phase 1 pipelines without retraining. Preparation
and artifact hashes were checked, and the same 1,409 reference rows / 374 churn
outcomes were evaluated. No metric limits were configured: six checks per model
were PASS as descriptive/class-support results, not model-quality approvals.

| Model | Precision at 0.5 | Recall | F1 | PR-AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.635179 | 0.521390 | 0.572687 | 0.659655 | 0.137061 |
| Random Forest | 0.652482 | 0.491979 | 0.560976 | 0.639817 | 0.139302 |

PR-AUC is trapezoidal; average precision remains separately reported
(LR 0.660382, RF 0.640588). The classification metrics match Phase 1. Binary
Brier was independently verified against Scikit-learn on both saved models.

### Threshold grid

PPR is predicted-positive rate, shown as a fraction. Counts use all 1,409 rows.
Displayed decimals are rounded; comparisons in the library are not rounded.

| Model | Threshold | Precision | Recall | F1 | PPR | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 0.30 | 0.539326 | 0.770053 | 0.634361 | 0.378992 | 288 | 246 | 789 | 86 |
| LR | 0.40 | 0.577566 | 0.647059 | 0.610340 | 0.297374 | 242 | 177 | 858 | 132 |
| LR | 0.50 | 0.635179 | 0.521390 | 0.572687 | 0.217885 | 195 | 112 | 923 | 179 |
| LR | 0.60 | 0.692308 | 0.385027 | 0.494845 | 0.147622 | 144 | 64 | 971 | 230 |
| LR | 0.70 | 0.835165 | 0.203209 | 0.326882 | 0.064585 | 76 | 15 | 1020 | 298 |
| RF | 0.30 | 0.530214 | 0.727273 | 0.613303 | 0.364088 | 272 | 241 | 794 | 102 |
| RF | 0.40 | 0.587500 | 0.628342 | 0.607235 | 0.283889 | 235 | 165 | 870 | 139 |
| RF | 0.50 | 0.652482 | 0.491979 | 0.560976 | 0.200142 | 184 | 98 | 937 | 190 |
| RF | 0.60 | 0.706522 | 0.347594 | 0.465950 | 0.130589 | 130 | 54 | 981 | 244 |
| RF | 0.70 | 0.732759 | 0.227273 | 0.346939 | 0.082328 | 85 | 31 | 1004 | 289 |

For LR, lowering the threshold from 0.50 to 0.30 caught 93 additional churners
and produced 134 additional false positives. This is a measured tradeoff, not a
business recommendation. No threshold was selected or called optimal.

### Calibration and repeatability

Both models report ten uniform bins with sample counts. LR's (0.8,0.9] bin had
only 8 rows, and (0.9,1.0] was empty; RF's highest bin had only 13 rows. Sparse
bins do not justify strong calibration claims. Nonempty bin means/fractions
were independently compared with Scikit-learn's calibration curve.

Two generated reports (`phase3_model_checks.json` and `phase3_repeat.json`)
were identical, and every check validated through `CheckResult`. Reports remain
local in ignored `reports/generated/`. Brier's small between-model difference
is descriptive; there was no significance test, confidence interval,
recalibration, temporal evaluation or threshold optimization. The Phase 2
predictor-overlap warning still applies; this phase did not change the split.

## Phase 4 — Contract segment evaluation on 2026-09-19

Used the same trusted saved pipelines and unchanged reference partition, with
one decision threshold 0.50 and minimum segment size 50. All three observed
Contract groups met the size minimum and contained both outcome classes.
No segment values were missing: named and eligible coverage were both 100%.

| Contract | Rows | Churn outcomes | Non-churn outcomes | Churn prevalence |
| --- | ---: | ---: | ---: | ---: |
| Month-to-month | 780 | 329 | 451 | 0.421795 |
| One year | 295 | 36 | 259 | 0.122034 |
| Two year | 334 | 9 | 325 | 0.026946 |

| Model | Contract | Precision | Recall | F1 | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic Regression | Month-to-month | 0.635179 | 0.592705 | 0.613208 | 0.694005 |
| Logistic Regression | One year | undefined | 0.000000 | 0.000000 | 0.260381 |
| Logistic Regression | Two year | undefined | 0.000000 | 0.000000 | 0.133427 |
| Random Forest | Month-to-month | 0.652482 | 0.559271 | 0.602291 | 0.672686 |
| Random Forest | One year | undefined | 0.000000 | 0.000000 | 0.235952 |
| Random Forest | Two year | undefined | 0.000000 | 0.000000 | 0.125875 |

PR-AUC is trapezoidal and uses continuous probabilities, so can be defined even
when threshold 0.50 yields no positive predictions. Undefined precision is null
with an explicit reason in JSON, not the numerical value zero.

Confusion matrices use `[[TN,FP],[FN,TP]]`:

- LR month-to-month: `[[339,112],[134,195]]`.
- RF month-to-month: `[[353,98],[145,184]]`.
- Both models, one-year: `[[259,0],[36,0]]`.
- Both models, two-year: `[[325,0],[9,0]]`.

The one-year and two-year recall gaps are -0.521390 relative to LR's overall
recall, and -0.592705 relative to its best segment. For RF, the corresponding
gaps are -0.491979 and -0.559271. These are absolute metric differences, not
relative percentage changes. The month-to-month recall is above pooled recall
by 0.071315 (LR) and 0.067292 (RF). The full JSON also contains precision/F1/
PR-AUC differences, all best ties and candidate counts.

Each model produced **20 PASS, 3 WARNING, 0 FAIL** results (23 checks). Two
warnings are undefined precision in the one-year and two-year groups; the third
warns that precision has only one eligible, defined comparator. A zero best-gap
for that remaining precision candidate is not evidence of peer parity. No
quality minima were configured, so defined zero recall/F1 are descriptive
measurements, not policy failures or quality approvals.

Independent Scikit-learn checks confirmed group confusion matrices, defined
precision/recall/F1 and PR-AUC. Undefined precision was independently checked
using `zero_division=np.nan` and matched the explicit null representation.
`phase4_verified.json` and `phase4_repeat.json` were identical; all 46 check
records validated with `CheckResult`. Prepared data hashes remained unchanged.
The reports stay local in ignored `reports/generated/`.

This evidence motivates inspecting segment performance, not automatically
changing thresholds or claiming unfair treatment. The two-year group has only
nine positives despite meeting the 50-row minimum, and group prevalence differs
substantially. There are no uncertainty intervals or significance tests. No
threshold tuning or retraining occurred; the Phase 2 overlap warning remains.

### Focused synthetic detection test

An invented 100-row fixture has an 80-row perfectly recalled group (40 positives)
and a 20-row group with ten missed positives. Overall recall is 40/50 = 0.80.
With minimum segment size 20 and an illustrative minimum recall of 0.75, the
weak segment fails with recall 0 while the large segment passes with recall 1.
The weak group's overall/best recall gaps are -0.80 / -1.00. Its precision is
undefined, warning rather than pretending to be zero. This is a focused unit
test, not the full controlled-experiment framework reserved for Phase 7.
