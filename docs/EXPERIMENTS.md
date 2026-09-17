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
