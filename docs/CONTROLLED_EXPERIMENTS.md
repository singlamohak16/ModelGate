# Phase 7: controlled experiments

The question is not just "does the command run?", but "does a deliberately
introduced defect produce the correct measurement and rule result?" This suite
is a software control, not a benchmark of real-world model quality or drift.

## Run

From the repository root with the editable package installed:

```powershell
.venv\Scripts\python scripts/run_experiments.py --output-dir artifacts/phase7
.venv\Scripts\python scripts/run_experiments.py --output-dir artifacts/phase7-repeat
```

Both use seed 42. An explicit `--seed` accepts integers from 0 to 2**32 - 1.
Existing output directories, including empty ones, are refused. A failed run may
leave partial evidence; inspect it and use a new destination rather than delete
or overwrite it. No network access, IBM data download or existing model is needed.
The default output is inside ignored `artifacts/`; if you choose a different
location, make sure it is ignored before committing anything.

The script fits synthetic baselines, then invokes
`python -m modelgate validate --config ...` in ten separate subprocesses using the
same Python environment. Each subprocess has a 120-second timeout. This verifies
the actual CLI and process exits, rather than reproducing validation logic in
the experiment runner. No new package dependency is introduced.

## Synthetic fixture design

Training and reference each have 1,000 rows, disjoint prefixed IDs and balanced
outcomes. Month-to-month has 900 rows (450 per outcome); One year has 100 rows
(50 per outcome). Independent RNG streams generated from the seed determine
continuous signal magnitudes and row permutations.

Tenure is `30 + sign * magnitude`, where sign is -1 for target 0 and +1 for
target 1, and magnitude is uniform in [10,20). Thus the two classes have a wide
gap around 30. The other two numerical features and non-segment categorical
features are constants; Contract is balanced with respect to the target.
Both training/reference rows are shuffled deterministically.

These are **invented Telco-shaped rows, not IBM records or simulated customers**.
Fractional tenure, constant fields and category `constant` are intentional. Names
allow reuse of the unchanged Phase 1 preprocessing and LR/RF builders. Synthetic
F1 = 1 does not replace or improve the recorded real Telco baseline results.

Each pipeline fits only the clean training features, excluding target and IDs.
It is saved once and its SHA256 is pinned in all five corresponding run configs.
Validation never refits it. A continuous predictive signal is not detected by
the exact two-valued leakage heuristic; perfect predictability alone cannot
establish leakage provenance.

## Fixed policy and expected outcomes

Decision threshold is 0.5; minimum overall F1 is 0.85; minimum segment recall is
0.8 with minimum group size 50. Drift limits are 0.2 for numerical KS, categorical
distance and prediction KS, with minimum sample size 50. Other earlier defaults
remain unchanged. These are predeclared illustrative controls, not statistically
calibrated production thresholds. No seed search or result-based threshold tuning
is performed. All scenarios use the same policy; only input/output paths and
current-label availability vary.

| Scenario | Mutation | Required observation |
| --- | --- | --- |
| Clean | None; labeled current is an exact reference copy | No warnings/failures; F1/segment recalls 1, zero feature/prediction drift and five labeled metric changes |
| Contamination | Replace reference positions 0–19 with complete training positions 0–19, including IDs | `overlap.identifiers` FAIL, count 20, denominator 1,000, fraction 0.02; predictor overlap also counts 20 |
| Target leakage | Append `injected_target_copy = Churn` to reference only | Exact-copy heuristic WARNING with coverage 1; feature excluded from model predictors; F1 unchanged |
| Feature drift | Unlabeled copy of reference, tenure increased by 100 | Tenure KS 1 and WARNING, other 18 feature distances zero; measured prediction drift and no labeled current claims |
| Segment degradation | Reflect tenure as `60 - tenure` only for One year rows | One year recall 0 and FAIL; Month-to-month recall 1; overall F1 0.90 still passes 0.85 |

Contamination replaces rather than appends rows, so the denominator stays 1,000
and reference IDs remain unique. The manifest records the precise positions and
copied IDs. Leakage adds an audit-only extra column; it also produces a schema
extra-column warning. It is not passed to the model, so this experiment does not
claim to demonstrate inflated evaluation metrics. A separate test verifies that
explicitly prohibiting that column produces a FAIL as well as the heuristic warning.

The drift current copy has labels removed and `current_labeled: false`.
Numerical KS needs no labels. Constant unchanged features have zero distance;
these are controls, not proof of real population stability. Prediction drift is
model-dependent: the recorded LR prediction KS is 1.0 and RF prediction KS is
0.5. Both exceed 0.2, but the common assertion requires evaluated, bounded
prediction evidence rather than forcing two estimators to have identical scores.
The target-like heuristic is unevaluated/WARNING on explicitly unlabeled current
data, accounting for an additional warning in that scenario.

Segment degradation leaves IDs, labels, segment membership and all other features
unchanged. It reverses the clean signal for exactly 100 rows, not model outputs.
The affected group has confusion matrix `[[0,50],[50,0]]`; the full reference
matrix is `[[450,50],[50,450]]`. Overall F1/recall/precision remain 0.90, while
group recall is 0. The recall gap is -0.90 from overall and -1.00 from the other
segment. This is deliberately constructed evidence of aggregation hiding a
group problem, not a fairness guarantee or a claim about real contract behavior.

## Measured audit counts (seed 42, both models)

| Scenario | PASS | WARNING | FAIL | Audit exit |
| --- | ---: | ---: | ---: | ---: |
| Clean | 276 | 0 | 0 | 0 |
| Contamination | 157 | 1 | 1 | 1 |
| Target leakage | 159 | 2 | 0 | 0 |
| Feature drift | 224 | 3 | 0 | 0 |
| Segment degradation | 158 | 0 | 1 | 1 |

The clean control has extra data/model/segment/performance checks because its
current copy is explicitly labeled. Comparisons across rows therefore must not
interpret a higher PASS count as greater model quality.

## Output and reproducibility

The fresh output directory contains:

```text
train.csv
clean/, contamination/, target_leakage/, feature_drift/, segment_degradation/
  reference.csv
  current.csv       (clean and feature_drift only)
models/            (one synthetic LR and one RF artifact)
configs/           (ten relative-path YAML configurations)
reports/           (ten ordinary Phase 6 audit reports)
manifest.json      (seed, mutation positions, input/config/model hashes)
summary.json       (628 expected/observed assertions across ten audits)
```

Manifest relative paths use forward slashes consistently on Windows and POSIX.
CSV encoding, line endings and float output are explicit. Repeated runs verify
identical CSV fingerprints and equal checks/results/coverage in the same
environment. Run IDs, timestamps and absolute report paths legitimately differ.
Model pickle bytes need not match across library/platform versions; record the
hash of each run's actual artifact, and compare measurements, not opaque bytes.
Package versions are retained in each Phase 6 report.

`assess_report` reads existing measurements/statuses rather than recomputing
metrics. It checks the intended rule ID, dataset, feature/group identity,
evaluation state, measurement, overall verdict, failed-rule IDs and process exit.
Missing or duplicate targeted checks fail the expectation. A check that did not
run cannot count as successful detection. Generated input/model/config hashes
must agree with the report and remain unchanged after validation.

## Two different exit-code contracts

The **audit CLI** keeps its Phase 6 semantics: 0 for completed PASS/WARNING, 1
for failed validation rules, 2 for execution errors.

The **experiment runner** returns:

- **0:** all expected findings matched, including deliberate audit FAIL/1 results.
- **1:** completed audits produced unexpected findings or evidence.
- **2:** generation, subprocess, report parsing, provenance or output execution
  failed. Available error details/partial evidence are retained.

Thus a working contamination detector does not make the suite fail. A missing
report, timeout or broken model does. This contract prepares for future CI;
no GitHub Actions workflow is included in Phase 7.

## Tests and limitations

Tests verify row-level mutation isolation, reproducible generation, segment/class
balance, unchanged labels/IDs, fixed policies, separate prohibited-column rules,
two complete repeated suites, exact confusion matrices and recall gaps, same-model
reuse, fitting only clean data, report identity, overwrite refusal, wrong/missing/
duplicated/blocked evidence, execution failures and wrapper exit propagation.

The clean control found no alerts in these deliberately easy fixtures. That is
not an estimate of false-positive rates. Correlated real features can create
legitimate target-like warnings; sampling noise or changing cohorts can create
drift without performance loss. Conversely, the two-valued heuristic misses many
forms of leakage. Extreme constructed shifts and a large signal margin make
detection easy; real-world detection power and threshold selection remain open.
Nothing here asserts causality, fairness, production readiness or automatic action.
