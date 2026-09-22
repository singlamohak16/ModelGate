# Phase 5: Drift and labeled performance changes

## Purpose and scope

Compare reference and optional current data offline using one already fitted
model. Numerical features use the two-sample KS statistic, categorical features
use Jensen-Shannon distance, and positive probabilities use KS. Labeled data
can additionally provide observed metric changes. Unlabeled data cannot establish
performance loss. There is no retraining, tuning, significance decision, temporal
monitoring, automatic deployment or complete Phase 6 audit runner.

## APIs

```python
from modelgate.checks.drift import DriftCheckPolicy, run_drift_checks
from modelgate.checks.performance import (
    PerformanceChangePolicy,
    run_performance_checks,
)

drift_policy = DriftCheckPolicy(max_numerical_ks=0.2)  # Illustrative, not universal.
checks = run_drift_checks(
    reference,
    current,
    drift_policy,
    reference_probabilities=p_reference,
    current_probabilities=p_current,
)
changes = run_performance_checks(
    reference,
    current,
    p_reference,
    p_current,
    PerformanceChangePolicy(max_recall_drop=0.1),
    current_labeled=True,
)
```

Both prediction vectors must come from the same fitted model, preprocessing and
positive-class definition. The array APIs cannot authenticate that provenance;
the caller must establish it. The demonstration uses one loaded artifact per
pair and the existing validated positive-probability extractor. Reference and
current may have different lengths and indexes. Within each dataset, Series
probability indexes must match the frame; arrays are positional.

The drift function returns existing `CheckResult` objects, one per configured
feature plus one for predictions. Feature identity is in `evidence.column`;
`check_id` identifies the rule. Performance returns a JSON-ready dictionary
with `comparison_available`, `reason`, `reference`, `current`, and serialized
`checks`. The availability flag means common comparison prerequisites passed,
not that every individual metric is defined or that quality passed.

## Numerical and prediction KS

For empirical cumulative distributions F and G, KS = max |F(x)-G(x)|, in [0,1].
The implementation sorts samples, evaluates both empirical CDFs on their pooled
distinct values with right-sided `searchsorted`, and takes the maximum gap.
This small direct calculation avoids computing an unused p-value and is tested
against `scipy.stats.ks_2samp(...).statistic`. Identical samples have distance 0;
fully separated samples have distance 1; repeated values and unequal sizes work.

We do not use SciPy's inferential p-value. Its test assumes independent samples
from continuous distributions; tied/discrete values and dependent samples make
that interpretation inappropriate without further analysis. KS remains a
descriptive empirical distance here. See
[SciPy's KS documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).

Numerical features accept CSV numeric text and numeric scalars. Null and
whitespace-only values are excluded with total, missing, invalid and usable
counts plus usable fraction. Invalid text, booleans or infinities block the
feature rather than being silently dropped. Fewer than the configured usable
observations in either sample yields an unevaluated WARNING. Missingness changes
can be hidden from KS on remaining values: inspect coverage and Phase 2 checks.
No imputation is fitted for drift measurement. Calculations use floating values.

Prediction vectors must be nonempty, one-dimensional, finite numeric values in
[0,1], not booleans/strings, and match their own frame's row count. Do not clip
invalid predictions. Missing or invalid vectors block prediction drift only;
valid feature comparisons still run. No prediction KS value is produced without
both datasets, even if two arrays were supplied.

## Categorical distance and smoothing

Categories must be strings or missing scalar values. Nonblank strings are not
trimmed or case-folded; literal 'NA' and 'missing' are ordinary categories.
Actual missing values, including blank strings, share a distinct typed bucket
that cannot collide with a literal category name. This comparison uses all rows,
including missing-category rows. An all-missing pair can have distance zero:
that describes equal distributions, not valid or useful data.

Build a sorted union of observed reference/current categories, including newly
appearing and disappearing categories. With K categories and epsilon=1e-6:

```
p_smoothed = (1 - epsilon) * p_observed + epsilon / K
q_smoothed = (1 - epsilon) * q_observed + epsilon / K
distance = sqrt((KL(p_smoothed || m) + KL(q_smoothed || m)) / 2)
m = (p_smoothed + q_smoothed) / 2
```

SciPy's `jensenshannon(..., base=2)` computes **distance**, not divergence, on
the normalized vectors. Base 2 gives the 0–1 scale. See
[SciPy's distance documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html).
Uniform-mixture smoothing is explicit, configurable in [0,1), and not necessary
to make Jensen-Shannon mathematically handle zero categories. It is a small
documented regularization convention; larger epsilon suppresses differences.
Unlike adding fixed pseudocounts, equal raw proportions at unequal sample sizes
remain equal after this smoothing. Epsilon zero is supported.

Record category count, reference/current sizes, missing counts, new/removed
category counts, base and smoothing formula. Example categories are ranked by
absolute raw proportion change, then deterministic typed name; default cap 10,
configurable 0–20. Each includes raw counts/fractions and smoothed fractions,
plus a truncation flag at the result level. The full support is always used in
the calculation, even when examples are truncated. A nonfinite numerical result
is unavailable evidence, never valid JSON NaN.

## Policy and status semantics

`DriftCheckPolicy` defaults to the 19 Telco predictors with target/IDs excluded.
Feature, target and ID names must be nonblank, unique and disjoint. Numeric and
categorical feature lists may be empty for prediction-only use. Unknown fields,
invalid ranges, booleans as numeric limits and nonfinite settings are rejected.

Minimum sample size is 50 by default, configurable as a positive integer. For
numerical features it applies to usable observations, for categorical features
to all rows including missing, and for predictions to valid probabilities.
Equality meets the minimum. This is illustrative support, not statistical proof.

Optional maxima: `max_numerical_ks`, `max_categorical_distance`,
`max_prediction_ks`. Strictly greater than a configured limit yields evaluated
WARNING; equality PASSes. Without a limit, PASS is explicitly descriptive, with
`applied=false`. Blocked comparisons WARNING with `evaluated=false`, null value,
reason and available evidence. A missing column blocks only that feature. Missing
current data or invalid frame structure blocks all comparisons, never reports zero.
Phase 2 remains responsible for the detailed data-quality audit.

## Labeled performance changes

`PerformanceChangePolicy` defines target column/pair, positive label, common
threshold (default 0.50), minimum sample size (50), and optional maxima:
`max_precision_drop`, `max_recall_drop`, `max_f1_drop`, `max_pr_auc_drop`,
`max_brier_increase`. All change limits are in absolute metric units, not relative
percentages. The same threshold is used for both populations, with p >= threshold.

- Always report current-minus-reference when comparison is available.
- For precision/recall/F1/PR-AUC, deterioration = reference minus current.
- For Brier, deterioration = current minus reference (larger Brier is worse).
- Negative deterioration means improvement; it is not clipped to zero.
- Exceeding a configured deterioration limit FAILs; equality PASSes.
- Without a configured limit, differences are descriptive only.

Reuse Phase 3 metrics and binary Brier. Trapezoidal PR-AUC and average precision
remain distinctly named; the change rule applies to PR-AUC. Record both sample
sizes, positive counts, prevalence and confusion matrices. Undefined metrics
produce unevaluated WARNINGs with null differences, independently of available
metrics. One-class samples add visible class-support WARNINGs; Brier can remain
available. Class support is not a release judgment.

`current_labeled=False` is an explicit mode: return no performance measurements
or checks, even if a target column is present. Do not inspect or infer labels.
Missing current data similarly returns unavailable metadata. A true flag with
missing/invalid labels or predictions produces blocked performance checks and
reasons; it never silently falls back to unlabeled mode. Too-small valid samples
retain descriptive per-dataset metrics but have no evaluated changes. Labels are
never partially filtered to manufacture a usable current dataset.

Differences may reflect prevalence, sampling noise, label collection changes or
model performance; they do not establish a cause, statistical significance or
future loss. Feature drift need not worsen performance, and performance can
worsen without marginal feature drift. Single-feature checks miss some changes
in relationships among features. There is no guarantee of drift detection.

## Demonstration and provenance

```powershell
.venv\Scripts\python scripts/demo_drift_checks.py --trust-local-models
# Select an unused filename for another run:
.venv\Scripts\python scripts/demo_drift_checks.py --trust-local-models --output reports/generated/phase5_review.json
```

No separate real current dataset exists. The demo uses an explicitly labeled
in-memory copy of reference, with reference/current overlap 100%, in labeled
and target-dropped unlabeled modes. It is a zero-change software control, not
an independent/temporal validation or production stability claim. Shifted small
invented tests exercise detection; no Phase 7 scenario framework is built.

Verify prepared hashes, manifest equality and both model artifact hashes before
loading either trusted local model. Joblib may execute code; fingerprints cannot
establish trust. Record policies, data/model provenance, seed and package versions.
Existing outputs cannot be overwritten. Generated reports/data/models stay
ignored. Exit 0 means the demonstration was written, not validation passed;
handled input/output errors exit 2. Full YAML/CLI/report aggregation is Phase 6.
