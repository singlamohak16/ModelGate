# Phase 4: Segment validation

## Purpose

Reveal differences hidden by overall model measurements. Evaluate one configured
categorical column, initially `Contract`, using the same saved model probabilities
and one shared decision threshold. No group-specific fitting, tuning, threshold
optimization, drift analysis or fairness guarantee is implemented.

## Library API and flow

```python
from modelgate.checks.segment import SegmentCheckPolicy, run_segment_checks

policy = SegmentCheckPolicy(
    segment_column="Contract",
    target_column="Churn",
    min_segment_size=50,
    decision_threshold=0.5,
    min_recall=0.4,  # Illustrative requirement, not a business recommendation.
)
# reference is a labeled DataFrame; probabilities are aligned positive probabilities.
report = run_segment_checks(reference, probabilities, policy)
```

`run_segment_checks` validates the frame's configured columns, calls
`analyze_segments`, and creates existing `CheckResult` objects before returning
their JSON-ready dictionaries. Its output includes:

- `overall`: Phase 3 metrics computed directly over every input row.
- `segments`: deterministic lexically sorted group records, sizes, class counts,
  prevalence, eligibility, metrics and differences.
- `best_by_metric`: highest defined value, all tied names, candidate count and
  a reason if unavailable.
- `coverage`: total/named/missing/eligible rows, fractions and segment counts.
- `checks`: completeness, comparison support, size, class support and metric rules.

`analyze_segments(labels, probabilities, segments, min_segment_size=50,
decision_threshold=0.5, target_labels=(0,1), positive_label=1)` is the
measurement-only helper; settings after the vectors are keyword-only. It reuses
Phase 3 input validation and `classification_metrics`. Labels are encoded once
to the configured positive direction, then every subgroup uses the same binary
definitions. It neither loads a model nor changes input values.

The named segment column accepts strings and missing scalar values. Non-string
non-missing category values raise `ValueError`; callers using numeric category
codes must explicitly supply their intended string labels first. Nonblank strings
are not trimmed or case-folded. Literal `"NA"` and `"missing"` are valid names,
not missing-value sentinels. Only observed groups are analyzed.

All vectors must be nonempty, one-dimensional and equally long. Every supplied
pandas Series must have identical indexes, including order; otherwise fail.
Arrays are positional, and row identity is the caller's responsibility. Matching
duplicate indexes do not cause a merge: rows are grouped with positional masks.
Labels/probabilities retain Phase 3's validity requirements. Empty frames,
duplicate columns, absent configured columns or invalid values raise input errors.
This focused API is not a replacement for Phase 2's schema audit.

## Measurements

For each size-eligible group report record count, positive/negative counts,
prevalence, precision, recall, F1, predicted-positive rate, confusion matrix,
trapezoidal PR-AUC and separately named average precision. Positive predictions
use `p >= decision_threshold`. Metrics retain Phase 3's formulas and null reasons.

Overall metrics are computed from all predictions, including rows without a
named segment and rows in small segments. Overall F1 is **not** an average of
segment F1 values. Overall PR-AUC is computed from the pooled scores, not averaged
group PR areas. The four comparison metrics are precision, recall, F1 and PR-AUC.

For each defined metric:

```
from_overall = segment_metric - overall_metric
from_best = segment_metric - maximum_eligible_defined_segment_metric
```

Differences are absolute metric units, not relative percentage changes. For
precision/recall/F1, -0.10 means ten percentage points lower. PR-AUC differences
are area differences. Unavailable comparisons are null with explicit reasons.
No gap thresholds or significance tests are applied.

Best is computed **separately for each metric**. Candidates must meet the size
minimum and have that metric defined. All exact ties are retained; comparisons
use unrounded floating values without an approximate-tie tolerance. A group
compares against itself if it is the only candidate, giving zero from-best; a
separate comparison-support WARNING makes clear that this is not peer evidence.
No eligible candidates means a null best value and empty name list, not zero.

## Size, missingness and undefined evidence

Default minimum is 50 rows; equality is eligible. This is an illustrative
reporting safeguard, **not a statistical reliability guarantee**. The policy
accepts any strictly positive integer. Class counts remain essential: 334 rows
with nine positives still provide limited positive-outcome evidence.

Below the minimum, retain count, positive/negative counts and prevalence, but set
the entire group `metrics` object to null. Four performance rules become
unevaluated WARNINGs, even for perfect predictions or a configured zero minimum.
Small groups cannot set best-segment benchmarks. Size itself is an evaluated
PASS/WARNING check because the row count is known.

One-class groups warn through class support. PR-AUC/average precision require
both observed classes and otherwise remain null. No predicted positives makes
precision undefined; if actual positives exist, recall and F1 are genuinely zero.
Defined metrics can still be measured and checked for an eligible one-class
group; the class-support warning remains visible.

Null/NaN/NA scalar segment values and empty/whitespace-only strings count as
missing. Any missing row fails completeness, with count and fraction evidence.
These rows remain in overall measurements but are excluded from named groups
and best comparisons. Coverage fractions use **all input rows** as denominator.
If every segment value is missing, overall measurements still exist, no groups
are invented, completeness fails and four comparison-support checks warn.

## Status and policy rules

`SegmentCheckPolicy` validates target/segment names, distinct roles, the binary
label pair, positive label, threshold, positive integer size and optional limits.
Unknown settings, booleans used as limits/counts and nonfinite values are rejected.
Default target/segment names are `Churn` / `Contract`, positive label 1 and
threshold 0.50. There are no new dependencies.

Optional common limits `min_precision`, `min_recall`, `min_f1`, `min_pr_auc`
apply to each eligible segment. A value strictly below its limit FAILs; equality
PASSes. No rounding/tolerance is applied. Limits are not adjusted to make a model
pass, and segment-specific overrides are not implemented.

With no limit, a defined metric is a **descriptive PASS**, with `applied=false`,
null threshold and an explicit message. Undefined or withheld metrics WARNING
with `evaluated=false`. No overall status is aggregated.

For G observed named groups, the output has 5 + 6G checks: one completeness,
four per-metric comparison-support checks, then size, class support and four
metric checks for each group. `check_id` identifies the rule, not a unique group;
use `evidence.segment` with it. Every result includes dataset, status, evaluated,
measurement, threshold, evidence and message. Metric evidence includes counts,
differences and best comparator, plus a confusion matrix when available.

## Local demonstration

```powershell
.venv\Scripts\python scripts/demo_segment_checks.py --trust-local-models
# Choose an unused output path on subsequent runs:
.venv\Scripts\python scripts/demo_segment_checks.py --trust-local-models --output reports/generated/phase4_review.json
```

Only use the trust flag for artifacts from your trusted local training run.
Joblib deserialization can execute code; matching fingerprints do not make an
untrusted artifact safe. The script checks the prepared data against its
manifest, matches that manifest to training provenance and verifies both model
hashes before loading either. It uses the original 19 predictors and no fitting.

It records data/model provenance, seed, policy, training/evaluation versions and
results. Dataset/model files and generated reports remain excluded from Git.
Existing outputs are protected. Exit 0 means the demo was written; handled
input/output errors exit 2. These are not Phase 6 validation exit codes; warnings
and failures in measured checks do not set a run-level exit status here.

## Findings and limits

Both baselines predict no positives for one-year and two-year customers at
threshold 0.50 on this reference split. Their recall/F1 are zero; precision is
undefined. Each model generates 20 PASS and 3 WARNING results (two undefined
precisions and insufficient precision-comparison support), with no configured
quality limits. Zero recall is therefore descriptive, not a policy FAIL by
default. Synthetic tests demonstrate an explicit recall limit catching a weak
group despite acceptable overall recall. See `EXPERIMENTS.md` for all numbers.

This does not establish discrimination, causality, fairness or unfairness.
Contract groups have very different churn prevalence and class counts; PR-AUC
and precision comparisons are especially sensitive to those differences. The
best observed segment is not a statistically proven winner. One split, no
confidence intervals and no temporal evaluation limit generalization. No
automated threshold changes, deployment or retraining follows from this report.
