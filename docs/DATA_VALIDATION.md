# Phase 2 data validation

## Library usage

The inputs are CSV-like pandas DataFrames. Read values before training preparation
so defects are observed rather than rejected or transformed by `prepare_frame`:

```python
from modelgate.checks import DataCheckPolicy, run_data_checks
from modelgate.dataset import read_csv_strings

train = read_csv_strings("data/prepared/train.csv")
reference = read_csv_strings("data/prepared/reference.csv")
policy = DataCheckPolicy()  # Telco predictors, prepared target labels 0/1
results = run_data_checks(train, reference, policy)
documents = [result.model_dump(mode="json") for result in results]
```

For raw target labels, construct
`DataCheckPolicy(target_labels=("No", "Yes"), positive_class_label="Yes")`.
The encoded policy accepts integers 0/1, integral numeric equivalents, and CSV
text `"0"`/`"1"`. It rejects Boolean targets and text `"0.0"`/`"1.0"`. Encoding is
explicit: raw Yes/No values are not silently accepted under the encoded policy.

Pass optional `current=frame` and `current_labeled=True` to check labeled current
data. The default `current_labeled=False` intentionally skips current target
content validation, even if a target column is supplied. No performance metrics
are computed. Train/reference are always treated as labeled.

Settings use a small Pydantic model, not the complete future YAML configuration.
Unknown settings, conflicting roles, invalid fractions, duplicate names, empty
ID sets, and unknown missingness overrides/exclusions fail before checking data.
To change settings, construct a new validated `DataCheckPolicy`.

## Result contract

Each `CheckResult` has `check_id`, `dataset`, `status`, `evaluated`, `measurement`,
`threshold`, `evidence`, and `message`. Status is PASS, WARNING, or FAIL. Results
are validated for JSON-compatible shape and finite measurements at construction.
Counts and thresholds are not rounded for decisions. No overall run status is
calculated in this phase.

When a prerequisite is absent, the dependent check is WARNING with
`evaluated=false`, an explicit reason, and a null measurement if unavailable.
It cannot be PASS or FAIL. An independent schema check reports the actual
structural defect as FAIL. For example, missing `customerID` fails schema columns
and prevents ID overlap measurement while still permitting predictor comparison.
Programming errors are not swallowed and converted into a validation result.

Examples are capped by `max_examples` (default 5, supported range 0-20). Set it to
0 to suppress example values. Row positions are zero-based offsets into the
supplied DataFrame, independent of its index labels. Identifier examples are
distinct keys in sorted order. Dataset/check ID pairs identify results uniquely.

## Rules and defaults

| Check | Measurement / rule | Default outcome |
| --- | --- | --- |
| Structure | Nonempty rows and columns; unique, nonblank string column names | Defect FAIL |
| Columns | Missing required columns and unexpected columns | Missing FAIL; extra WARNING |
| Types | Invalid nonmissing numerical/integer values; allowed categorical values when configured | Invalid values FAIL |
| Target | Missing/invalid labels and class counts | Invalid FAIL; one-class train FAIL; one-class reference/current WARNING |
| Identifiers | Rows missing any component of the configured key | Any missing FAIL |
| Segment | Rows missing the one configured segment | Any missing FAIL |
| Missingness | Missing rows / total rows, per column | Above 0.05 FAIL |
| Duplicate rows | Occurrences after the first / total rows | Above 0 FAIL |
| Duplicate IDs | Repeated complete key occurrences after the first / total rows | Above 0 FAIL |
| Identifier overlap | Matching reference rows / all reference rows | Above 0 FAIL |
| Predictor overlap | Exact predictor matches / all reference rows | Any match WARNING |
| Prohibited columns | Configured forbidden features present | Any present FAIL |
| Target-like columns | Exact two-value feature/target bijection on comparable rows | Relationship WARNING |

Every maximum-fraction check uses strict `>`: equality passes. For example, a
missing fraction of 0.05 passes a 0.05 limit; 0.051 fails. Fractions are 0-1 and
display percentages are 0-100. Settings can override missingness per column and
set separate row-duplicate, ID-duplicate, and ID-overlap limits.

## Schema and missingness details

The default predictor types reuse Phase 1's three numerical and sixteen
categorical columns. `column_types` may additionally declare integer-valued
features. Numerical parsing rejects nonblank invalid text, infinity, and Boolean
values. Integer rules also reject nonintegral numbers. Category rules represent
scalar CSV values as text; they restrict values only when `allowed_categories`
is supplied. Unrestricted category types are not a physical pandas dtype check.

The target and IDs must be outside the predictor type map. The one segment must
be a categorical predictor. All configured roles/types must be in required
columns. Unexpected columns remain visible to missingness and leakage scans;
set `strict_extra_columns=True` to make their schema result FAIL.

Missingness means pandas null, empty string, or whitespace-only string. Other
text, such as `"NA"` or `"NaN"`, stays literal. Literal `"NaN"` in a numerical
column is invalid text, while a real floating NaN is missing. Missing cells are
handled separately from nonmissing type validity. Empty datasets fail structure;
undefined percentages are not replaced with zero.

`missing_exclusions` skips that column's missingness policy and records PASS with
`excluded=true`, no applied maximum, and the observed measurement retained. It
does not disable target, ID, or segment requirements. An excluded-but-absent
required column still cannot have its missingness measured and is blocked.

## Duplicates and overlap

Exact-row duplication uses all supplied columns and pandas equality semantics,
counting each occurrence after the first. It does not perform the predictor
comparison's numerical/text normalization. A row appearing three times counts
as two duplicates. Repeated nulls in otherwise identical rows follow pandas'
duplicate behavior.

Multiple configured ID columns form a composite tuple. Nonmissing values retain
their text representation: `"001"` differs from `"1"`. Fields are never joined
into an ambiguous delimiter string. Missing ID components exclude that row from
ID matching. A below-limit ID result with excluded rows becomes WARNING, not
PASS; excluded row counts are recorded. An above-limit measurement remains FAIL.

ID overlap counts each reference row once even when training has repeated keys.
Its denominator includes every reference row, with excluded rows separately
reported. It also reports the number of distinct overlapping keys.

Predictor comparison uses exactly `column_types` fields, excluding target and IDs.
It uses structured tuples, normalized missing values, and Decimal numerical keys
to avoid rounding distinct large integer strings to the same float. Categories
retain literal text. Invalid numerical values or absent predictors block the
entire comparison instead of silently omitting rows. No fuzzy matching occurs.
Rows with missing predictor values may match another row with the same missing
pattern; separate missingness checks still apply.

## Leakage heuristic and interpretation

Explicit feature prohibitions are policy failures even if the target is absent.
They do not by themselves prove that a feature leaked future information.

The target-like heuristic scans present columns, including unexpected features,
excluding target and IDs. It requires two valid target classes, at least
`min_leakage_rows=20` comparable rows, and `min_leakage_coverage=0.8`. Comparable
rows have a nonmissing feature; their target labels must already be valid.
Insufficient coverage or a one-class comparable subset produces an unevaluated
WARNING. The intentionally unlabeled current dataset also gets an explicit
unevaluated target-like result.

Exactly two feature values must map bijectively to the two target labels.
Evidence distinguishes a copy, inverted labels, and another two-value recoding.
It records comparison count, coverage, distinct values, and bounded mapping
examples. A relationship found only on comparable rows is not presented as
covering the full dataset.

A high-cardinality identifier-like feature is not flagged simply because every
value uniquely identifies a class. This avoids a trivial false positive but also
means complex, noisy, or high-cardinality leakage can be missed. Legitimate
strong binary predictors can warn. No statistical significance or causality
claim is made. There is no single-feature model fitting in this phase. A PASS
only means this particular heuristic found no exact two-value relationship.

## Reproducible demonstration and observed outcome

```powershell
.venv\Scripts\python scripts/demo_data_checks.py
# For an additional run without overwriting an earlier report:
.venv\Scripts\python scripts/demo_data_checks.py --output reports/generated/recheck.json
```

The demonstration uses prepared CSVs directly and writes hashes plus individual
results into a local JSON document. Its zero exit code means writing succeeded;
it is not the final validation exit-code policy. The generic YAML run schema,
status aggregation, and public `modelgate validate` CLI remain Phase 6.

On 2026-09-17, the original Phase 1 split produced 136 check results:
135 PASS, 1 WARNING, and 0 FAIL. All were evaluated. The warning was predictor
overlap: 10 of 1,409 reference rows (0.709723%) matched training predictor tuples.
There were zero overlapping customer IDs. An independent pandas merge using
deduplicated training predictor rows confirmed the count of 10.

Missing TotalCharges: 10/5,634 training rows (0.177494%) and 1/1,409 reference rows
(0.070972%). Both were below the default 5% limit. Exact-row and ID duplicates
were zero. No two-value target relationship was found among the checked features.

These results do not establish that the dataset is leakage-free or suitable for
release. Matching attributes among distinct IDs are evidence worth inspecting,
not proof of duplicated people. The data and split were not modified to remove
this warning. Data and generated JSON remain ignored by Git.
