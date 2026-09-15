# Dataset: IBM Telco Customer Churn

## Source and licence review

The user approved local use of this IBM-hosted CSV on 2026-09-15, with raw and
prepared data kept out of Git.

- Publisher repository: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- Source revision: `d5371f5d83a446ad5673cbcca3b814b926491f8a`
- [Pinned CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv)
- SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- Size verified from downloaded bytes: 970,457 bytes

The repository's [LICENSE](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/d5371f5d83a446ad5673cbcca3b814b926491f8a/LICENSE)
is Apache 2.0, while its README describes licensing of the code pattern. No
separate dataset-specific licence was found in the data directory during the
review. We therefore document the repository licence without claiming a
separately verified CSV redistribution grant. ModelGate's MIT licence applies
to our code, not automatically to this data. We publish source references and
preparation code, not the raw or prepared CSV. Local-only storage does not itself
settle every question about permitted data use.

This repository was archived by IBM in July 2024. Source availability is not
guaranteed; the downloader fails clearly on network errors or changed bytes.
It does not silently substitute a mirror or disable checksum verification.

## Observed audit (2026-09-15)

- 7,043 rows and 21 columns: one ID, one target, 19 predictors.
- `Churn`: 5,174 `No`, 1,869 `Yes`.
- 11 whitespace-only `TotalCharges` cells (0.156183% of rows).
- No other missing cells after whitespace-only normalization.
- Zero exact duplicate rows; zero duplicate IDs.
- Zero invalid nonblank numerical values under the documented schema.

The audit counts duplicates after the first occurrence. Raw cells are initially
read as strings to preserve lexical values; inferred raw types are therefore
strings. The audit also reports numerical parsing failures and the resulting
prepared dtypes. It is a descriptive dataset audit, not a rules-based ModelGate
PASS/WARNING/FAIL report.

## Explicit schema

| Columns | Prepared type / role |
| --- | --- |
| `customerID` | Unique, nonmissing string; identifier only |
| `Churn` | Integer target: `No=0`, `Yes=1`; both classes required |
| `tenure`, `MonthlyCharges`, `TotalCharges` | Float64; missing numerical values allowed |
| `Contract` | Nonmissing string; predictor and future segment column |
| `SeniorCitizen` | Categorical string flag, including `0` and `1` |
| `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling`, `PaymentMethod` | Categorical strings; missing cells allowed |

This preparation code accepts exactly these columns. Missing, unexpected, or
duplicate column names produce errors. Domain validation for every categorical
value and numerical range is not implemented in Phase 1. Nonblank category
spelling and whitespace are preserved; whitespace-only cells become missing.
Unexpected nonblank numerical text and infinity fail parsing rather than being
silently converted to missing values. The raw download remains unchanged.

Prepared CSVs contain the encoded target and must be read using
`prepare_frame(read_csv_strings(path), encoded_target=True)` before prediction.
`read_csv_strings` alone intentionally leaves numeric values as strings.

## Split and preprocessing

Records are sorted by `customerID`, then split once with `test_size=0.2`,
`stratify=Churn`, `random_state=42`. Each resulting partition is ID-sorted again.

| Partition | Rows | Positive labels | Positive prevalence |
| --- | ---: | ---: | ---: |
| Train | 5,634 | 1,495 | 0.26535321 |
| Reference | 1,409 | 374 | 0.26543648 |

Both models use these exact partitions. Split IDs are recoverable from the local
CSV files; the manifest records hashes, counts, ordering, and parameters.

- Training CSV SHA-256: `54b450a3742f15c9dc6cfb5f43dca14ffcb86d26af302489791996dcf46a6667`
- Reference CSV SHA-256: `59a0aefabe95f2a822ac256205825487dcaee9dc0e65c0f6369a212d667377e4`

Only deterministic cell parsing and fixed target encoding happen before the
split. Median imputation, scaling, and category vocabularies are fitted on the
training partition. Reference-only categories encode as all zeros for that
feature, which avoids an inference error but loses information about the new
category. Unknown-category counts will require future validation checks.

Numerical preprocessing: median imputation followed by standard scaling.
Categorical preprocessing: `__MISSING__` imputation followed by one-hot encoding.
`keep_empty_features=True` preserves feature shape; an all-missing numerical
training column uses sklearn's zero fallback and should be investigated.
The fixed missing-category marker could collide with a genuine category in
other datasets; this code targets the pinned Telco sample only.

## Interpretation limits

This sample is useful for learning validation mechanics. A random split is not
an evaluation of future customer behavior: the selected schema has no sequence
of dated customer snapshots for temporal backtesting. It cannot establish the
causal effect of retention interventions or production business value. Repeated
reference-set tuning would also bias its metrics. Both models are retained;
no deployment decision follows from this comparison.
