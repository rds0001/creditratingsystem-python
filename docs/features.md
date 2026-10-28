# Frozen feature engineering

`fit_preprocessor` estimates transformation parameters from the rows explicitly
passed as training data. It returns a `FitResult` containing a declarative
`ModelArtifact`. Provide `target`, `training`, `model_id` and `model_version`
using the common artifact contract. `training` includes `data_fingerprint`,
`split_id`, `known_at`, and optionally `data_origin`. Fit-time parameters and an
input fingerprint are stored; split metadata is not independently verified.
Fit preprocessing within each training fold, then apply the frozen artifact to
calibration and test folds. Both languages use their native implementation.

Python input tables are lists of row dictionaries (for example,
`dataframe.to_dict('records')`). R accepts lists of named row lists or data frames.
Feature order comes from the ordered `feature_schema`, not incoming column order.
Required columns must exist. Null/None and numeric missing values are handled
according to the declared policy; infinity, numeric strings, and category type
changes are rejected. Extra columns are ignored.

## Preprocessing contract

Each schema entry has `name` and `type`:

| Type | Required policies | Optional settings |
|---|---|---|
| numeric | `imputation`: error, mean, median, constant; `scaling`: none, standard, robust | `fill_value` for constant; `missing_indicator` boolean |
| categorical | `missing_policy`: error, separate, mode; `unknown_policy`: error, separate | `drop_reference`, a training category |

Numeric centering/scaling is estimated **after training imputation**. Standard
scaling uses the population standard deviation (ddof=0); robust scaling uses the
median and the 75th-minus-25th quantile, linearly interpolated (R type 7). A zero
training scale is replaced by one and explicitly reported. An entirely missing
column cannot estimate its mean or median, but can use an explicit constant.

Categorical vocabularies are sorted lexically and frozen. Mode ties select the
first sorted category. Separate missing and unknown indicators are distinct;
unknown values never silently become the reference category. One-hot columns
have stable `name::level:index` names, with separate `::missing` and `::unknown`
columns. The artifact records their meanings. Columns with no training variation
are retained: decide their model treatment explicitly. In particular, unknown
indicators cannot estimate a coefficient from a training sample containing no
unknowns, and a complete one-hot set plus intercept is rank deficient unless a
reference is dropped.

`transform_features(features,fitted_preprocessor)` returns `design_matrix`,
`feature_order` and `lineage`, with missing/unknown counts. It never refits.
The same function also applies a monotone-binning artifact. All these artifacts
support the public JSON serializer and cross-language replay.

## Monotone bins and WOE

`fit_monotone_bins(x_train,y_train,feature_name=...,direction=...,
min_bin_count=...,max_prebins=...,target=...,training=...,model_id=...,
model_version=...)` groups tied observed values without splitting ties. It builds
approximately equal-count prebins, merges undersupported adjacent bins (nearest
observed default-rate neighbor; ties merge left), then pools adjacent monotonicity
violations. Direction is explicitly increasing or decreasing. This deterministic
supervised procedure is not claimed to find a globally optimal partition.

Every nonmissing final bin satisfies minimum training support. Missing values
have a separate bin, exempt from support and monotonicity, with an undefined rate
when unobserved. Boundaries are right-closed observed upper values; the last
boundary is null, meaning no upper limit. New out-of-range values enter the
outermost regular bin; missing values enter the missing bin. `transform_features`
returns a numeric **bin ID**, not a risk estimate or WOE value. Join those IDs to
the frozen WOE table when building a WOE design.

`compute_woe_iv(bin_counts,smoothing=...,woe_sign=...)` uses explicit defaults and
nondefaults per unique `bin_id`. Additive smoothing is applied separately to both
class-conditional distributions within that partition. Both classes must already
exist before smoothing. With zero class cells, zero smoothing is rejected.
`nondefault_over_default` means log(nondefault share/default share); the inverse
convention changes only the WOE sign. IV contributions remain
`(nondefault share-default share)*log(nondefault share/default share)` and sum to
total IV. Missing and unobserved bins are not automatically removed: choose the
intended partition explicitly. No universal IV cutoff or predictive validation
claim is attached to the statistic.

## Ratios and dated lags

`build_financial_ratios(financial_statements,ratio_spec)` takes statement records
with entity_id, as_of, period_start, period_end, currency, values and units.
`period_end` equals `as_of`. Each values field has a unit equal to the statement
currency or dimensionless. A ratio specification contains name, numerator and
denominator maps of field-to-coefficient, `denominator_policy` (positive or
nonzero), and `output_unit='dimensionless'`. Units must match within and between
both formula sides. Zero, prohibited negative and missing denominators produce
null values with explicit statuses. For example, cashflow/(interest+principal)
uses numerator `{'cashflow':1}` and denominator `{'interest':1,'principal':1}`.
Accounting definitions, currency and reporting-period alignment remain explicit
input responsibilities; the method does not combine different statements.

`build_distributed_lags(series,observations,lag_spec,frequency=...,
gap_policy='null')` uses **actual calendar offsets**, not previous row positions.
Series records contain entity_id, feature, observed_at, available_at, known_at,
value, with observed <= available <= known. Query rows contain entity_id, as_of,
known_at. For each exact lag date, select the latest vintage available and known
at the query cutoff; unresolved equal-vintage duplicates are rejected. Lag specs
contain name, feature, nonnegative integer lag and operation level, difference or
relative_change. Difference uses current minus lagged; relative change uses
current/lagged minus one. Zero bases and absent dates produce explicit statuses
and null values, or errors under `gap_policy='error'`. Frequency is days, months
or years; monthly/yearly offsets clamp to valid month ends. Each row reports
eligibility. Future revisions cannot alter an earlier information-cutoff replay.

The pool-adjacent-violators principle is documented in
[SciPy's isotonic-regression reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.isotonic_regression.html).
Support merging, missing bins, units, availability and artifact contracts are
explicit library specifications. Tests use independently hand-computed training
statistics, bin risks, WOE/IV and calendar-vintage examples.
