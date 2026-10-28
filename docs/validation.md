# Validation of observed default predictions

These methods return `CalcResult` and keep a statistic, its assumptions and its
applicability separate. Python uses `.value`; native R uses `$value`. Outcomes are
observed numeric zero/one values, never inferred from rejected applications or
incomplete follow-up. PDs must describe the same event, horizon and population as
the outcomes. A numerical metric cannot establish that those data were measured
correctly. No p-value is an automatic model approval, and multiple testing is not
silently adjusted.

The five basic metrics accept scalars as one observation, or equal-length numeric
vectors. Matrices, nonfinite values, missing values, boolean outcomes, recycling,
negative weights and nonpositive weight totals fail. Weights are descriptive
nonnegative weights, with one per row by default. Zero-weight observations remain
subject to input validation. Weights do not create independent borrowers.

## Discrimination and scoring

`auc_gini(outcomes, scores, weights=None, *, score_direction, context=None)` uses
explicit `higher_is_risk` or `lower_is_risk`. It compares every positive-weight
event/non-event pair, assigns one to correctly ordered pairs and one half to ties,
and normalizes pair weights within each class. Gini = 2 AUC − 1. Returned raw pair
counts count observations, not frequency-weight replications. Concordant/tied
pair masses are normalized probabilities, not unnormalized exposure amounts.
One class with no positive weight yields `NOT_APPLICABLE`, reason `ONE_CLASS`,
and missing AUC/Gini. No confidence interval follows from row count alone.

`ks_discrimination(outcomes, scores, weights=None, *, score_direction,
context=None)` reports the maximum absolute difference of the class empirical
CDFs of oriented risk score. Ties move together. `signed_difference` is the
non-event CDF minus event CDF. Equal maximum distances select the first threshold
in ascending oriented score order; original thresholds run backwards for
`lower_is_risk`. The full `cdf` table is returned. A missing weighted class yields
`NOT_APPLICABLE`. This descriptive distance has no classical continuous-IID KS
p-value attached to weighted, tied credit data.

`brier_score(outcomes, predicted_pd, weights=None, *, context=None)` calculates
Σ(wᵢ/Σw)(yᵢ−pᵢ)². This is the binary [0,1] convention. It returns normalized
contributions, individual squared errors, observed rate and
`outcome_variance_baseline = observed_rate*(1-observed_rate)`. That baseline is
an in-sample descriptive constant predictor, not an independent benchmark.

`log_loss(outcomes, predicted_pd, weights=None, *, boundary_policy="exact",
epsilon=None, context=None)` uses natural logarithms. Correct boundary predictions
have zero loss. An impossible prediction with positive weight has `log_loss=None`
and `loss_status="INFINITE"`; its contribution is missing and its infinite flag
true. This deliberately preserves positive infinity in finite JSON. A zero-weight
impossible prediction contributes zero. Optional `boundary_policy="clip"` requires
explicit epsilon in (0,0.5), with representable interior bounds. Diagnostics record
epsilon and clipping count. No default clipping hides impossible predictions.

These conventions correspond to the binary metrics documented by
[scikit-learn AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html),
[Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
and [log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
The explicit infinity and caller-selected clipping contract above is a deliberate
library interface choice; it does not silently reproduce another package's
implicit precision clipping.

## Exact common-probability count test

`binomial_pd_test(n, observed_defaults, common_pd, *, alternative,
independence_assumption, confidence_level=.95, context=None)` requires integer
0 ≤ observed_defaults ≤ n, n ≤ 2⁵³−1, PD in [0,1], and the declaration
`independence_assumption="independent_common_probability"`. Alternatives are
`less`, `greater`, `two-sided`. The two-sided test orders outcomes by null
probability mass; the exact Clopper-Pearson interval matches the alternative.
An empty cohort yields `NOT_APPLICABLE`, `EMPTY_COHORT` and missing metrics.
Python uses SciPy's exact binomial implementation and native R uses
[`stats::binom.test`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html).
Correlated defaults or heterogeneous individual PDs do not satisfy this null.

## Frozen-design metadata for extended diagnostics

The remaining statistical diagnostics use explicit `design`:

```python
{
    "forecast_fingerprint": "<optional expected canonical hash of the PD array>",
    "prediction_origin": "HOLDOUT",  # or EXTERNAL
    "forecast_known_at": "2023-12-31",
    "evaluation_at": "2025-01-01",
    "split_id": "untouched-test",
    "target": {"event": "first_default", "horizon": 1, "time_unit": "years",
               "probability_measure": "PHYSICAL", "conditioning": "PIT"}
}
```

The method always calculates and returns the canonical SHA-256 forecast fingerprint.
When an expected fingerprint from an earlier result/manifest is supplied, the PD
array must match it. Omit this optional field for the initial calculation; no private
hashing function is needed to call a diagnostic. Dates are ISO
calendar dates; forecast knowledge must not follow evaluation. Target metadata
must be complete and physical. HOLDOUT/EXTERNAL describe supplied evidence, not
an automated proof of sample independence. Ordinary fitted-sample predictions
are rejected except by the explicitly declared fitted-logistic HL approximation.

## Calibration intercept and slope

`calibration_intercept_slope(predicted_pd, outcomes, *, design,
covariance_policy, cluster_ids=None, confidence_level=.95,
boundary_policy="reject", epsilon=None, max_iter=100, tolerance=1e-10,
context=None)` fits the diagnostic model

logit Pr(Y=1) = intercept + slope × logit(frozen PD).

A calibrated population suggests (0,1), subject to sampling uncertainty. The
fit returns coefficients, covariance, standard errors and confidence intervals;
it never replaces or changes the original model/predictions. The native GLM
estimator supplies rank, separation and convergence checks. Constant PDs yield
`CONSTANT_PD` because the joint intercept/slope is unidentified. Boundary PDs
yield `BOUNDARY_PD` by default; explicit representable epsilon clipping is
available and disclosed. One observed class, separation, failed convergence or
singular information produce missing coefficients with an explicit reason.

Covariance policies are `none`, `iid`, and `cluster_cr1`. IID uses inverse expected
Fisher information and normal intervals. Cluster CR1 requires one nonempty cluster
ID per row, at least two clusters and n>2. It aggregates likelihood score vectors
within clusters, sandwiches their outer products with inverse information, and
multiplies by G/(G−1) × (n−1)/(n−2). Intervals use Student t with G−1 degrees of
freedom. Independent clusters and adequate cluster count are assumptions; arbitrary
cross-cluster/time dependence is not repaired by naming a cluster column. Original
forecast-estimation uncertainty is conditional/fixed here. The sandwich convention
is documented by [statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html).

## Reliability bins

`calibration_curve(predicted_pd, outcomes, bin_edges, *, design,
interval_method, dependence_assumption, confidence_level=.95,
minimum_count=5, context=None)` uses fixed increasing edges covering [0,1]. Bins
are left-closed/right-open; the final bin includes 1. Every observation enters
exactly one bin. Empty bins are retained with count zero and missing rates,
residuals and intervals. The returned row assignments and counts allow checks
against the original total. `small_bin` compares count with the supplied minimum;
it is descriptive, not an inferred reliability threshold.

Use `interval_method="none"`, `dependence_assumption="descriptive_only"` when a
count interval is not supported. `clopper_pearson` requires
`independent_common_probability_within_bins`, an explicit assumption about the
underlying event probability within each bin. A bin of heterogeneous dependent
borrowers does not acquire an exact binomial interval merely because it is
shown in a reliability chart. Outcomes must not determine opportunistic bin edges.

## Spiegelhalter calibration moment test

`spiegelhalter_test(predicted_pd, outcomes, *, design, alternative,
independence_assumption, context=None)` requires `independent_fixed_bernoulli`.
It computes numerator Σ(yᵢ−pᵢ)(1−2pᵢ), variance Σpᵢ(1−pᵢ)(1−2pᵢ)² and their
standardized Z. The p-value uses a normal approximation and the declared
alternative. Positive/negative refer to this moment, not universally to PD
under-/overprediction when probabilities cross 0.5. Zero variance, including all
PDs equal to 0.5, yields `NOT_APPLICABLE`, `ZERO_VARIANCE`. Diagnostics expose the
largest individual variance share; small or dominant-term samples can invalidate
the approximation. This is a single calibration moment, not an omnibus test.
The formula was checked against the author's
[Spiegelhalter paper record](https://doi.org/10.1002/sim.4780050506) and
[Harrell's primary implementation](https://github.com/harrelfe/rms/blob/master/R/val.prob.s).

## Heterogeneous independent count test

`poisson_binomial_pd_test(individual_pd, observed_default_count, *, design,
alternative, independence_assumption, method="dynamic_programming",
context=None)` requires `independent_fixed_bernoulli`. The distribution is the
product polynomial ∏[(1−pᵢ)+pᵢz]; coefficient k is P(default count=k). Successive
Bernoulli convolutions produce the complete count distribution with O(n²) work
and linear working memory. Identical PDs recover the binomial law. Endpoint PDs
are deterministic trials. Two-sided probability-mass ordering uses relative
comparison tolerance 1e−7; one-sided tails are inclusive.

The result includes mass, expectation, variance and p-value. Diagnostics expose
mass error and zeroed cells inside feasible support. If underflow prevents a
requested positive tail from being represented, the p-value is missing with
`NUMERICAL_UNDERFLOW`; a truly impossible count outside deterministic support can
still have exact zero probability. Empty cohorts are not applicable. This
conditional distribution excludes dependence and parameter estimation uncertainty.
See the primary [SciPy distribution definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson_binom.html);
the implementation here uses its own elementary convolution rather than requiring
a newer SciPy distribution implementation at runtime.

## Grouped Hosmer-Lemeshow/Pearson diagnostic

`hosmer_lemeshow_test(predicted_pd, outcomes, bin_edges, *, design,
degrees_of_freedom_policy, independence_assumption, minimum_expected=5.,
context=None)` uses the same prespecified bin convention. Per-bin contribution is
(O−E)²/[E(1−E/n)]. Counts, expectations, mean PD, variance and contributions remain
visible even when asymptotic inference is inappropriate.

`external_common_bin_pd` requires HOLDOUT/EXTERNAL forecasts and genuinely equal
PDs within each bin. Degrees of freedom equal the nonempty, non-deterministic
bin count. `fitted_logistic_approximation` requires `prediction_origin=FITTED_SAMPLE`
and uses G−2, explicitly as the conventional approximation rather than a universal
correction for any fitted model. Both require `independent_fixed_bernoulli`.
Empty and deterministic bins are disclosed. Small expected event or non-event
cells, heterogeneous external-bin probabilities, impossible deterministic-bin
outcomes or nonpositive degrees of freedom withhold a p-value. The
[BIS validation study](https://www.bis.org/publ/bcbs_wp14.pdf) describes the
independent fixed-rating-bin chi-square construction and its small-PD limitations;
this library makes the external/fitted design distinction explicit.

## Dependence-aware simulation backtest

`dependent_pd_backtest(predicted_pd, outcomes, cluster_ids, time_ids, *, design,
null_model, innovations, alternative, confidence_level=.95, context=None)` tests
observed count minus the fixed expected count under a declared Gaussian factor
null. It does not attach binomial significance to correlated panel rows.
`design.observation_ids` is an ordered array of unique row IDs. Each row also has
nonempty cluster/time keys. Distinct `(cluster_id,time_id)` pairs, sorted by cluster
then time, define the factor columns.

`null_model` contains:

- `family="gaussian_factor"`;
- `asset_correlation`: scalar or per-observation vector in [0,1];
- `factor_correlation`: symmetric positive-definite matrix with unit diagonal;
- `parameter_uncertainty="conditional_fixed"`, or
  `"supplied_predictive_pd_draws"` plus `parameter_draw_source`.

In JSON/R artifacts, matrices are explicit arrays of row arrays. `innovations`
contains `draw_assumption="independent_standard_normal_rows"`, matching ordered
`observation_ids`, matching `factor_keys` objects, `factor_normals` (B×G), and
`idiosyncratic_normals` (B×n), with B≥2. No local random draws or hidden seeds are
used. The caller supplies independent standard normal innovation rows; the library
cannot certify their distribution from their realized values.

A lower Cholesky factor L maps factor innovations Z to Z Lᵀ. For observation i,
latent value is √ρᵢ times its correlated cluster/time factor plus √(1−ρᵢ) times its
idiosyncratic innovation. Default occurs at or below Φ⁻¹(pᵢ). Shared factor keys
preserve common shocks; factor correlation preserves declared cross-time or
cross-cluster dependence. At ρ=0 the conditional null is independent; at ρ=1
rows sharing a factor have a common latent shock. Perfectly common factors should
share a key instead of supplying a singular factor-correlation matrix.

For supplied predictive parameter uncertainty, `parameter_pd_draws` is B×n and
replaces each row's threshold; source and uncertainty interpretation remain
explicit. This is a caller-defined predictive mixture, not an automatically
estimated frequentist distribution. IID innovation rows and their relationship
to supplied parameter draws remain part of the caller's null-design assumption.

The returned complete simulated counts/statistics make the null reviewable.
One-sided tail comparisons are inclusive; two-sided ordering uses absolute count
residual. The Monte Carlo p-value is (extreme+1)/(B+1), with explicit finite
resolution, following the nonzero Monte Carlo test principle discussed by
[Phipson and Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
The count interval is an empirical type-1 null prediction interval. The separate
Clopper-Pearson tail-probability interval measures Monte Carlo error, not borrower
PD uncertainty. The innovation fingerprint binds the exact paths for native
Python/R comparison and replay.

## Cohort counts, frozen forecasts and maturity

`default_count_backtest(observation_panel, frozen_forecasts, group_by, *, unit,
horizon_days, default_definition, redefault_policy, evaluation_cutoff,
context=None)` is a descriptive aggregation with no automatic significance test.
`unit` is `obligor` or `facility`; `redefault_policy` must be
`exclude_previous_default`. Horizon is an explicit positive number of calendar
days; default definition is a caller-declared string.

Observation rows contain unique `observation_id`, `entity_id` for obligor units
or `facility_id` for facility units, `as_of`, `performance_start`, `outcome_end`,
boolean `eligible` and `prior_default`, `default_definition`, explicit `target`
(possibly null), and grouping fields such as vintage, grade and segment.
An observed target requires `outcome_known_at`; a nondefault zero cannot be known
before horizon completion. Start must not precede as_of, and window length must
match the horizon.

Forecast rows contain `observation_id`, scalar `pd`, `forecast_at`, `known_at`,
`default_definition`, `horizon_days`, `unit`, `probability_measure="PHYSICAL"`,
`model_id`, `model_version`. Forecast creation ≤ forecast knowledge ≤ observation
as_of is required. Unknown observation references, duplicate forecast IDs,
incompatible units/targets and future forecasts fail.

Repeated facilities representing the same obligor, observation date and outcome
window are counted once under the obligor policy. PD, outcome, grouping and
eligibility must agree; conflicting duplicates fail. Under a facility policy,
distinct facilities remain distinct. The function does not infer an obligor
outcome by taking arbitrary facility labels.

Only mature, complete, known and eligible first-default windows contribute to
**both** observed and expected counts. Even an early observed default waits for
its cohort window to mature, avoiding a denominator selected by early events.
Future/ineligible/prior-default/immature/missing outcomes are explicitly excluded.
Missing forecasts are separately excluded before unit aggregation; `unit_windows`
therefore counts units with available compatible forecasts. Returned group totals
reconstruct the overall compared population. Hashes bind all supplied forecasts
and observation rows. Repeated time windows may remain dependent.

O/E is missing with `UNDEFINED` when both expected and observed counts are zero,
or with `INFINITE` when expectation is zero but defaults occurred. It is never
silently capped or replaced by a benign zero. Returned rates, counts, duplicate
counts and row-level exclusions expose exactly which cohort was compared.
