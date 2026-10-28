# Calibration and model selection

These functions estimate explicit statistical transformations and record the full
selection procedure. Supply independent calibration observations; fitting a
calibrator is not validation of the resulting model on the final test set.
Python functions are available from `creditratingsystem.calibration`; R uses the
same function names. Array indices in supplied plans are **zero-based in both
languages**, including R lists.

## Fitted transformations

```python
calibrate_beta(base_pd, outcomes, weights=None, *, metadata,
    monotonic=True, boundary_policy="reject", epsilon=1e-12, context=None)
fit_hierarchical_calibration(base_pd, outcomes, groups, weights=None, *,
    metadata, variance_bounds=(1e-8, 25.0), context=None)
calibrate_lgd_level(base_lgd, target_lgd, weights=None, *, metadata,
    transformation="additive", context=None)
apply_calibrator(calibrator, values, *, target, base_model_version,
    context=None, groups=None)
```

`base_pd` contains probabilities and `outcomes` observed binary events. `weights`
are nonnegative integer frequency weights, with positive total. `groups` are
nonempty group strings, supplied again when applying hierarchical calibration.
`base_lgd` contains loss fractions, including raw negative or above-one values for
the additive transformation. `target_lgd` is the desired weighted loss mean.
LGD weights may instead be nonnegative real exposure weights when metadata says
`weight_type="exposure"`. This weights the calibration target by exposure;
it does not silently replace the arithmetic loss mean with an EAD-weighted mean.

`metadata` follows the existing calibration contract: explicit `target` with
`event`, `horizon`, `time_unit`, `probability_measure`, `conditioning`;
`base_model_version`, `population_definition`, `outcome_policy`, `weight_type`,
`data_origin`; and `calibration` with `start`, `end`, `known_at`, `split_id`,
`prediction_origin`. Empirical fits require `HOLDOUT` or `OUT_OF_FOLD` prediction
origin. External LGD level targets also allow `EXTERNAL_TARGET`. Dates are ISO
calendar strings. `context` is the optional calculation context and must agree
with the declared target.

Beta calibration fits
`expit(intercept + a*log(p) - b*log(1-p))` by weighted Bernoulli likelihood.
`monotonic=True` constrains both `a` and `b` to be nonnegative; unrestricted mode
may reverse ranking. `boundary_policy="reject"` rejects zero and one, while
`"clip"` applies the declared `epsilon` at fit and application. Coefficients
`(0,1,1)` give the identity. Singular and non-finite fits fail. Inverse-information
covariance is an interior approximation; a reported active inequality constraint
requires boundary-aware inference. This is the probability transformation of
[Kull, Silva Filho and Flach (2017)](https://proceedings.mlr.press/v54/kull17a.html),
not a beta-distributed LGD response model.

Hierarchical calibration uses the input logit as an offset, a global intercept,
and independent normally distributed group intercepts. It estimates the global
intercept and **variance** by a Laplace marginal likelihood, with analytic
profile derivatives and scalar solutions for conditional group modes.
`variance_bounds` are explicit positive optimization bounds; a boundary estimate
is reported. Existing groups use their conditional mode. Unknown groups use a
40-node Gaussian quadrature integral over the fitted normal population, so their
prediction generally differs from simply inserting a zero intercept. Group
curvature uncertainty conditions on fitted hyperparameters; it omits their
estimation error and the Laplace approximation error.

LGD `transformation` is `additive`, `multiplicative`, or `logit`. The first shifts
raw economic losses without clipping; the second requires nonnegative losses,
a positive base mean and a nonnegative target; the third requires strictly
interior fractions and solves a finite log-odds shift. The artifact reports the
achieved mean and all fitted values. Positive multipliers preserve strict order;
a zero target gives a zero multiplier and ties. Frozen application checks the
artifact hash, schema, transformation parameters, target and model version.
For hierarchical application, `groups` must align with `values`. Plain JSON
round trips preserve each artifact; independently fitted floating coefficients
need numerical comparison, not equality of independently generated hashes.

## Grade estimates, low defaults, uncertainty and weighting

```python
calibrate_grade_pd(counts, defaults, *, grades=None, prior_alpha=0.0,
    prior_beta=0.0, monotonic=True, target_default_rate=None,
    default_grade=None, confidence=0.95, target_weights=None, context=None)
low_default_bound(defaults, counts, *, confidence=0.95,
    dependence="independent", correlation=0.0, pooling="none",
    integration_tolerance=1e-8, context=None)
estimate_margin_of_conservatism(base_estimate, components, *,
    confidence=0.95, correlation=None, already_included=(), bounds=None,
    context=None)
population_reweighting(sample_features, target_distribution, *,
    method="poststratification", cap=None, outcome_dependent=False, context=None)
```

`counts` and `defaults` are aligned integer observations and events, with positive
counts and `0 <= defaults <= counts`. `grades` names best-to-worst performing
grades in their declared order. Without names, zero-based strings are used.
`prior_alpha` and `prior_beta` are explicit beta pseudo-counts **per grade**;
both default to zero. `monotonic` applies weighted pool-adjacent-violators to
posterior means (or empirical rates without a prior). The result reports each
pooled membership. `default_grade`, when supplied, is excluded from performing
calibration and must contain only defaults. `target_default_rate` optionally
shifts pooled logits to a target weighted mean, preserving structural zeros and
ones; unattainable finite shifts fail. `target_weights` controls this target
reconciliation, defaulting to grade counts, and does not change the binomial
pooling likelihood. A prior gives equal-tailed beta posterior intervals; no
prior gives exact binomial intervals. Intervals describe **pre-target pooled**
estimates conditional on selected pools, not post-selection, simultaneous or
external-target uncertainty. A zero-default cell retains a positive upper bound.

Low-default `dependence` is `independent` or `gaussian_one_factor`.
`correlation` is a fixed Gaussian asset correlation, zero for independence.
For `pooling="ordered_tail"`, each best-to-worst grade is pooled with every
subsequent grade; both raw tail bounds and cumulative monotone upper bounds are
reported. `pooling="none"` treats rows separately. The upper root solves
`P(D <= observed_defaults | pd)=1-confidence`. For no independent defaults this
is exactly `1-(1-confidence)**(1/n)`. Correlated mode integrates conditional
binomial probabilities over a shared standard normal factor on [-10,10], with
`integration_tolerance` controlling absolute/relative integration error; the
reported numerical error excludes the negligible normal-tail truncation.
This supplies the ordered conservative construction discussed by
[Pluto and Tasche](https://arxiv.org/abs/cond-mat/0411699), under a declared
one-period dependence model. It does not give a universal confidence guarantee
under unknown correlation or simultaneous confidence across multiple grades.

MoC `base_estimate` is a scalar. Every entry of `components` declares a unique
`source_id`, `category` (`data`, `method`, `general`), nonnegative `adverse_bias`
and `standard_error`. `already_included` lists sources already included in the
base; their contributions are zeroed. Duplicate component sources fail.
`correlation` is a symmetric positive-semidefinite correlation matrix in component
order; omission explicitly means independence. The margin is total adverse bias
plus `normal_quantile(confidence)*sqrt(s' C s)`. `bounds`, if supplied as lower and
upper limits, clips the final estimate and separately reports the clipping
adjustment and unbounded estimate. This is a declared normal uncertainty policy,
not an automatic regulatory MoC. Zero declared uncertainty produces zero margin.

For poststratification, `sample_features` is a stratum-string vector and
`target_distribution` maps strata to proportions summing to one. A positive
target stratum absent from the sample fails. For `method="membership_odds"`,
`sample_features` contains externally estimated strictly interior
`P(target membership | features)`; `target_distribution` contains the design
`sample_prior` and `target_prior`, summing to one. The density ratio is
`q/(1-q) * sample_prior/target_prior`. This function does not fit propensity
models or certify covariate overlap. `outcome_dependent` explicitly declares
outcome-dependent sampling. Optional `cap` clips raw ratios **before**
normalization to mean one; final normalized weights may exceed that cap.
The result reports raw/final weights, clipping count, effective sample size and
achieved stratum distribution, exposing the cap's effect on target balance.

## Selection and resampling

```python
fit_stepwise(X, outcomes, *, feature_order, candidate_groups,
    mandatory_terms=(), hierarchy=None, direction="bidirectional",
    criterion="AIC", estimator="GLM", weights=None, max_steps=100,
    lr_alpha=0.05, training=None, context=None)
fit_regularization_path(X, outcomes, *, feature_order, penalty_grid,
    split_plan, estimator="GLM", elastic_net_alpha=0.0,
    scoring="log_loss", weights=None, context=None)
resample_fit_uncertainty(X, outcomes, *, feature_order, resampling_indices,
    resampling_units, unit_kind, estimator="GLM", weights=None,
    confidence=0.95, prediction_X=None, selection_spec=None, context=None)
fit_model_selection_protocol(X, outcomes, *, feature_order, candidates,
    split_plan, metadata, search_budget, scoring="log_loss", context=None)
```

`X` is a finite numeric matrix; `feature_order` contains distinct column names.
`outcomes` is aligned and fully observed. Estimator and criterion names accept either letter case. The estimator is `OLS` (unit weights),
`WLS` (positive inverse-variance weights) or `GLM` (Bernoulli logit, positive
integer frequency weights). The Gaussian likelihood estimates its residual
variance and includes that parameter in AIC/BIC; WLS includes the weight
determinant. No columns or rows are silently dropped. Rank deficiency,
separation, zero Gaussian residual variance and failed convergence are explicit.

`candidate_groups` maps group names to disjoint lists of columns; columns not
listed cannot enter. `mandatory_terms` names groups always present. `hierarchy`
maps interaction/group names to required parent groups; cycles and missing
mandatory parents fail. `direction` is `forward`, `backward` or `bidirectional`.
Backward starts at all candidate groups; other modes start at mandatory groups
plus an intercept. `criterion` is `AIC`, `BIC` or `LR`. AIC/BIC compare full
likelihoods on identical rows. LR adds if the nested chi-square p-value is below
`lr_alpha`, removes if it exceeds that threshold, and reports degrees of freedom
and statistic. A fixed visited-state set prevents cycles. Improvements tied
within 1e-10 use lexical action/group order. `max_steps` limits accepted moves.
All attempted candidates, including failures, remain in the evidence. These
choices follow likelihood selection principles described in
[R's step documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/step.html).
No fixed-model inferential confidence claim is made after adaptive selection.

A `SelectionModel` is a frozen JSON record with feature order, intercept,
coefficients, center, scale, link, penalty, likelihood, and content hash.
Its scoring equation is `intercept + ((X-center)/scale) @ coefficients`, followed
by `expit` for GLM. It is returned directly with the evidence; it is not silently
converted into an unrelated production-model class. `artifact_hash` hashes the
record without the hash field. Callers must preserve feature order and verify
the digest when replaying externally.

`penalty_grid` lists unique nonnegative lambda values. `elastic_net_alpha` is zero
for ridge, one for lasso, or intermediate for elastic net. The objective is
weighted **average** negative log likelihood (Gaussian half squared error), plus
`lambda*((1-alpha)*sum(beta**2)/2 + alpha*sum(abs(beta)))`; the intercept is
unpenalized. Features are centered/scaled independently inside each training
fold. After selection the model is refitted only on development rows, using its
own scaling. Scores are `log_loss` or `brier` for GLM, `mse` for Gaussian models.
Equal final scores prefer stronger regularization. Every fold must succeed for
a penalty to remain eligible.

`split_plan` is an object with `rows` and `folds`. Each row declares `entity_id`,
`as_of`, `label_known_at`, and `role` (`development`, `calibration`, `test`). Each
fold declares `train` and `validation` lists of zero-based row indices. Fold
entities must be disjoint and all training labels must mature strictly before
validation begins. Selection folds may use development rows only. Stage entity
sets are disjoint, with all earlier-stage labels mature before the next stage.
These checks are applied to the supplied rows; responsibility for omitted
relations and data definitions remains with the user.

`resampling_indices` contains the complete shared zero-based row-index list for
each replicate, including repeats. `resampling_units` labels every original row
with a predeclared cluster or time block; `unit_kind` is `cluster` or
`time_block`. Each chosen unit must be repeated in full with equal row
multiplicity. Blocks must already reflect the relevant time dependence; this
API does not infer block length or bootstrap stationarity. `prediction_X`
optionally supplies aligned prediction columns, defaulting to original X.
`selection_spec` optionally supplies the stepwise keyword arguments; selection
then repeats independently in every replicate. Each failure remains counted.
Percentile intervals at `confidence` use successful replicates and are omitted
when fewer than two succeed; failure selection can compromise their coverage.

The full protocol uses GLM candidates, each declaring unique `id`, optional
`features`, `penalty_grid`, and `elastic_net_alpha`. `search_budget` must cover
all candidates; none are silently skipped. Inner development folds choose the
candidate/path, the winning specification refits on development rows, and
independent calibration rows fit a logistic calibrator (score tolerance 1e-8).
The `metadata` calibration window must cover the actual calibration rows and
label maturity and declare `HOLDOUT`. The final test receives no predictions or
metrics during selection. Shuffling its labels leaves both frozen selected and
calibration artifacts unchanged. All unsuccessful candidates are reported.

Standalone models explicitly record a `training_only` role and unknown cutoff
when no dates are supplied. `fit_stepwise(training={"known_at": "2020-12-31",
"role": "training_only"})` declares a cutoff without inferring chronology.
Regularization paths derive their development-only cutoff from checked row dates.
Every SelectionModel includes a diagnostic hash. The full protocol also returns
`selected_frozen_model`, a common classifier artifact usable with `predict_pd` and
standard model serialization; its frozen preprocessing and target are verified.

The role of the Laplace likelihood approximation is also documented by the
[lme4 authors](https://lme4.github.io/lme4/reference/glmer.html). The elastic-net
penalty family follows [Friedman, Hastie and Tibshirani](https://www.jstatsoft.org/article/view/v033i01);
this implementation solves it with proximal gradient (and the direct Gaussian
ridge solution), not their coordinate-descent implementation.

Default tests retain small real fits and analytical checks for every API. Set
`CREDITRATINGSYSTEM_EXTENDED_TESTS=true` for the complete 27-case stepwise
direction/criterion/estimator grid and all three penalty modes. Python additionally
marks these cases `extended`; native R uses the same environment flag.
