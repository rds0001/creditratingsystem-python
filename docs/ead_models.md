# Conditional EAD and conversion-factor regression

`fit_ead_model(exposure_episodes, X_train, ...)` offers two explicit target modes.
Both require a physical `given_default` target with an integer-day `horizon`,
`time_unit='days'`, `event='ead'`, and `currency` matching the method argument. Training metadata
includes `data_fingerprint`, `split_id`, `known_at`, and optional `data_origin`.
Predictors are an identity-preprocessed numeric matrix with explicit `feature_order`.
Integer frequency weights represent independently replicated observations.

Each episode has unique `episode_id`, `reference_date`, `reference_known_at`,
`default_date`, `default_known_at`, `features_observed_at`, `features_known_at`,
nonnegative `reference_drawn`, `reference_limit`, `ead_at_default`, and `currency`.
The reference and all feature knowledge must be available at reference date;
default occurs exactly the target horizon later and is known by the training
cutoff. Features cannot become known before their observation. These explicit
row-level dates are checked; source-document truth remains the caller's responsibility.

For `target_mode='ccf'`, specify
`model_spec={'mean_model':'identity','covariance':'HC0'}`. Raw target CCF is
(EAD−reference drawn)/(reference limit−reference drawn). Nonpositive or at/below
`minimum_undrawn` denominators are excluded with episode IDs and reasons.
Remaining negative and above-one CCF values are retained. Frequency least squares
estimates the conditional mean with HC0 sandwich covariance. It does not identify
a full exposure distribution. Predictions reconstruct drawn+undrawn×mean CCF;
negative implied EAD is a controlled support error, never silently floored.
Mean standard errors propagate the coefficient covariance, holding reference
amounts fixed. These are parameter-mean errors, not individual-exposure prediction
intervals. Quantile requests are rejected for this mean-only model.

For `target_mode='direct_ead'`, specify
`model_spec={'positive_distribution':'lognormal','zero_model':...}`.
`zero_model='none'` requires all active targets positive; `'logit'` requires both
zero and positive observations; `'all_zero'` requires every active target zero.
The logit gate estimates positive-EAD probability once. The positive component
fits log(EAD) by frequency least squares and estimates lognormal residual variance
by its maximum-likelihood denominator N. Nonzero residual dispersion and full
positive-stage design rank are required. A zero-only model has an identified
point distribution without invented positive-stage parameters.

If positive probability is p, log amount mean is µ and residual variance is s²,
conditional expected EAD is p×exp(µ+s²/2). The distribution has mass1−p at zero;
quantiles above that atom use the rescaled positive lognormal probability.
Coefficient covariance is HC0; predictive quantiles depend on the full lognormal
assumption. They do not include estimation uncertainty or arbitrary cluster effects.
Positive-stage marginal feature ranges are diagnosed separately from overall ranges.

`predict_ead` matches the frozen target, feature order, currency, and supplied
context. `reference_dates` cannot precede training knowledge, and `default_dates`
must preserve the exact horizon. CCF predictions additionally require aligned
`reference_drawn` and `reference_limit`; direct EAD rejects these extra amounts.
Optional `quantile_levels` are strictly increasing interior probabilities.
Models retain training provenance and input fingerprints and replay in Python or R.

The same predictor accepts an `exposure_schedule` result: supply
`facility_covariates=None`, `feature_order=[]`, `expected_target=None`, currency,
reference dates equal to the schedule's as_of and exact period-end default dates.
No interpolation, regression predictors or model quantiles apply. Complete repayment
before the selected exposure snapshot produces zero under the declared schedule
ordering. These are conditional exposure methods; they do not estimate PD or
regulatory conversion factors.
