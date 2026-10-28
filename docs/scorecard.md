# Logistic scorecards and explanations

`fit_scorecard` estimates a logistic model on a **previously frozen** WOE or expert
feature matrix. Inputs include binary outcomes (default=1), ordered feature names,
target/training metadata, model ID/version and optional integer frequency weights.
`fit_spec` requires one `feature_kinds` entry (woe or expert) and one finite
`neutral_profile` value for each feature. Feature preprocessing and supervised
binning must be fitted within the training fold; declaring a feature as WOE does
not prove it was fitted without leakage.

Estimation uses the native binary-GLM engine, with intercept, logit link, explicit
rank and separation checks, convergence and likelihood-score checks, and expected
Fisher covariance under IID frequency-weight assumptions. No proprietary
coefficients or neutral profiles are supplied. Complete/quasi separation raises
`SEPARATION`; finite unpenalized coefficients are not fabricated. Model artifacts
include the neutral profile and fitted-design fingerprint.

`score_scorecard(fitted_model,transformed_features,feature_order=...,
expected_target=...)` requires the original feature order and an exactly compatible
target. It returns PD, default/nondefault log-odds, intercept, per-feature
contributions, reference contributions and differences from the neutral profile.
Intercept plus feature contributions reconstructs log-odds. The artifact can be
serialized and applied natively in the other language; applying it never refits.
Target horizon, physical/risk-neutral measure and conditioning remain those
explicitly declared for the model.

## Points and odds

`scale_score_points(log_odds,base_score=...,base_odds=...,
points_to_double_odds=...,odds_convention=...,score_direction=...)` always accepts
**default/nondefault** log-odds. `odds_convention` declares whether base_odds is
default_over_nondefault or nondefault_over_default. `score_direction` is
higher_is_safer or higher_is_riskier. Let eta0 be the declared base odds converted
to default log-odds, and F=PDO/log(2). Then:

    points = base_score + s * F * (log_odds - eta0)
    s = -1 for higher_is_safer, +1 for higher_is_riskier

Thus doubling default/nondefault odds reduces a safer score by PDO; doubling
nondefault/default odds increases it by PDO. No rounding or clipping occurs.
Returned inverse parameters reconstruct log-odds and PD from any point value.
The score has no interpretation without its odds and direction convention.

## Qualitative adjustments and reasons

`apply_qualitative_adjustments(qualitative_inputs,coefficient_spec,neutral_levels,
base_value=...,adjustment_scale=...)` accepts named observed levels, per-feature
level coefficients and a named neutral level. Each contribution is the observed
coefficient minus the neutral coefficient. Missing is an error, not an implicit
neutral answer. For `log_odds`, the output adds contributions to the base predictor
and calculates PD. For `rating_notches`, coefficient differences must be integer
notches; the output does not invent a PD or apply unknown rating-scale boundaries.
Using this function after a model already includes the same qualitative factor
would double count its effect; the caller must assign it to one stage.

`scorecard_reason_codes(score_contributions,reference_profile,reason_policy)`
compares matching named contribution maps. Declare `adverse_direction` as positive
or negative, `max_reasons` and a `reason_codes` map. It selects adverse differences
by decreasing magnitude, with feature-name lexical ties. `all_differences`,
`total_difference` and `unreported_difference` preserve exact reconciliation even
when only a few reasons are displayed. Inputs must already share a contribution
scale and intercept convention. These are model-contribution explanations,
not causal claims or an automatic lending decision.

The estimation backend follows
[R's GLM documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html).
Tests independently recover the grouped binomial MLE, reconstruct predictor
contributions, invert scores and reconcile neutral/reason arithmetic.
