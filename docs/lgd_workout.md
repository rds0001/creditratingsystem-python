# Residual recovery and open workouts

`fit_recovery_hurdle` fits a deliberately defined residual settlement stage. It combines a cure probability, a positive-recovery probability among noncured resolutions, a lognormal positive recovery ratio and a censored accelerated failure time model for resolution. `estimate_open_workout_lgd` then combines known past cashflows with the conditional remaining terminal value. This is useful for accounts that enter a final settlement stage; intermediate future recovery streams need a different payment model.

The horizon is explicit: `target.event="residual_terminal_loss_fraction"`, `target.horizon_role="TIMING_REPORTING_REFERENCE"`, `target.probability_measure="PHYSICAL"`, `target.conditioning="given_default"`, `target.time_unit="days"`, and `target.currency` must equal the reporting currency. The core positive `target.horizon` is a resolution reporting reference, **not a truncation of the ultimate payout**. `predict_lgd` returns `payout_horizon="ULTIMATE"`; its mean and quantiles describe the undiscounted residual terminal loss fraction, not whole-workout discounted LGD or loss at that finite reporting horizon.

## Training contract

```python
fit_recovery_hurdle(
    episodes, cashflows, X_train, *, feature_order, stage_spec, target,
    training, model_id, model_version, currency, fit_intercept=True,
    context=None,
)
```

`X_train` has one row per episode and the declared feature order. An empty feature order uses a matrix with one empty row per episode. Independent observations have unit frequency; no exposure weights or automatically inferred sampling weights are used. Training must declare `workout_population="open_and_closed_defaulted_landmarks"` and the ordinary fingerprint, split ID, knowledge date and provenance fields.

Every `stage_spec` must contain:

| Field | Required value |
|---|---|
| `payout_definition` | `RESIDUAL_TERMINAL_GROSS_RECOVERY_OR_CURE_VALUE` |
| `cashflow_pattern` | `SINGLE_TERMINAL_PAYOUT_AFTER_REFERENCE` |
| `timing_outcome_independence` | `CONDITIONAL_ON_REFERENCE_COVARIATES` |
| `cashflow_completeness` | `COMPLETE_THROUGH_OBSERVATION_END` |
| `payout_horizon` | `ULTIMATE` |
| `positive_distribution` | `lognormal` |
| `timing_distribution` | `weibull`, `lognormal` or `loglogistic` |
| `cure_residual_value_ratio` | Nonnegative declared ratio |
| `landmark_definition` | Nonempty description of the common landmark selection rule |

The exact stage object is frozen and must match open-workout prediction. Covariates describe the reference landmark; they are not silently replaced by later updated information. A new landmark is possible only if it follows the same training-stage selection rule and covariate definition.

Episode fields are `episode_id`, `default_date`, `default_known_at`, `reference_date`, `reference_known_at`, `features_observed_at`, `features_known_at`, `currency`, `ead_at_default`, `residual_exposure_at_reference`, `observation_end`, and explicit nullable `resolution_date`/`resolution_known_at`. Observed resolutions also require numeric `cured` equal to zero or one and nonnegative `cure_value`. Default must already be known at the landmark, the reference snapshot must be known on its reference date, and features must be observed and known no later than that date. Follow-up must be positive and end no later than the training cutoff. Resolution contributes an event only if it occurred by the observation end and was known at the training cutoff; otherwise the row is right-censored. Future cure labels are ignored. Positive residual exposure is required for training ratios; default EAD is a separate amount.

Cashflows have unique `cashflow_id`, matching `episode_id`, `cashflow_date`, `known_at`, `currency`, `kind` (`RECOVERY`, `DIRECT_COST` or `INDIRECT_COST`) and nonnegative `amount`. This stage requires amounts already converted to the declared reporting currency. A realized flow cannot be known before it occurs. Only cutoff-known flows enter estimation. Positive post-landmark recoveries must occur at the observed terminal resolution date; intermediate recovery payments violate this model's scope. Prior cashflows are not part of the residual payout target. Costs are separate from gross recovery.

For cured cases, the declared cure asset value must equal residual exposure times the fixed cure ratio, and no terminal cash recovery may duplicate it. Noncured cases have zero cure value. A cure value is the restored asset's economic value, not automatically a cash collection or a par asset value.

At least one cure, one noncured zero payout and two positive noncured payouts are required, together with full-rank, nonseparated gate designs and positive lognormal residual dispersion. Cure and positive-payout gates use closed observations; the timing likelihood uses closed and censored observations. This factorization assumes that terminal outcome and amount are independent of resolution time conditional on landmark covariates, and that right censoring is conditionally independent. The completeness declaration is a user assertion about the supplied history, not a reconstruction of unreported payments. Missing cashflows must not be represented as observed zeros.

## Frozen residual prediction

Use `predict_lgd(fitted_model, X, feature_order=..., expected_target=..., default_condition="given_default", quantile_levels=...)`. The frozen target, feature order and context must match. Let `p_c` be cure probability, `p_r` positive recovery probability conditional on noncure, `c` the fixed cure ratio and `mu,sigma²` the positive log-ratio parameters. The expected residual terminal value ratio is

`p_c*c + (1-p_c)*p_r*exp(mu + sigma²/2)`.

Residual mean loss is one minus this ratio. Quantiles include the zero-recovery loss atom at one and the cure-value atom at `1-c`; positive recovery follows its fitted lognormal distribution. Negative economic losses are retained. Marginal support violations include both gates and the positive-amount subsample. They do not establish joint support or validate the timing/outcome independence assumption.

The lognormal moment and inverse-survival formulas follow the [NIST distribution reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3669.htm). Timing reuses `survival.fit_aft_model` and its native R counterpart; the Weibull survival parameterization is documented by [lifelines](https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html). The cure/recovery factorization is an explicit model assumption, not a claim established by those references.

## Open-workout valuation

```python
estimate_open_workout_lgd(
    episodes, cashflows, fitted_recovery_model, X,
    valuation_contracts, future_cost_cashflows, *,
    feature_order, expected_target, stage_spec, as_of, currency,
    extrapolation, context=None,
)
```

The episode schema is the same, with `as_of` replacing the training observation end. Both the model and all component models must have been available by the evaluation date. `residual_exposure_at_reference` can be zero at prediction. Each valuation contract has a matching `episode_id`, the exact-date `discount_curve` and `fx_contract` used by `discount_recoveries`, strictly increasing `future_resolution_dates`, `resolution_time_convention="RIGHT_ENDPOINT"`, explicit nullable `tail_discount_factor` and `tail_assumption_reference`, and `unmodeled_future_costs_assumption="NONE_BEYOND_SUPPLIED"`. Curves are anchored at default and known at `as_of`. No interpolation is inferred. All future value and cost amounts are already in reporting currency.

Future costs use the cashflow row schema, must have a cost kind, future payment date, current knowledge and a distinct ID. They are unconditional dated costs: they are not multiplied by survival probabilities, and no additional future costs are estimated. A closed final workout cannot retain an unpaid future cost schedule.

For age `a` since the reference date, conditional remaining survival is `S(a+t)/S(a)`. Each future grid interval's resolution mass is multiplied by its right-endpoint discount factor and the expected residual terminal value. `extrapolation` explicitly selects `error`, `flat` or `parametric` according to the survival API. No last observed resolution is treated as the end of all possible future recovery.

The result reports known discounted net cashflows, realized cure asset value, expected discounted future terminal value within the grid, supplied future costs, unresolved probability, undiscounted value beyond the grid and final economic loss/LGD. Known past net flows enter once. If remaining value has unknown tail discounting, `expected_final_loss` and `expected_final_lgd` are null; `expected_loss_within_grid` and `expected_lgd_within_grid` remain available and visibly omit the unresolved tail value. A supplied tail factor is the user's conditional average discount factor beyond the last grid date and requires a nonempty economic assumption reference. Closed cases use their actual cashflows and cure value without a forecast. Zero EAD leaves the ratio null while preserving the monetary loss.

## Small fitted example

```python
from datetime import date, timedelta
from creditratingsystem.lgd import fit_recovery_hurdle, predict_lgd

stage = dict(
    payout_definition="RESIDUAL_TERMINAL_GROSS_RECOVERY_OR_CURE_VALUE",
    cashflow_pattern="SINGLE_TERMINAL_PAYOUT_AFTER_REFERENCE",
    timing_outcome_independence="CONDITIONAL_ON_REFERENCE_COVARIATES",
    cashflow_completeness="COMPLETE_THROUGH_OBSERVATION_END",
    payout_horizon="ULTIMATE", positive_distribution="lognormal",
    timing_distribution="weibull", cure_residual_value_ratio=1.,
    landmark_definition="Entry into terminal settlement stage",
)
target = dict(event="residual_terminal_loss_fraction", horizon=365,
    horizon_role="TIMING_REPORTING_REFERENCE", time_unit="days",
    probability_measure="PHYSICAL", conditioning="given_default", currency="EUR")
training = dict(data_fingerprint="example", split_id="training",
    known_at="2024-12-31", data_origin="SYNTHETIC",
    workout_population="open_and_closed_defaulted_landmarks")
episodes, flows = [], []
for i in range(8):
    resolution = (date(2024, 1, 1) + timedelta(days=10 + 5*i)).isoformat()
    episodes.append(dict(episode_id=str(i), default_date="2023-12-01",
        default_known_at="2023-12-01", reference_date="2024-01-01",
        reference_known_at="2024-01-01", features_observed_at="2024-01-01",
        features_known_at="2024-01-01", currency="EUR", ead_at_default=100.,
        residual_exposure_at_reference=80., observation_end="2024-04-01",
        resolution_date=resolution if i < 6 else None,
        resolution_known_at=resolution if i < 6 else None,
        cured=int(i < 2), cure_value=80. if i < 2 else 0.))
    if i in (4, 5):
        flows.append(dict(episode_id=str(i), cashflow_id=str(i),
            cashflow_date=resolution, known_at=resolution, currency="EUR",
            kind="RECOVERY", amount=30. if i == 4 else 100.))
fitted = fit_recovery_hurdle(episodes, flows, [[] for _ in episodes],
    feature_order=[], stage_spec=stage, target=target, training=training,
    model_id="settlement", model_version="1", currency="EUR")
result = predict_lgd(fitted, [[]], feature_order=[], expected_target=target,
    default_condition="given_default", quantile_levels=[.1, .5, .9])
print(result.value["mean_lgd"])
```
