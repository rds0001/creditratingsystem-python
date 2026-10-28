# Fixed-origination vintage backtesting

`vintage_backtest(vintage_table, account_predictions, account_outcomes, *,
origin_cohorts, denominator_contract, design, uncertainty,
cluster_resamples=None, confidence_level=.95, context=None)` validates supplied
operational vintage cells, such as `build_vintage_performance` output, against
separately supplied account-level frozen probabilities and observed targets.
It does not construct an alternative target model.

Origin membership has unique `account_id`, `opened_at`, `known_at`,
`original_balance`, `currency`, `cluster_id`. Original balances are nonnegative;
opening cannot follow knowledge, and origins must be known by evaluation.
The denominator contract declares `method:'fixed_origination'`, weighting count
or original_balance, `exit_treatment:'first_competing_exit'`, `reference`,
`default_definition` and
`horizon_convention:'origination_month_end_plus_mob'`. The design target is
physical default conditional on performing at origin, with whole-month horizon
specifying the largest permissible MOB. MOB zero is the opening month's end,
not zero elapsed days for every account.

Each account prediction supplies `months_on_book`, matching model/version/
parameter IDs, `forecast_at`, `known_at`, `cumulative_pd`, `interval_pd`.
It must be frozen at actual account opening. Every original member, including
exits, needs a prediction for each requested cell. Cumulative PD must increase;
consecutive MOB interval probabilities must equal cumulative increments.

Each supplied account outcome has account/MOB, `period_end`, `known_at`,
`default_definition`, Boolean `complete`, Boolean `exited`, and binary-or-null
`cumulative_default`, `interval_default`. Observed default or first competing exit
requires `event_id` and `first_event_at` inside the opening-to-period-end window
and known by the outcome knowledge date. Interval defaults must fall within the
named calendar MOB interval. A complete no-event observation requires
`complete_through >= period_end`, without coverage extending beyond knowledge.
Future-known observations do not enter counts; missing outcomes remain explicit.
Default and competing exit cannot both be the first absorbing event.

The validator reconciles all original counts/balances, observed defaults,
incidence, exits, complete-account counts, calendar maturity and rates to the
operational table. Exits never shrink the original denominator. If any original
account lacks complete endpoint follow-up, the whole cell's outcome metrics are
NOT_APPLICABLE; the validator does not reweight survivors into a false full-cohort
claim. Known event counts remain visible. No censoring correction is inferred.

For complete mature cells, output contains cumulative and unconditional interval
prediction/outcome means, prediction-minus-outcome bias, MAE, RMSE and Brier.
The interval denominator remains the original cohort; it is not a conditional
hazard among current survivors. A separate remaining-first-event-risk count
shows original accounts minus cumulative first defaults and competing exits.
Original-balance weights measure monetary-weighted incidence, not account PD.

Uncertainty is none or actual supplied cluster_bootstrap. Each draw samples the
original number of clusters and is shared over all MOB cells, preserving repeated
accounts and overlapping horizons. Undefined draws suppress affected intervals.
There is no homogeneous-binomial or independent-MOB fiction.

The independent 100-account oracle includes 20 prepayments and 10 later defaults:
the original denominator remains 100, observed cumulative rate is .10, remaining
first-event risk count is 70. Constant forecast .15 gives bias .05 and Brier
.0925. Removing one final follow-up makes that cell non-estimable rather than
changing its denominator to 99 or 79. Tests construct the operational table using
the public native producer in each language.
