# Residual workout estimation for specialized finance

Run `examples/specialized_open_workout.py`, or source its native R counterpart and
call `run_specialized_open_workout()`. Python `run(data=None,
verify_reference=True)` and R `run_specialized_open_workout(data=NULL,
verify_reference=TRUE)` accept the dictionary/list schema in
`specialized_open_workout.json`; disabling reference verification removes only
the hand-oracle assertions.

Each product fits its own recovery hurdle and censored Weibull timing model on
16 explicitly specified historical reference-date workouts. Twelve resolve:
four cure, four yield zero terminal recovery, and four yield positive terminal
asset recovery. Four remain right censored through their observation end. No
unobserved resolution is fabricated. Product-specific durations and positive
recovery scales differ; this is not a shared fitted corporate model applied to
unrelated assets. All historical data are available before the new workout's
reference date. The examples are small method acceptance cohorts, not claims of
statistical adequacy for a bank's calibration.

For positive outcomes, log recovery ratios are `log(s)+[-.3,-.1,.1,.3]`.
The intercept-only lognormal maximum likelihood mean is `log(s)` and variance
is `.05`. Thus expected terminal value per residual exposure is
`(1 + s*exp(.025))/3`: one third cure at the explicit residual cure value, and
one third positive lognormal recovery. Product scales are `.6,.7,.55,.5,.4`.
Known interim recovery is 10% of original EAD; residual exposure is 90%. The
independent ultimate LGD oracle is `1 - [.1 + .9*(1+s*exp(.025))/3]`.

Valuation deliberately declares a flat, zero-interest discount curve, with
explicit ultimate-tail discount factor one. This isolates the ultimate recovery
calculation: the full unresolved timing mass is included, rather than discarded
at the end of a finite grid. Positive future receipts and explicit supplied costs
are treated by the granular estimator; the example assumes no further unmodelled
costs. These are open workouts and the output is an **estimated** final LGD, not
an observed closed-workout outcome.

The example returns training and stage metadata, original EAD, known discounted
cashflows, estimated ultimate loss/LGD, and independent reference algebra.
Timing models are actually fitted and used by the estimator. This example reports
ultimate financial quantities only; it does not claim a separate timing-grid
validation or a full product acceptance. The library's granular estimator retains
resolution-grid probabilities and tail values for callers requiring those outputs.
