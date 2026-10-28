Behavioral development and monthly monitoring
============================================

Run ``examples/behavioral_development.py`` or source the native R example and
call ``run_behavioral_development``. The packaged JSON has three disjoint
customer populations: ten 2020 training accounts, thirty 2022 pool/calibration
accounts, and ten 2024 held-out accounts. Both calendar availability and customer
identity are checked. Pool fitting and grade calibration use the second prior
population; no held-out outcome selects parameters or thresholds.

The ten training accounts have four first defaults within sixty days. Real ridge
logistic fitting has an unpenalized intercept, giving the independent anchor
PD .4. Real discrete hazards use ten complete scheduled monthly intervals with
two defaults, then eight intervals with two defaults: hazards .2 and .25 give
survival .8 and .6. Scheduled interval endings represent a period event, not a
claim that the event happened on that endpoint. Original event dates remain in
the history. January31 and February29 two-month windows have sixty days in these
leap years; no thirty-day approximation replaces the calendar target.

Set ``CREDITRATINGSYSTEM_EXTENDED_TESTS=true`` or pass ``measured_fit=True``
(``TRUE`` in R) for the actual nonzero utilization model. It fits observed
balances divided by observed limits from the prior snapshots, with ridge .1.
Its coefficient is about .7574. An independent analytic gradient checks the
unpenalized intercept score and slope score plus .1 times the slope. This fit
then supplies the held-out behavioral predictions and expected default counts.
The full installed acceptance must call this variant as well as the tiny anchor.
Native glmnet warns about this deliberately small illustrative class; neither
fit is an empirical bank model, and no meaningful population precision is claimed.

Prior calibration grades have ten observations each and 1,4,2 defaults.
Adjacent pooling produces .1,.3,.3. Separately fitted retail risk pools merge
the second and third utilization bins and estimate their risks from those same
prior calibration observations. The known date is 2022-03-03. Each held-out
account is assigned using its contemporary feature and the frozen boundaries;
outputs show model PD, prior pool PD and prior utilization-grade PD separately.
They are alternative views, not multiplicative adjustments or silently combined
probabilities. Missing prior default workouts remain missing LGD/EAD evidence.
The pool API's ``not_defaulted_at_index`` and behavioral ``performing`` targets
refer to the same explicitly first-default account risk set in this recipe.

Payment obligations and dated payments produce DPD, state and payment-history
features from known information. Timely payment is not inferred from absent
coverage. Statement utilization drives the measured model; the displayed payment
history explains the observed account state. Eighteen eligible January/February
snapshots remain after the first two defaults. Their overlapping sixty-day
windows contain eight positive labels, but only ten distinct customer clusters.
The count report is descriptive: it has no binomial independence p-value.
Survival Brier values are 3.12/18 and4.48/18. Bootstrap draws always resample whole
customers, including every eligible window. Four illustrative draws are not an
adequate empirical confidence interval; undefined rank intervals stay null.
Early-known positive outcomes are retained in target history but every window
is still calendar-immature at the early evaluation cutoff. They are never used
as mature observations in a backtest or model fit.

Prior February-to-March roll counts produce [[.75,.25],[0,1]]. Two applications
of this explicitly homogeneous projection give default mass .4375. The exposure
path drops from100 to80 independently of default probability. This is a declared
projection, not a claim that calendar-month hazards are always equal. Vintage
uses the actual held-out events and preserves null results for immature ages.
Already-defaulted ``hold0`` has no performing prediction; its February15 default
is explicitly bound to the same open-workout episode and a genuinely earlier
recovery model. The March29 recovery estimate uses only cash known then, with
an explicit residual-tail assumption. Later observed cash remains separate.

``return_artifacts=True`` (R ``TRUE``) returns the original fitted models, pool
calibration, behavioral predictions, ValidationResult and sealed report, including
its exact replay inputs and policy. These original artifacts are not rounded or
re-sealed for cross-language comparison. The ordinary output is a labeled summary:
PD reporting uses eight decimals, pool summaries show actual counts/estimates,
and report findings come from the actual unmodified validation report. Local
report replay checks the original bindings. Independently fitted models may have
minute optimizer differences; frozen Python-origin and R-origin models replay in
both languages. No numerical tolerance exception is required for the summary.

The illustrative predeclared Brier threshold .3 applies only to this synthetic
review criterion. The actual report decision does not approve a bank model,
production release, fairness, or independent calibration. Manifests bind the
actual training rows, held-out data and customer split. All eighteen behavioral
monitoring recipe methods execute with linked inputs; supporting prediction and
open-workout APIs are used as part of the same account history.

Method reference: Friedman, Hastie and Tibshirani (2010), Regularization Paths
for Generalized Linear Models, Journal of Statistical Software33(1),
https://www.jstatsoft.org/article/view/v033i01 . The numerical count, hazard,
PAVA and stationarity oracles above are independently derived from the supplied
synthetic records; they are not proprietary bank calibrations.
