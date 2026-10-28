Paired model review, rollback and replay
======================================

The Python and native R ``champion_challenger`` examples compose local registry,
role assignment, prediction, mature cohort construction, paired model comparison,
feature and calibration drift, supplied group diagnostics, linear and Shapley
explanations, policy findings, explicit rollback and historical replay.

The two frozen synthetic models were specified before three untouched 2025 test
cohorts. They are analytically chosen reference coefficients, not a fitted real
bank model. Each month has four independent obligors with actual labels 0,1,0,1.
The reference model predicts .25,.75,.75,.75; the candidate predicts .8,.9,.8,.9.
Thus AUC improves from .75 to 1, while Brier worsens from .1875 to .325 and the
observed-minus-expected rate gap moves from -.125 to -.35. No single metric
promotes a model. Twelve observed outcomes illustrate arithmetic and workflow,
not enough data for a production approval.

A prior explicit assignment activates version 2 before these outcomes exist.
After the monitoring evaluation, a separately supplied local review action rolls
back to version 1. The historical version 1 prediction is .75 before and after
that action. Both original model versions and their recorded seals remain intact.
All assignments are local artifacts; the workflow performs no external action.
The fixed 2024/2025 dates are part of this reference history. Bank users replace
models, case features, actually observed events, coverage, resampling plans and
review policy with their own chronologically consistent inputs.

The monthly calibration residuals feed EWMA with smoothing .5, giving absolute
statistics .175, .2625 and .30625. Its nondegenerate null uses explicit independent
Bernoulli draws at the candidate's four probabilities, stored as common reference
paths before monitoring; 99 separate paths validate its finite-horizon false
alarm rate. These parameters and the illustrative .2 review threshold are not
regulatory standards. Cluster bootstrap draws resample complete independent
obligors; 32 replicates keep this example quick and do not establish precise tail
confidence bounds. With bank data, use an adequate prespecified resampling plan.

Supplied synthetic group labels are never inferred. Each group has only six
observations, so the eight-observation minimum explicitly reports insufficient
information. No credit decisions are invented: selection rates are unidentified.
Removing nondefault coverage leaves only six observed defaults; the workflow
preserves missing outcomes and cannot identify AUC from a one-class sample.

The linear explanation sums on the log-odds scale. Exact two-group Shapley values
use the actual candidate prediction callback and an explicit zero-feature
background: the unused x contributes zero, z contributes .9 - 6/7 probability
points. The background is a mathematical reference, not a feasible intervention
or a causal claim. All result fields and historical hashes are compared between
the independent native implementations; no optimizer or hash exception is used.
