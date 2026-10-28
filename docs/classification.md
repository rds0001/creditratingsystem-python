# Binary PD estimation

`fit_binary_glm` estimates a Bernoulli likelihood with an explicit `logit`,
`probit` or `cloglog` link. `predict_pd` applies the resulting frozen model and
also dispatches the supported frozen model families through their explicit contracts.
Scorecard offsets are unsupported, and feature/target checks remain identical.
Python uses statsmodels IRLS; R uses native `stats::glm.fit`. Both return the
same model structure, and both require explicit feature names, intercept choice,
event, horizon, probability measure, conditioning and training identity.

The input matrix contains already transformed numeric features. No imputation,
scaling, variable selection, calibration or implicit intercept is performed.
Use a matrix with zero columns for an intercept-only model. Training targets
must be observed numeric zeros or ones. Outcomes of unfinanced applications
are unavailable, and must not be encoded as zero.

```python
import numpy as np
from creditratingsystem.classification import fit_binary_glm, predict_pd

target = {"event": "default", "horizon": 1, "time_unit": "years",
          "probability_measure": "PHYSICAL", "conditioning": "performing"}
training = {"data_fingerprint": "analytic-counts", "split_id": "train",
            "known_at": "2025-12-31", "data_origin": "SYNTHETIC"}
fit = fit_binary_glm(np.empty((2, 0)), [0, 1], link="logit",
    feature_order=[], fit_intercept=True, target=target, training=training,
    model_id="intercept-example", model_version="1", weights=[7, 3])
pd = predict_pd(fit, np.empty((1, 0)), feature_order=[], expected_target=target)
assert abs(pd.value[0] - 0.3) < 1e-8
```

`weights` are nonnegative integer frequency counts: weight seven repeats the
same observed row seven times in the likelihood. Zero-weight rows are excluded.
Fractional, sampling, class-balancing and exposure weights are rejected by this
API. The fitted covariance is inverse expected Fisher information with unit
binomial dispersion and independent observations. It is not a cluster-robust
or survey covariance. AIC and BIC use the unpenalized Bernoulli log-likelihood;
BIC sample size is the sum of frequency weights.

An `offset` is a known additive link-scale quantity with one value per row.
If supplied during fitting it must be supplied during prediction, even if the
training offset was zero. It is never estimated or copied from training rows.
Prediction requires the complete matching `expected_target` and exact feature
order. Any supplied context horizon, time unit, measure or conditioning must
also match. Model reload changes neither coefficients nor target semantics.

Before estimation the implementation checks full design rank at relative
singular-value tolerance 1e-10 and solves a bounded linear program for complete
or quasi separation. For signed, column-normalized design rows A, the program
maximizes the average of A*b subject to A*b >= 0 and -1 <= b <= 1. A positive
optimum identifies a direction that improves at least one likelihood term
without worsening another. A mean margin above 1e-8 is rejected; feasibility
tolerance is 1e-7. These are declared numerical tolerances, not proofs at
arbitrary precision. A failed LP, rank deficiency, nonconvergence, large score,
unusable covariance or fitted numerical probability boundary raises an error.
There is no implicit penalized fallback.

Diagnostics include likelihood, deviance, AIC/BIC, effective frequency count,
rank, condition number, score, scaled-score threshold, iteration count, solver
warnings and separation check. Standard errors follow from the covariance
diagonal. The model records identity preprocessing and preserves coefficients,
expected-Fisher covariance, offset contract, target, training metadata and
parameter origin `ESTIMATED` in declarative JSON-compatible data.

The inverse links are logistic expit, standard-normal CDF, and
`-expm1(-exp(eta))`. The covariance uses information
`X' diag(w * (dp/deta)^2 / (p*(1-p))) X`. Independent test cases use two groups
with observed default proportions 0.2 and 0.6: the exact intercept is g(0.2),
and the exact group coefficient is g(0.6)-g(0.2), for each link separately.
Intercept, offset, frequency replication, covariance, both separation forms,
JSON reload and target mismatch cases are also tested. These tests establish
computational behavior; real-portfolio performance needs independent validation.

Primary implementation references: [R GLM documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html),
[R binomial families](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/family.html),
[statsmodels GLM](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html),
[SciPy linear programming](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html),
[lpSolve manual](https://cran.r-project.org/web/packages/lpSolve/lpSolve.pdf).


## Generic frozen prediction

`predict_pd(fitted_model, X, feature_order=..., expected_target=..., offset=None,
context=None)` supports binary GLM, discriminant models, naive Bayes, Firth and
penalized logit, explicitly default-mapped ordinal models, GAM, random forests,
gradient boosting, Bayesian models, MLP, GRU/LSTM sequence models, logistic
scorecards, stacked PD models, frozen model-selection protocols, joint-selection
PD models and empirical market-to-physical bridges. An SVM margin remains an
uncalibrated score and is rejected here; use its score API and an explicitly fitted
calibrator. Unknown model families and conflicting method/parameter kinds fail.

Numeric models use a transformed feature matrix; naive Bayes also accepts typed
categorical/mixed row tables with declared missing/unknown handling. Grouped Bayesian
models use `{X, groups}`. Sequence models use their dated/masked sequence records,
not a flattened matrix. Native R accepts numeric row-list matrices on the public
prediction path, preserving the other structures. Integer-valued categorical
indices round-trip through language-independent JSON numeric representations.

Feature order and the entire expected target must match. Any supplied context event,
horizon, time unit, probability measure and conditioning must agree, and context
`known_at` cannot precede model training availability. Stored transformations are
applied by the owning frozen model; no preprocessing refit, automatic holdout
calibration or change of target is introduced. In particular a PIT prediction is
never labelled TTC by this dispatcher.

Every successful result contains one finite probability per input row plus common
lineage: `model_method_id`, `model_hash`, `prediction_input_hash`, `target`,
`prediction_count`, `training_data_fingerprint`, `training_known_at` and
`calibration_applied`. Input hashes use the semantic data array/record representation
(with missing NaN values represented as JSON null), exact feature order and supplied
offset. `FROZEN_PROTOCOL` identifies a pre-fitted selection protocol;
`NONE_IMPLICIT` means the dispatcher added no calibration stage. Existing model
uncertainty and diagnostic fields are preserved. Independently supplied or fitted
models retain their provenance and validation limitations after reload.
