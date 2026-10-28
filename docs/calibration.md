# Calibrating and applying probabilities

`calibrate_intercept`, `calibrate_logistic` and `calibrate_isotonic` fit a
versioned `CalibrationArtifact` in `CalcResult.value`. `apply_calibrator` uses
that frozen artifact on new inputs, preserving their order. Native R provides
the same functions and uses `$value`.

Supply a declared target, calibration window and data split before fitting.
Calibration predictions must come from a separate holdout or properly produced
out-of-fold predictions. An external target may instead support an intercept
shift. These functions record these declarations; they cannot establish that
the caller kept training, calibration and final evaluation data separate.
Independent calibration predictions avoid the bias from calibrating a model on
its own training predictions. [Probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html)

## A complete synthetic example

```python
from creditratingsystem import calibrate_intercept, apply_calibrator

metadata = {
    "target": {
        "event": "first_default", "horizon": 12, "time_unit": "months",
        "probability_measure": "PHYSICAL", "conditioning": "PIT",
    },
    "calibration": {
        "start": "2020-01-01", "end": "2021-12-31", "known_at": "2022-01-01",
        "split_id": "synthetic_calibration", "prediction_origin": "EXTERNAL_TARGET",
    },
    "base_model_version": "synthetic_score_v1",
    "population_definition": "Handcrafted demonstration",
    "outcome_policy": "Fully observed first-default outcomes",
    "weight_type": "frequency", "data_origin": "SYNTHETIC",
}
fit = calibrate_intercept([0.1, 0.1], 2 / 11, metadata=metadata)
prediction = apply_calibrator(
    fit, [0.1, 0.2, 0.5], target=metadata["target"],
    base_model_version=metadata["base_model_version"],
)
# fit.value["parameters"]["intercept"] is log(2).
# prediction.value is [2/11, 1/3, 2/3].
```

Metadata dates must satisfy `start <= end <= known_at`. Target horizon is a
positive number with unit `days`, `months` or `years`. Measure is explicitly
`PHYSICAL` or `RISK_NEUTRAL`; `conditioning` is a caller-defined nonempty string.
Neither calibration nor application changes the horizon, measure or
conditioning. Supplied calculation contexts must agree with the target.

For empirical logistic and isotonic fitting, `prediction_origin` must be
`HOLDOUT` or `OUT_OF_FOLD`. `EXTERNAL_TARGET` is permitted only for the intercept
fit. `data_origin` is `USER_SUPPLIED`, `SYNTHETIC` or `REFERENCE`. The outcome
policy should identify maturity, censoring treatment and first-default rules;
input outcomes must already be observed binary values. Unobserved outcomes
cannot be substituted with zero.

## Method contracts

| Function | Inputs and result |
|---|---|
| `calibrate_intercept(base_pd, target_default_rate, weights=None, *, metadata, context=None, tolerance=1e-12, max_iterations=200)` | Fits a finite shift `a` in `expit(logit(p)+a)` to reconcile the weighted target mean. |
| `calibrate_logistic(base_pd, outcomes, weights=None, *, metadata, context=None, tolerance=1e-10, max_iterations=100)` | Fits unpenalized binomial maximum likelihood `expit(a+b*logit(p))`. |
| `calibrate_isotonic(scores, outcomes, weights=None, *, metadata, increasing=True, out_of_bounds="error", context=None)` | Fits weighted monotone probabilities by tied-score pooling and the pool-adjacent-violators algorithm. |
| `apply_calibrator(calibrator, values, *, target, base_model_version, context=None, groups=None)` | Checks the frozen artifact, target and model version, then returns a probability vector. |

R uses the same names with `NULL` defaults, `TRUE`/`FALSE`, and named arguments
instead of Python keyword-only syntax. Python fits require one-dimensional
sequences, including for one observation. All application outputs are vectors;
an empty application vector returns an empty vector.

Weights have the explicitly declared `frequency` meaning: nonnegative integer
counts with a positive finite sum. Omitting them assigns one to each row.
Zero-weight rows remain in the input fingerprint but do not identify parameters
or determine isotonic support. Sampling weights, EAD weights and fractional
class weights require a different contract and are rejected here.

The intercept fit keeps structural zero and one probabilities unchanged. The
target must lie strictly between the weighted mass already at one and the
weighted mass above zero. Endpoint targets need infinite shifts and are
rejected. A sample containing no positive-weight interior probability cannot
identify the shift. Diagnostics report the root bracket, convergence,
iterations, achieved mean, target and residual.

The logistic fit requires base PDs strictly inside `(0, 1)`. It does not clip
endpoint probabilities or substitute a penalized estimate. Constant base logits
are unidentifiable; complete and quasi separation are rejected. A negative or
zero fitted slope is returned with `slope_nonpositive=True`, making reversal or
loss of ranking visible. Standard errors and covariance use binomial Fisher
information under independent frequency repetitions. They do not provide
clustered, serially dependent or model-selection-adjusted uncertainty. R fits
the binomial model using `stats::glm.fit`; its weight and fit diagnostics are
explicit rather than inherited as unstated modelling choices.
[R generalized linear model documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html)

Isotonic calibration accepts arbitrary finite scores. It first sorts scores
stably and combines equal scores using their frequency weights. Adjacent
violating blocks are pooled until fitted probabilities follow the requested
increasing or decreasing direction. This is weighted least-squares isotonic
regression with PAVA. [SciPy isotonic regression documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.isotonic_regression.html)

Application is a **left-constant step function**: a fitted value at score `x_i`
applies on `[x_i, x_(i+1))`, and the maximum training score uses its own fitted
value. It does not linearly interpolate between training scores. Outside the
positive-weight training support, `out_of_bounds="error"` rejects the input;
`"clip"` explicitly uses the nearest endpoint and reports the number of affected
inputs. The policy is frozen in the artifact.

## Storage and checks

Artifacts contain method and parameter versions, target, full supplied
metadata, input fingerprint, fitted parameters and diagnostics. They carry a
content hash checked at application. This detects accidental changes and is
not a signature or authorization mechanism. Results remain `NOT_VALIDATED`.

To store a calibrator together with its base model, add its plain mapping as a
`calibration_artifact` field to a valid model artifact and use the public
`serialize_fitted_model` and `load_fitted_model` functions. The base model must
already meet their model contract; a standalone calibrator does not become a
base model. The shared JSON codec preserves numeric precision across Python and
native R. Ordinary R JSON serialization may round numbers even with
`digits=NA`, causing an intentional integrity failure on replay.

```python
# base_model is your valid, frozen base model mapping.
# base_model["calibration_artifact"] = fit.value
# serialize_fitted_model(base_model, "explicit/local/model.json")
# loaded = load_fitted_model("explicit/local/model.json").value
# frozen = loaded["calibration_artifact"]
```

Controlled failures include `INVALID_CALIBRATION_METADATA`, `INVALID_WEIGHTS`,
`INVALID_OUTCOME`, `INFEASIBLE_TARGET`, `NOT_IDENTIFIABLE`, `SEPARATION`,
`NON_CONVERGENCE`, `OUT_OF_SUPPORT`, `INCOMPATIBLE_TARGET`,
`INCOMPATIBLE_MODEL_VERSION` and `INTEGRITY_ERROR`, in addition to the common
numeric input errors. No calibration result is an independent validation or
approval for a bank's portfolio.

Additional beta, hierarchical, grade, loss, uncertainty and selection methods are
documented in [calibration_models.md](calibration_models.md).
