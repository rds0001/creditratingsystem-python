"""Binary likelihood models with explicit design and target contracts."""
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.special import expit, ndtr
import statsmodels.api as sm

from .core import (CalcResult, FitResult, _context_value, _fail, _fit_result,
                   _model_artifact, _numeric, _result, _fingerprint, _iso_date)

_LINKS = ("logit", "probit", "cloglog")


def _design(X, feature_order, fit_intercept):
    x = _numeric(X, "X", allow_empty=True)
    if x.ndim != 2:
        _fail("INVALID_INPUT", "X must be a two-dimensional numeric matrix")
    if not isinstance(feature_order, (list, tuple)) or any(not isinstance(f, str) or not f.strip() for f in feature_order):
        _fail("FEATURE_MISMATCH", "feature_order must contain explicit nonempty names")
    if len(feature_order) != x.shape[1] or len(set(feature_order)) != len(feature_order):
        _fail("FEATURE_MISMATCH", "Feature names must be distinct and match matrix columns")
    if not isinstance(fit_intercept, bool):
        _fail("INVALID_INPUT", "fit_intercept must be explicitly boolean")
    d = np.column_stack((np.ones(len(x)), x)) if fit_intercept else x.copy()
    if d.shape[1] == 0:
        _fail("INVALID_INPUT", "At least one estimable parameter is required")
    return x, d


def _vector(x, n, name, default):
    if x is None:
        return np.full(n, default, dtype=float)
    v = _numeric(x, name, allow_empty=True, vector_only=True)
    if len(v) != n:
        _fail("INCOMPATIBLE_SHAPE", f"{name} must have one value per row")
    return v


def _inverse(eta, link):
    if link == "logit":
        return expit(eta)
    if link == "probit":
        return ndtr(eta)
    with np.errstate(over="ignore", under="ignore"):
        return -np.expm1(-np.exp(eta))


def _derivative(eta, p, link):
    if link == "logit":
        return p * (1 - p)
    if link == "probit":
        return np.exp(-eta * eta / 2) / np.sqrt(2 * np.pi)
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(eta - np.exp(eta))


def _separation(d, y):
    scale = np.max(np.abs(d), axis=0)
    signed = (2 * y - 1)[:, None] * d / scale
    result = linprog(-signed.mean(axis=0), A_ub=-signed,
                     b_ub=np.zeros(len(y)), bounds=[(-1, 1)] * d.shape[1], method="highs")
    if not result.success:
        _fail("NUMERICAL_FAILURE", "Separation linear program failed", {"solver_status": int(result.status)})
    margins = signed @ result.x
    if margins.min() < -1e-7:
        _fail("NUMERICAL_FAILURE", "Separation certificate violates feasibility tolerance")
    objective = float(margins.mean())
    if objective > 1e-8:
        _fail("SEPARATION", "Complete or quasi separation: finite unpenalized MLE is unavailable",
              {"mean_margin": objective, "separation_tolerance": 1e-8})
    return objective


def fit_binary_glm(X_train, y_train, *, link, feature_order, fit_intercept,
                   target, training, model_id, model_version, weights=None,
                   weight_type="frequency", offset=None, max_iter=100,
                   tolerance=1e-10, context=None):
    """Fit logit, probit or cloglog Bernoulli MLE using frequency weights.

    Features are already transformed training values. Offsets are known additive
    link-scale quantities; a supplied offset must also be supplied at prediction.
    Rank deficiency, separation and unconverged solutions raise controlled errors.
    """
    if not isinstance(link, str) or link not in _LINKS:
        _fail("INVALID_INPUT", "link must be logit, probit or cloglog")
    x, d = _design(X_train, feature_order, fit_intercept)
    n = len(x)
    if y_train is None:
        _fail("MISSING_INPUT", "Observed y_train is required")
    y = _vector(y_train, n, "y_train", np.nan)
    if n == 0:
        _fail("EMPTY_INPUT", "Training requires observations")
    if not np.all(np.isin(y, (0, 1))):
        _fail("INVALID_OUTCOME", "y_train must contain observed numeric zero/one outcomes")
    if weight_type != "frequency":
        _fail("INVALID_WEIGHT_TYPE", "Only integer frequency weights are supported by this MLE")
    w = _vector(weights, n, "weights", 1)
    if np.any(w < 0) or np.any(w != np.floor(w)) or not np.any(w > 0) or w.sum() > 2**53 - 1:
        _fail("INVALID_WEIGHTS", "Frequency weights must be nonnegative exact integers with positive finite total")
    o = _vector(offset, n, "offset", 0)
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        _fail("INVALID_INPUT", "max_iter must be a positive integer")
    t = _numeric(tolerance, "tolerance")
    if t.ndim or not 0 < float(t) < 1:
        _fail("INVALID_INPUT", "tolerance must be a scalar in (0,1)")
    tolerance = float(t)
    ctx = _context_value(context)
    manifest = ctx.get("data_manifest")
    context_origin = manifest.get("data_origin", "USER_SUPPLIED") if isinstance(manifest, dict) else "USER_SUPPLIED"
    origin = training.get("data_origin", context_origin) if isinstance(training, dict) else context_origin
    model = _model_artifact(dict(schema_version="1.0", model_id=model_id,
        model_version=model_version, method_id="classification.fit_binary_glm",
        feature_order=list(feature_order), parameters={}, preprocessing={"transformation": "identity"},
        target=target, training=training, diagnostics={},
        provenance={"data_origin": origin, "parameter_origin": "ESTIMATED"}))
    active = w > 0
    d, y, w, o = d[active], y[active], w[active], o[active]
    if len(np.unique(y)) != 2:
        _fail("INSUFFICIENT_CLASS_SUPPORT", "Both observed classes need positive frequency weight")
    singular_values = np.linalg.svd(d, compute_uv=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-10))
    if rank < d.shape[1]:
        _fail("RANK_DEFICIENT", "Design is not identifiable at relative rank tolerance 1e-10")
    separation_objective = _separation(d, y)
    link_object = {"logit": sm.families.links.Logit,
                   "probit": sm.families.links.Probit,
                   "cloglog": sm.families.links.CLogLog}[link]()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            backend = sm.GLM(y, d, family=sm.families.Binomial(link=link_object),
                             freq_weights=w, offset=o)
            fitted = backend.fit(maxiter=int(max_iter), tol=tolerance, method="IRLS")
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            _fail("NUMERICAL_FAILURE", "GLM estimation failed", {"reason": str(exc)})
    if not fitted.converged or not np.all(np.isfinite(fitted.params)):
        _fail("NON_CONVERGENCE", "GLM did not converge to finite coefficients")
    eta = d @ fitted.params + o
    p = _inverse(eta, link)
    derivative = _derivative(eta, p, link)
    if np.any((p <= 0) | (p >= 1)) or np.any(derivative <= 0):
        _fail("NUMERICAL_FAILURE", "Fitted probabilities or derivatives reached a numerical boundary")
    information_weights = w * derivative**2 / (p * (1 - p))
    information = d.T @ (information_weights[:, None] * d)
    gradient = d.T @ (w * (y - p) * derivative / (p * (1 - p)))
    gradient_scaled = float(np.max(np.abs(gradient) / np.sqrt(np.diag(information))))
    if not np.isfinite(gradient_scaled) or gradient_scaled > max(np.sqrt(tolerance), 1e-6):
        _fail("NON_CONVERGENCE", "Likelihood score exceeds tolerance", {"scaled_gradient": gradient_scaled})
    try:
        chol = np.linalg.cholesky(information)
        inverse_chol = np.linalg.solve(chol, np.eye(rank))
        covariance = inverse_chol.T @ inverse_chol
    except np.linalg.LinAlgError:
        _fail("NON_IDENTIFIABLE", "Expected Fisher information is singular")
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0):
        _fail("NON_IDENTIFIABLE", "Coefficient covariance is not finite positive definite")
    ll = float(np.sum(w * (y * np.log(p) + (1 - y) * np.log1p(-p))))
    diagnostics = dict(converged=True, iterations=int(fitted.fit_history["iteration"]),
        log_likelihood=ll, deviance=-2 * ll, aic=-2 * ll + 2 * rank,
        bic=-2 * ll + np.log(w.sum()) * rank, n_rows=n, n_active_rows=len(y),
        frequency_total=float(w.sum()), rank=rank, rank_tolerance=1e-10,
        condition_number=float(singular_values[0] / singular_values[-1]),
        scaled_gradient=gradient_scaled, gradient_tolerance=max(np.sqrt(tolerance), 1e-6), score=gradient.tolist(),
        separation_mean_margin=separation_objective, separation_tolerance=1e-8,
        covariance_type="EXPECTED_FISHER_IID", weight_type=weight_type,
        solver="IRLS", max_iter=int(max_iter), tolerance=tolerance,
        warnings=[str(c.message) for c in caught])
    model["parameters"] = dict(link=link, fit_intercept=fit_intercept,
        coefficients=fitted.params.tolist(), covariance=covariance.tolist(),
        covariance_type="EXPECTED_FISHER_IID", offset_required=offset is not None)
    model["diagnostics"] = diagnostics
    model = _model_artifact(model)
    return _fit_result(model, "classification.fit_binary_glm", diagnostics=diagnostics, context=context)


_PD_METHOD_KINDS = {
    "classification.fit_discriminant_pd": "discriminant",
    "classification.fit_naive_bayes_pd": "naive_bayes",
    "classification.fit_svm_score": "svm",
    "classification.fit_firth_logit": "firth_logit",
    "classification.fit_penalized_glm": "penalized_logit",
    "classification.fit_ordinal_model": "ordinal",
    "classification.fit_gam_pd": "gam",
    "classification.fit_random_forest_pd": "random_forest",
    "classification.fit_gradient_boosting_pd": "gradient_boosting",
    "classification.fit_bayesian_pd": "bayesian",
    "classification.fit_neural_pd": "neural",
    "classification.fit_sequence_pd": "sequence",
}
_PD_OTHER_METHODS = {
    "classification.fit_binary_glm", "hybrid.fit_stacked_pd",
    "selection_bias.fit_joint_selection_model", "market_implied.fit_market_to_physical_pd_bridge",
    "calibration.fit_model_selection_protocol", "scorecard.fit_scorecard",
}


def _prediction_data(value):
    """Normalize missing table cells to JSON null for semantic input lineage."""
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if isinstance(value, np.ndarray):
        return _prediction_data(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: _prediction_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_prediction_data(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def predict_pd(fitted_model, X, *, feature_order, expected_target, offset=None, context=None):
    """Apply an explicit frozen classifier with common contracts and content lineage.

    Structured sequence/grouped/categorical inputs retain their native structure.
    This operation neither fits a model nor silently calibrates an uncalibrated score.
    """
    model = _model_artifact(fitted_model)
    method = model["method_id"]
    if method not in _PD_METHOD_KINDS and method not in _PD_OTHER_METHODS:
        _fail("UNSUPPORTED_MODEL", "Unsupported frozen classification model")
    if method in _PD_METHOD_KINDS and model["parameters"].get("kind") != _PD_METHOD_KINDS[method]:
        _fail("INVALID_MODEL", "Frozen method identifier and parameter kind disagree")
    if not isinstance(feature_order, (list, tuple)) or list(feature_order) != model["feature_order"]:
        _fail("FEATURE_MISMATCH", "Feature order must exactly match the fitted model")
    if expected_target != model["target"]:
        _fail("TARGET_MISMATCH", "Expected event, horizon, measure and conditioning must match the model")
    ctx = _context_value(context)
    for key in ("event", "horizon", "time_unit", "probability_measure", "conditioning"):
        if ctx.get(key) is not None and ctx[key] != model["target"].get(key):
            _fail("TARGET_MISMATCH", "Context " + key + " conflicts with frozen target")
    if ctx.get("known_at") is not None and _iso_date(model["training"]["known_at"], "model.known_at") > _iso_date(ctx["known_at"], "context.known_at"):
        _fail("LOOKAHEAD_BIAS", "Frozen model was unavailable at prediction knowledge cutoff")
    # JSON numbers from native R have no separate integer type. Normalize only
    # validated categorical indices; canonical numeric content remains unchanged.
    if method == "classification.fit_naive_bayes_pd":
        for feature in model["parameters"].get("features", []):
            for key in ("missing_index", "unknown_index"):
                index = feature.get(key)
                if index is not None:
                    if isinstance(index, bool) or not isinstance(index, (int, float)) or not np.isfinite(index) or index < 0 or index != int(index):
                        _fail("INVALID_MODEL", "Categorical index must be a nonnegative integer")
                    feature[key] = int(index)
    result = _predict_pd_dispatch(model, X, feature_order=feature_order,
        expected_target=expected_target, offset=offset, context=context)
    p = _numeric(result.value, "predicted PD", allow_empty=True, vector_only=True)
    if np.any((p < 0) | (p > 1)):
        _fail("INVALID_PROBABILITY", "Frozen inference returned values outside [0,1]")
    table = X.get("X") if isinstance(X, dict) and "X" in X else X
    if len(p) != len(table):
        _fail("INCOMPATIBLE_SHAPE", "Prediction must preserve one PD per supplied row")
    diagnostics = dict(result.diagnostics)
    diagnostics.update(model_id=model["model_id"], model_version=model["model_version"],
        model_method_id=method, model_hash=_fingerprint(model),
        prediction_input_hash=_fingerprint(dict(X=_prediction_data(X), feature_order=list(feature_order), offset=_prediction_data(offset))),
        target=model["target"], prediction_count=len(p),
        training_data_fingerprint=model["training"]["data_fingerprint"],
        training_known_at=model["training"]["known_at"],
        calibration_applied="FROZEN_PROTOCOL" if method == "calibration.fit_model_selection_protocol" else "NONE_IMPLICIT")
    return _result(result.value, "classification.predict_pd", "probability", context,
        diagnostics=diagnostics, provenance=model["provenance"], limitations=result.limitations)


def _predict_pd_dispatch(fitted_model, X, *, feature_order, expected_target, offset=None, context=None):
    """Apply a frozen supported model to an explicitly matching target/design."""
    if isinstance(fitted_model, FitResult):
        fitted_model = fitted_model.model
    model = _model_artifact(fitted_model)
    ctx = _context_value(context)
    for key in ("horizon", "time_unit", "probability_measure", "conditioning"):
        if ctx.get(key) is not None and ctx[key] != model["target"].get(key):
            _fail("TARGET_MISMATCH", f"Context {key} conflicts with fitted target")
    if model["method_id"] in ('classification.fit_discriminant_pd', 'classification.fit_naive_bayes_pd', 'classification.fit_svm_score', 'classification.fit_firth_logit', 'classification.fit_penalized_glm', 'classification.fit_ordinal_model', 'classification.fit_gam_pd', 'classification.fit_random_forest_pd', 'classification.fit_gradient_boosting_pd', 'classification.fit_bayesian_pd', 'classification.fit_neural_pd', 'classification.fit_sequence_pd'):
        if offset is not None:
            _fail("OFFSET_MISMATCH", "This fitted model does not accept prediction offsets")
        from ._classification_models import _predict_extended_pd
        return _predict_extended_pd(model, X, feature_order=feature_order, expected_target=expected_target, context=context)
    if model["method_id"] == "hybrid.fit_stacked_pd":
        if offset is not None:
            _fail("OFFSET_MISMATCH", "Stacked PD models do not accept prediction offsets")
        from .hybrid import _predict_stacked_pd
        return _predict_stacked_pd(model, X, feature_order=feature_order, expected_target=expected_target, context=context)
    if model["method_id"] == "selection_bias.fit_joint_selection_model":
        if offset is not None:
            _fail("OFFSET_MISMATCH", "Joint selection models do not accept prediction offsets")
        from .selection_bias import _predict_joint_pd
        return _predict_joint_pd(model, X, feature_order=feature_order, expected_target=expected_target, context=context)
    if model["method_id"] == "market_implied.fit_market_to_physical_pd_bridge":
        if offset is not None:
            _fail("OFFSET_MISMATCH", "Market bridge models do not accept prediction offsets")
        from ._market_bridge import _predict_market_bridge
        return _predict_market_bridge(model, X, feature_order=feature_order, expected_target=expected_target, context=context)
    if model["method_id"] == "calibration.fit_model_selection_protocol":
        if offset is not None:
            _fail("OFFSET_MISMATCH", "Selected frozen models do not accept prediction offsets")
        from ._calibration_models import _predict_selection_pd
        return _predict_selection_pd(model, X, feature_order=feature_order, expected_target=expected_target, context=context)
    if model["method_id"] == "scorecard.fit_scorecard":
        from .scorecard import score_scorecard
        if offset is not None:
            _fail("OFFSET_MISMATCH", "Frozen scorecards do not accept prediction offsets")
        scored = score_scorecard(model, X, feature_order=feature_order, expected_target=expected_target, context=context)
        return _result(scored.value["pd"], "classification.predict_pd", "probability", context,
            diagnostics={"model_id":model["model_id"],"model_version":model["model_version"],
                         "prediction_family":"scorecard","target":model["target"]},
            provenance=model["provenance"],limitations=("Frozen scorecard inference requires its trained feature transformation and matching target.",))
    if model["method_id"] != "classification.fit_binary_glm":
        _fail("UNSUPPORTED_MODEL", "Unsupported frozen classification model")
    if not isinstance(feature_order, (list, tuple)) or list(feature_order) != model["feature_order"]:
        _fail("FEATURE_MISMATCH", "Feature order must exactly match the fitted model")
    if expected_target != model["target"]:
        _fail("TARGET_MISMATCH", "Expected event, horizon, measure and conditioning must match the model")
    ctx = _context_value(context)
    for key in ("horizon", "time_unit", "probability_measure", "conditioning"):
        if ctx.get(key) is not None and ctx[key] != model["target"].get(key):
            _fail("TARGET_MISMATCH", f"Context {key} conflicts with fitted target")
    if model["preprocessing"] != {"transformation": "identity"}:
        _fail("UNSUPPORTED_MODEL", "This GLM expects already transformed features with identity preprocessing")
    parameters = model["parameters"]
    if parameters.get("link") not in _LINKS or not isinstance(parameters.get("offset_required"), bool):
        _fail("INVALID_MODEL", "Incomplete or unsupported GLM parameter schema")
    _, d = _design(X, feature_order, parameters.get("fit_intercept"))
    coefficients = _numeric(parameters.get("coefficients"), "coefficients", vector_only=True)
    if len(coefficients) != d.shape[1]:
        _fail("INVALID_MODEL", "Coefficient count does not match fitted design")
    if parameters["offset_required"] != (offset is not None):
        _fail("OFFSET_MISMATCH", "Prediction offset presence must match the fitted contract")
    o = _vector(offset, len(d), "offset", 0)
    eta = d @ coefficients + o
    if not np.all(np.isfinite(eta)):
        _fail("NUMERICAL_FAILURE", "Prediction linear predictor overflowed")
    p = _inverse(eta, parameters["link"])
    return _result(p, "classification.predict_pd", "probability", context,
        diagnostics={"model_id": model["model_id"], "model_version": model["model_version"],
                     "link": parameters["link"], "numerical_boundary_count": int(np.sum((p == 0) | (p == 1)))},
        provenance=model["provenance"],
        limitations=("IID likelihood estimation does not establish calibration on future or dependent observations.",))

# Granular public estimators implemented in the native backend module.
from ._classification_models import (fit_discriminant_pd, fit_naive_bayes_pd, fit_svm_score, predict_model_score, fit_firth_logit, fit_penalized_glm, fit_ordinal_model)

from ._classification_flexible import (fit_gam_pd, fit_random_forest_pd, fit_gradient_boosting_pd)

from ._classification_bayesian import fit_bayesian_pd
from ._classification_neural import fit_neural_pd, fit_sequence_pd
