"""Frozen PD calibration with explicit target, sample and boundary contracts."""
from ._long_run import estimate_long_run_pd
from collections.abc import Mapping

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit

from .core import (
    CalcResult, _context_value, _fail, _fingerprint, _iso_date, _json_data, _numeric,
    _probability, _result,
)


def _target(target):
    if not isinstance(target, Mapping):
        _fail("INVALID_CALIBRATION_METADATA", "target must be an object")
    target = _json_data(target)
    for name in ("event", "conditioning"):
        if not isinstance(target.get(name), str) or not target[name].strip():
            _fail("INVALID_CALIBRATION_METADATA", f"target.{name} is required")
    if target.get("probability_measure") not in ("PHYSICAL", "RISK_NEUTRAL"):
        _fail("INVALID_CALIBRATION_METADATA", "Unknown probability measure")
    if target.get("time_unit") not in ("days", "months", "years"):
        _fail("INVALID_CALIBRATION_METADATA", "Unknown horizon time unit")
    horizon = _numeric(target.get("horizon"), "target.horizon")
    if horizon.ndim != 0 or horizon <= 0:
        _fail("INVALID_CALIBRATION_METADATA", "horizon must be a positive scalar")
    target["horizon"] = float(horizon)
    return target


def _metadata(metadata, empirical):
    if not isinstance(metadata, Mapping):
        _fail("INVALID_CALIBRATION_METADATA", "Explicit calibration metadata is required")
    meta = _json_data(metadata)
    meta["target"] = _target(meta.get("target"))
    for name in ("base_model_version", "population_definition", "outcome_policy"):
        if not isinstance(meta.get(name), str) or not meta[name].strip():
            _fail("INVALID_CALIBRATION_METADATA", f"{name} is required")
    if meta.get("weight_type") != "frequency":
        _fail("INVALID_WEIGHT_TYPE", "This fit uses frequency weights only")
    if meta.get("data_origin") not in ("USER_SUPPLIED", "SYNTHETIC", "REFERENCE"):
        _fail("INVALID_CALIBRATION_METADATA", "Declare data_origin")
    sample = meta.get("calibration")
    if not isinstance(sample, dict):
        _fail("INVALID_CALIBRATION_METADATA", "calibration window and split are required")
    for key in ("start", "end", "known_at"):
        sample[key] = _iso_date(sample.get(key), f"calibration.{key}")
    if not sample["start"] <= sample["end"] <= sample["known_at"]:
        _fail("INVALID_CALIBRATION_METADATA", "Require start <= end <= known_at")
    if not isinstance(sample.get("split_id"), str) or not sample["split_id"].strip():
        _fail("INVALID_CALIBRATION_METADATA", "Calibration split_id is required")
    allowed = ("HOLDOUT", "OUT_OF_FOLD") if empirical else ("HOLDOUT", "OUT_OF_FOLD", "EXTERNAL_TARGET")
    if sample.get("prediction_origin") not in allowed:
        _fail("INVALID_CALIBRATION_METADATA", "Declare independent calibration prediction origin")
    return meta


def _data(values, weights, outcomes=None, probability=False, open_domain=False):
    x = (_probability(values, "open" if open_domain else "closed", vector_only=True)
         if probability else _numeric(values, "values", vector_only=True))
    if len(x) == 0:
        _fail("EMPTY_INPUT", "Calibration requires observations")
    w = np.ones(len(x)) if weights is None else _numeric(weights, "weights", vector_only=True)
    if len(w) != len(x):
        _fail("INCOMPATIBLE_SHAPE", "Weights and values must have equal length")
    if np.any(w < 0) or np.any(w != np.floor(w)) or not np.any(w > 0):
        _fail("INVALID_WEIGHTS", "Frequency weights must be nonnegative integers with positive total")
    if not np.isfinite(w.sum()):
        _fail("INVALID_WEIGHTS", "Total frequency weight overflowed")
    y = None
    if outcomes is not None:
        y = _numeric(outcomes, "outcomes", vector_only=True)
        if len(y) != len(x):
            _fail("INCOMPATIBLE_SHAPE", "Outcomes and values must have equal length")
        if np.any((y != 0) & (y != 1)):
            _fail("INVALID_OUTCOME", "Outcomes must be binary and fully observed")
    return x, w, y


def _controls(tolerance, max_iterations):
    tol = _numeric(tolerance, "tolerance")
    count = _numeric(max_iterations, "max_iterations")
    if tol.ndim != 0 or not 0 < tol < 0.01 or count.ndim != 0 or count < 1 or count != np.floor(count):
        _fail("INVALID_INPUT", "Require tolerance in (0,0.01) and positive integer max_iterations")
    return float(tol), int(count)


def _cal_context(context, target):
    ctx = _context_value(context)
    for key in ("horizon", "time_unit", "probability_measure", "conditioning"):
        if ctx.get(key) is not None and ctx[key] != target[key]:
            _fail("INCOMPATIBLE_TARGET", f"Context {key} differs from calibration target")
    return ctx


def _artifact(kind, parameters, meta, x, w, y, diagnostics, context):
    method = f"calibration.calibrate_{kind}"
    fingerprint = _fingerprint({"values": x.tolist(), "weights": w.tolist(),
                                "outcomes": None if y is None else y.tolist()})
    artifact = {"schema_version": "1.0", "artifact_type": "CalibrationArtifact",
                "calibration_type": kind, "method_id": method, "parameter_version": "1.0",
                "input_kind": "score" if kind == "isotonic" else "probability",
                "parameters": parameters, "target": meta["target"], "metadata": meta,
                "calibration_input_fingerprint": fingerprint, "diagnostics": diagnostics}
    artifact["artifact_hash"] = _fingerprint(artifact)
    return _result(artifact, method, "calibration_artifact", _cal_context(context, meta["target"]), diagnostics=diagnostics,
                   provenance={"data_origin": meta["data_origin"], "parameter_origin": "ESTIMATED"},
                   limitations=("Sample independence and outcome maturity are caller declarations; this fit does not verify them.",
                                "Calibration is not out-of-sample validation or model approval."))


def calibrate_intercept(base_pd, target_default_rate, weights=None, *, metadata,
                        context=None, tolerance=1e-12, max_iterations=200):
    """Fit one finite log-odds shift; preserve structural zero/one probabilities."""
    meta = _metadata(metadata, False)
    p, w, _ = _data(base_pd, weights, probability=True)
    tol, limit = _controls(tolerance, max_iterations)
    target = _probability(target_default_rate)
    if target.ndim != 0:
        _fail("INVALID_INPUT", "target_default_rate must be scalar")
    target = float(target)
    wn = w / w.sum()
    active = (p > 0) & (p < 1) & (w > 0)
    lower, upper = float(wn[p == 1].sum()), float(wn[p > 0].sum())
    if not np.any(active):
        _fail("NOT_IDENTIFIABLE", "No interior positive-weight probabilities identify a finite shift")
    if not lower < target < upper:
        _fail("INFEASIBLE_TARGET", "Target cannot be attained by a finite shift preserving 0/1",
              {"lower_limit": lower, "upper_limit": upper})
    scores = logit(p[active])
    def residual(shift):
        return lower + float(wn[active] @ expit(scores + shift)) - target
    initial = residual(0)
    if initial == 0:
        shift, iterations, bracket = 0.0, 0, [0.0, 0.0]
    else:
        low, high = -1.0, 1.0
        while residual(low) > 0 and abs(low) < 4096:
            low *= 2
        while residual(high) < 0 and high < 4096:
            high *= 2
        try:
            shift, solver = brentq(residual, low, high, xtol=tol, rtol=4*np.finfo(float).eps,
                                   maxiter=limit, full_output=True, disp=False)
        except (ValueError, RuntimeError):
            _fail("NON_CONVERGENCE", "Cannot solve the finite intercept shift")
        if not solver.converged:
            _fail("NON_CONVERGENCE", "Intercept solver exhausted iterations")
        iterations, bracket = solver.iterations, [low, high]
    achieved = target + residual(shift)
    if abs(achieved - target) > tol:
        _fail("NON_CONVERGENCE", "Target reconciliation exceeds tolerance")
    diagnostics = {"converged": True, "iterations": int(iterations), "bracket": bracket,
                   "target_default_rate": target, "achieved_mean_pd": achieved,
                   "residual": achieved-target, "tolerance": tol,
                   "n_observations": len(p), "weight_sum": float(w.sum()),
                   "zero_weight_count": int(np.sum(w == 0)),
                   "boundary_policy": "preserve", "identifiable": True}
    return _artifact("intercept", {"intercept": float(shift), "slope": 1.0,
                                  "boundary_policy": "preserve"},
                     meta, p, w, None, diagnostics, context)


def calibrate_logistic(base_pd, outcomes, weights=None, *, metadata, context=None,
                       tolerance=1e-10, max_iterations=100):
    """Unpenalized Bernoulli MLE of intercept and slope on the base logit."""
    meta = _metadata(metadata, True)
    p, w, y = _data(base_pd, weights, outcomes, probability=True, open_domain=True)
    if y is None:
        _fail("MISSING_INPUT", "Observed outcomes are required")
    tol, limit = _controls(tolerance, max_iterations)
    keep = w > 0
    x, fit_w, fit_y = logit(p[keep]), w[keep], y[keep]
    if len(np.unique(x)) < 2:
        _fail("NOT_IDENTIFIABLE", "A constant base logit cannot identify intercept and slope")
    if len(np.unique(fit_y)) < 2:
        _fail("SEPARATION", "Both outcome classes are required for finite logistic MLE")
    x0, x1 = x[fit_y == 0], x[fit_y == 1]
    if max(x0) <= min(x1) or max(x1) <= min(x0):
        _fail("SEPARATION", "Complete or quasi separation has no finite logistic MLE")
    center, scale = float(np.mean(x)), float(np.max(np.abs(x - np.mean(x))))
    design = np.column_stack([np.ones(len(x)), (x-center)/scale])
    wn = fit_w/fit_w.sum()
    def objective(beta):
        eta = design @ beta
        return float(wn @ (np.logaddexp(0, eta)-fit_y*eta))
    def gradient(beta):
        return design.T @ (wn*(expit(design @ beta)-fit_y))
    def hessian(beta):
        probability = expit(design @ beta)
        return design.T @ ((wn*probability*(1-probability))[:, None]*design)
    fit = minimize(objective, [float(logit(wn @ fit_y)), 0.0], jac=gradient, hess=hessian,
                   method="trust-exact", options={"gtol": tol, "maxiter": limit})
    error = float(np.max(np.abs(gradient(fit.x))))
    if not np.all(np.isfinite(fit.x)) or error > tol:
        _fail("NON_CONVERGENCE", "Logistic calibration score equation did not converge",
              {"gradient_max": error, "iterations": int(fit.nit)})
    transform = np.array([[1, -center/scale], [0, 1/scale]])
    beta = transform @ fit.x
    try:
        covariance = transform @ np.linalg.inv(hessian(fit.x)*fit_w.sum()) @ transform.T
    except np.linalg.LinAlgError:
        _fail("NOT_IDENTIFIABLE", "Singular observed information")
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0):
        _fail("NOT_IDENTIFIABLE", "Calibration information is not positive definite")
    diagnostics = {"converged": True, "iterations": int(fit.nit), "gradient_max": error,
                   "solver_converged": bool(fit.success),
                   "negative_log_likelihood": objective(fit.x)*float(fit_w.sum()),
                   "standard_errors": np.sqrt(np.diag(covariance)).tolist(),
                   "covariance": covariance.tolist(), "uncertainty": "IID_FREQUENCY_FISHER",
                   "slope_nonpositive": bool(beta[1] <= 0),
                   "n_observations": len(p), "weight_sum": float(w.sum()),
                   "zero_weight_count": int(np.sum(w == 0)), "boundary_policy": "reject"}
    return _artifact("logistic", {"intercept": float(beta[0]), "slope": float(beta[1]),
                                 "boundary_policy": "reject"},
                     meta, p, w, y, diagnostics, context)


def calibrate_isotonic(scores, outcomes, weights=None, *, metadata, increasing=True,
                       out_of_bounds="error", context=None):
    """Weighted PAVA with tied-score pooling and a frozen left-constant step map."""
    meta = _metadata(metadata, True)
    x, w, y = _data(scores, weights, outcomes)
    if y is None:
        _fail("MISSING_INPUT", "Observed outcomes are required")
    if not isinstance(increasing, bool) or out_of_bounds not in ("error", "clip"):
        _fail("INVALID_INPUT", "Declare increasing boolean and error/clip boundary policy")
    keep = w > 0
    order = np.argsort(x[keep], kind="stable")
    xx, ww, yy = x[keep][order], w[keep][order], y[keep][order]
    knots, starts = np.unique(xx, return_index=True)
    mass = np.add.reduceat(ww, starts)
    positives = np.add.reduceat(ww*yy, starts)
    blocks = []
    direction = 1 if increasing else -1
    for i, (weight, events) in enumerate(zip(mass, positives)):
        blocks.append([i, i, float(weight), float(events)])
        while len(blocks) > 1 and direction*blocks[-2][3]/blocks[-2][2] > direction*blocks[-1][3]/blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    values = np.empty(len(knots))
    for start, end, weight, events in blocks:
        values[start:end+1] = events/weight
    parameters = {"knots": knots.tolist(), "values": values.tolist(), "increasing": increasing,
                  "out_of_bounds": out_of_bounds, "interpolation": "left_constant"}
    diagnostics = {"n_observations": len(x), "weight_sum": float(w.sum()),
                   "zero_weight_count": int(np.sum(w == 0)), "n_unique_scores": len(knots),
                   "n_blocks": len(blocks), "blocks": blocks,
                   "weighted_outcome_mean": float(w@y/w.sum()),
                   "weighted_fitted_mean": float(mass@values/w.sum())}
    return _artifact("isotonic", parameters, meta, x, w, y, diagnostics, context)


def _checked_artifact(calibrator):
    if isinstance(calibrator, CalcResult):
        calibrator = calibrator.value
    if not isinstance(calibrator, Mapping):
        _fail("INVALID_CALIBRATOR", "Expected a CalibrationArtifact")
    artifact = _json_data(calibrator)
    if artifact.get("schema_version") != "1.0":
        _fail("UNSUPPORTED_SCHEMA_VERSION", "Unsupported calibration schema")
    if artifact.get("artifact_type") != "CalibrationArtifact" or artifact.get("calibration_type") not in ("intercept", "logistic", "isotonic"):
        _fail("INVALID_CALIBRATOR", "Unknown calibration artifact type")
    digest = artifact.pop("artifact_hash", None)
    if digest != _fingerprint(artifact):
        _fail("INTEGRITY_ERROR", "Calibration artifact content differs from its hash")
    artifact["artifact_hash"] = digest
    kind = artifact["calibration_type"]
    if (artifact.get("method_id") != f"calibration.calibrate_{kind}"
            or artifact.get("parameter_version") != "1.0"
            or artifact.get("input_kind") != ("score" if kind == "isotonic" else "probability")
            or not isinstance(artifact.get("calibration_input_fingerprint"), str)
            or len(artifact["calibration_input_fingerprint"]) != 64):
        _fail("INVALID_CALIBRATOR", "Invalid method, parameter version, input kind or fingerprint")
    meta = _metadata(artifact.get("metadata"), artifact["calibration_type"] != "intercept")
    if artifact.get("target") != meta["target"] or not isinstance(artifact.get("parameters"), dict):
        _fail("INVALID_CALIBRATOR", "Inconsistent calibration target or parameters")
    return artifact


def apply_calibrator(calibrator, values, *, target, base_model_version, context=None, groups=None):
    """Apply frozen calibration in original input order after target/version checks."""
    raw = calibrator.value if isinstance(calibrator, CalcResult) else calibrator
    if isinstance(raw, Mapping) and raw.get("calibration_type") in ("beta", "hierarchical", "lgd_level"):
        from ._calibration_models import _apply_extended_calibrator
        return _apply_extended_calibrator(calibrator, values, target=target, base_model_version=base_model_version, groups=groups, context=context)
    artifact = _checked_artifact(calibrator)
    if _target(target) != artifact["target"]:
        _fail("INCOMPATIBLE_TARGET", "Calibration target, horizon, measure or conditioning differs")
    if base_model_version != artifact["metadata"]["base_model_version"]:
        _fail("INCOMPATIBLE_MODEL_VERSION", "Base model version differs from calibration")
    params, kind = artifact["parameters"], artifact["calibration_type"]
    if kind == "isotonic":
        x = _numeric(values, "values", allow_empty=True, vector_only=True)
        knots = _numeric(params.get("knots"), "knots", vector_only=True)
        fitted = _probability(params.get("values"), vector_only=True)
        if len(knots) != len(fitted) or np.any(np.diff(knots) <= 0):
            _fail("INVALID_CALIBRATOR", "Invalid isotonic knots")
        increasing = params.get("increasing")
        if not isinstance(increasing, bool) or np.any((1 if increasing else -1)*np.diff(fitted) < 0):
            _fail("INVALID_CALIBRATOR", "Invalid isotonic monotonicity")
        if params.get("out_of_bounds") not in ("error", "clip") or params.get("interpolation") != "left_constant":
            _fail("INVALID_CALIBRATOR", "Unknown isotonic interpolation or boundary policy")
        outside = (x < knots[0]) | (x > knots[-1])
        if np.any(outside) and params["out_of_bounds"] == "error":
            _fail("OUT_OF_SUPPORT", "Score lies outside positive-weight calibration support")
        value = fitted[np.clip(np.searchsorted(knots, x, side="right")-1, 0, len(knots)-1)]
        trace = {"out_of_support_count": int(np.sum(outside)), "interpolation": "left_constant"}
    else:
        p = _probability(values, "closed" if kind == "intercept" else "open", vector_only=True)
        a, b = _numeric(params.get("intercept")), _numeric(params.get("slope"))
        if a.ndim or b.ndim or (kind == "intercept" and b != 1):
            _fail("INVALID_CALIBRATOR", "Invalid scalar calibration coefficients")
        if params.get("boundary_policy") != ("preserve" if kind == "intercept" else "reject"):
            _fail("INVALID_CALIBRATOR", "Invalid calibration boundary policy")
        value = p.copy()
        interior = (p > 0) & (p < 1)
        value[interior] = expit(float(a)+float(b)*logit(p[interior]))
        trace = {"intercept": float(a), "slope": float(b), "boundary_policy": params["boundary_policy"]}
    trace.update({"calibration_hash": artifact["artifact_hash"], "calibration_type": kind})
    return _result(value, "calibration.apply_calibrator", "probability", _cal_context(context, artifact["target"]),
                   diagnostics=trace, provenance={"parameter_origin": "ESTIMATED"},
                   limitations=("Frozen calibration does not establish performance on new observations.",))

from ._calibration_models import (fit_stepwise, fit_regularization_path, resample_fit_uncertainty, calibrate_beta, calibrate_grade_pd, low_default_bound, fit_hierarchical_calibration, calibrate_lgd_level, estimate_margin_of_conservatism, population_reweighting, fit_model_selection_protocol)
