"""Explicit calibration, uncertainty and leakage-controlled model selection."""

from collections.abc import Mapping
from collections import Counter
from math import log, pi
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize, linprog
from scipy.special import expit, logit
from scipy.stats import beta as beta_distribution, binom, chi2, norm
from .core import CalcResult, _fail, _fingerprint, _json_data, _numeric, _probability, _result


def _base():
    from . import calibration

    return calibration


def _out(value, name, context=None, diagnostics=None, data_origin=None):
    return _result(
        value,
        "calibration." + name,
        context=context,
        diagnostics=diagnostics,
        provenance={
            "parameter_origin": "ESTIMATED",
            **({"data_origin": data_origin} if data_origin is not None else {}),
        },
        limitations=(
            "Estimation uncertainty does not establish external validity or regulatory acceptance.",
        ),
    )


def _scalar(x, name, lower=None, upper=None):
    a = _numeric(x, name)
    if a.ndim or (lower is not None and a < lower) or (upper is not None and a > upper):
        _fail("INVALID_INPUT", name + " outside scalar domain")
    return float(a)


def _ext_artifact(kind, params, meta, inputs, diagnostics, context):
    method = {
        "beta": "calibrate_beta",
        "hierarchical": "fit_hierarchical_calibration",
        "lgd_level": "calibrate_lgd_level",
    }[kind]
    a = {
        "schema_version": "1.0",
        "artifact_type": "CalibrationArtifact",
        "calibration_type": kind,
        "method_id": "calibration." + method,
        "parameter_version": "1.0",
        "input_kind": "loss" if kind == "lgd_level" else "probability",
        "parameters": params,
        "target": meta["target"],
        "metadata": meta,
        "calibration_input_fingerprint": _fingerprint(inputs),
        "diagnostics": diagnostics,
    }
    a["artifact_hash"] = _fingerprint(a)
    return _result(
        a,
        "calibration." + method,
        "calibration_artifact",
        _base()._cal_context(context, meta["target"]),
        diagnostics=diagnostics,
        provenance={"data_origin": meta["data_origin"], "parameter_origin": "ESTIMATED"},
    )


def calibrate_beta(
    base_pd,
    outcomes,
    weights=None,
    *,
    metadata,
    monotonic=True,
    boundary_policy="reject",
    epsilon=1e-12,
    context=None,
):
    """Bernoulli beta calibration: expit(c+a log(p)-b log(1-p))."""
    meta = _base()._metadata(metadata, True)
    p, w, y = _base()._data(base_pd, weights, outcomes, probability=True)
    if type(monotonic) is not bool or boundary_policy not in ("reject", "clip"):
        _fail("INVALID_INPUT", "Declare monotonic boolean and reject/clip boundary policy")
    eps = _scalar(epsilon, "epsilon", 1e-15, 0.01)
    if boundary_policy == "reject" and np.any((p == 0) | (p == 1)):
        _fail("INVALID_PROBABILITY", "Beta calibration requires interior probabilities")
    q = np.clip(p, eps, 1 - eps) if boundary_policy == "clip" else p
    X = np.column_stack([np.ones(len(p)), np.log(q), -np.log1p(-q)])
    if np.linalg.matrix_rank(X[w > 0]) < 3:
        _fail("NOT_IDENTIFIABLE", "Beta calibration requires rank-three design")
    if len(set(y[w > 0])) < 2:
        _fail("SEPARATION", "Both observed classes required")
    active = X[w > 0]
    signed = (2 * y[w > 0] - 1)[:, None] * active / np.max(abs(active), axis=0)
    separation = linprog(
        -signed.mean(axis=0),
        A_ub=-signed,
        b_ub=np.zeros(len(active)),
        bounds=[(-1, 1), (0, 1), (0, 1)] if monotonic else [(-1, 1)] * 3,
        method="highs",
    )
    if not separation.success:
        _fail("NUMERICAL_FAILURE", "Beta separation certificate failed")
    if -separation.fun > 1e-8:
        _fail("SEPARATION", "Finite beta MLE unavailable under declared coefficient constraints")
    objective = lambda b: float(w @ (np.logaddexp(0, X @ b) - y * (X @ b)))
    gradient = lambda b: X.T @ (w * (expit(X @ b) - y))
    fit = minimize(
        objective,
        [0.0, 1.0, 1.0],
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(None, None), (0, None), (0, None)] if monotonic else None,
        options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 2000},
    )
    if not fit.success or np.max(abs(fit.x)) > 100:
        _fail("NONCONVERGENCE", "Beta MLE did not converge to a finite estimate")
    fitted = expit(X @ fit.x)
    H = X.T @ ((w * fitted * (1 - fitted))[:, None] * X)
    if np.linalg.matrix_rank(H) < 3:
        _fail("NOT_IDENTIFIABLE", "Singular calibration information")
    params = {
        "intercept": float(fit.x[0]),
        "a": float(fit.x[1]),
        "b": float(fit.x[2]),
        "monotonic": monotonic,
        "boundary_policy": boundary_policy,
        "epsilon": eps,
    }
    diag = {
        "log_likelihood": -objective(fit.x),
        "information": H.tolist(),
        "conditional_covariance": np.linalg.inv(H).tolist(),
        "boundary_active": bool(monotonic and min(fit.x[1:]) < 1e-7),
        "uncertainty_note": "Interior inverse-information approximation; active constraints require boundary-aware inference",
        "clipped_count": int(np.sum(q != p)),
    }
    return _ext_artifact(
        "beta", params, meta, {"p": p.tolist(), "w": w.tolist(), "y": y.tolist()}, diag, context
    )


def _pava(values, mass):
    blocks = []
    for i, (v, w) in enumerate(zip(values, mass)):
        blocks.append([i, i, float(w), float(v * w)])
        while len(blocks) > 1 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    fitted = np.zeros(len(values))
    membership = np.zeros(len(values), dtype=int)
    for j, b in enumerate(blocks):
        fitted[b[0] : b[1] + 1] = b[3] / b[2]
        membership[b[0] : b[1] + 1] = j
    return fitted, membership, blocks


def calibrate_grade_pd(
    counts,
    defaults,
    *,
    grades=None,
    prior_alpha=0.0,
    prior_beta=0.0,
    monotonic=True,
    target_default_rate=None,
    default_grade=None,
    confidence=0.95,
    target_weights=None,
    context=None,
):
    """Ordered grade PAVA with explicit beta pseudo-counts and optional level target."""
    n = _numeric(counts, "counts", vector_only=True)
    d = _numeric(defaults, "defaults", vector_only=True)
    if (
        len(n) != len(d)
        or not len(n)
        or np.any(n <= 0)
        or np.any(d < 0)
        or np.any(d > n)
        or np.any(n != np.floor(n))
        or np.any(d != np.floor(d))
    ):
        _fail("INVALID_COUNTS", "Require integer 0<=defaults<=positive counts")
    names = list(grades) if grades is not None else [str(i) for i in range(len(n))]
    if (
        len(names) != len(n)
        or any(not isinstance(x, str) or not x for x in names)
        or len(set(names)) != len(names)
    ):
        _fail("INVALID_INPUT", "Unique ordered grade names required")
    a = _scalar(prior_alpha, "prior_alpha", 0)
    b = _scalar(prior_beta, "prior_beta", 0)
    conf = _scalar(confidence, "confidence", 1e-8, 1 - 1e-8)
    if type(monotonic) is not bool:
        _fail("INVALID_INPUT", "monotonic must be boolean")
    mask = np.ones(len(n), dtype=bool)
    if default_grade is not None:
        if default_grade not in names:
            _fail("INVALID_INPUT", "Unknown default grade")
        j = names.index(default_grade)
        if d[j] != n[j]:
            _fail("INVALID_COUNTS", "Default grade must contain only defaults")
        mask[j] = False
    if not mask.any():
        _fail("EMPTY_INPUT", "At least one performing grade required")
    tw = (
        n
        if target_weights is None
        else _numeric(target_weights, "target_weights", vector_only=True)
    )
    if len(tw) != len(n) or np.any(tw < 0) or not np.isfinite(tw.sum()) or tw[mask].sum() <= 0:
        _fail(
            "INVALID_WEIGHTS", "Target weights must be nonnegative with positive performing total"
        )
    means = (d[mask] + a) / (n[mask] + a + b)
    mass = n[mask] + a + b
    fitted, membership, blocks = (
        _pava(means, mass)
        if monotonic
        else (
            means,
            np.arange(len(means)),
            [[i, i, mass[i], means[i] * mass[i]] for i in range(len(means))],
        )
    )
    before = fitted.copy()
    shift = 0.0
    if target_default_rate is not None:
        t = _scalar(target_default_rate, "target_default_rate", 0, 1)
        positive = (fitted > 0) & (fitted < 1)
        lo = float(tw[mask][fitted == 1].sum() / tw[mask].sum())
        hi = float(tw[mask][fitted > 0].sum() / tw[mask].sum())
        if not lo < t < hi:
            _fail("INFEASIBLE_TARGET", "Target cannot preserve structural grade boundaries")
        f = lambda z: float(
            tw[mask]
            @ np.where(positive, expit(z + logit(np.clip(fitted, 1e-300, 1 - 1e-15))), fitted)
            / tw[mask].sum()
            - t
        )
        shift = brentq(f, -1000, 1000, xtol=1e-12)
        fitted[positive] = expit(logit(fitted[positive]) + shift)
    result = []
    k = 0
    for i, g in enumerate(names):
        if not mask[i]:
            result.append(
                {
                    "grade": g,
                    "pd": 1.0,
                    "pool": None,
                    "count": int(n[i]),
                    "defaults": int(d[i]),
                    "lower": 1.0,
                    "upper": 1.0,
                    "default_grade": True,
                }
            )
            continue
        bl = blocks[int(membership[k])]
        ix = np.flatnonzero(mask)[bl[0] : bl[1] + 1]
        nn = float(n[ix].sum())
        dd = float(d[ix].sum())
        aa = dd + a * len(ix)
        bb = nn - dd + b * len(ix)
        # Exact binomial bounds without a prior; posterior equal-tailed bounds with a prior.
        lower = (
            0.0
            if aa == 0
            else float(beta_distribution.ppf((1 - conf) / 2, aa, bb + (1 if a + b == 0 else 0)))
        )
        upper = (
            1.0
            if bb == 0
            else float(beta_distribution.ppf((1 + conf) / 2, aa + (1 if a + b == 0 else 0), bb))
        )
        result.append(
            {
                "grade": g,
                "pd": float(fitted[k]),
                "pool": int(membership[k]),
                "count": int(n[i]),
                "defaults": int(d[i]),
                "lower": lower,
                "upper": upper,
                "default_grade": False,
            }
        )
        k += 1
    return _out(
        {
            "grades": result,
            "intercept_shift": shift,
            "achieved_mean_pd": float(tw[mask] @ fitted / tw[mask].sum()),
            "pre_target_mean_pd": float(tw[mask] @ before / tw[mask].sum()),
            "interval_scope": "Pre-target pooled intervals, conditional on selected pools; no simultaneous/post-selection guarantee",
            "target_weighting": "counts" if target_weights is None else "user_supplied",
            "interval_method": "beta_posterior" if a + b else "exact_binomial",
        },
        "calibrate_grade_pd",
        context,
    )


def low_default_bound(
    defaults,
    counts,
    *,
    confidence=0.95,
    dependence="independent",
    correlation=0.0,
    pooling="none",
    integration_tolerance=1e-8,
    context=None,
):
    """One-sided binomial or Gaussian one-factor bounds; optional ordered tail pooling."""
    d = np.atleast_1d(_numeric(defaults))
    n = np.atleast_1d(_numeric(counts))
    conf = _scalar(confidence, "confidence", 1e-8, 1 - 1e-8)
    rho = _scalar(correlation, "correlation", 0, 0.999)
    tol = _scalar(integration_tolerance, "integration_tolerance", 1e-12, 1e-3)
    if (
        d.ndim != 1
        or n.shape != d.shape
        or np.any(n <= 0)
        or np.any(d < 0)
        or np.any(d > n)
        or np.any(n != np.floor(n))
        or np.any(d != np.floor(d))
    ):
        _fail("INVALID_COUNTS", "Require integer counts/defaults")
    if dependence not in ("independent", "gaussian_one_factor") or pooling not in (
        "none",
        "ordered_tail",
    ):
        _fail("INVALID_INPUT", "Unknown dependence/pooling mode")
    if dependence == "independent" and rho:
        _fail("INVALID_INPUT", "Independent mode requires zero correlation")
    if pooling == "ordered_tail":
        d = np.cumsum(d[::-1])[::-1]
        n = np.cumsum(n[::-1])[::-1]
    rows = []
    for dd, nn in zip(d, n):
        worst = 0.0
        evaluations = 0

        def cdf(p):
            nonlocal worst, evaluations
            evaluations += 1
            if p <= 0:
                return 1.0
            if p >= 1:
                return float(dd == nn)
            if not rho:
                return float(binom.cdf(dd, int(nn), p))
            value, error = quad(
                lambda z: float(
                    binom.cdf(
                        dd, int(nn), norm.cdf((norm.ppf(p) - np.sqrt(rho) * z) / np.sqrt(1 - rho))
                    )
                    * norm.pdf(z)
                ),
                -10,
                10,
                epsabs=tol,
                epsrel=tol,
                limit=200,
            )
            worst = max(worst, error)
            return value

        bound = 1.0 if dd == nn else brentq(lambda p: cdf(p) - (1 - conf), 0, 1, xtol=1e-12)
        rows.append(
            {
                "defaults": int(dd),
                "count": int(nn),
                "upper_pd": bound,
                "cdf_at_bound": cdf(bound),
                "integration_error": worst,
                "evaluations": evaluations,
            }
        )
    if pooling == "ordered_tail":
        bound = 0.0
        for row in rows:
            bound = max(bound, row["upper_pd"])
            row["ordered_upper_pd"] = bound
    return _out(
        {
            "bounds": rows,
            "confidence": conf,
            "dependence": dependence,
            "correlation": rho,
            "pooling": pooling,
            "assumption": "Conditional independent Bernoulli given one shared standard normal factor; declared correlation is fixed, not estimated",
            "confidence_scope": "Model-conditional pointwise bounds; unknown dependence and multiple grades lack universal simultaneous coverage",
        },
        "low_default_bound",
        context,
    )


def fit_hierarchical_calibration(
    base_pd, outcomes, groups, weights=None, *, metadata, variance_bounds=(1e-8, 25.0), context=None
):
    """Offset-logit normal random intercepts fitted by Laplace marginal likelihood."""
    meta = _base()._metadata(metadata, True)
    p, w, y = _base()._data(base_pd, weights, outcomes, probability=True, open_domain=True)
    g = list(groups)
    if len(g) != len(p) or any(not isinstance(x, str) or not x for x in g):
        _fail("INVALID_INPUT", "Nonempty group IDs required")
    levels = sorted(set(g))
    ix = [np.flatnonzero(np.array(g) == v) for v in levels]
    if len(levels) < 2 or any(w[k].sum() == 0 for k in ix) or len(set(y[w > 0])) < 2:
        _fail("NOT_IDENTIFIABLE", "At least two positive-weight groups and both classes required")
    bounds = _numeric(variance_bounds, "variance_bounds", vector_only=True)
    if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] <= 100:
        _fail("INVALID_INPUT", "Positive ordered variance bounds <=100 required")
    offset = logit(p)

    def modes(a, v):
        ans = []
        total = 0.0
        gradient = np.zeros(2)
        for k in ix:
            f = lambda b: float(w[k] @ (y[k] - expit(offset[k] + a + b)) - b / v)
            b = brentq(
                f, -max(100, float(w[k].sum() * v)), max(100, float(w[k].sum() * v)), xtol=1e-11
            )
            q = expit(offset[k] + a + b)
            h = float(w[k] @ (q * (1 - q)) + 1 / v)
            ll = (
                float(w[k] @ (y[k] * (offset[k] + a + b) - np.logaddexp(0, offset[k] + a + b)))
                - b * b / (2 * v)
                - 0.5 * np.log(v * h)
            )
            total += ll
            ans.append((b, 1 / h))
            third = float(w[k] @ (q * (1 - q) * (1 - 2 * q)))
            gradient[0] += b / v - 0.5 * third / (v * h * h)
            gradient[1] += b * b / (2 * v) - 0.5 - 0.5 * third * b / (v * h * h) + 0.5 / (v * h)
        return total, ans, gradient

    fit = minimize(
        lambda x: -modes(x[0], np.exp(x[1]))[0],
        [0.0, np.log(np.clip(0.1, *bounds))],
        jac=lambda x: -modes(x[0], np.exp(x[1]))[2],
        method="L-BFGS-B",
        bounds=[(-30, 30), (np.log(bounds[0]), np.log(bounds[1]))],
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000},
    )
    if not fit.success or abs(fit.x[0]) >= 29.99:
        _fail("NONCONVERGENCE", "Hierarchical Laplace fit did not converge")
    a = float(fit.x[0])
    v = float(np.exp(fit.x[1]))
    ll, posterior, gradient = modes(a, v)
    effects = [
        {
            "group": g,
            "mode": float(b),
            "conditional_variance": float(var),
            "shrinkage_information_fraction": float(1 - var / v),
        }
        for g, (b, var) in zip(levels, posterior)
    ]
    params = {
        "intercept": a,
        "variance": v,
        "groups": effects,
        "unknown_group": "logistic_normal_marginal",
        "known_group": "conditional_mode",
        "quadrature_points": 40,
    }
    return _ext_artifact(
        "hierarchical",
        params,
        meta,
        {"p": p.tolist(), "w": w.tolist(), "y": y.tolist(), "groups": g},
        {
            "laplace_log_likelihood": ll,
            "gradient": gradient.tolist(),
            "variance_at_boundary": bool(v < bounds[0] * 1.01 or v > bounds[1] * 0.99),
            "variance_bounds": bounds.tolist(),
            "uncertainty_note": "Conditional group inverse curvature; hyperparameter and approximation uncertainty not included",
        },
        context,
    )


def _loss_metadata(metadata):
    if not isinstance(metadata, Mapping) or metadata.get("weight_type") not in (
        "frequency",
        "exposure",
    ):
        _fail(
            "INVALID_WEIGHT_TYPE",
            "Loss calibration requires declared frequency or exposure weights",
        )
    kind = metadata["weight_type"]
    copy = dict(metadata)
    copy["weight_type"] = "frequency"
    meta = _base()._metadata(copy, False)
    meta["weight_type"] = kind
    return meta


def calibrate_lgd_level(
    base_lgd, target_lgd, weights=None, *, metadata, transformation="additive", context=None
):
    """Shift raw losses, multiply nonnegative losses, or shift bounded loss logits."""
    meta = _loss_metadata(metadata)
    x = _numeric(base_lgd, "base_lgd", vector_only=True)
    w = np.ones(len(x)) if weights is None else _numeric(weights, "weights", vector_only=True)
    target = _scalar(target_lgd, "target_lgd")
    if (
        len(w) != len(x)
        or np.any(w < 0)
        or not np.isfinite(w.sum())
        or w.sum() <= 0
        or (meta["weight_type"] == "frequency" and np.any(w != np.floor(w)))
    ):
        _fail("INVALID_WEIGHTS", "Valid declared frequency/exposure weights required")
    if transformation not in ("additive", "multiplicative", "logit"):
        _fail("INVALID_INPUT", "Unknown loss transformation")
    if transformation == "additive":
        parameter = target - float(w @ x / w.sum())
        adjusted = x + parameter
    elif transformation == "multiplicative":
        mean = float(w @ x / w.sum())
        if np.any(x < 0) or mean <= 0 or target < 0:
            _fail("INFEASIBLE_TARGET", "Positive base mean and nonnegative losses/target required")
        parameter = target / mean
        adjusted = x * parameter
    else:
        _probability(x, "open")
        if not 0 < target < 1:
            _fail("INFEASIBLE_TARGET", "Logit target must be interior")
        parameter = brentq(lambda z: float(w @ expit(logit(x) + z) / w.sum() - target), -1000, 1000)
        adjusted = expit(logit(x) + parameter)
    params = {
        "transformation": transformation,
        "parameter": float(parameter),
        "target_lgd": target,
        "boundary_policy": "raw"
        if transformation == "additive"
        else "nonnegative"
        if transformation == "multiplicative"
        else "reject",
    }
    return _ext_artifact(
        "lgd_level",
        params,
        meta,
        {"x": x.tolist(), "w": w.tolist()},
        {
            "achieved_mean": float(w @ adjusted / w.sum()),
            "adjusted_lgd": adjusted.tolist(),
            "weighting": meta["weight_type"],
        },
        context,
    )


def estimate_margin_of_conservatism(
    base_estimate,
    components,
    *,
    confidence=0.95,
    correlation=None,
    already_included=(),
    bounds=None,
    context=None,
):
    """Explicit bias plus joint normal uncertainty uplift, with source deduplication."""
    base = _scalar(base_estimate, "base_estimate")
    conf = _scalar(confidence, "confidence", 0.5, 1 - 1e-8)
    if not isinstance(components, (list, tuple)):
        _fail("INVALID_INPUT", "List of uncertainty components required")
    included = list(already_included)
    ids = []
    bias = []
    sd = []
    active = []
    detail = []
    if len(set(included)) != len(included):
        _fail("DUPLICATE_UNCERTAINTY", "Duplicate already-included source")
    for row in components:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("source_id"), str)
            or not row["source_id"]
            or row.get("category") not in ("data", "method", "general")
        ):
            _fail("INVALID_INPUT", "source_id and data/method/general category required")
        sid = row["source_id"]
        if sid in ids:
            _fail("DUPLICATE_UNCERTAINTY", "One source cannot be counted twice")
        ids.append(sid)
        b = _scalar(row.get("adverse_bias"), "adverse_bias", 0)
        s = _scalar(row.get("standard_error"), "standard_error", 0)
        use = sid not in included
        bias.append(b if use else 0)
        sd.append(s if use else 0)
        active.append(use)
        detail.append(
            {
                "source_id": sid,
                "category": row["category"],
                "included": use,
                "bias": b,
                "standard_error": s,
            }
        )
    m = len(ids)
    C = np.eye(m) if correlation is None else _numeric(correlation, "correlation")
    if C.shape != (m, m) or (
        m
        and (
            not np.allclose(C, C.T, atol=1e-12)
            or not np.allclose(np.diag(C), 1)
            or np.linalg.eigvalsh(C).min() < -1e-10
        )
    ):
        _fail("INVALID_CORRELATION", "Symmetric PSD correlation matrix required")
    uncertainty = (
        float(norm.ppf(conf) * np.sqrt(max(0, np.array(sd) @ C @ np.array(sd)))) if m else 0.0
    )
    moc = float(sum(bias) + uncertainty)
    raw = base + moc
    adjusted = raw
    if bounds is not None:
        limit = _numeric(bounds, "bounds", vector_only=True)
        if len(limit) != 2 or limit[0] >= limit[1]:
            _fail("INVALID_INPUT", "Ordered bounds required")
        adjusted = float(np.clip(raw, *limit))
    return _out(
        {
            "components": detail,
            "confidence": conf,
            "correlation": C.tolist(),
            "bias_margin": float(sum(bias)),
            "uncertainty_margin": uncertainty,
            "margin": moc,
            "unbounded_adjusted_estimate": raw,
            "adjusted_estimate": adjusted,
            "clipping_adjustment": adjusted - raw,
            "assumption": "Declared joint normal estimation errors; one-sided quantile; no universal regulatory default",
        },
        "estimate_margin_of_conservatism",
        context,
    )


def population_reweighting(
    sample_features,
    target_distribution,
    *,
    method="poststratification",
    cap=None,
    outcome_dependent=False,
    context=None,
):
    """Finite strata poststratification or supplied sample-membership density ratios."""
    if type(outcome_dependent) is not bool:
        _fail("INVALID_INPUT", "Declare outcome_dependent boolean")
    if method == "poststratification":
        groups = list(sample_features)
        if (
            not groups
            or any(not isinstance(g, str) or not g for g in groups)
            or not isinstance(target_distribution, Mapping)
        ):
            _fail("INVALID_INPUT", "Sample stratum strings and target proportions object required")
        target = {k: _scalar(v, "target proportion", 0, 1) for k, v in target_distribution.items()}
        if not np.isclose(sum(target.values()), 1, atol=1e-12, rtol=0):
            _fail("INVALID_INPUT", "Target proportions must sum to one")
        missing = [k for k, v in target.items() if v > 0 and k not in groups]
        if missing:
            _fail(
                "NO_COMMON_SUPPORT",
                "Positive target strata absent from sample",
                {"strata": missing},
            )
        freq = {k: v / len(groups) for k, v in Counter(groups).items()}
        raw = np.array([target.get(g, 0) / freq[g] for g in groups])
    elif method == "membership_odds":
        q = _probability(sample_features, "open", vector_only=True)
        if not isinstance(target_distribution, Mapping):
            _fail("INVALID_INPUT", "Supply sample_prior and target_prior")
        ps = _scalar(target_distribution.get("sample_prior"), "sample_prior", 1e-12, 1 - 1e-12)
        pt = _scalar(target_distribution.get("target_prior"), "target_prior", 1e-12, 1 - 1e-12)
        if abs(ps + pt - 1) > 1e-12:
            _fail("INVALID_INPUT", "Membership design priors must sum to one")
        # q=P(target membership|x), priors are membership sampling fractions.
        raw = q / (1 - q) * ps / pt
        groups = None
        target = None
    else:
        _fail("INVALID_INPUT", "Unknown population weighting method")
    if cap is not None:
        cap = _scalar(cap, "cap", 1e-12)
    clipped = raw if cap is None else np.minimum(raw, cap)
    if not np.isfinite(raw).all() or not np.isfinite(clipped.sum()) or clipped.sum() <= 0:
        _fail("NO_COMMON_SUPPORT", "No positive population weights")
    w = clipped * len(clipped) / clipped.sum()
    realized = (
        {g: float(w[np.array(groups) == g].sum() / w.sum()) for g in sorted(set(groups))}
        if groups
        else None
    )
    return _out(
        {
            "weights": w.tolist(),
            "raw_weights": raw.tolist(),
            "effective_sample_size": float(w.sum() ** 2 / (w @ w)),
            "clipped_count": int(np.sum(clipped != raw)),
            "cap_before_normalization": cap,
            "max_final_weight": float(w.max()),
            "realized_distribution": realized,
            "target_distribution": target,
            "outcome_dependent_sampling": outcome_dependent,
            "normalization": "Mean one after cap; final maximum can exceed pre-normalization cap",
            "support_note": "Overlap is explicit by stratum; supplied membership propensities require externally assessed covariate overlap",
        },
        "population_reweighting",
        context,
    )


def _estimator(value):
    if not isinstance(value, str) or value.upper() not in ("OLS", "WLS", "GLM"):
        _fail("INVALID_INPUT", "estimator must be OLS, WLS or GLM")
    return value.upper()


def _xy(X, y, feature_order, weights=None, estimator="OLS"):
    x = _numeric(X, "X")
    y = _numeric(y, "outcomes", vector_only=True)
    if (
        x.ndim != 2
        or len(x) != len(y)
        or not len(y)
        or x.shape[1] != len(feature_order)
        or len(set(feature_order)) != len(feature_order)
        or any(not isinstance(f, str) or not f for f in feature_order)
    ):
        _fail("INVALID_DESIGN", "Finite matrix and unique feature_order required")
    w = np.ones(len(y)) if weights is None else _numeric(weights, "weights", vector_only=True)
    if len(w) != len(y) or np.any(w <= 0) or not np.isfinite(w.sum()):
        _fail("INVALID_WEIGHTS", "Strictly positive row weights required")
    if estimator not in ("OLS", "WLS", "GLM"):
        _fail("INVALID_INPUT", "estimator must be OLS, WLS or GLM")
    if estimator == "OLS" and np.any(w != 1):
        _fail("INVALID_WEIGHTS", "OLS uses unit weights; choose WLS")
    if estimator == "GLM" and (np.any((y != 0) & (y != 1)) or np.any(w != np.floor(w))):
        _fail("INVALID_OUTCOME", "GLM requires binary outcomes and integer frequency weights")
    return x, y, w


def _fit_model(x, y, w, features, estimator, penalty=0.0, alpha=0.0, standardize=False):
    center = np.average(x, axis=0, weights=w) if standardize else np.zeros(x.shape[1])
    scale = (
        np.sqrt(np.average((x - center) ** 2, axis=0, weights=w))
        if standardize
        else np.ones(x.shape[1])
    )
    scale = np.where(scale > 1e-12, scale, 1.0)
    D = np.column_stack([np.ones(len(x)), (x - center) / scale])
    n = len(y)
    if not penalty and np.linalg.matrix_rank(D * np.sqrt(w[:, None])) < D.shape[1]:
        _fail("RANK_DEFICIENT", "Selected fixed-row design is singular")
    if estimator == "GLM" and len(set(y)) < 2:
        _fail("SEPARATION", "Both GLM classes required")
    if estimator == "GLM" and not penalty:
        from .classification import _separation

        _separation(D, y)
    b = np.zeros(D.shape[1])
    b[0] = logit(np.average(y, weights=w)) if estimator == "GLM" else np.average(y, weights=w)
    if estimator != "GLM" and alpha == 0:
        b = np.linalg.solve(
            D.T @ (w[:, None] * D)
            + np.diag([0.0] + [penalty * (1 - alpha) * w.sum()] * (len(b) - 1)),
            D.T @ (w * y),
        )
    else:
        # Proximal gradient of average loss, exact intercept unpenalized.
        L = float(np.linalg.eigvalsh(D.T @ (w[:, None] * D) / w.sum()).max()) * (
            0.25 if estimator == "GLM" else 1
        ) + penalty * (1 - alpha)
        for it in range(50000):
            eta = D @ b
            pred = expit(eta) if estimator == "GLM" else eta
            grad = D.T @ (w * (pred - y)) / w.sum()
            grad[1:] += penalty * (1 - alpha) * b[1:]
            q = b - grad / L
            q[1:] = np.sign(q[1:]) * np.maximum(0, np.abs(q[1:]) - penalty * alpha / L)
            if np.max(abs(q - b)) < 1e-10:
                b = q
                break
            b = q
        else:
            _fail("NONCONVERGENCE", "Penalized candidate did not converge")
    eta = D @ b
    if estimator == "GLM":
        ll = float(w @ (y * eta - np.logaddexp(0, eta)))
        k = len(b)
        variance = None
    else:
        rss = float(w @ ((y - eta) ** 2))
        variance = rss / n
        if variance <= 1e-20:
            _fail("DEGENERATE_VARIANCE", "Gaussian likelihood has zero residual variance")
        ll = float(-0.5 * n * (log(2 * pi) + 1 + log(variance)) + 0.5 * np.log(w).sum())
        k = len(b) + 1
    model = {
        "schema_version": "1.0",
        "artifact_type": "SelectionModel",
        "estimator": estimator,
        "link": "logit" if estimator == "GLM" else "identity",
        "feature_order": list(features),
        "intercept": float(b[0]),
        "coefficients": b[1:].tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "penalty": float(penalty),
        "elastic_net_alpha": float(alpha),
        "residual_variance": variance,
        "log_likelihood": ll,
        "parameter_count": k,
        "row_count": n,
        "weight_sum": float(w.sum()),
    }
    model["training"] = {
        "role": "training_only",
        "known_at": None,
        "cutoff_status": "not_supplied",
        "data_fingerprint": _fingerprint({"X": x.tolist(), "y": y.tolist(), "weights": w.tolist()}),
    }
    model["diagnostic_hash"] = _fingerprint(
        {"log_likelihood": ll, "parameter_count": k, "row_count": n, "weight_sum": float(w.sum())}
    )
    model["artifact_hash"] = _fingerprint(model)
    return model


def _prediction(model, x):
    eta = model["intercept"] + (
        (x - np.array(model["center"])) / np.array(model["scale"])
    ) @ np.array(model["coefficients"])
    return expit(eta) if model["estimator"] == "GLM" else eta


def fit_stepwise(
    X,
    outcomes,
    *,
    feature_order,
    candidate_groups,
    mandatory_terms=(),
    hierarchy=None,
    direction="bidirectional",
    criterion="AIC",
    estimator="GLM",
    weights=None,
    max_steps=100,
    lr_alpha=0.05,
    training=None,
    context=None,
):
    """Deterministic grouped stepwise likelihood search on one complete fixed sample."""
    estimator = _estimator(estimator)
    x, y, w = _xy(X, outcomes, feature_order, weights, estimator)
    if not isinstance(candidate_groups, Mapping) or not candidate_groups:
        _fail("INVALID_INPUT", "Named candidate groups required")
    groups = {g: list(cols) for g, cols in candidate_groups.items()}
    names = sorted(groups)
    if any(
        not isinstance(g, str)
        or not g
        or not cols
        or len(set(cols)) != len(cols)
        or any(c not in feature_order for c in cols)
        for g, cols in groups.items()
    ):
        _fail("INVALID_INPUT", "Groups require unique known feature names")
    if len([c for cols in groups.values() for c in cols]) != len(
        set(c for cols in groups.values() for c in cols)
    ):
        _fail("INVALID_INPUT", "Candidate groups must not overlap")
    mandatory = set(mandatory_terms)
    parents = {} if hierarchy is None else {k: set(v) for k, v in hierarchy.items()}
    if not mandatory <= set(names) or any(
        k not in groups or not v <= set(names) or k in v for k, v in parents.items()
    ):
        _fail("INVALID_INPUT", "Invalid mandatory terms/hierarchy")

    def valid(s):
        return all(parents.get(g, set()) <= s for g in s)

    for g in names:
        seen = set()
        stack = list(parents.get(g, set()))
        while stack:
            h = stack.pop()
            if h == g:
                _fail("INVALID_INPUT", "Cyclic interaction hierarchy")
            if h not in seen:
                seen.add(h)
                stack.extend(parents.get(h, set()))
    if not valid(mandatory):
        _fail("INVALID_INPUT", "Mandatory set must include hierarchy parents")
    if not isinstance(criterion, str):
        _fail("INVALID_INPUT", "String criterion required")
    criterion = criterion.upper()
    if direction not in ("forward", "backward", "bidirectional") or criterion not in (
        "AIC",
        "BIC",
        "LR",
    ):
        _fail("INVALID_INPUT", "Unknown direction or selection criterion")
    steps = int(_scalar(max_steps, "max_steps", 1, 10000))
    if steps != max_steps:
        _fail("INVALID_INPUT", "Integer max_steps required")
    sig = _scalar(lr_alpha, "lr_alpha", 1e-8, 1 - 1e-8)
    selected = set(names) if direction == "backward" else mandatory.copy()
    cache = {}
    evidence = []

    def fit(s):
        key = tuple(sorted(s))
        if key not in cache:
            feats = [f for f in feature_order if any(f in groups[g] for g in s)]
            ix = [feature_order.index(f) for f in feats]
            cache[key] = _fit_model(x[:, ix], y, w, feats, estimator)
        return cache[key]

    def score(m):
        return (
            -2 * m["log_likelihood"]
            + (2 if criterion == "AIC" else log(w.sum() if estimator == "GLM" else len(y)))
            * m["parameter_count"]
        )

    current = fit(selected)
    path = [
        {
            "step": 0,
            "groups": sorted(selected),
            "log_likelihood": current["log_likelihood"],
            "criterion_value": None if criterion == "LR" else score(current),
        }
    ]
    visited = {tuple(sorted(selected))}
    reason = "no_improvement"
    for step in range(1, steps + 1):
        candidates = []
        moves = [("add", g) for g in names if g not in selected] if direction != "backward" else []
        if direction != "forward":
            moves += [("remove", g) for g in names if g in selected and g not in mandatory]
        for action, g in moves:
            trial = selected | {g} if action == "add" else selected - {g}
            if not valid(trial) or tuple(sorted(trial)) in visited:
                continue
            try:
                m = fit(trial)
                df = abs(m["parameter_count"] - current["parameter_count"])
                stat = max(0, 2 * abs(m["log_likelihood"] - current["log_likelihood"]))
                p = float(chi2.sf(stat, df))
                improvement = (
                    score(current) - score(m)
                    if criterion != "LR"
                    else (
                        -log(max(p, 1e-300))
                        if action == "add" and p < sig
                        else log(p / sig)
                        if action == "remove" and p > sig
                        else -1.0
                    )
                )
                row = {
                    "step": step,
                    "action": action,
                    "group": g,
                    "groups": sorted(trial),
                    "log_likelihood": m["log_likelihood"],
                    "criterion_value": None if criterion == "LR" else score(m),
                    "lr_statistic": stat,
                    "lr_df": df,
                    "lr_p": p,
                    "improvement": improvement,
                    "failure": None,
                }
                evidence.append(row)
                if improvement > 1e-10:
                    candidates.append((improvement, action, g, trial, m, row))
            except ValueError as error:
                evidence.append(
                    {
                        "step": step,
                        "action": action,
                        "group": g,
                        "failure": getattr(error, "code", str(error)),
                    }
                )
        if not candidates:
            break
        # Stable tolerance tie: lexicographic action/group order among equal best improvements.
        best = max(z[0] for z in candidates)
        choice = min((z for z in candidates if best - z[0] <= 1e-10), key=lambda z: (z[1], z[2]))
        _, _, _, selected, current, row = choice
        visited.add(tuple(sorted(selected)))
        path.append(row)
    else:
        reason = "maximum_steps"
    current = _stamp_selection_training(current, training)
    return _out(
        {
            "selected_model": current,
            "selected_groups": sorted(selected),
            "path": path,
            "candidate_evidence": evidence,
            "stopping_reason": reason,
            "criterion": criterion,
            "direction": direction,
            "fixed_sample_fingerprint": _fingerprint(
                {"X": x.tolist(), "y": y.tolist(), "w": w.tolist()}
            ),
            "inference_note": "Selection path is adaptive; selected-model fixed-model standard errors are not post-selection inference",
        },
        "fit_stepwise",
        context,
    )


def _split_plan(plan, n):
    from .core import _iso_date

    if (
        not isinstance(plan, Mapping)
        or not isinstance(plan.get("rows"), list)
        or len(plan["rows"]) != n
        or not isinstance(plan.get("folds"), list)
        or not plan["folds"]
    ):
        _fail("INVALID_SPLIT", "rows and nonempty folds required")
    rows = plan["rows"]
    ids = []
    for r in rows:
        if (
            not isinstance(r, Mapping)
            or not isinstance(r.get("entity_id"), str)
            or not r["entity_id"]
            or r.get("role") not in ("development", "calibration", "test")
        ):
            _fail("INVALID_SPLIT", "Entity and development/calibration/test role required")
        for k in ("as_of", "label_known_at"):
            _iso_date(r.get(k), k)
        if r["label_known_at"] < r["as_of"]:
            _fail("INVALID_SPLIT", "Outcome cannot be known before observation")
        ids.append(r["entity_id"])

    def indices(value):
        if (
            not isinstance(value, list)
            or not value
            or any(type(v) is not int or not 0 <= v < n for v in value)
            or len(set(value)) != len(value)
        ):
            _fail("INVALID_SPLIT", "Nonempty unique zero-based integer row indices required")
        return value

    folds = []
    for fold in plan["folds"]:
        if not isinstance(fold, Mapping):
            _fail("INVALID_SPLIT", "Fold must be an object")
        tr = indices(fold.get("train"))
        va = indices(fold.get("validation"))
        if set(tr) & set(va) or set(ids[i] for i in tr) & set(ids[i] for i in va):
            _fail("LEAKAGE", "Train and validation entities must be disjoint")
        if any(rows[i]["role"] != "development" for i in tr + va):
            _fail("LEAKAGE", "Fold selection may use development rows only")
        if max(rows[i]["label_known_at"] for i in tr) >= min(rows[i]["as_of"] for i in va):
            _fail("LEAKAGE", "All training labels must be known strictly before validation begins")
        folds.append((tr, va))
    byrole = {
        role: [i for i, r in enumerate(rows) if r["role"] == role]
        for role in ("development", "calibration", "test")
    }
    for i, a in enumerate(byrole):
        for b in list(byrole)[i + 1 :]:
            if set(ids[k] for k in byrole[a]) & set(ids[k] for k in byrole[b]):
                _fail("LEAKAGE", "Development, calibration and test entity sets overlap")
            if (
                byrole[a]
                and byrole[b]
                and max(rows[k]["label_known_at"] for k in byrole[a])
                >= min(rows[k]["as_of"] for k in byrole[b])
            ):
                _fail("LEAKAGE", "Earlier-stage labels not mature before later stage")
    return folds, byrole


def _loss(y, p, w, scoring):
    if scoring == "mse":
        return float(w @ ((y - p) ** 2) / w.sum())
    if scoring == "brier":
        return float(w @ ((y - p) ** 2) / w.sum())
    if scoring == "log_loss":
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return float(-w @ (y * np.log(p) + (1 - y) * np.log1p(-p)) / w.sum())
    _fail("INVALID_INPUT", "Unknown scoring rule")


def fit_regularization_path(
    X,
    outcomes,
    *,
    feature_order,
    penalty_grid,
    split_plan,
    estimator="GLM",
    elastic_net_alpha=0.0,
    scoring="log_loss",
    weights=None,
    context=None,
):
    """Fold-local scaled elastic-net paths with group/time leakage checks."""
    estimator = _estimator(estimator)
    x, y, w = _xy(X, outcomes, feature_order, weights, estimator)
    folds, roles = _split_plan(split_plan, len(y))
    grid = _numeric(penalty_grid, "penalty_grid", vector_only=True)
    alpha = _scalar(elastic_net_alpha, "elastic_net_alpha", 0, 1)
    if not len(grid) or np.any(grid < 0) or len(set(grid)) != len(grid):
        _fail("INVALID_INPUT", "Unique nonnegative penalties required")
    if scoring not in (("log_loss", "brier") if estimator == "GLM" else ("mse",)):
        _fail("INVALID_INPUT", "Scoring must match response family")
    results = []
    for penalty in grid:
        records = []
        for j, (tr, va) in enumerate(folds):
            try:
                m = _fit_model(
                    x[tr], y[tr], w[tr], feature_order, estimator, float(penalty), alpha, True
                )
                value = _loss(y[va], _prediction(m, x[va]), w[va], scoring)
                records.append(
                    {
                        "fold": j,
                        "score": value,
                        "failure": None,
                        "training_center": m["center"],
                        "training_scale": m["scale"],
                    }
                )
            except ValueError as error:
                records.append(
                    {"fold": j, "score": None, "failure": getattr(error, "code", str(error))}
                )
        score = (
            float(np.mean([r["score"] for r in records]))
            if all(r["failure"] is None for r in records)
            else None
        )
        results.append({"penalty": float(penalty), "mean_score": score, "folds": records})
    valid = [r for r in results if r["mean_score"] is not None]
    if not valid:
        _fail("ALL_CANDIDATES_FAILED", "No penalty completed every fold", {"path": results})
    selected = min(valid, key=lambda r: (r["mean_score"], -r["penalty"]))
    ix = roles["development"]
    model = _fit_model(
        x[ix], y[ix], w[ix], feature_order, estimator, selected["penalty"], alpha, True
    )
    model = _stamp_selection_training(
        model,
        {
            "known_at": max(split_plan["rows"][i]["label_known_at"] for i in ix),
            "role": "development_only",
        },
    )
    return _out(
        {
            "selected_penalty": selected["penalty"],
            "selected_model": model,
            "path": results,
            "selection_rule": "Mean fold loss; exact ties prefer stronger penalty",
            "untouched_rows": roles["calibration"] + roles["test"],
            "split_fingerprint": _fingerprint(split_plan),
        },
        "fit_regularization_path",
        context,
    )


def resample_fit_uncertainty(
    X,
    outcomes,
    *,
    feature_order,
    resampling_indices,
    resampling_units,
    unit_kind,
    estimator="GLM",
    weights=None,
    confidence=0.95,
    prediction_X=None,
    selection_spec=None,
    context=None,
):
    """Fit every supplied whole-cluster/time-block replicate, retaining failures."""
    estimator = _estimator(estimator)
    x, y, w = _xy(X, outcomes, feature_order, weights, estimator)
    units = list(resampling_units)
    conf = _scalar(confidence, "confidence", 1e-8, 1 - 1e-8)
    if (
        unit_kind not in ("cluster", "time_block")
        or len(units) != len(y)
        or any(not isinstance(u, str) or not u for u in units)
    ):
        _fail("INVALID_RESAMPLING", "Declare whole-cluster or predeclared time-block units")
    if not isinstance(resampling_indices, list) or not resampling_indices:
        _fail("INVALID_RESAMPLING", "Explicit shared zero-based replicate indices required")
    px = x if prediction_X is None else _numeric(prediction_X, "prediction_X")
    if px.ndim != 2 or px.shape[1] != x.shape[1]:
        _fail("INVALID_DESIGN", "Prediction matrix columns mismatch")
    results = []
    predictions = []
    coefficients = []
    for j, idx in enumerate(resampling_indices):
        if (
            not isinstance(idx, list)
            or not idx
            or any(type(i) is not int or not 0 <= i < len(y) for i in idx)
        ):
            _fail("INVALID_RESAMPLING", "Invalid zero-based resampling index")
        freq = np.bincount(idx, minlength=len(y))
        for unit in set(units):
            if len(set(freq[np.array(units) == unit])) != 1:
                _fail(
                    "INVALID_RESAMPLING",
                    "Every selected cluster/time block must be repeated in full",
                )
        try:
            if selection_spec is None:
                m = _fit_model(x[idx], y[idx], w[idx], feature_order, estimator)
            else:
                if not isinstance(selection_spec, Mapping):
                    _fail("INVALID_INPUT", "selection_spec must be stepwise keyword arguments")
                m = fit_stepwise(
                    x[idx],
                    y[idx],
                    feature_order=feature_order,
                    weights=w[idx],
                    estimator=estimator,
                    **selection_spec,
                ).value["selected_model"]
            cols = [feature_order.index(f) for f in m["feature_order"]]
            p = _prediction(m, px[:, cols])
            predictions.append(p)
            beta = np.zeros(x.shape[1] + 1)
            beta[0] = m["intercept"]
            beta[np.array(cols, dtype=int) + 1] = m["coefficients"]
            coefficients.append(beta)
            results.append({"replicate": j, "model": m, "failure": None})
        except ValueError as error:
            results.append(
                {"replicate": j, "model": None, "failure": getattr(error, "code", str(error))}
            )
    effective = len(predictions)
    q = [(1 - conf) / 2, (1 + conf) / 2]
    return _out(
        {
            "replicates": results,
            "effective_replicates": effective,
            "failed_replicates": len(results) - effective,
            "coefficient_intervals": np.quantile(coefficients, q, axis=0).T.tolist()
            if effective >= 2
            else None,
            "prediction_intervals": np.quantile(predictions, q, axis=0).T.tolist()
            if effective >= 2
            else None,
            "confidence": conf,
            "unit_kind": unit_kind,
            "selection_repeated": selection_spec is not None,
            "interval_note": "Percentile intervals conditional on successful replicates; failure proportion must be reviewed",
            "indices_fingerprint": _fingerprint(resampling_indices),
        },
        "resample_fit_uncertainty",
        context,
    )


def fit_model_selection_protocol(
    X,
    outcomes,
    *,
    feature_order,
    candidates,
    split_plan,
    metadata,
    search_budget,
    scoring="log_loss",
    context=None,
):
    """Inner development CV selection, disjoint calibration, untouched final test."""
    x, y, w = _xy(X, outcomes, feature_order, estimator="GLM")
    folds, roles = _split_plan(split_plan, len(y))
    meta = _base()._metadata(metadata, True)
    if not roles["calibration"] or not roles["test"]:
        _fail("INVALID_SPLIT", "Separate nonempty calibration/test roles required")
    calibration_rows = [split_plan["rows"][i] for i in roles["calibration"]]
    if (
        meta["calibration"]["prediction_origin"] != "HOLDOUT"
        or min(r["as_of"] for r in calibration_rows) < meta["calibration"]["start"]
        or max(r["as_of"] for r in calibration_rows) > meta["calibration"]["end"]
        or max(r["label_known_at"] for r in calibration_rows) > meta["calibration"]["known_at"]
    ):
        _fail(
            "INVALID_CALIBRATION_METADATA",
            "Calibration metadata must cover actual independent calibration rows and label maturity",
        )
    budget = _scalar(search_budget, "search_budget", 1, 1000000)
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) > budget
        or int(budget) != budget
    ):
        _fail("INVALID_INPUT", "Positive search budget must cover all declared candidates")
    evidence = []
    best = None
    names = set()
    for candidate in candidates:
        if (
            not isinstance(candidate, Mapping)
            or not isinstance(candidate.get("id"), str)
            or not candidate["id"]
            or candidate["id"] in names
        ):
            _fail("INVALID_INPUT", "Unique candidate IDs required")
        names.add(candidate["id"])
        try:
            columns = candidate.get("features", feature_order)
            if (
                not isinstance(columns, list)
                or len(set(columns)) != len(columns)
                or any(c not in feature_order for c in columns)
            ):
                _fail("INVALID_INPUT", "Invalid candidate feature subset")
            indices = [feature_order.index(f) for f in columns]
            result = fit_regularization_path(
                x[:, indices],
                y,
                feature_order=columns,
                penalty_grid=candidate.get("penalty_grid", [0.0]),
                elastic_net_alpha=candidate.get("elastic_net_alpha", 0.0),
                split_plan=split_plan,
                scoring=scoring,
            ).value
            score = next(
                row["mean_score"]
                for row in result["path"]
                if row["penalty"] == result["selected_penalty"]
            )
            evidence.append(
                {
                    "candidate_id": candidate["id"],
                    "score": score,
                    "failure": None,
                    "path": result["path"],
                }
            )
            if best is None or (score, candidate["id"]) < (best[0], best[1]):
                best = (score, candidate["id"], result["selected_model"], indices)
        except ValueError as error:
            evidence.append(
                {
                    "candidate_id": candidate["id"],
                    "score": None,
                    "failure": getattr(error, "code", str(error)),
                }
            )
    if best is None:
        _fail("ALL_CANDIDATES_FAILED", "Every candidate failed", {"evidence": evidence})
    _, name, model, cols = best
    ix = roles["calibration"]
    p = _prediction(model, x[ix][:, cols])
    cal = _base().calibrate_logistic(p, y[ix], metadata=meta, tolerance=1e-8).value
    frozen = _freeze_protocol_classifier(model, name, meta, split_plan)
    return _out(
        {
            "selected_candidate": name,
            "selected_model": model,
            "selected_frozen_model": frozen,
            "calibration_artifact": cal,
            "candidate_evidence": evidence,
            "untouched_test_indices": roles["test"],
            "test_used_for_selection": False,
            "test_metrics": None,
            "split_fingerprint": _fingerprint(split_plan),
            "protocol": "Development inner grouped temporal CV; final development refit; independent logistic calibration; no test predictions or labels used in optimization",
        },
        "fit_model_selection_protocol",
        context,
        data_origin=meta["data_origin"],
    )


def _apply_extended_calibrator(
    calibrator, values, *, target, base_model_version, groups=None, context=None
):
    raw = calibrator.value if isinstance(calibrator, CalcResult) else calibrator
    if not isinstance(raw, Mapping):
        _fail("INVALID_CALIBRATOR", "Expected calibration artifact")
    a = _json_data(raw)
    digest = a.pop("artifact_hash", None)
    if digest != _fingerprint(a):
        _fail("INTEGRITY_ERROR", "Calibration artifact differs from hash")
    kind = a.get("calibration_type")
    method = {
        "beta": "calibrate_beta",
        "hierarchical": "fit_hierarchical_calibration",
        "lgd_level": "calibrate_lgd_level",
    }.get(kind)
    if (
        a.get("schema_version") != "1.0"
        or a.get("artifact_type") != "CalibrationArtifact"
        or a.get("parameter_version") != "1.0"
        or a.get("method_id") != "calibration." + str(method)
        or a.get("input_kind") != ("loss" if kind == "lgd_level" else "probability")
        or not isinstance(a.get("calibration_input_fingerprint"), str)
        or len(a["calibration_input_fingerprint"]) != 64
    ):
        _fail("INVALID_CALIBRATOR", "Invalid extended calibration schema")
    meta = (
        _loss_metadata(a.get("metadata"))
        if kind == "lgd_level"
        else _base()._metadata(a.get("metadata"), True)
    )
    if a.get("target") != meta["target"] or not isinstance(a.get("parameters"), dict):
        _fail("INVALID_CALIBRATOR", "Invalid calibration target or parameters")
    if _base()._target(target) != a["target"]:
        _fail("INCOMPATIBLE_TARGET", "Calibration target differs")
    if base_model_version != meta["base_model_version"]:
        _fail("INCOMPATIBLE_MODEL_VERSION", "Calibration model version differs")
    q = a["parameters"]
    trace = {"calibration_hash": digest, "calibration_type": kind}
    if kind == "lgd_level":
        x = _numeric(values, "values", allow_empty=True, vector_only=True)
        parameter = _scalar(q.get("parameter"), "parameter")
        mode = q.get("transformation")
        _scalar(q.get("target_lgd"), "target_lgd")
        if mode not in ("additive", "multiplicative", "logit") or q.get("boundary_policy") != {
            "additive": "raw",
            "multiplicative": "nonnegative",
            "logit": "reject",
        }.get(mode):
            _fail("INVALID_CALIBRATOR", "Invalid loss transformation")
        if mode == "additive":
            v = x + parameter
        elif mode == "multiplicative":
            if parameter < 0 or np.any(x < 0):
                _fail("INVALID_CALIBRATOR", "Nonnegative multiplicative domain required")
            v = x * parameter
        else:
            _probability(x, "open")
            v = expit(logit(x) + parameter)
    elif kind == "beta":
        p = _probability(values, vector_only=True)
        eps = _scalar(q.get("epsilon"), "epsilon", 1e-15, 0.01)
        b0 = _scalar(q.get("intercept"), "intercept")
        b1 = _scalar(q.get("a"), "a")
        b2 = _scalar(q.get("b"), "b")
        if (
            type(q.get("monotonic")) is not bool
            or (q["monotonic"] and min(b1, b2) < 0)
            or q.get("boundary_policy") not in ("reject", "clip")
        ):
            _fail("INVALID_CALIBRATOR", "Invalid beta parameters")
        if q["boundary_policy"] == "reject" and np.any((p == 0) | (p == 1)):
            _fail("INVALID_PROBABILITY", "Interior beta input required")
        x = np.clip(p, eps, 1 - eps) if q["boundary_policy"] == "clip" else p
        v = expit(b0 + b1 * np.log(x) - b2 * np.log1p(-x))
        trace["clipped_count"] = int(np.sum(x != p))
    else:
        p = _probability(values, "open", vector_only=True)
        b0 = _scalar(q.get("intercept"), "intercept")
        variance = _scalar(q.get("variance"), "variance", 1e-12, 100)
        if (
            q.get("unknown_group") != "logistic_normal_marginal"
            or q.get("known_group") != "conditional_mode"
            or q.get("quadrature_points") != 40
            or not isinstance(q.get("groups"), list)
        ):
            _fail("INVALID_CALIBRATOR", "Invalid hierarchical policy")
        effects = {}
        for row in q["groups"]:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("group"), str)
                or row["group"] in effects
            ):
                _fail("INVALID_CALIBRATOR", "Invalid group effects")
            effects[row["group"]] = _scalar(row.get("mode"), "group mode")
            _scalar(
                row.get("conditional_variance"), "conditional_variance", 0, variance * (1 + 1e-10)
            )
        if (
            groups is None
            or len(groups) != len(p)
            or any(not isinstance(g, str) or not g for g in groups)
        ):
            _fail("INVALID_INPUT", "Group IDs required at hierarchical application")
        nodes, mass = np.polynomial.hermite.hermgauss(40)
        v = []
        unknown = 0
        for prob, g in zip(p, groups):
            eta = logit(prob) + b0
            if g in effects:
                value = expit(eta + effects[g])
            else:
                value = float(mass @ expit(eta + np.sqrt(2 * variance) * nodes) / np.sqrt(pi))
                unknown += 1
            v.append(value)
        v = np.array(v)
        trace["unknown_group_count"] = unknown
    return _result(
        v,
        "calibration.apply_calibrator",
        "loss_fraction" if kind == "lgd_level" else "probability",
        _base()._cal_context(context, a["target"]),
        diagnostics=trace,
        provenance={"parameter_origin": "ESTIMATED"},
    )


def _stamp_selection_training(model, training):
    from .core import _iso_date

    model = _json_data(model)
    model.pop("artifact_hash", None)
    if training is not None:
        if not isinstance(training, Mapping) or training.get("role", "training_only") not in (
            "training_only",
            "development_only",
        ):
            _fail("INVALID_TRAINING", "Training-only role required")
        model["training"].update(
            {
                "known_at": _iso_date(training.get("known_at"), "training.known_at"),
                "cutoff_status": "declared",
                "role": training.get("role", "training_only"),
            }
        )
    model["artifact_hash"] = _fingerprint(model)
    return model


def _freeze_protocol_classifier(model, name, metadata, split_plan):
    frozen = {
        "schema_version": "1.0",
        "model_id": name,
        "model_version": metadata["base_model_version"],
        "method_id": "calibration.fit_model_selection_protocol",
        "feature_order": model["feature_order"],
        "parameters": {"selection_model": model},
        "preprocessing": {"transformation": "stored_training_center_scale"},
        "target": metadata["target"],
        "training": {
            "data_fingerprint": model["training"]["data_fingerprint"],
            "split_id": "development:" + metadata["calibration"]["split_id"],
            "known_at": model["training"]["known_at"],
            "role": "development_only",
        },
        "diagnostics": {
            "selection_model_hash": model["artifact_hash"],
            "diagnostic_hash": model["diagnostic_hash"],
            "split_fingerprint": _fingerprint(split_plan),
        },
        "provenance": {"data_origin": metadata["data_origin"], "parameter_origin": "ESTIMATED"},
    }
    frozen["artifact_hash"] = _fingerprint(frozen)
    return frozen


def _predict_selection_pd(fitted_model, X, *, feature_order, expected_target, context=None):
    from .core import _model_artifact

    model = _model_artifact(fitted_model)
    raw = dict(model)
    digest = raw.pop("artifact_hash", None)
    if digest != _fingerprint(raw):
        _fail("INTEGRITY_ERROR", "Frozen selection classifier hash mismatch")
    if model["method_id"] != "calibration.fit_model_selection_protocol" or model[
        "preprocessing"
    ] != {"transformation": "stored_training_center_scale"}:
        _fail("INVALID_MODEL", "Invalid selection classifier schema")
    if list(feature_order) != model["feature_order"]:
        _fail("FEATURE_MISMATCH", "Feature order differs")
    if _base()._target(expected_target) != model["target"]:
        _fail("TARGET_MISMATCH", "Target differs")
    inner = model["parameters"].get("selection_model")
    if not isinstance(inner, dict):
        _fail("INVALID_MODEL", "Missing selection parameters")
    body = dict(inner)
    h = body.pop("artifact_hash", None)
    if h != _fingerprint(body):
        _fail("INTEGRITY_ERROR", "Inner selection hash mismatch")
    if (
        inner.get("artifact_type") != "SelectionModel"
        or inner.get("estimator") != "GLM"
        or inner.get("link") != "logit"
        or inner.get("feature_order") != model["feature_order"]
    ):
        _fail("INVALID_MODEL", "Invalid Bernoulli selection schema")
    size = len(model["feature_order"])
    coef = _numeric(inner.get("coefficients"), "coefficients", allow_empty=True, vector_only=True)
    center = _numeric(inner.get("center"), "center", allow_empty=True, vector_only=True)
    scale = _numeric(inner.get("scale"), "scale", allow_empty=True, vector_only=True)
    _scalar(inner.get("intercept"), "intercept")
    if any(len(z) != size for z in (coef, center, scale)) or np.any(scale <= 0):
        _fail("INVALID_MODEL", "Invalid stored coefficients or scaling")
    x = _numeric(X, "X", allow_empty=True)
    if x.ndim != 2 or x.shape[1] != size:
        _fail("FEATURE_MISMATCH", "Prediction matrix differs")
    values = _prediction(inner, x)
    if not np.isfinite(values).all():
        _fail("NUMERICAL_FAILURE", "Prediction overflow")
    return _result(
        values,
        "classification.predict_pd",
        "probability",
        _base()._cal_context(context, model["target"]),
        diagnostics={
            "model_id": model["model_id"],
            "model_version": model["model_version"],
            "prediction_family": "selection_protocol",
        },
        provenance=model["provenance"],
    )
