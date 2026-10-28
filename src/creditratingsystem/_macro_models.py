"""Native dynamic macro estimators and frozen latent-state/satellite mappings."""

import numpy as np
from .core import _fail, _numeric
from .macro import _history, _integer, _macro_model, _sqrt_cov


def _vecm_design(x, p, deterministic):
    n, k = x.shape
    delta = np.diff(x, axis=0)
    response = delta[p - 1 :]
    levels = x[p - 1 : -1]
    outside = deterministic in (
        "constant_outside",
        "constant_outside_trend_inside",
        "constant_trend_outside",
    )
    if deterministic == "constant_inside":
        levels = np.c_[levels, np.ones(n - p)]
    elif deterministic == "constant_outside_trend_inside":
        levels = np.c_[levels, np.arange(p, n)]
    parts = [delta[p - 1 - j : n - 1 - j] for j in range(1, p)]
    if outside:
        parts.append(np.ones((n - p, 1)))
    if deterministic == "constant_trend_outside":
        parts.append(np.arange(p + 1, n + 1)[:, None])
    short = np.column_stack(parts) if parts else np.empty((n - p, 0))
    return response, levels, short


def fit_vecm(
    level_history,
    *,
    rank,
    lags,
    deterministic="constant_inside",
    integration_order="I1",
    training,
    target,
    model_id,
    model_version,
    context=None,
):
    """Johansen reduced-rank Gaussian ML with explicit inside/outside deterministic terms."""
    from statsmodels.tsa.vector_ar.vecm import VECM

    x, dates, variables, h, window = _history(level_history, training)
    n, k = x.shape
    p = _integer(lags, "lags", 1)
    r = _integer(rank, "rank", 0)
    mapping = {
        "none": "n",
        "constant_inside": "ci",
        "constant_outside": "co",
        "constant_outside_trend_inside": "coli",
        "constant_trend_outside": "colo",
    }
    if k < 2 or r >= k or deterministic not in mapping or integration_order != "I1":
        _fail(
            "MACRO_VECM_SPEC",
            "Require at least two I(1) level series, rank 0..k-1 and explicit supported deterministic terms",
        )
    if any(
        s["transform"] in ("difference", "log_difference") for s in h.get("inverse_state", []) or []
    ):
        _fail("MACRO_VECM_SPEC", "VECM requires levels, not already differenced history")
    y, z, d = _vecm_design(x, p, deterministic)
    if len(y) <= d.shape[1] + z.shape[1] + k or (
        d.shape[1] and np.linalg.matrix_rank(d) < d.shape[1]
    ):
        _fail(
            "INSUFFICIENT_OBSERVATIONS",
            "Full-rank short-run design and enough observations required",
        )
    r0 = y - d @ np.linalg.lstsq(d, y, rcond=None)[0] if d.shape[1] else y
    r1 = z - d @ np.linalg.lstsq(d, z, rcond=None)[0] if d.shape[1] else z
    s00 = r0.T @ r0 / len(y)
    s11 = r1.T @ r1 / len(y)
    s01 = r0.T @ r1 / len(y)
    try:
        chol = np.linalg.cholesky(s11)
        middle = s01.T @ np.linalg.solve(s00, s01)
        whitened = np.linalg.solve(chol, middle) @ np.linalg.inv(chol.T)
        eigen = np.linalg.eigvalsh((whitened + whitened.T) / 2)[::-1][:k]
        fitted = VECM(x, k_ar_diff=p - 1, coint_rank=r, deterministic=mapping[deterministic]).fit()
    except np.linalg.LinAlgError as exc:
        _fail(
            "RANK_DEFICIENT",
            "Johansen covariance or normalization is singular",
            {"reason": str(exc)},
        )
    beta = (
        np.vstack((fitted.beta, fitted.det_coef_coint))
        if z.shape[1] > k
        else np.asarray(fitted.beta)
    )
    if r:
        try:
            beta = beta @ np.linalg.inv(beta[:r])
        except np.linalg.LinAlgError:
            _fail(
                "IDENTIFICATION_FAILURE",
                "First rank variables cannot identify the cointegration basis",
            )
    design = np.column_stack((z @ beta, d))
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    alpha = coefficient[:r].T
    gamma = coefficient[r : r + k * (p - 1)].T
    outside = coefficient[r + k * (p - 1) :].T
    residual = y - design @ coefficient
    covariance = residual.T @ residual / len(y)
    _, factorization = _sqrt_cov(covariance)
    pi = alpha @ beta[:k].T
    coefs = np.zeros((p, k, k))
    coefs[0] = np.eye(k) + pi
    for j in range(p - 1):
        coefs[j] += gamma[:, j * k : (j + 1) * k]
        coefs[j + 1] -= gamma[:, j * k : (j + 1) * k]
    intercept = np.zeros(k)
    trend = np.zeros(k)
    if deterministic in (
        "constant_outside",
        "constant_outside_trend_inside",
        "constant_trend_outside",
    ):
        intercept = outside[:, 0].copy()
    if deterministic == "constant_trend_outside":
        trend = outside[:, 1].copy()
    if deterministic == "constant_inside" and r:
        intercept += alpha @ beta[k]
    if deterministic == "constant_outside_trend_inside" and r:
        trend += alpha @ beta[k]
        intercept -= alpha @ beta[k]
    companion = np.zeros((k * p, k * p))
    companion[:k] = np.concatenate(coefs, axis=1)
    if p > 1:
        companion[k:, :-k] = np.eye(k * (p - 1))
    roots = sorted(np.linalg.eigvals(companion), key=lambda v: (v.real, v.imag))
    radius = float(np.max(np.abs(roots)))
    if np.any(eigen > 1 + 1e-8) or np.any(eigen < -1e-8):
        _fail("NUMERICAL_FAILURE", "Johansen eigenvalues outside covariance-projection domain")
    traces = [float(-len(y) * np.log1p(-np.clip(eigen[j:], 0, 1 - 1e-15)).sum()) for j in range(k)]
    params = dict(
        dynamics="vecm",
        lags=p,
        rank=r,
        integration_order="I1",
        deterministic=deterministic,
        alpha=alpha.tolist(),
        beta=beta[:k].tolist(),
        cointegration_deterministic=beta[k:].tolist(),
        short_run=gamma.tolist(),
        outside_deterministic=outside.tolist(),
        lag_coefficients=coefs.tolist(),
        intercept=intercept.tolist(),
        trend=trend.tolist(),
        covariance=covariance.tolist(),
        covariance_estimator="mle",
        initial_history=x[-p:].tolist(),
        next_time_index=n + 1,
        _training_values=x.tolist(),
    )
    diag = dict(
        backend="statsmodels.VECM Johansen ML",
        observations=len(y),
        residuals=residual.tolist(),
        residual_dates=dates[p:],
        rank_assumed=r,
        integration_order_assumed="I1",
        normalization="FIRST_R_VARIABLE_ROWS_IDENTITY",
        johansen_eigenvalues=eigen.tolist(),
        rank_trace_statistics=[dict(null_rank=j, statistic=traces[j]) for j in range(k)],
        critical_values_not_inferred=True,
        cointegration_residuals=(z @ beta).tolist(),
        companion_eigenvalues=[[float(v.real), float(v.imag)] for v in roots],
        spectral_radius=radius,
        expected_unit_roots=k - r,
        explosive=radius > 1 + 1e-8,
        eigenvalue_repair="NONE",
        factorization=factorization,
        training_only=True,
    )
    return _macro_model(
        "fit_vecm", params, h, dates, training, target, model_id, model_version, diag, context
    )


def _hmm_expectation(y, design, coefficients, covariances, transition, initial):
    from scipy.special import logsumexp

    n, k = y.shape
    r = len(initial)
    emission = np.empty((n, r))
    for j in range(r):
        cov = covariances[j]
        sign, logdet = np.linalg.slogdet(cov)
        if (
            sign <= 0
            or np.min(np.linalg.eigvalsh(cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))))
            <= 1e-12
        ):
            _fail(
                "REGIME_DEGENERACY",
                "Gaussian regime covariance must be positive definite without inferred ridge",
            )
        residual = y - design @ coefficients[j]
        emission[:, j] = -0.5 * (
            k * np.log(2 * np.pi)
            + logdet
            + np.sum(residual * np.linalg.solve(cov, residual.T).T, axis=1)
        )
    with np.errstate(divide="ignore"):
        logp = np.log(transition)
        logi = np.log(initial)
    alpha = np.empty_like(emission)
    normal = np.empty(n)
    alpha[0] = logi + emission[0]
    normal[0] = logsumexp(alpha[0])
    alpha[0] -= normal[0]
    for t in range(1, n):
        alpha[t] = emission[t] + logsumexp(alpha[t - 1, :, None] + logp, axis=0)
        normal[t] = logsumexp(alpha[t])
        alpha[t] -= normal[t]
    beta = np.zeros_like(emission)
    xi = np.zeros((r, r))
    for t in range(n - 2, -1, -1):
        beta[t] = (
            logsumexp(logp + emission[t + 1, None, :] + beta[t + 1, None, :], axis=1)
            - normal[t + 1]
        )
        pair = alpha[t, :, None] + logp + emission[t + 1, None, :] + beta[t + 1, None, :]
        xi += np.exp(pair - logsumexp(pair))
    gamma = alpha + beta
    gamma = np.exp(gamma - logsumexp(gamma, axis=1)[:, None])
    return float(normal.sum()), gamma, xi, np.exp(alpha)


def fit_regime_switching(
    transformed_history,
    *,
    regime_count,
    lags=0,
    initialization,
    max_iterations=200,
    tolerance=1e-8,
    training,
    target,
    model_id,
    model_version,
    context=None,
):
    """Estimate a joint Gaussian autoregressive HMM by explicit-start EM.

    Regime-specific intercepts, lag coefficients and full innovation covariance
    govern observed lagged levels, not deviations from past latent regime means.
    Transition rows are origin regimes. Every initialization is evaluated.
    """
    from .core import CreditRiskError

    x, dates, variables, h, _ = _history(transformed_history, training)
    n, k = x.shape
    r = _integer(regime_count, "regime_count", 1)
    p = _integer(lags, "lags", 0)
    maxit = _integer(max_iterations, "max_iterations", 1)
    if not np.isscalar(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        _fail("INVALID_TOLERANCE", "Positive finite convergence tolerance required")
    if n - p <= 1 + k * p or not isinstance(initialization, list) or not initialization:
        _fail(
            "REGIME_SPEC",
            "Sufficient observations and explicit nonempty initialization list required",
        )
    design = np.column_stack([np.ones(n - p)] + [x[p - j : n - j] for j in range(1, p + 1)])
    y = x[p:]
    reports = []
    best = None
    for start_id, start in enumerate(initialization):
        try:
            transition = _numeric(start.get("transition"), "transition")
            initial = _numeric(
                start.get("initial_probabilities"), "initial_probabilities", vector_only=True
            )
            coef = _numeric(start.get("coefficients"), "coefficients")
            cov = _numeric(start.get("covariances"), "covariances")
            if (
                transition.shape != (r, r)
                or initial.shape != (r,)
                or coef.shape != (r, 1 + k * p, k)
                or cov.shape != (r, k, k)
            ):
                _fail(
                    "INCOMPATIBLE_SHAPE",
                    "Regime start parameter dimensions differ from specification",
                )
            if (
                np.any(transition < 0)
                or np.any(initial < 0)
                or not np.allclose(transition.sum(axis=1), 1, rtol=0, atol=1e-12)
                or abs(initial.sum() - 1) > 1e-12
            ):
                _fail(
                    "INVALID_TRANSITION", "Initial and transition probabilities must be normalized"
                )
            for c in cov:
                _sqrt_cov(c)
            trace = []
            converged = False
            for iteration in range(maxit + 1):
                ll, gamma, xi, filtered = _hmm_expectation(
                    y, design, coef, cov, transition, initial
                )
                trace.append(ll)
                if len(trace) > 1:
                    change = trace[-1] - trace[-2]
                    if change < -1e-7 * (1 + abs(trace[-2])):
                        _fail(
                            "NUMERICAL_FAILURE", "EM likelihood decreased beyond rounding tolerance"
                        )
                    if abs(change) <= tolerance * (1 + abs(trace[-2])):
                        converged = True
                        break
                if iteration == maxit:
                    break
                for j in range(r):
                    weights = gamma[:, j]
                    weighted = design * np.sqrt(weights)[:, None]
                    if (
                        weights.sum() <= design.shape[1]
                        or np.linalg.matrix_rank(weighted) < design.shape[1]
                    ):
                        _fail(
                            "REGIME_DEGENERACY",
                            "Regime occupancy or weighted design is insufficient",
                        )
                    coef[j] = np.linalg.lstsq(weighted, y * np.sqrt(weights)[:, None], rcond=None)[
                        0
                    ]
                    resid = y - design @ coef[j]
                    cov[j] = (resid * weights[:, None]).T @ resid / weights.sum()
                transition = xi / xi.sum(axis=1)[:, None]
                initial = gamma[0]
            reports.append(
                dict(
                    start_index=start_id,
                    status="CONVERGED" if converged else "MAX_ITERATIONS",
                    log_likelihood=ll,
                    iterations=iteration,
                    likelihood_trace=trace,
                )
            )
            if best is None or ll > best[0]:
                best = (
                    ll,
                    coef.copy(),
                    cov.copy(),
                    transition.copy(),
                    initial.copy(),
                    gamma.copy(),
                    filtered.copy(),
                    start_id,
                    converged,
                )
        except CreditRiskError as exc:
            reports.append(dict(start_index=start_id, status="FAILED", error_code=exc.code))
    if best is None:
        _fail(
            "REGIME_ESTIMATION_FAILED",
            "All explicit HMM starts failed; check occupancy, covariance and design",
        )
    ll, coef, cov, transition, initial, gamma, filtered, start_id, converged = best
    order = np.array(sorted(range(r), key=lambda j: tuple(coef[j, 0].tolist()) + (j,)))
    coef = coef[order]
    cov = cov[order]
    transition = transition[np.ix_(order, order)]
    initial = initial[order]
    gamma = gamma[:, order]
    filtered = filtered[:, order]
    params = dict(
        dynamics="regime",
        regime_count=r,
        lags=p,
        coefficients=coef.tolist(),
        covariances=cov.tolist(),
        transition=transition.tolist(),
        initial_probabilities=initial.tolist(),
        filtered_probabilities_end=filtered[-1].tolist(),
        initial_history=x[-p:].tolist() if p else [],
        _training_values=x.tolist(),
    )
    diag = dict(
        backend="native Gaussian autoregressive HMM EM",
        log_likelihood=ll,
        converged=converged,
        selected_start=start_id,
        starts=reports,
        filtered_probabilities=filtered.tolist(),
        smoothed_training_probabilities=gamma.tolist(),
        probability_dates=dates[p:],
        ordering_rule="LEXICOGRAPHIC_INTERCEPT_ASCENDING",
        original_regime_order=order.tolist(),
        transition_orientation="ROW_ORIGIN_COLUMN_DESTINATION",
        training_only=True,
        global_optimum_claimed=False,
        smoothing_uses_entire_training_window=True,
        emission_specification="OBSERVED_LAGS_REGIME_SPECIFIC_COEFFICIENTS",
    )
    return _macro_model(
        "fit_regime_switching",
        params,
        h,
        dates,
        training,
        target,
        model_id,
        model_version,
        diag,
        context,
    )


def _simulate_latent_macro(
    model,
    innovations,
    *,
    initial_state=None,
    regime_uniforms=None,
    output_space="model",
    sampling_contract=None,
    context=None,
):
    from .macro import _scenario_inputs, _measure_contract, _cube, _inverse_paths
    from .core import _fingerprint
    from copy import deepcopy

    p = model["parameters"]
    ctx = _measure_contract(model, context)
    if p["dynamics"] == "dynamic_factor":
        return _simulate_factor(
            model,
            innovations,
            initial_state,
            regime_uniforms,
            output_space,
            sampling_contract,
            context,
        )
    z, ids, times, w, kind = _scenario_inputs(
        innovations, model["feature_order"], model["preprocessing"]["frequency"]
    )
    s, h, k = z.shape
    if kind != "STANDARD_NORMAL":
        _fail(
            "MACRO_INNOVATIONS",
            "Regime-dependent covariances require named standard-normal shared draws",
        )
    u = _numeric(regime_uniforms, "regime_uniforms")
    r = _integer(p["regime_count"], "regime_count", 1)
    lags = _integer(p["lags"], "lags", 0)
    if u.shape != (s, h + 1) or np.any((u < 0) | (u >= 1)):
        _fail(
            "MACRO_INNOVATIONS",
            "Regime uniforms require scenario by (initial plus forecast steps), each in [0,1)",
        )
    transition = _numeric(p["transition"], "transition")
    initial = _numeric(
        p["filtered_probabilities_end"], "filtered_probabilities_end", vector_only=True
    )
    if (
        transition.shape != (r, r)
        or np.any(transition < 0)
        or not np.allclose(transition.sum(axis=1), 1)
        or initial.shape != (r,)
        or np.any(initial < 0)
        or not np.isclose(initial.sum(), 1)
    ):
        _fail("INVALID_MODEL", "Invalid frozen regime probabilities")
    coef = _numeric(p["coefficients"], "coefficients")
    cov = _numeric(p["covariances"], "covariances")
    if coef.shape != (r, 1 + k * lags, k) or cov.shape != (r, k, k):
        _fail("INVALID_MODEL", "Invalid frozen regime coefficient dimensions")
    roots = np.array([_sqrt_cov(c)[0] for c in cov])
    sampling = (
        deepcopy(sampling_contract)
        if sampling_contract is not None
        else dict(law="FORECAST_CONDITIONAL")
    )
    if sampling.get("law") == "FORECAST_CONDITIONAL":
        if ctx.get("conditioning") == "TTC":
            _fail("TTC_INITIALIZATION", "Filtered regime forecasts do not constitute TTC")
        sampling.update(conditioning_scope="PIT", initialization="FILTERED_REGIME_POSTERIOR")
    elif sampling.get("law") == "STATIONARY_MARKOV_EMISSION":
        if lags or initial_state is not None:
            _fail(
                "TTC_INITIALIZATION",
                "Markov stationary emission law is only valid for zero observed lags",
            )
        if not all(
            isinstance(sampling.get(v), str) and sampling[v].strip()
            for v in ("cycle_law_id", "equilibrium_assumptions")
        ):
            _fail("TTC_INITIALIZATION", "Cycle-law and assumptions required")
        if np.linalg.matrix_rank(np.eye(r) - transition.T) != r - 1:
            _fail("NONSTATIONARY_TTC", "Unique stationary regime law required")
        initial = np.linalg.lstsq(
            np.vstack([np.eye(r) - transition.T, np.ones((1, r))]),
            np.r_[np.zeros(r), 1],
            rcond=None,
        )[0]
        if np.any(initial < -1e-10):
            _fail("NONSTATIONARY_TTC", "Stationary regime weights invalid")
        initial = np.maximum(initial, 0)
        initial /= initial.sum()
        sampling.update(
            conditioning_scope="TTC_STATIONARY", stationary_regime_probabilities=initial.tolist()
        )
    else:
        _fail(
            "TTC_INITIALIZATION",
            "Regime dynamics require fixed posterior or stationary zero-lag Markov-emission law",
        )
    raw_state = _numeric(
        p["initial_history"] if initial_state is None else initial_state,
        "initial_state",
        allow_empty=True,
    )
    if raw_state.size != lags * k or (lags and raw_state.shape != (lags, k)):
        _fail("INCOMPATIBLE_SHAPE", "Regime initial history dimensions differ")
    state = raw_state.reshape(lags, k)
    states = np.broadcast_to(state, (s, lags, k)).copy()
    paths = np.zeros((s, h, k))
    regimes = np.zeros((s, h), dtype=int)
    previous = np.array([np.searchsorted(np.cumsum(initial), v, side="right") for v in u[:, 0]])
    for t in range(h):
        for i in range(s):
            j = min(
                r - 1,
                np.searchsorted(np.cumsum(transition[previous[i]]), u[i, t + 1], side="right"),
            )
            regimes[i, t] = j
            design = np.r_[1, states[i, ::-1, :].reshape(-1)]
            value = design @ coef[j] + roots[j] @ z[i, t]
            paths[i, t] = value
            previous[i] = j
            if lags:
                states[i] = np.concatenate((states[i, 1:], value[None, :]))
    if output_space == "level":
        if initial_state is not None or sampling["law"] != "FORECAST_CONDITIONAL":
            _fail(
                "MACRO_TRANSFORM",
                "Alternative initial law requires explicit original-level anchors",
            )
        paths = _inverse_paths(paths, model["preprocessing"]["inverse_state"])
    elif output_space != "model":
        _fail("MACRO_TRANSFORM", "output_space must be model or level")
    return _cube(
        model,
        paths,
        ids,
        times,
        w,
        dict(
            output_space=output_space,
            sampling_contract=sampling,
            regime_paths=regimes.tolist(),
            regime_uniform_hash=_fingerprint(u.tolist()),
            innovation_hash=_fingerprint(innovations),
            shared_macro_across_entities=True,
        ),
        context,
        "simulate_macro_scenarios",
    )


def _factor_layout(k, q):
    return [(i, j) for i in range(k) for j in range(q) if i >= q or j < i]


def _factor_unpack(theta, k, q):
    positions = _factor_layout(k, q)
    load = np.zeros((k, q))
    load[:q] = np.eye(q)
    for a, (i, j) in enumerate(positions):
        load[i, j] = theta[a]
    offset = len(positions)
    mean = theta[offset : offset + k]
    ar = theta[offset + k : offset + k + q + k]
    variance = np.exp(theta[offset + k + q + k :])
    return load, mean, ar, variance


def _factor_filter(x, load, mean, ar, variance):
    from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

    k, q = load.shape
    m = q + k
    model = KalmanFilter(k_endog=k, k_states=m, k_posdef=m)
    model.bind(np.asfortranarray(x.T))
    model.design = np.c_[load, np.eye(k)]
    model.obs_intercept = mean
    model.transition = np.diag(ar)
    model.selection = np.eye(m)
    model.state_cov = np.diag(variance)
    model.obs_cov = np.zeros((k, k))
    model.initialize_known(np.zeros(m), np.diag(variance / (1 - ar**2)))
    return model.filter()


def fit_dynamic_factor(
    transformed_history,
    *,
    factor_count,
    initialization,
    ar_limit=0.995,
    variance_bounds=(1e-8, 1e4),
    max_iterations=300,
    tolerance=1e-8,
    training,
    target,
    model_id,
    model_version,
    context=None,
):
    """Fit identified Gaussian factors and serial idiosyncratic states by Kalman likelihood.

    First q loading rows have unit diagonal and zero upper triangle. Factors
    and idiosyncratic components each follow an independent stationary AR(1).
    Missing observed entries are integrated out, never interpolated.
    """
    from scipy.optimize import minimize

    x, dates, variables, h, _ = _history(transformed_history, training, missing=True)
    n, k = x.shape
    q = _integer(factor_count, "factor_count", 1)
    m = q + k
    if q >= k or any(np.sum(np.isfinite(x), axis=0) <= 2 * m):
        _fail(
            "FACTOR_IDENTIFICATION",
            "More observed variables than factors and sufficient released observations required",
        )
    limits = _numeric(variance_bounds, "variance_bounds", vector_only=True)
    if (
        limits.shape != (2,)
        or limits[0] <= 0
        or limits[1] <= limits[0]
        or not np.isfinite(ar_limit)
        or not 0 < ar_limit < 1
    ):
        _fail("FACTOR_SPEC", "Explicit positive variance bounds and stationary AR bound required")
    if not isinstance(initialization, dict):
        _fail("FACTOR_SPEC", "Explicit identified initial parameters required")
    load = _numeric(initialization.get("loadings"), "loadings")
    mean = _numeric(initialization.get("mean"), "mean", vector_only=True)
    ar = _numeric(initialization.get("ar"), "ar", vector_only=True)
    variance = _numeric(initialization.get("variances"), "variances", vector_only=True)
    if (
        load.shape != (k, q)
        or mean.shape != (k,)
        or ar.shape != (m,)
        or variance.shape != (m,)
        or np.any(np.abs(ar) > ar_limit)
        or np.any((variance < limits[0]) | (variance > limits[1]))
    ):
        _fail("FACTOR_SPEC", "Initial parameter dimensions or estimation bounds invalid")
    if not np.array_equal(np.diag(load[:q]), np.ones(q)) or np.any(np.triu(load[:q], 1) != 0):
        _fail(
            "FACTOR_IDENTIFICATION",
            "Initial loadings must obey unit-diagonal lower-triangular anchors",
        )
    positions = _factor_layout(k, q)
    theta = np.r_[[load[i, j] for i, j in positions], mean, ar, np.log(variance)]
    free = len(positions) + k
    bounds = (
        [(None, None)] * free
        + [(-ar_limit, ar_limit)] * m
        + [(np.log(limits[0]), np.log(limits[1]))] * m
    )

    def objective(v):
        loading, b, a, qv = _factor_unpack(v, k, q)
        return -float(_factor_filter(x, loading, b, a, qv).llf)

    initial_objective = objective(theta)
    maxit = _integer(max_iterations, "max_iterations", 1)
    if not np.isfinite(tolerance) or tolerance <= 0:
        _fail("INVALID_TOLERANCE", "Positive finite optimizer tolerance required")
    fit = minimize(
        objective,
        theta,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=maxit, ftol=tolerance, gtol=tolerance),
    )
    load, mean, ar, variance = _factor_unpack(fit.x, k, q)
    filtered = _factor_filter(x, load, mean, ar, variance)
    design = np.c_[load, np.eye(k)]
    if not np.isfinite(filtered.llf) or filtered.llf < -initial_objective - 1e-7:
        _fail("NUMERICAL_FAILURE", "State-space optimizer produced invalid or inferior likelihood")
    names = [f"factor_{j + 1}" for j in range(q)] + ["idiosyncratic_" + v for v in variables]
    params = dict(
        dynamics="dynamic_factor",
        factor_count=q,
        loadings=load.tolist(),
        mean=mean.tolist(),
        ar=ar.tolist(),
        variances=variance.tolist(),
        state_transition=np.diag(ar).tolist(),
        state_covariance=np.diag(variance).tolist(),
        observation_matrix=design.tolist(),
        state_ids=names,
        filtered_state_end=filtered.filtered_state[:, -1].tolist(),
        filtered_covariance_end=filtered.filtered_state_cov[:, :, -1].tolist(),
        stationary_covariance=np.diag(variance / (1 - ar**2)).tolist(),
        _training_values=[[None if not np.isfinite(v) else float(v) for v in row] for row in x],
    )
    diag = dict(
        backend="statsmodels KalmanFilter likelihood / scipy L-BFGS-B",
        converged=bool(fit.success),
        optimizer_status=int(fit.status),
        optimizer_message=str(fit.message),
        iterations=int(fit.nit),
        log_likelihood=float(filtered.llf),
        initial_log_likelihood=-initial_objective,
        identification="FIRST_FACTOR_ROWS_UNIT_DIAGONAL_LOWER_TRIANGULAR",
        factor_lag_order=1,
        idiosyncratic_lag_order=1,
        ar_limit=float(ar_limit),
        variance_bounds=limits.tolist(),
        boundary_parameters=[
            int(i)
            for i, (v, (lo, hi)) in enumerate(zip(fit.x, bounds))
            if (lo is not None and abs(v - lo) < 1e-6) or (hi is not None and abs(v - hi) < 1e-6)
        ],
        missing_entries=int(np.isnan(x).sum()),
        missing_policy="OBSERVED_SUBVECTOR_KALMAN_LIKELIHOOD",
        filtered_states=filtered.filtered_state.T.tolist(),
        state_dates=dates,
        training_only=True,
        global_optimum_claimed=False,
    )
    return _macro_model(
        "fit_dynamic_factor",
        params,
        h,
        dates,
        training,
        target,
        model_id,
        model_version,
        diag,
        context,
    )


def _factor_state(model, initial_state, sampling_contract, scenario_count, context):
    from .macro import _measure_contract
    from .core import _fingerprint
    from copy import deepcopy

    p = model["parameters"]
    ctx = _measure_contract(model, context)
    m = len(p["state_ids"])
    sampling = (
        deepcopy(sampling_contract)
        if sampling_contract is not None
        else dict(law="FILTERED_GAUSSIAN")
    )
    law = sampling.get("law")
    if law == "FILTERED_GAUSSIAN":
        if ctx.get("conditioning") == "TTC":
            _fail("TTC_INITIALIZATION", "Filtered latent posterior is PIT, not a cycle law")
        mean = np.asarray(p["filtered_state_end"])
        cov = np.asarray(p["filtered_covariance_end"])
        sampling.update(conditioning_scope="PIT", initialization="FILTERED_STATE_DISTRIBUTION")
    elif law == "STATIONARY_GAUSSIAN":
        if not all(
            isinstance(sampling.get(v), str) and sampling[v].strip()
            for v in ("cycle_law_id", "equilibrium_assumptions")
        ):
            _fail("TTC_INITIALIZATION", "Cycle law and equilibrium assumptions required")
        transition = _numeric(p["state_transition"], "state_transition")
        radius = max(abs(np.linalg.eigvals(transition)))
        if radius >= 1 - 1e-10:
            _fail(
                "NONSTATIONARY_TTC",
                "Stationary latent Gaussian law requires stable state transition",
            )
        from scipy.linalg import solve_discrete_lyapunov

        mean = np.zeros(m)
        cov = solve_discrete_lyapunov(transition, np.asarray(p["state_covariance"]))
        sampling.update(conditioning_scope="TTC_STATIONARY", spectral_radius=float(radius))
    elif law == "DECLARED_FIXED_STATE":
        if initial_state is None:
            _fail("MACRO_INITIAL_STATE", "Fixed-state law requires an explicit state vector")
        mean = _numeric(initial_state, "initial_state", vector_only=True)
        cov = np.zeros((m, m))
        sampling.update(conditioning_scope="PIT", initialization="DECLARED_FIXED_STATE")
        if ctx.get("conditioning") == "TTC":
            _fail("TTC_INITIALIZATION", "A fixed latent state is not TTC")
    else:
        _fail(
            "TTC_INITIALIZATION",
            "Factor law requires filtered Gaussian, stationary Gaussian or declared fixed latent state",
        )
    if mean.shape != (m,) or cov.shape != (m, m):
        _fail("INVALID_MODEL", "Latent initial distribution dimensions invalid")
    if law != "DECLARED_FIXED_STATE" and initial_state is not None:
        _fail(
            "MACRO_INITIAL_STATE",
            "Use a declared fixed-state law to override a latent distribution",
        )
    if law == "DECLARED_FIXED_STATE":
        states = np.repeat(mean[None, :], scenario_count, axis=0)
    else:
        draws = _numeric(sampling.pop("initial_state_draws", None), "initial_state_draws")
        if draws.shape != (scenario_count, m):
            _fail(
                "INCOMPATIBLE_SHAPE", "Initial latent normal draws require scenario by state axes"
            )
        root, fd = _sqrt_cov(cov)
        states = mean + draws @ root.T
        sampling.update(initial_draw_hash=_fingerprint(draws.tolist()), initial_factorization=fd)
    sampling.update(
        initial_mean=mean.tolist(), initial_covariance=cov.tolist(), state_ids=p["state_ids"]
    )
    return states, mean, cov, sampling


def _simulate_factor(
    model, innovations, initial_state, regime_uniforms, output_space, sampling_contract, context
):
    from .macro import _scenario_inputs, _cube, _inverse_paths
    from .core import _fingerprint

    p = model["parameters"]
    if regime_uniforms is not None:
        _fail("MACRO_INNOVATIONS", "Linear factors do not consume regime uniforms")
    z, ids, times, w, kind = _scenario_inputs(
        innovations, p["state_ids"], model["preprocessing"]["frequency"]
    )
    s, h, m = z.shape
    if kind != "STANDARD_NORMAL":
        _fail(
            "MACRO_INNOVATIONS",
            "Latent Gaussian factor model requires standard-normal state innovations",
        )
    states, _, _, sampling = _factor_state(model, initial_state, sampling_contract, s, context)
    root, fd = _sqrt_cov(p["state_covariance"])
    transition = np.asarray(p["state_transition"])
    design = np.asarray(p["observation_matrix"])
    mean = np.asarray(p["mean"])
    paths = np.zeros((s, h, len(mean)))
    if transition.shape != (m, m) or design.shape != (len(mean), m):
        _fail("INVALID_MODEL", "Latent dynamics dimensions invalid")
    for t in range(h):
        states = states @ transition.T + z[:, t] @ root.T
        paths[:, t] = mean + states @ design.T
    if output_space == "level":
        if sampling["law"] != "FILTERED_GAUSSIAN":
            _fail("MACRO_TRANSFORM", "Alternative initial law requires original-level anchors")
        paths = _inverse_paths(paths, model["preprocessing"]["inverse_state"])
    elif output_space != "model":
        _fail("MACRO_TRANSFORM", "output_space must be model or level")
    return _cube(
        model,
        paths,
        ids,
        times,
        w,
        dict(
            output_space=output_space,
            sampling_contract=sampling,
            factorization=fd,
            innovation_hash=_fingerprint(innovations),
            shared_macro_across_entities=True,
        ),
        context,
        "simulate_macro_scenarios",
    )


def _factor_gaussian_moments(model, horizon, initial_state=None, sampling_contract=None):
    p = model["parameters"]
    m = len(p["state_ids"])
    contract = dict(sampling_contract or {"law": "FILTERED_GAUSSIAN"})
    if contract.get("law") != "DECLARED_FIXED_STATE":
        contract["initial_state_draws"] = [[0] * m]
    _, initial, cov, _ = _factor_state(model, initial_state, contract, 1, None)
    transition = np.asarray(p["state_transition"])
    q = np.asarray(p["state_covariance"])
    design = np.asarray(p["observation_matrix"])
    offset = np.asarray(p["mean"])
    k = len(offset)
    means = []
    state_cov = []
    for _ in range(horizon):
        initial = transition @ initial
        cov = transition @ cov @ transition.T + q
        means.append(offset + design @ initial)
        state_cov.append(cov.copy())
    covariance = np.zeros((horizon * k, horizon * k))
    for t in range(horizon):
        power = np.eye(m)
        for u in range(t, horizon):
            block = design @ state_cov[t] @ power.T @ design.T
            covariance[t * k : (t + 1) * k, u * k : (u + 1) * k] = block
            covariance[u * k : (u + 1) * k, t * k : (t + 1) * k] = block.T
            power = transition @ power
    return np.array(means).reshape(-1), covariance
