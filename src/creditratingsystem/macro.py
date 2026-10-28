"""Release-aware macro dynamics and shared, explicitly conditioned scenario paths."""

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .core import (
    CalcResult,
    _numeric,
    _fail,
    _result,
    _fit_result,
    _model_artifact,
    _fingerprint,
    _iso_date,
    _context_value,
)
from .calibration import _target
from ._loss_paths import _ids, _weights


def _unwrap(value):
    return value.value if isinstance(value, CalcResult) else value


def _integer(value, name, minimum=0):
    x = _numeric(value, name)
    if x.ndim or x != np.floor(x) or x < minimum:
        _fail("MACRO_INPUT", name + " must be an integer at least " + str(minimum))
    return int(x)


def _dates(values, frequency):
    if frequency not in ("monthly", "quarterly", "annual"):
        _fail("MACRO_FREQUENCY", "Declare monthly, quarterly or annual observations")
    ds = [_iso_date(v, "date") for v in values]
    if not ds or len(set(ds)) != len(ds) or ds != sorted(ds):
        _fail("INVALID_TIME_GRID", "Dates must be unique and increasing")
    step = {"monthly": 1, "quarterly": 3, "annual": 12}[frequency]
    months = [date.fromisoformat(v).year * 12 + date.fromisoformat(v).month for v in ds]
    if any(b - a != step for a, b in zip(months, months[1:])):
        _fail(
            "MACRO_FREQUENCY",
            "History must have a complete regular calendar grid; no silent interpolation",
        )
    return ds


def _window(window):
    if not isinstance(window, Mapping):
        _fail("MACRO_WINDOW", "Explicit start/end/known_at window required")
    w = {k: _iso_date(window.get(k), k) for k in ("start", "end", "known_at")}
    if not w["start"] <= w["end"] <= w["known_at"]:
        _fail("MACRO_WINDOW", "Require start <= end <= known_at")
    return w


def transform_macro_history(
    history, variable_transforms, *, frequency, fit_window, information_cutoff, context=None
):
    """Select released vintages and fit transformations only inside the declared window."""
    cutoff = _iso_date(information_cutoff, "information_cutoff")
    window = _window(fit_window)
    if window["known_at"] > cutoff:
        _fail("LEAKAGE", "Transformation fit cutoff exceeds information cutoff")
    if not isinstance(variable_transforms, (list, tuple)) or not variable_transforms:
        _fail("MACRO_TRANSFORM", "Explicit per-variable transformation metadata required")
    variables = _ids([v.get("variable_id") for v in variable_transforms], "variable_ids")
    specs = deepcopy(list(variable_transforms))
    for spec in specs:
        if (
            spec.get("transform") not in ("identity", "log", "difference", "log_difference")
            or not isinstance(spec.get("standardize"), bool)
            or not isinstance(spec.get("unit"), str)
            or not spec["unit"]
        ):
            _fail("MACRO_TRANSFORM", "Declare transform, boolean standardize and original unit")
    if not isinstance(history, (list, tuple)) or not history:
        _fail("MACRO_HISTORY", "Release-dated records required")
    selected = {}
    seen = set()
    excluded = 0
    for row in history:
        if not isinstance(row, Mapping) or row.get("variable_id") not in variables:
            _fail("MACRO_HISTORY", "Every history record needs a declared variable_id")
        obs = _iso_date(row.get("date"), "date")
        released = _iso_date(row.get("release_date"), "release_date")
        key = (row["variable_id"], obs, released)
        value = _numeric(row.get("value"), "value")
        if value.ndim or released < obs:
            _fail(
                "MACRO_HISTORY",
                "Scalar observed value and release no earlier than observation required",
            )
        if key in seen:
            _fail("DUPLICATE_ID", "Duplicate variable/date/release vintage")
        seen.add(key)
        if released > cutoff:
            excluded += 1
            continue
        cell = key[:2]
        if cell not in selected or selected[cell]["release_date"] < released:
            selected[cell] = dict(row, value=float(value))
    dates = _dates(sorted(set(k[1] for k in selected)), frequency)
    if any((v, t) not in selected for t in dates for v in variables):
        _fail(
            "MACRO_MISSING",
            "Transformation history requires aligned complete variables; no imputation",
        )
    raw = np.array([[selected[(v, t)]["value"] for v in variables] for t in dates])
    releases = [[selected[(v, t)]["release_date"] for v in variables] for t in dates]
    differenced = any(v["transform"] in ("difference", "log_difference") for v in specs)
    drop = 1 if differenced else 0
    output_dates = dates[drop:]
    x = np.empty((len(output_dates), len(variables)))
    state = []
    # Transform fitting uses only vintages already known at the fit cutoff, even
    # when later observations/revisions are supplied for prospective application.
    fit_rows = [i for i, t in enumerate(output_dates) if window["start"] <= t <= window["end"]]
    if len(fit_rows) < 2:
        _fail("INSUFFICIENT_OBSERVATIONS", "At least two transformation fit observations required")
    fit_values = []
    for t in dates:
        row = []
        for v in variables:
            eligible = [
                r
                for r in history
                if r["variable_id"] == v
                and r["date"] == t
                and r["release_date"] <= window["known_at"]
            ]
            row.append(
                None
                if not eligible
                else float(max(eligible, key=lambda r: r["release_date"])["value"])
            )
        fit_values.append(row)
    for j, spec in enumerate(specs):
        kind = spec["transform"]
        level = raw[:, j]
        if kind in ("log", "log_difference") and np.any(level <= 0):
            _fail("MACRO_LOG_DOMAIN", "Log-transformed original values must be strictly positive")
        base = np.log(level) if kind in ("log", "log_difference") else level
        transformed = np.diff(base) if kind in ("difference", "log_difference") else base[drop:]
        fitting = []
        for i in fit_rows:
            original = i + drop
            needed = (
                [original - 1, original] if kind in ("difference", "log_difference") else [original]
            )
            vals = [fit_values[k][j] for k in needed]
            if any(v is None for v in vals):
                _fail("LEAKAGE", "Transformation fit uses data not yet released by fit cutoff")
            if kind in ("log", "log_difference"):
                if any(v <= 0 for v in vals):
                    _fail("MACRO_LOG_DOMAIN", "Training log values must be positive")
                vals = np.log(vals).tolist()
            fitting.append(vals[-1] - vals[0] if len(vals) == 2 else vals[0])
        center = float(np.mean(fitting)) if spec["standardize"] else 0.0
        scale = float(np.std(fitting, ddof=0)) if spec["standardize"] else 1.0
        if scale <= 0:
            _fail("MACRO_ZERO_VARIANCE", "A constant training series cannot be standardized")
        x[:, j] = (transformed - center) / scale
        state.append(
            dict(
                variable_id=variables[j],
                transform=kind,
                unit=spec["unit"],
                standardize=spec["standardize"],
                center=center,
                scale=scale,
                initial_original_value=float(level[0]),
                last_original_value=float(level[-1]),
                training_last_original_value=float(fit_values[fit_rows[-1] + drop][j]),
                standard_deviation_ddof=0,
            )
        )
    # Preserve a separately transformed training-vintage matrix to prevent a fit
    # on revised validation-vintage values of dates inside the training window.
    train_x = []
    for i in fit_rows:
        row = []
        for j, spec in enumerate(state):
            original = i + drop
            kind = spec["transform"]
            val = fit_values[original][j]
            if kind in ("log", "log_difference"):
                val = float(np.log(val))
            if kind in ("difference", "log_difference"):
                previous = fit_values[original - 1][j]
                val -= float(np.log(previous)) if kind == "log_difference" else previous
            row.append((val - spec["center"]) / spec["scale"])
        train_x.append(row)
    result = dict(
        schema_version="1.0",
        kind="MacroHistory",
        dates=output_dates,
        values=x.tolist(),
        variable_ids=variables,
        frequency=frequency,
        release_dates=releases[drop:],
        units=[
            "standard_deviation"
            if s["standardize"]
            else "log_ratio"
            if s["transform"] == "log_difference"
            else "log(" + s["unit"] + ")"
            if s["transform"] == "log"
            else "change(" + s["unit"] + ")"
            if s["transform"] == "difference"
            else s["unit"]
            for s in specs
        ],
        information_cutoff=cutoff,
        fit_window=window,
        inverse_state=state,
        training_dates=[output_dates[i] for i in fit_rows],
        training_values=train_x,
        training_vintage=window["known_at"],
        dropped_initial_dates=dates[:drop],
        data_origin="USER_SUPPLIED",
    )
    return _result(
        result,
        "macro.transform_macro_history",
        "macro_history",
        context,
        diagnostics=dict(
            excluded_future_releases=excluded,
            training_vintage_separated=True,
            data_fingerprint=_fingerprint(history),
        ),
        limitations=(
            "No seasonal adjustment, frequency conversion or missing-data imputation is inferred.",
        ),
    )


def _history(history, training, missing=False):
    h = deepcopy(_unwrap(history))
    if not isinstance(h, Mapping):
        _fail("MACRO_HISTORY", "Explicit history object required")
    variables = _ids(h.get("variable_ids"), "variable_ids")
    dates = _dates(h.get("dates", []), h.get("frequency"))
    window = _window(training)
    source = h.get("values")
    use_dates = dates
    if h.get("training_vintage") is not None:
        if h["training_vintage"] != window["known_at"] or h.get("fit_window") != window:
            _fail(
                "LEAKAGE",
                "Model fit must use exactly the frozen transformation training window/vintage",
            )
        source = h["training_values"]
        use_dates = h["training_dates"]
    if missing:
        try:
            x = np.asarray(
                [[np.nan if v is None else v for v in row] for row in source], dtype=float
            )
        except (TypeError, ValueError):
            _fail("MACRO_HISTORY", "Numeric/missing observation matrix required")
        if np.any(np.isinf(x)):
            _fail("NON_FINITE_INPUT", "Infinite macro observation")
    else:
        x = _numeric(source, "history_values")
    if x.ndim != 2 or x.shape != (len(use_dates), len(variables)):
        _fail("INCOMPATIBLE_SHAPE", "History values must match dates and variables")
    ix = [i for i, t in enumerate(use_dates) if window["start"] <= t <= window["end"]]
    if not ix:
        _fail("INSUFFICIENT_OBSERVATIONS", "No observations in training window")
    if h.get("training_vintage") is None:
        releases = h.get("release_dates")
        if not isinstance(releases, list) or len(releases) != len(dates):
            _fail("MACRO_HISTORY", "Release dates required for every observation and variable")
        for i in ix:
            if len(releases[i]) != len(variables) or any(
                _iso_date(v, "release_date") > window["known_at"] or v < dates[i]
                for v in releases[i]
            ):
                _fail("LEAKAGE", "Training history contains a future or invalid release")
    units = h.get("units")
    if (
        not isinstance(units, list)
        or len(units) != len(variables)
        or any(not isinstance(u, str) or not u for u in units)
    ):
        _fail("MACRO_UNITS", "One explicit unit per macro variable required")
    fit_dates = [use_dates[i] for i in ix]
    _dates(fit_dates, h["frequency"])
    return x[ix], fit_dates, variables, h, window


def _sqrt_cov(covariance):
    a = _numeric(covariance, "covariance")
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a) or np.any(np.diag(a) < 0):
        _fail("INVALID_COVARIANCE", "Square covariance with nonnegative variances required")
    scale = np.sqrt(np.diag(a))
    active = scale > 0
    if np.any(a[~active, :] != 0) or np.any(a[:, ~active] != 0):
        _fail("NON_PSD_MATRIX", "Zero-variance coordinates must have zero covariance")
    root = np.zeros_like(a)
    eigen = np.array([])
    if np.any(active):
        corr = a[np.ix_(active, active)] / np.outer(scale[active], scale[active])
        if not np.allclose(corr, corr.T, rtol=0, atol=1e-10):
            _fail("INVALID_COVARIANCE", "Covariance must be symmetric after unit normalization")
        eigen, vectors = np.linalg.eigh((corr + corr.T) / 2)
        if eigen.min() < -1e-10:
            _fail("NON_PSD_MATRIX", "Covariance is not PSD; no spectral repair is inferred")
        root[np.ix_(active, active)] = scale[active, None] * (
            (vectors * np.sqrt(np.maximum(eigen, 0))) @ vectors.T
        )
    return root, dict(
        factorization="CORRELATION_STANDARDIZED_PSD_FACTOR",
        minimum_correlation_eigenvalue=float(eigen.min()) if len(eigen) else 0.0,
        roundoff_clipped_eigenvalues=eigen[eigen < 0].tolist(),
        maximum_covariance_roundoff=float(np.max(np.abs(root @ root.T - a))),
    )


def _measure_contract(model, context):
    ctx = _context_value(context)
    if ctx.get("probability_measure") not in (None, model["target"]["probability_measure"]):
        _fail("TARGET_MISMATCH", "Context measure conflicts with frozen macro model")
    return ctx


def _stationary_linear(model):
    p = model["parameters"]
    k = len(model["feature_order"])
    lags = _integer(p["lags"], "lags", 1)
    if p["dynamics"] not in ("var", "vecm") or np.any(np.asarray(p["trend"]) != 0):
        _fail(
            "TTC_INITIALIZATION",
            "Gaussian equilibrium requires constant linear dynamics without a deterministic trend",
        )
    coefs = np.asarray(p["lag_coefficients"])
    f = np.zeros((k * lags, k * lags))
    f[:k] = np.concatenate(coefs, axis=1)
    if lags > 1:
        f[k:, :-k] = np.eye(k * (lags - 1))
    radius = float(np.max(np.abs(np.linalg.eigvals(f))))
    if radius >= 1 - 1e-10:
        _fail(
            "NONSTATIONARY_TTC",
            "A unit-root or explosive process has no finite Gaussian stationary law",
        )
    mean = np.tile(np.linalg.solve(np.eye(k) - coefs.sum(axis=0), p["intercept"]), lags)
    covariance = np.zeros_like(f)
    covariance[:k, :k] = p["covariance"]
    power = f.copy()
    for _ in range(100):
        addition = power @ covariance @ power.T
        covariance += addition
        power = power @ power
        if np.max(np.abs(addition)) <= 1e-12 * max(
            np.max(np.abs(covariance)), np.finfo(float).tiny
        ):
            break
    else:
        _fail("NUMERICAL_FAILURE", "Stationary covariance doubling did not converge")
    return mean, covariance, f, radius


def _sampling_initial(model, z, w, kind, initial_state, sampling_contract, context):
    p = model["parameters"]
    s, _, k = z.shape
    lags = _integer(p["lags"], "lags", 1)
    contract = (
        deepcopy(sampling_contract)
        if sampling_contract is not None
        else {"law": "FORECAST_CONDITIONAL"}
    )
    if not isinstance(contract, Mapping):
        _fail("TTC_INITIALIZATION", "Sampling contract must be an object")
    law = contract.get("law")
    ctx = _measure_contract(model, context)
    initial = p["initial_history"] if initial_state is None else initial_state
    if law == "FORECAST_CONDITIONAL":
        if ctx.get("conditioning") == "TTC":
            _fail("TTC_INITIALIZATION", "TTC requires an explicit equilibrium or cycle-mixture law")
        contract.update(
            conditioning_scope="PIT",
            initialization="LAST_OBSERVED_STATE"
            if initial_state is None
            else "DECLARED_FIXED_STATE",
        )
    elif law == "STATIONARY_GAUSSIAN":
        if initial_state is not None or kind != "STANDARD_NORMAL":
            _fail(
                "TTC_INITIALIZATION",
                "Stationary Gaussian law consumes initial and future normal draws, not fixed states or empirical residuals",
            )
        if not all(
            isinstance(contract.get(v), str) and contract[v].strip()
            for v in ("cycle_law_id", "equilibrium_assumptions")
        ):
            _fail("TTC_INITIALIZATION", "Cycle-law ID and equilibrium assumptions required")
        draws = _numeric(contract.pop("initial_state_draws", None), "initial_state_draws")
        if draws.shape != (s, lags * k):
            _fail(
                "INCOMPATIBLE_SHAPE",
                "Stationary initial draws require scenario by companion-state axes",
            )
        mean, covariance, f, radius = _stationary_linear(model)
        root, fd = _sqrt_cov(covariance)
        initial = (mean + draws @ root.T).reshape(s, lags, k)[:, ::-1, :]
        contract.update(
            conditioning_scope="TTC_STATIONARY",
            initial_draw_hash=_fingerprint(draws.tolist()),
            stationary_mean=mean.tolist(),
            stationary_covariance=covariance.tolist(),
            spectral_radius=radius,
            state_order="MOST_RECENT_FIRST",
            factorization=fd,
        )
    elif law == "DECLARED_CYCLE_MIXTURE":
        if not all(
            isinstance(contract.get(v), str) and contract[v].strip()
            for v in ("cycle_law_id", "equilibrium_assumptions")
        ):
            _fail("TTC_INITIALIZATION", "Declared cycle law and assumptions required")
        phases = contract.get("phase_ids")
        prob = contract.get("phase_probabilities")
        if (
            not isinstance(phases, list)
            or len(phases) != s
            or not isinstance(prob, Mapping)
            or not prob
            or any(not isinstance(v, str) or v not in prob for v in phases)
        ):
            _fail("TTC_INITIALIZATION", "Every scenario requires a declared cycle phase")
        keys = list(prob)
        pw = _numeric([prob[v] for v in keys], "phase_probabilities", vector_only=True)
        if (
            np.any(pw < 0)
            or abs(pw.sum() - 1) > 1e-10
            or any(abs(w[np.array(phases) == v].sum() - prob[v]) > 1e-10 for v in keys)
        ):
            _fail("INVALID_WEIGHTS", "Scenario weight mass must match the declared cycle-phase law")
        initial = _numeric(initial_state, "initial_state")
        if initial.shape != (s, lags, k):
            _fail(
                "TTC_INITIALIZATION", "Cycle mixture requires explicit scenario-specific histories"
            )
        contract.update(
            conditioning_scope="TTC_DECLARED_CYCLE_MIXTURE",
            initial_state_hash=_fingerprint(initial.tolist()),
            equilibrium_verified=False,
        )
    else:
        _fail("TTC_INITIALIZATION", "Unknown sampling law")
    return initial, dict(contract)


def _macro_model(
    method, parameters, h, dates, training, target, model_id, model_version, diagnostics, context
):
    train = dict(
        training,
        data_fingerprint=_fingerprint(dict(dates=dates, values=parameters.pop("_training_values"))),
    )
    inverse_state = deepcopy(h.get("inverse_state"))
    if inverse_state is not None:
        for state in inverse_state:
            state["last_original_value"] = state["training_last_original_value"]
    preprocessing = dict(
        frequency=h["frequency"],
        units=h["units"],
        inverse_state=inverse_state,
        training_end=dates[-1],
        training_observation_count=len(dates),
    )
    model = _model_artifact(
        dict(
            schema_version="1.0",
            method_id="macro." + method,
            model_id=model_id,
            model_version=model_version,
            feature_order=h["variable_ids"],
            parameters=parameters,
            preprocessing=preprocessing,
            target=_target(target),
            training=train,
            diagnostics=diagnostics,
            provenance=dict(
                data_origin=training.get("data_origin", "USER_SUPPLIED"),
                parameter_origin="ESTIMATED",
            ),
        )
    )
    _measure_contract(model, context)
    return _fit_result(model, "macro." + method, diagnostics, context)


def fit_var(
    transformed_history,
    *,
    lags,
    deterministic="constant",
    covariance_estimator="df_adjusted",
    training,
    target,
    model_id,
    model_version,
    context=None,
):
    """Actual multivariate OLS VAR backend with explicit deterministic and covariance conventions."""
    from statsmodels.tsa.api import VAR

    x, dates, variables, h, window = _history(transformed_history, training)
    n, k = x.shape
    p = _integer(lags, "lags", 1)
    if deterministic not in (
        "none",
        "constant",
        "trend",
        "constant_trend",
    ) or covariance_estimator not in ("mle", "df_adjusted"):
        _fail("MACRO_SPEC", "Declare deterministic terms and mle/df_adjusted innovation covariance")
    constant = deterministic in ("constant", "constant_trend")
    trend = deterministic in ("trend", "constant_trend")
    d = k * p + int(constant) + int(trend)
    if n - p <= d:
        _fail("INSUFFICIENT_OBSERVATIONS", "VAR requires positive residual degrees of freedom")
    design = np.column_stack(
        ([np.ones(n - p)] if constant else [])
        + ([np.arange(p + 1, n + 1, dtype=float)] if trend else [])
        + [x[p - j : n - j] for j in range(1, p + 1)]
    )
    if np.linalg.matrix_rank(design) < d:
        _fail("RANK_DEFICIENT", "VAR lag/deterministic design is rank deficient")
    if k == 1:
        # statsmodels VAR requires >=2 variables; its univariate analogue is
        # the exact OLS AR equation, not an artificial second response.
        import statsmodels.api as sm

        fit = sm.OLS(x[p:, 0], design).fit()
        b = np.asarray(fit.params)[:, None]
        residual = np.asarray(fit.resid)[:, None]
        intercept = b[0] if constant else np.zeros(1)
        trend_coef = b[int(constant)] if trend else np.zeros(1)
        coefs = np.array([b[int(constant) + int(trend) + j].reshape(1, 1) for j in range(p)])
        backend = "statsmodels.OLS (univariate AR)"
    else:
        fit = VAR(x, exog=np.arange(1, n + 1, dtype=float)[:, None] if trend else None).fit(
            p, trend="c" if constant else "n"
        )
        coefs = np.asarray(fit.coefs)
        residual = np.asarray(fit.resid)
        b = np.asarray(fit.params)
        intercept = b[0] if constant else np.zeros(k)
        trend_coef = b[int(constant)] if trend else np.zeros(k)
        backend = "statsmodels.VAR"
    covariance = (
        residual.T @ residual / (n - p - d if covariance_estimator == "df_adjusted" else n - p)
    )
    _, factorization = _sqrt_cov(covariance)
    companion = np.zeros((k * p, k * p))
    companion[:k, :] = np.concatenate(coefs, axis=1)
    if p > 1:
        companion[k:, :-k] = np.eye(k * (p - 1))
    roots = sorted(np.linalg.eigvals(companion), key=lambda v: (v.real, v.imag))
    radius = float(np.max(np.abs(roots)))
    params = dict(
        dynamics="var",
        lags=p,
        lag_coefficients=coefs.tolist(),
        intercept=intercept.tolist(),
        trend=trend_coef.tolist(),
        covariance=covariance.tolist(),
        initial_history=x[-p:].tolist(),
        next_time_index=n + 1,
        deterministic=deterministic,
        covariance_estimator=covariance_estimator,
        _training_values=x.tolist(),
    )
    diag = dict(
        backend=backend,
        observations=n - p,
        residual_df=n - p - d,
        residuals=residual.tolist(),
        residual_dates=dates[p:],
        companion_eigenvalues=[[float(v.real), float(v.imag)] for v in roots],
        spectral_radius=radius,
        stable=radius < 1,
        stationarity_assumed=True,
        eigenvalue_repair="NONE",
        factorization=factorization,
        training_only=True,
    )
    return _macro_model(
        "fit_var", params, h, dates, training, target, model_id, model_version, diag, context
    )


def _scenario_inputs(innovations, variables, frequency):
    value = _unwrap(innovations)
    if not isinstance(value, Mapping):
        _fail("MACRO_INNOVATIONS", "Explicit shared innovation cube and axis IDs required")
    ids = _ids(value.get("scenario_ids"), "scenario_ids")
    factors = _ids(value.get("factor_ids"), "factor_ids")
    if factors != list(variables):
        _fail(
            "FEATURE_MISMATCH", "Innovation factor order must match the model innovation channels"
        )
    z = _numeric(value.get("values"), "innovations")
    times = _numeric(value.get("time_grid"), "time_grid", vector_only=True)
    if z.ndim != 3 or z.shape != (len(ids), len(times), len(factors)) or len(times) == 0:
        _fail("INCOMPATIBLE_SHAPE", "Innovation axes must be scenario,time,factor")
    periods = {"monthly": 12, "quarterly": 4, "annual": 1}[frequency]
    if not np.allclose(times, np.arange(1, len(times) + 1) / periods, rtol=0, atol=1e-12):
        _fail(
            "INVALID_TIME_GRID",
            "Forecast times in years must match consecutive model frequency steps",
        )
    w, _ = _weights(value.get("weights", [1 / len(ids)] * len(ids)), len(ids))
    if value.get("kind") not in ("STANDARD_NORMAL", "RAW_INNOVATIONS"):
        _fail("MACRO_INNOVATIONS", "Declare STANDARD_NORMAL or RAW_INNOVATIONS")
    return z, ids, times, w, value["kind"]


def _scenario_model(model):
    m = _model_artifact(model)
    p = m["parameters"]
    k = len(m["feature_order"])
    if m["method_id"] not in (
        "macro.fit_var",
        "macro.fit_vecm",
        "macro.fit_regime_switching",
        "macro.fit_dynamic_factor",
    ):
        _fail("UNSUPPORTED_MODEL", "Frozen macro scenario model required")
    if p.get("dynamics") not in ("var", "vecm", "regime", "dynamic_factor"):
        _fail("INVALID_MODEL", "Unknown macro dynamics")
    expected = {
        "macro.fit_var": "var",
        "macro.fit_vecm": "vecm",
        "macro.fit_regime_switching": "regime",
        "macro.fit_dynamic_factor": "dynamic_factor",
    }
    if p["dynamics"] != expected[m["method_id"]]:
        _fail("INVALID_MODEL", "Frozen model method and dynamics disagree")
    if p["dynamics"] in ("var", "vecm"):
        if _numeric(p.get("covariance"), "covariance").shape != (k, k):
            _fail("INVALID_MODEL", "Innovation covariance differs from macro dimension")
        _integer(p.get("next_time_index"), "next_time_index", 1)
    return m, p, k


def _linear_paths(parameters, initial_history, innovations):
    e = np.asarray(innovations)
    s, h, k = e.shape
    p = _integer(parameters.get("lags"), "lags", 1)
    a = _numeric(parameters.get("lag_coefficients"), "lag_coefficients")
    c = _numeric(parameters.get("intercept"), "intercept", vector_only=True)
    trend = _numeric(parameters.get("trend"), "trend", vector_only=True)
    initial = _numeric(initial_history, "initial_history")
    if (
        initial.shape not in ((p, k), (s, p, k))
        or a.shape != (p, k, k)
        or c.shape != (k,)
        or trend.shape != (k,)
    ):
        _fail("INVALID_MODEL", "VAR/VECM recursion dimensions are inconsistent")
    paths = np.empty((s, h, k))
    state = np.broadcast_to(initial, (s, p, k)).copy()
    for t in range(h):
        value = (
            np.broadcast_to(c + trend * (parameters["next_time_index"] + t), (s, k)).copy()
            + e[:, t, :]
        )
        for lag in range(p):
            value += state[:, -1 - lag, :] @ a[lag].T
        paths[:, t, :] = value
        state = np.concatenate((state[:, 1:, :], value[:, None, :]), axis=1)
    if not np.all(np.isfinite(paths)):
        _fail("NUMERICAL_FAILURE", "Explosive recursion exceeded finite representation")
    return paths


def _cube(model, paths, ids, times, weights, diagnostics, context, method):
    if not np.all(np.isfinite(paths)):
        _fail("NUMERICAL_FAILURE", "Macro recursion exceeded finite representation")
    result = dict(
        schema_version="1.0",
        kind="ScenarioCube",
        axes=["scenario", "time", "factor"],
        scenario_ids=ids,
        time_grid=times.tolist(),
        time_unit="years",
        factor_ids=model["feature_order"],
        values=paths.tolist(),
        weights=weights.tolist(),
        units=[v["unit"] for v in model["preprocessing"]["inverse_state"]]
        if diagnostics.get("output_space") == "level"
        else model["preprocessing"]["units"],
        frequency=model["preprocessing"]["frequency"],
        origin=model["preprocessing"]["training_end"],
        information_cutoff=model["training"]["known_at"],
        sampling_contract=diagnostics.get("sampling_contract"),
        probability_measure=model["target"]["probability_measure"],
        model_hash=_fingerprint(model),
        scenario_origin="SIMULATED",
        data_origin=model["provenance"]["data_origin"],
    )
    return _result(
        result,
        "macro." + method,
        "scenario_cube",
        context,
        diagnostics=diagnostics,
        provenance=model["provenance"],
    )


def simulate_macro_scenarios(
    fitted_model,
    innovations,
    *,
    initial_state=None,
    regime_uniforms=None,
    output_space="model",
    sampling_contract=None,
    context=None,
):
    """Propagate one frozen joint macro model using explicit shared innovations."""
    model, p, k = _scenario_model(fitted_model)
    if p["dynamics"] not in ("var", "vecm"):
        from ._macro_models import _simulate_latent_macro

        return _simulate_latent_macro(
            model,
            innovations,
            initial_state=initial_state,
            regime_uniforms=regime_uniforms,
            output_space=output_space,
            sampling_contract=sampling_contract,
            context=context,
        )
    if regime_uniforms is not None:
        _fail("MACRO_INNOVATIONS", "Linear dynamics do not consume regime uniforms")
    z, ids, times, w, kind = _scenario_inputs(
        innovations, model["feature_order"], model["preprocessing"]["frequency"]
    )
    root, fd = _sqrt_cov(p["covariance"])
    e = z @ root.T if kind == "STANDARD_NORMAL" else z
    initial, sampling = _sampling_initial(
        model, z, w, kind, initial_state, sampling_contract, context
    )
    paths = _linear_paths(p, initial, e)
    if output_space == "level":
        if initial_state is not None or sampling["law"] != "FORECAST_CONDITIONAL":
            _fail(
                "MACRO_TRANSFORM",
                "Level reconstruction requires the fitted fixed-state anchor; alternative cycle anchors must be declared separately",
            )
        paths = _inverse_paths(paths, model["preprocessing"].get("inverse_state"))
    elif output_space != "model":
        _fail("MACRO_TRANSFORM", "output_space must be model or level")
    return _cube(
        model,
        paths,
        ids,
        times,
        w,
        dict(
            innovation_hash=_fingerprint(_unwrap(innovations)),
            factorization=fd,
            output_space=output_space,
            shared_macro_across_entities=True,
            sampling_contract=sampling,
        ),
        context,
        "simulate_macro_scenarios",
    )


def _inverse_paths(paths, state):
    if not isinstance(state, list) or len(state) != paths.shape[2]:
        _fail(
            "MACRO_TRANSFORM",
            "Frozen inverse transformation state required for level reconstruction",
        )
    out = np.empty_like(paths)
    for j, spec in enumerate(state):
        value = paths[:, :, j] * spec["scale"] + spec["center"]
        kind = spec["transform"]
        if kind in ("difference", "log_difference"):
            initial = (
                np.log(spec["last_original_value"])
                if kind == "log_difference"
                else spec["last_original_value"]
            )
            value = initial + np.cumsum(value, axis=1)
        if kind in ("log", "log_difference"):
            value = np.exp(value)
        out[:, :, j] = value
    if not np.all(np.isfinite(out)):
        _fail("NUMERICAL_FAILURE", "Inverse macro transformation exceeds finite representation")
    return out


def block_bootstrap_scenarios(
    residual_history,
    shared_draws,
    *,
    factor_ids,
    scenario_ids,
    time_grid,
    frequency,
    block_length,
    scheme="moving",
    weights=None,
    context=None,
):
    """Resample multivariate blocks jointly with explicit starts/restart uniforms."""
    residual = _numeric(residual_history, "residual_history")
    factors = _ids(factor_ids, "factor_ids")
    ids = _ids(scenario_ids, "scenario_ids")
    if residual.ndim != 2 or residual.shape[1] != len(factors):
        _fail("INCOMPATIBLE_SHAPE", "Residual rows must match named factors")
    n = len(residual)
    draws = _numeric(shared_draws, "shared_draws")
    times = _numeric(time_grid, "time_grid", vector_only=True)
    h = len(times)
    if draws.shape != (len(ids), h, 2) or np.any((draws < 0) | (draws >= 1)):
        _fail(
            "INVALID_DRAWS", "Shared draws require scenario,time,[start,restart] uniforms in [0,1)"
        )
    length = _numeric(block_length, "block_length")
    if (
        length.ndim
        or length < 1
        or length > n
        or scheme not in ("moving", "circular", "stationary")
    ):
        _fail("MACRO_BOOTSTRAP", "Explicit valid block scheme/length required")
    if scheme != "stationary" and length != np.floor(length):
        _fail("MACRO_BOOTSTRAP", "Moving/circular block lengths must be integers")
    selected = np.empty((len(ids), h), dtype=int)
    blocks = np.empty_like(selected)
    for i in range(len(ids)):
        position = 0
        block = -1
        for t in range(h):
            restart = t == 0 or (
                draws[i, t, 1] < 1 / length if scheme == "stationary" else t % int(length) == 0
            )
            if restart:
                choices = n - int(length) + 1 if scheme == "moving" else n
                position = int(np.floor(draws[i, t, 0] * choices))
                block += 1
            else:
                position = (position + 1) % n
            selected[i, t] = position
            blocks[i, t] = block
    result = dict(
        kind="RAW_INNOVATIONS",
        values=residual[selected].tolist(),
        factor_ids=factors,
        scenario_ids=ids,
        time_grid=times.tolist(),
        weights=[1 / len(ids)] * len(ids) if weights is None else list(weights),
    )
    _scenario_inputs(result, factors, frequency)
    return _result(
        result,
        "macro.block_bootstrap_scenarios",
        "innovations_cube",
        context,
        diagnostics=dict(
            scheme=scheme,
            block_length=float(length),
            selected_indices=selected.tolist(),
            block_ids=blocks.tolist(),
            shared_draw_hash=_fingerprint(shared_draws),
            joint_variables=True,
        ),
        limitations=(
            "Historical blocks retain empirical local dependence, not a guarantee about future regimes or tail events.",
        ),
    )


def _linear_gaussian_moments(model, horizon, initial_state=None):
    p = model["parameters"]
    k = len(model["feature_order"])
    if p["dynamics"] == "dynamic_factor":
        from ._macro_models import _factor_gaussian_moments

        return _factor_gaussian_moments(model, horizon, initial_state)
    if p["dynamics"] not in ("var", "vecm"):
        _fail(
            "UNSUPPORTED_CONDITIONER",
            "Gaussian conditioning does not silently replace a hidden-regime mixture",
        )
    initial = p["initial_history"] if initial_state is None else initial_state
    mean = _linear_paths(p, initial, np.zeros((1, horizon, k)))[0].reshape(-1)
    impulses = np.zeros((horizon * k, horizon, k))
    for j in range(horizon * k):
        impulses[j, j // k, j % k] = 1
    response = _linear_paths(
        p, np.zeros((_integer(p["lags"], "lags", 1), k)), impulses
    ) - _linear_paths(p, np.zeros((_integer(p["lags"], "lags", 1), k)), np.zeros_like(impulses))
    mapping = response.reshape(horizon * k, horizon * k).T
    covariance = mapping @ np.kron(np.eye(horizon), np.asarray(p["covariance"])) @ mapping.T
    return mean, covariance


def condition_macro_paths(
    fitted_model,
    innovations,
    conditioning_points,
    *,
    initial_state=None,
    sampler="gaussian_equality",
    sampling_contract=None,
    context=None,
):
    """Condition the full joint Gaussian path distribution, never overwrite isolated cells."""
    model, p, k = _scenario_model(fitted_model)
    if sampler != "gaussian_equality":
        _fail(
            "UNSUPPORTED_CONDITIONER",
            "Only explicit linear Gaussian equality conditioning is supported",
        )
    if p["dynamics"] == "regime":
        _fail(
            "UNSUPPORTED_CONDITIONER",
            "Hidden-regime mixtures require a mixture-specific conditioner",
        )
    unconditional = simulate_macro_scenarios(
        model,
        innovations,
        initial_state=initial_state,
        sampling_contract=sampling_contract,
        context=context,
    )
    cube = unconditional.value
    paths = np.asarray(cube["values"])
    s, h, k = paths.shape
    if not isinstance(conditioning_points, (list, tuple)):
        _fail("MACRO_CONDITION", "Conditioning points must be explicit records")
    if not conditioning_points:
        return _result(
            cube,
            "macro.condition_macro_paths",
            "scenario_cube",
            context,
            diagnostics=dict(
                constraint_count=0, maximum_constraint_error=0.0, conditioning="UNCONDITIONAL"
            ),
            provenance=model["provenance"],
        )
    if _unwrap(innovations).get("kind") != "STANDARD_NORMAL":
        _fail(
            "UNSUPPORTED_CONDITIONER",
            "Gaussian conditioning requires explicitly standard-normal shared draws, not empirical bootstrap residuals",
        )
    indexes = []
    values = []
    for point in conditioning_points:
        t = _integer(point.get("time_index"), "time_index", 0)
        name = point.get("factor_id")
        value = _numeric(point.get("value"), "condition_value")
        if t >= h or name not in model["feature_order"] or value.ndim:
            _fail(
                "MACRO_CONDITION",
                "Condition must identify a valid zero-based time, factor and scalar value",
            )
        ix = t * k + model["feature_order"].index(name)
        if ix in indexes:
            _fail("DUPLICATE_CONSTRAINT", "Each constrained path coordinate must be unique")
        indexes.append(ix)
        values.append(float(value))
    law = cube["sampling_contract"]["law"]
    if law == "DECLARED_CYCLE_MIXTURE":
        _fail(
            "UNSUPPORTED_CONDITIONER",
            "A declared cycle mixture requires a mixture-specific conditioner",
        )
    if p["dynamics"] == "dynamic_factor":
        from ._macro_models import _factor_gaussian_moments

        mean, covariance = _factor_gaussian_moments(model, h, initial_state, sampling_contract)
    elif law == "STATIONARY_GAUSSIAN":
        state_mean, state_cov, f, _ = _stationary_linear(model)
        mean, covariance = _linear_gaussian_moments(
            model, h, state_mean.reshape(_integer(p["lags"], "lags", 1), k)[::-1]
        )
        power = np.eye(len(f))
        mapping = []
        for _ in range(h):
            power = power @ f
            mapping.append(power[:k])
        mapping = np.concatenate(mapping, axis=0)
        covariance += mapping @ state_cov @ mapping.T
    else:
        mean, covariance = _linear_gaussian_moments(model, h, initial_state)
    c = covariance[np.ix_(indexes, indexes)]
    delta = np.asarray(values) - mean[indexes]
    _sqrt_cov(c)
    scales = np.sqrt(np.diag(c))
    active = scales > 0
    if np.any(delta[~active] != 0):
        _fail("INFEASIBLE_CONSTRAINT", "Nonzero departure from a deterministic constraint")
    inverse = np.zeros_like(c)
    eigen = np.array([])
    positive = np.array([], dtype=bool)
    if np.any(active):
        corr = c[np.ix_(active, active)] / np.outer(scales[active], scales[active])
        eigen, vectors = np.linalg.eigh((corr + corr.T) / 2)
        positive = eigen > 1e-10
        corr_inverse = (vectors[:, positive] / eigen[positive]) @ vectors[:, positive].T
        standardized = delta[active] / scales[active]
        if np.max(np.abs(standardized - corr @ corr_inverse @ standardized)) > 1e-8:
            _fail("INFEASIBLE_CONSTRAINT", "Constraint is outside the Gaussian covariance support")
        inverse[np.ix_(active, active)] = corr_inverse / np.outer(scales[active], scales[active])
    gain = covariance[:, indexes] @ inverse
    flat = paths.reshape(s, h * k)
    conditioned = flat + (np.asarray(values)[None, :] - flat[:, indexes]) @ gain.T
    error = float(np.max(np.abs(conditioned[:, indexes] - values)))
    normalized_error = np.max(
        np.abs(conditioned[:, indexes] - values) / np.where(active, scales, 1)
    )
    if normalized_error > 1e-7:
        _fail(
            "NUMERICAL_FAILURE",
            "Gaussian conditional constraint residual exceeds scale-normalized tolerance",
        )
    conditional_cov = covariance - gain @ covariance[indexes, :]
    conditional_mean = mean + gain @ delta
    rank = int(np.sum(positive))
    logdensity = (
        -0.5 * (rank * np.log(2 * np.pi) + np.log(eigen[positive]).sum() + delta @ inverse @ delta)
        if rank
        else 0.0
    )
    if rank == len(indexes):
        logdensity -= np.log(scales).sum()
    cube = dict(
        cube,
        values=conditioned.reshape(s, h, k).tolist(),
        conditioning_points=deepcopy(list(conditioning_points)),
        sampling_contract=dict(
            cube["sampling_contract"],
            conditioning_scope="CONDITIONAL_STRESS",
            parent_conditioning_scope=cube["sampling_contract"]["conditioning_scope"],
        ),
    )
    return _result(
        cube,
        "macro.condition_macro_paths",
        "scenario_cube",
        context,
        diagnostics=dict(
            constraint_count=len(indexes),
            constraint_rank=rank,
            maximum_constraint_error=error,
            conditional_mean=conditional_mean.reshape(h, k).tolist(),
            conditional_covariance=conditional_cov.tolist(),
            log_constraint_density_on_support=float(logdensity),
            density_coordinates="PHYSICAL_CONSTRAINTS"
            if rank == len(indexes)
            else "STANDARDIZED_SUPPORT",
            density_is_not_event_probability=True,
            conditioning="JOINT_GAUSSIAN_CROSS_COVARIANCE",
            mean_adjustment=(gain @ delta).reshape(h, k).tolist(),
        ),
        provenance=model["provenance"],
    )


def reverse_stress_scenario(
    fitted_model,
    loss_model,
    *,
    horizon,
    target_loss,
    shock_metric,
    initial_state=None,
    shock_bounds=None,
    initial_shock=None,
    tolerance=1e-7,
    max_iterations=500,
    context=None,
):
    """Minimize declared shock distance subject to a loss threshold using augmented Lagrangian optimization."""
    model, p, k = _scenario_model(fitted_model)
    _measure_contract(model, context)
    h = _integer(horizon, "horizon", 1)
    if p["dynamics"] not in ("var", "vecm"):
        _fail(
            "UNSUPPORTED_REVERSE_MODEL",
            "Reverse optimization currently requires linear VAR/VECM dynamics",
        )
    threshold = _numeric(target_loss, "target_loss")
    tol = _numeric(tolerance, "tolerance")
    limit = _integer(max_iterations, "max_iterations", 1)
    if threshold.ndim or tol.ndim or not 0 < tol < 0.01:
        _fail("MACRO_OPTIMIZATION", "Scalar target and tolerance in (0,.01) required")
    dimension = h * k
    metric = _numeric(shock_metric, "shock_metric")
    if (
        metric.shape != (dimension, dimension)
        or not np.allclose(metric, metric.T, rtol=0, atol=1e-12)
        or np.min(np.linalg.eigvalsh(metric)) <= 0
    ):
        _fail("MACRO_OPTIMIZATION", "Explicit symmetric positive-definite shock metric required")
    if shock_bounds is None:
        bounds = [(None, None)] * dimension
    else:
        bound = _numeric(shock_bounds, "shock_bounds")
        if bound.shape != (dimension, 2) or np.any(bound[:, 0] > bound[:, 1]):
            _fail("MACRO_OPTIMIZATION", "Lower/upper bounds per shock required")
        bounds = [tuple(v) for v in bound]
    start = (
        np.zeros(dimension)
        if initial_shock is None
        else _numeric(initial_shock, "initial_shock", vector_only=True)
    )
    if start.shape != (dimension,) or any(
        (lo is not None and v < lo) or (hi is not None and v > hi)
        for v, (lo, hi) in zip(start, bounds)
    ):
        _fail("MACRO_OPTIMIZATION", "Initial shock must satisfy its bounds")
    root, fd = _sqrt_cov(p["covariance"])
    initial = p["initial_history"] if initial_state is None else initial_state

    def paths(z):
        return _linear_paths(p, initial, z.reshape(1, h, k) @ root.T)[0]

    if callable(loss_model):

        def response(z):
            value = _numeric(loss_model(paths(z)), "loss_model output")
            if value.ndim:
                _fail("MACRO_OPTIMIZATION", "Loss callback must return a finite scalar")
            return float(value)

        response_metadata = dict(kind="LOCAL_CALLBACK", portable=False)
    else:
        if (
            not isinstance(loss_model, Mapping)
            or loss_model.get("kind") != "linear_predictor"
            or loss_model.get("link") not in ("identity", "logit", "log")
        ):
            _fail(
                "MACRO_OPTIMIZATION",
                "Use callable or explicit linear_predictor response with identity/logit/log link",
            )
        coefficients = _numeric(loss_model.get("coefficients"), "loss_coefficients")
        intercept = _numeric(loss_model.get("intercept"), "loss_intercept")
        if coefficients.shape != (h, k) or intercept.ndim:
            _fail("INCOMPATIBLE_SHAPE", "Response coefficients must match horizon and variables")

        def response(z):
            eta = float(intercept + np.sum(coefficients * paths(z)))
            value = (
                eta
                if loss_model["link"] == "identity"
                else float(expit(eta))
                if loss_model["link"] == "logit"
                else float(np.exp(eta))
            )
            if not np.isfinite(value):
                _fail("NUMERICAL_FAILURE", "Loss response overflow")
            return value

        response_metadata = deepcopy(loss_model)
    z = start.copy()
    multiplier = 0.0
    penalty = 10.0
    trace = []
    success = False
    for outer in range(12):

        def objective(v):
            violation = float(threshold) - response(v)
            return float(
                0.5 * v @ metric @ v
                + (max(0.0, multiplier + penalty * violation) ** 2 - multiplier**2) / (2 * penalty)
            )

        fit = minimize(
            objective,
            z,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": limit, "ftol": 1e-14, "gtol": float(tol) * 0.1, "maxls": 50},
        )
        z = fit.x
        achieved = response(z)
        violation = float(threshold) - achieved
        new_multiplier = max(0.0, multiplier + penalty * violation)
        trace.append(
            dict(
                outer_iteration=outer,
                penalty=penalty,
                achieved_loss=achieved,
                positive_violation=max(0.0, violation),
                inner_converged=bool(fit.success),
                inner_message=str(fit.message),
                metric_distance_squared=float(z @ metric @ z),
            )
        )
        if max(0.0, violation) <= tol and abs(new_multiplier - multiplier) <= max(
            float(tol), 1e-6
        ) * max(1.0, new_multiplier):
            success = True
            break
        multiplier = new_multiplier
        penalty *= 10
    achieved = response(z)
    feasible = achieved >= float(threshold) - float(tol)
    result = dict(
        status="FEASIBLE_LOCAL_SOLUTION"
        if feasible and success
        else "FEASIBLE_OPTIMIZATION_INCOMPLETE"
        if feasible
        else "INFEASIBLE_OR_LOCAL_SEARCH_FAILED",
        shocks=z.reshape(h, k).tolist(),
        candidate_path=paths(z).tolist(),
        factor_ids=model["feature_order"],
        target_loss=float(threshold),
        achieved_loss=achieved,
        feasible=bool(feasible),
        maximum_constraint_violation=max(0.0, float(threshold) - achieved),
        metric_distance_squared=float(z @ metric @ z),
        objective=0.5 * float(z @ metric @ z),
        shock_metric=metric.tolist(),
        response=response_metadata,
        model_hash=_fingerprint(model),
        global_optimality_guaranteed=False,
    )
    return _result(
        result,
        "macro.reverse_stress_scenario",
        "reverse_stress_result",
        context,
        diagnostics=dict(
            algorithm="POWELL_HESTENES_AUGMENTED_LAGRANGIAN_LBFGSB",
            trace=trace,
            innovation_factorization=fd,
            metric_is_not_probability=True,
        ),
        provenance=model["provenance"],
        limitations=(
            "A local feasible stress point is not a global minimum or an event probability. Failure to find feasibility does not generally certify global infeasibility.",
        ),
    )


# Public family entry points; estimation implementations retain native artifacts.
from ._macro_models import (  # noqa: E402
    fit_vecm as fit_vecm,
    fit_regime_switching as fit_regime_switching,
    fit_dynamic_factor as fit_dynamic_factor,
)


def _validate_macro_cube(cube):
    """Validate a frozen shared macro cube; return its plain value without mutation."""
    value = _unwrap(cube)
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != "1.0"
        or value.get("kind") != "ScenarioCube"
        or value.get("axes") != ["scenario", "time", "factor"]
    ):
        _fail("INVALID_SCENARIO_CUBE", "Versioned scenario,time,factor macro cube required")
    ids = _ids(value.get("scenario_ids"), "scenario_ids")
    factors = _ids(value.get("factor_ids"), "factor_ids")
    x = _numeric(value.get("values"), "values")
    time = _numeric(value.get("time_grid"), "time_grid", vector_only=True)
    if (
        x.shape != (len(ids), len(time), len(factors))
        or len(time) == 0
        or np.any(np.diff(time) <= 0)
        or time[0] <= 0
        or value.get("time_unit") != "years"
    ):
        _fail(
            "INVALID_SCENARIO_CUBE",
            "Finite values must match named axes with increasing positive year grid",
        )
    frequency = value.get("frequency")
    periods = {"monthly": 12, "quarterly": 4, "annual": 1}.get(frequency)
    if periods is None or not np.allclose(
        time, np.arange(1, len(time) + 1) / periods, rtol=0, atol=1e-12
    ):
        _fail(
            "INVALID_TIME_GRID", "Macro cube grid must match consecutive declared frequency periods"
        )
    weight = _numeric(value.get("weights"), "weights", vector_only=True)
    if weight.shape != (len(ids),) or np.any(weight < 0) or abs(weight.sum() - 1) > 1e-10:
        _fail("INVALID_WEIGHTS", "Cube weights must already be normalized")
    units = value.get("units")
    if (
        not isinstance(units, list)
        or len(units) != len(factors)
        or any(not isinstance(v, str) or not v for v in units)
    ):
        _fail("MACRO_UNITS", "One explicit nonempty unit per factor required")
    if value.get("probability_measure") not in ("PHYSICAL", "RISK_NEUTRAL"):
        _fail("INCOMPATIBLE_MEASURE", "Explicit macro probability measure required")
    origin = _iso_date(value.get("origin"), "origin")
    cutoff = _iso_date(value.get("information_cutoff"), "information_cutoff")
    if cutoff < origin:
        _fail("INVALID_DATE", "Information cutoff cannot precede last observed macro date")
    sampling = value.get("sampling_contract")
    if (
        not isinstance(sampling, Mapping)
        or not isinstance(sampling.get("law"), str)
        or sampling.get("conditioning_scope")
        not in ("PIT", "TTC_STATIONARY", "TTC_DECLARED_CYCLE_MIXTURE", "CONDITIONAL_STRESS")
    ):
        _fail("TTC_INITIALIZATION", "Explicit PIT/TTC/stress sampling contract required")
    return dict(value)


from ._macro_satellites import (  # noqa: E402
    fit_satellite_model as fit_satellite_model,
    apply_satellite_model as apply_satellite_model,
)
