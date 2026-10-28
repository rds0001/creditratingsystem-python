"""Frozen release-aware satellite regressions and shared macro transmission."""

from collections.abc import Mapping
from copy import deepcopy
import numpy as np
from scipy.special import expit, ndtr
from .core import _fail, _numeric, _iso_date, _fingerprint, _model_artifact, _fit_result, _result
from .macro import _unwrap, _dates, _integer, _measure_contract, _validate_macro_cube
from .calibration import _target
from ._loss_paths import _ids


def _satellite_design(target_series, macro_history, feature_spec, split_plan, training):
    if not isinstance(target_series, list) or not target_series:
        _fail("SATELLITE_DATA", "Dated target records required")
    ids = _ids([v.get("observation_id") for v in target_series], "observation_ids")
    rows = {v["observation_id"]: v for v in target_series}
    if (
        not isinstance(split_plan, Mapping)
        or split_plan.get("mode") not in ("TEMPORAL", "ENTITY_TIME")
        or split_plan.get("split_id") != training.get("split_id")
    ):
        _fail("LEAKAGE", "Explicit matching temporal SplitPlan required")
    roles = {
        key: split_plan.get(key, [])
        for key in ("train_ids", "calibration_ids", "validation_ids", "test_ids")
    }
    assigned = sum(roles.values(), [])
    if (
        len(assigned) != len(set(assigned))
        or set(assigned) != set(ids)
        or not roles["train_ids"]
        or not sum([roles[k] for k in ("calibration_ids", "validation_ids", "test_ids")], [])
    ):
        _fail("LEAKAGE", "Disjoint exhaustive training and later holdout IDs required")
    cutoff = _iso_date(training.get("known_at"), "known_at")
    start = _iso_date(training.get("start"), "start")
    end = _iso_date(training.get("end"), "end")
    if split_plan.get("training_cutoff") != cutoff or not start <= end <= cutoff:
        _fail("LEAKAGE", "Training cutoff and window must agree")
    byrole = []
    for role, keys in roles.items():
        dates = [_iso_date(rows[v].get("date"), "date") for v in keys]
        entities = [rows[v].get("entity_id") for v in keys]
        if any(not isinstance(v, str) or not v for v in entities):
            _fail("INVALID_IDENTIFIERS", "Target entity IDs required")
        if dates:
            byrole.append((min(dates), max(dates), set(entities)))
        if role == "train_ids" and any(not start <= v <= end for v in dates):
            _fail("LEAKAGE", "Training row lies outside declared window")
    if len(byrole) > 1 and byrole[1][0] <= cutoff:
        _fail("LEAKAGE", "First holdout observation must follow model information cutoff")
    for before, after in zip(byrole, byrole[1:]):
        if before[1] >= after[0]:
            _fail("LEAKAGE", "Split roles must be strictly time ordered")
    if split_plan["mode"] == "ENTITY_TIME":
        for i, a in enumerate(byrole):
            if any(a[2] & b[2] for b in byrole[i + 1 :]):
                _fail("LEAKAGE", "Entity-time roles require disjoint entities")
    h = _unwrap(macro_history)
    if isinstance(h, Mapping) and h.get("fit_window") is not None:
        transform_window = h["fit_window"]
        if (
            _iso_date(transform_window.get("known_at"), "transform_known_at") > cutoff
            or _iso_date(transform_window.get("end"), "transform_end") > end
        ):
            _fail(
                "LEAKAGE",
                "Macro transformation was fitted beyond the satellite training information window",
            )
    dates = _dates(h.get("dates"), h.get("frequency"))
    factors = _ids(h.get("variable_ids"), "variable_ids")
    x = _numeric(h.get("values"), "macro_values")
    units = h.get("units")
    releases = h.get("release_dates")
    if (
        x.shape != (len(dates), len(factors))
        or not isinstance(units, list)
        or len(units) != len(factors)
        or not isinstance(releases, list)
        or len(releases) != len(dates)
    ):
        _fail("SATELLITE_DATA", "Aligned macro values, units and release dates required")
    if not isinstance(feature_spec, list) or not feature_spec:
        _fail("SATELLITE_SPEC", "Explicit lagged feature specification required")
    names = _ids([v.get("feature_id") for v in feature_spec], "feature_ids")
    spec = deepcopy(feature_spec)
    for v in spec:
        lag = _integer(v.get("lag"), "lag", 0)
        name = v.get("variable_id")
        v["lag"] = lag
        if (
            name not in factors
            or v.get("unit") != units[factors.index(name)]
            or v.get("expected_sign") not in ("positive", "negative", "unrestricted")
        ):
            _fail("SATELLITE_SPEC", "Factor, physical unit and expected sign hypothesis required")
    train_rows = sorted(
        [rows[v] for v in roles["train_ids"]],
        key=lambda r: (r["date"], r["entity_id"], r["observation_id"]),
    )
    seen = set()
    design = []
    outcomes = []
    lineage = []
    for row in train_rows:
        key = (row["entity_id"], row["date"])
        if key in seen:
            _fail("DUPLICATE_RECORD", "At most one target observation per entity/date")
        seen.add(key)
        known = _iso_date(row.get("known_at"), "target_known_at")
        info = _iso_date(row.get("information_date"), "information_date")
        if known < row["date"] or known > cutoff or info > row["date"] or info > cutoff:
            _fail("LEAKAGE", "Target maturity or feature information date violates training cutoff")
        if row["date"] not in dates:
            _fail("SATELLITE_DATA", "Target period absent from macro calendar")
        t = dates.index(row["date"])
        features = []
        for v in spec:
            ix = t - v["lag"]
            j = factors.index(v["variable_id"])
            if ix < 0:
                _fail("INSUFFICIENT_HISTORY", "Declared macro lag requires earlier observations")
            release = _iso_date(releases[ix][j], "release_date")
            if release < dates[ix] or release > info:
                _fail("LEAKAGE", "Macro vintage was not released at the target information date")
            features.append(x[ix, j])
            lineage.append(
                dict(
                    observation_id=row["observation_id"],
                    feature_id=v["feature_id"],
                    macro_date=dates[ix],
                    release_date=release,
                )
            )
        design.append(features)
        outcomes.append(row.get("value"))
    y = _numeric(outcomes, "training_targets", vector_only=True)
    return np.asarray(design), y, train_rows, h, spec, names, lineage


def fit_satellite_model(
    target_series,
    macro_history,
    *,
    feature_spec,
    estimator,
    link,
    split_plan,
    training,
    target,
    target_unit,
    model_id,
    model_version,
    weights=None,
    covariance="classical",
    hac_lags=0,
    residual_process="white_noise",
    context=None,
):
    """Estimate only dated training targets with released, explicitly lagged macro predictors."""
    import statsmodels.api as sm

    x, y, rows, h, spec, names, lineage = _satellite_design(
        target_series, macro_history, feature_spec, split_plan, training
    )
    n, d = x.shape
    design = np.c_[np.ones(n), x]
    dimension = d + 1
    if (
        not isinstance(target_unit, str)
        or not target_unit
        or estimator not in ("OLS", "WLS", "GLM")
        or covariance not in ("classical", "HC1", "HAC", "cluster")
        or residual_process not in ("none", "white_noise", "AR1")
    ):
        _fail(
            "SATELLITE_SPEC",
            "Explicit estimator, target unit, covariance and residual process required",
        )
    if n <= dimension or np.linalg.matrix_rank(design) < dimension:
        _fail(
            "RANK_DEFICIENT",
            "Satellite training design needs positive residual degrees of freedom and full rank",
        )
    if weights is None:
        w = np.ones(n)
    else:
        if not isinstance(weights, Mapping) or set(weights) != set(
            r["observation_id"] for r in rows
        ):
            _fail("INVALID_WEIGHTS", "Weights must map exactly the training observation IDs")
        w = _numeric([weights[r["observation_id"]] for r in rows], "weights", vector_only=True)
    if np.any(w <= 0) or (estimator == "OLS" and np.any(w != 1)):
        _fail("INVALID_WEIGHTS", "Positive WLS precision weights or unit OLS weights required")
    transformed = y.copy()
    binary = estimator == "GLM"
    if binary:
        if (
            link not in ("logit", "probit", "cloglog")
            or residual_process != "none"
            or covariance != "classical"
        ):
            _fail(
                "SATELLITE_SPEC",
                "Binary GLM uses explicit link, Fisher covariance and no inferred Gaussian residual process",
            )
        from .classification import fit_binary_glm

        glm = fit_binary_glm(
            x,
            y,
            link=link,
            feature_order=names,
            fit_intercept=True,
            target=target,
            training=dict(
                training,
                data_fingerprint=_fingerprint(dict(x=x.tolist(), y=y.tolist(), w=w.tolist())),
            ),
            model_id=model_id,
            model_version=model_version,
            weights=w,
        )
        coefficients = np.asarray(glm.model["parameters"]["coefficients"])
        cov = np.asarray(glm.model["parameters"]["covariance"])
        backend = "statsmodels GLM Bernoulli"
        residual = np.zeros(n)
        phi = 0.0
        sigma = 0.0
        pairs = 0
    else:
        if link not in ("identity", "log"):
            _fail("SATELLITE_SPEC", "Continuous satellites use identity or log response transform")
        if link == "log":
            if np.any(y <= 0):
                _fail("SATELLITE_DOMAIN", "Log response requires strictly positive observed values")
            transformed = np.log(y)
        fitted = sm.WLS(transformed, design, weights=w).fit()
        coefficients = np.asarray(fitted.params)
        residual = transformed - design @ coefficients
        bread = np.linalg.inv(design.T @ (w[:, None] * design))
        scores = design * (w * residual)[:, None]
        if covariance == "classical":
            cov = bread * np.sum(w * residual**2) / (n - dimension)
        else:
            if covariance == "HC1":
                meat = scores.T @ scores * n / (n - dimension)
            elif covariance == "cluster":
                entities = sorted(set(r["entity_id"] for r in rows))
                g = len(entities)
                if g < 2:
                    _fail(
                        "INSUFFICIENT_CLUSTERS",
                        "Cluster covariance needs at least two target entities",
                    )
                summed = np.array(
                    [
                        scores[[i for i, r in enumerate(rows) if r["entity_id"] == v]].sum(axis=0)
                        for v in entities
                    ]
                )
                meat = summed.T @ summed * g / (g - 1) * (n - 1) / (n - dimension)
            else:
                length = _integer(hac_lags, "hac_lags", 0)
                if len(set(r["entity_id"] for r in rows)) != 1 or length >= n:
                    _fail(
                        "SATELLITE_SPEC",
                        "HAC requires one contiguous entity series and lag below sample size",
                    )
                _dates([r["date"] for r in rows], h["frequency"])
                meat = scores.T @ scores
                for lag in range(1, length + 1):
                    cross = scores[lag:].T @ scores[:-lag]
                    meat += (1 - lag / (length + 1)) * (cross + cross.T)
            cov = bread @ meat @ bread
        phi = 0.0
        pairs = 0
        errors = residual
        if residual_process == "AR1":
            pairs_list = []
            frequency = {"monthly": 1, "quarterly": 3, "annual": 12}[h["frequency"]]
            for entity in sorted(set(r["entity_id"] for r in rows)):
                ix = [i for i, r in enumerate(rows) if r["entity_id"] == entity]
                for a, b in zip(ix, ix[1:]):
                    da, db = rows[a]["date"], rows[b]["date"]
                    gap = (int(db[:4]) - int(da[:4])) * 12 + int(db[5:7]) - int(da[5:7])
                    if gap == frequency:
                        pairs_list.append((a, b))
            if len(pairs_list) < 3:
                _fail(
                    "INSUFFICIENT_HISTORY",
                    "Residual AR1 needs at least three consecutive within-entity pairs",
                )
            previous = np.array([residual[a] for a, b in pairs_list])
            current = np.array([residual[b] for a, b in pairs_list])
            den = float(previous @ previous)
            if den <= 0:
                _fail("NON_IDENTIFIABLE", "Residual AR1 variance is zero")
            phi = float(previous @ current / den)
            if abs(phi) >= 1:
                _fail("NONSTATIONARY_RESIDUAL", "Estimated residual AR1 is not stationary")
            errors = current - phi * previous
            pairs = len(pairs_list)
        sigma = float(np.sqrt(np.mean(errors**2))) if residual_process != "none" else 0.0
        backend = "statsmodels WLS"
    train = dict(
        training,
        data_fingerprint=_fingerprint(
            dict(
                rows=[
                    dict(
                        observation_id=r["observation_id"],
                        date=r["date"],
                        entity_id=r["entity_id"],
                        known_at=r["known_at"],
                        information_date=r["information_date"],
                    )
                    for r in rows
                ],
                x=x.tolist(),
                y=y.tolist(),
                weights=w.tolist(),
            )
        ),
    )
    sign_violations = [
        v["feature_id"]
        for j, v in enumerate(spec)
        if (v["expected_sign"] == "positive" and coefficients[j + 1] < 0)
        or (v["expected_sign"] == "negative" and coefficients[j + 1] > 0)
    ]
    parameters = dict(
        dynamics="satellite",
        estimator=estimator,
        link=link,
        coefficients=coefficients.tolist(),
        covariance=cov.tolist(),
        residual_process=residual_process,
        residual_ar=phi,
        residual_sigma=sigma,
        target_unit=target_unit,
    )
    prep = dict(
        feature_spec=spec,
        macro_factor_ids=h["variable_ids"],
        macro_units=h["units"],
        frequency=h["frequency"],
        target_transform=link,
        application_modes=["absolute", "baseline_delta"],
    )
    diagnostics = dict(
        backend=backend,
        training_only=True,
        training_rows=n,
        training_ids=[r["observation_id"] for r in rows],
        split_plan_hash=_fingerprint(split_plan),
        coefficient_covariance=covariance,
        hac_lags=hac_lags if covariance == "HAC" else None,
        residual_ar_pairs=pairs,
        residuals=residual.tolist(),
        feature_lineage=lineage,
        sign_hypothesis_violations=sign_violations,
        sign_constraints_enforced=False,
        log_inverse_interpretation="CONDITIONAL_MEDIAN_WITHOUT_STOCHASTIC_DRAWS"
        if link == "log"
        else None,
        holdout_used_for_estimation=False,
    )
    model = _model_artifact(
        dict(
            schema_version="1.0",
            method_id="macro.fit_satellite_model",
            model_id=model_id,
            model_version=model_version,
            feature_order=names,
            parameters=parameters,
            preprocessing=prep,
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
    return _fit_result(model, "macro.fit_satellite_model", diagnostics, context)


def apply_satellite_model(
    fitted_satellite_model,
    scenario_cube,
    idiosyncratic_innovations,
    initial_target_state,
    *,
    context=None,
):
    """Transmit shared macro paths through a frozen link equation and fitted residual process."""
    model = _model_artifact(fitted_satellite_model)
    _measure_contract(model, context)
    cube = _validate_macro_cube(scenario_cube)
    p = model["parameters"]
    prep = model["preprocessing"]
    if (
        model["method_id"] != "macro.fit_satellite_model"
        or cube["probability_measure"] != model["target"]["probability_measure"]
    ):
        _fail("TARGET_MISMATCH", "Compatible frozen satellite and macro measures required")
    if (
        cube["factor_ids"] != prep["macro_factor_ids"]
        or cube["units"] != prep["macro_units"]
        or cube.get("frequency") != prep["frequency"]
    ):
        _fail(
            "FEATURE_MISMATCH",
            "Macro factor order, units and frequency differ from frozen satellite",
        )
    if (
        not isinstance(initial_target_state, Mapping)
        or initial_target_state.get("as_of") != cube["origin"]
    ):
        _fail("INVALID_DATE", "Initial target and macro origins must agree")
    if cube["origin"] < model["training"]["end"]:
        _fail("LEAKAGE", "Satellite training ends after the requested macro origin")
    knowledge = _iso_date(initial_target_state.get("known_at"), "known_at")
    if knowledge < cube["information_cutoff"] or knowledge < model["training"]["known_at"]:
        _fail("LEAKAGE", "Target application precedes model or macro information availability")
    paths = np.asarray(cube["values"])
    s, h, k = paths.shape
    innov = _unwrap(idiosyncratic_innovations)
    if (
        not isinstance(innov, Mapping)
        or innov.get("scenario_ids") != cube["scenario_ids"]
        or innov.get("time_grid") != cube["time_grid"]
        or innov.get("kind") != "STANDARD_NORMAL"
    ):
        _fail(
            "MACRO_INNOVATIONS",
            "Idiosyncratic draws must name exactly the shared scenario and time axes",
        )
    z = _numeric(innov.get("values"), "idiosyncratic_innovations")
    if z.shape != (s, h):
        _fail("INCOMPATIBLE_SHAPE", "Idiosyncratic draws require scenario by time axes")
    specs = [dict(v, lag=_integer(v.get("lag"), "lag", 0)) for v in prep["feature_spec"]]
    maxlag = max(v["lag"] for v in specs)
    history = (
        _numeric(initial_target_state.get("macro_history"), "macro_history")
        if maxlag
        else np.zeros((0, k))
    )
    if history.shape != (maxlag, k):
        _fail(
            "INSUFFICIENT_HISTORY",
            "Initial macro history must cover exactly maximum declared lag ending at cube origin",
        )
    full = np.concatenate((np.broadcast_to(history, (s, maxlag, k)), paths), axis=1)
    features = np.empty((s, h, len(specs)))
    for j, v in enumerate(specs):
        features[:, :, j] = full[
            :, maxlag - v["lag"] : maxlag - v["lag"] + h, cube["factor_ids"].index(v["variable_id"])
        ]
    coefficients = _numeric(p["coefficients"], "coefficients", vector_only=True)
    mode = initial_target_state.get("mode")
    base = np.full((s, h), coefficients[0])
    drivers = features
    if mode == "baseline_delta":
        baseline = _numeric(
            initial_target_state.get("baseline_link_path"), "baseline_link_path", vector_only=True
        )
        reference = _numeric(
            initial_target_state.get("reference_feature_path"), "reference_feature_path"
        )
        if baseline.shape != (h,) or reference.shape != (h, len(specs)):
            _fail(
                "INCOMPATIBLE_SHAPE",
                "Baseline and reference lagged feature paths must match forecast axes",
            )
        base = np.broadcast_to(baseline, (s, h)).copy()
        drivers = features - reference[None, :, :]
    elif mode != "absolute":
        _fail("SATELLITE_SPEC", "Application mode must be absolute or baseline_delta")
    macro = drivers @ coefficients[1:]
    state = _numeric(initial_target_state.get("residual_state"), "residual_state")
    if state.ndim:
        _fail(
            "INCOMPATIBLE_SHAPE", "Initial residual is an explicit scalar on the fitted link scale"
        )
    state = np.full(s, float(state))
    idio = np.empty((s, h))
    if p["residual_process"] == "none" and np.any(state != 0):
        _fail("SATELLITE_SPEC", "No residual process implies zero initial residual")
    for t in range(h):
        state = p["residual_ar"] * state + p["residual_sigma"] * z[:, t]
        idio[:, t] = state
    eta = base + macro + idio
    link = p["link"]
    out = (
        eta
        if link == "identity"
        else np.exp(eta)
        if link == "log"
        else expit(eta)
        if link == "logit"
        else ndtr(eta)
        if link == "probit"
        else -np.expm1(-np.exp(eta))
    )
    if not np.all(np.isfinite(out)):
        _fail("NUMERICAL_FAILURE", "Satellite inverse link exceeded finite representation")
    provenance = dict(
        model["provenance"],
        data_origin="SYNTHETIC"
        if cube.get("data_origin") == "SYNTHETIC"
        or model["provenance"]["data_origin"] == "SYNTHETIC"
        else "USER_SUPPLIED",
    )
    result = dict(
        cube,
        data_origin=provenance["data_origin"],
        values=out[:, :, None].tolist(),
        factor_ids=[model["target"]["event"]],
        units=[p["target_unit"]],
        model_hash=_fingerprint(model),
        source_macro_hash=_fingerprint(cube),
        scenario_origin="DERIVED_SATELLITE",
        macro_sampling_contract=deepcopy(cube["sampling_contract"]),
    )
    return _result(
        result,
        "macro.apply_satellite_model",
        "scenario_cube",
        context,
        diagnostics=dict(
            application_mode=mode,
            baseline_link=base.tolist(),
            macro_link=macro.tolist(),
            idiosyncratic_link=idio.tolist(),
            linear_predictor=eta.tolist(),
            link=link,
            source_scenario_ids=cube["scenario_ids"],
            shared_macro_across_entities=True,
        ),
        provenance=provenance,
    )
