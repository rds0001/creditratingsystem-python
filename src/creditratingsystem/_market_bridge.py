"""Historically estimated bridge from eligible Q-market features to observed P defaults."""

from copy import deepcopy
import numpy as np
from .core import _fail, _iso_date, _fingerprint, _fit_result, _model_artifact, _context_value
from .classification import fit_binary_glm, predict_pd, _design
from .selection_bias import _folds
from .hybrid import _hy_horizon_end
from .calibration import _target
from ._market_pricing import _text, _scalar


def fit_market_to_physical_pd_bridge(
    observations, feature_order, bridge_contract, folds, *, model_id, model_version, context=None
):
    """Estimate a physical-default GLM with grouped forward evaluation and mature labels."""
    if not isinstance(observations, (list, tuple)) or not observations:
        _fail("INVALID_INPUT", "Observed dated market/default rows required")
    c = deepcopy(dict(bridge_contract))
    target = _target(c.get("target"))
    if (
        target["probability_measure"] != "PHYSICAL"
        or target["conditioning"] != "initially_performing"
        or target["event"] != "first_default"
    ):
        _fail(
            "TARGET_MISMATCH",
            "Bridge needs observed initially-performing first-default PHYSICAL target",
        )
    if (
        c.get("feature_probability_measure") != "RISK_NEUTRAL"
        or c.get("outcome_origin") != "OBSERVED_ISSUER_DEFAULT"
    ):
        _fail(
            "MEASURE_MISMATCH",
            "Explicit Q market features and observed issuer-default outcomes required",
        )
    _text(c.get("target_population"), "target_population")
    if (
        c.get("population_selection") != "MARKET_QUOTED_ISSUERS"
        or c.get("transformation") != "IDENTITY_PREDECLARED_FEATURES"
    ):
        _fail("INVALID_INPUT", "Declare market-quoted population and predeclared identity features")
    feature_contracts = c.get("market_feature_contracts")
    if not isinstance(feature_contracts, dict) or set(feature_contracts) != set(feature_order):
        _fail(
            "FEATURE_MISMATCH",
            "Each predeclared feature needs a measure/horizon/definition contract",
        )
    for feature in feature_order:
        fc = feature_contracts[feature]
        if (
            not isinstance(fc, dict)
            or fc.get("probability_measure") != "RISK_NEUTRAL"
            or fc.get("time_unit") not in ("days", "months", "years")
        ):
            _fail("MEASURE_MISMATCH", "Market feature measure and horizon units required")
        if _scalar(fc.get("horizon"), "market feature horizon", 0) <= 0:
            _fail("INVALID_INPUT", "Positive market feature horizon required")
        _text(fc.get("definition"), "market feature definition")
    cutoff = _iso_date(c.get("known_at"), "bridge known_at")
    ctx = _context_value(context)
    if ctx.get("probability_measure") not in (None, "PHYSICAL"):
        _fail("MEASURE_MISMATCH", "Bridge output context is PHYSICAL")
    ids = []
    metadata = []
    values = []
    outcomes = []
    for row in observations:
        ids.append(_text(row.get("observation_id"), "observation_id"))
        asof = _iso_date(row.get("as_of"), "as_of")
        end = _hy_horizon_end(asof, target).isoformat()
        label = _iso_date(row.get("label_known_at"), "label_known_at")
        complete = _iso_date(row.get("complete_through"), "complete_through")
        if complete < end or label < end or label > cutoff:
            _fail(
                "IMMATURE_OUTCOME",
                "Every physical target window must be mature and available at fit cutoff",
            )
        curve_date = _iso_date(row.get("market_curve_as_of"), "market_curve_as_of")
        curve_known = _iso_date(row.get("market_curve_known_at"), "market_curve_known_at")
        features_known = _iso_date(row.get("features_known_at"), "features_known_at")
        if (
            curve_known > curve_date
            or curve_date > asof
            or features_known > asof
            or features_known < curve_known
        ):
            _fail(
                "LOOKAHEAD_BIAS",
                "Market curve vintage and feature availability must precede prediction",
            )
        _text(row.get("market_curve_hash"), "market_curve_hash")
        if (
            row.get("outcome_origin") != "OBSERVED_ISSUER_DEFAULT"
            or row.get("initial_state") != "PERFORMING"
        ):
            _fail(
                "INVALID_OUTCOME", "Actual observed initially-performing issuer outcomes required"
            )
        y = _scalar(row.get("outcome"), "outcome", 0, 1)
        if y not in (0.0, 1.0):
            _fail("INVALID_OUTCOME", "Binary physical default outcome required")
        default = row.get("default_date")
        if y == 1:
            default = _iso_date(default, "default_date")
            if not asof < default <= end:
                _fail("TARGET_MISMATCH", "Default must occur inside declared horizon")
        elif default is not None and _iso_date(default, "default_date") <= end:
            _fail("TARGET_MISMATCH", "Nondefault label conflicts with dated default")
        features = row.get("features")
        if not isinstance(features, dict) or set(features) != set(feature_order):
            _fail("FEATURE_MISMATCH", "Each row requires exactly the declared market features")
        values.append([features[f] for f in feature_order])
        outcomes.append(y)
        metadata.append(
            dict(
                entity_id=_text(row.get("entity_id"), "entity_id"),
                as_of=asof,
                label_known_at=label,
                features_known_at=features_known,
            )
        )
    if len(ids) != len(set(ids)):
        _fail("DUPLICATE_ID", "Duplicate observations")
    x, _ = _design(values, feature_order, True)
    y = np.array(outcomes)
    plans = _folds(folds, metadata, len(y))
    training = dict(
        data_fingerprint=_fingerprint(observations),
        split_id=_text(c.get("split_id"), "split_id"),
        known_at=cutoff,
        population_selection=c["population_selection"],
        target_population=c["target_population"],
        data_origin=c.get("data_origin", "USER_SUPPLIED"),
    )
    common = dict(
        link=c.get("link"),
        feature_order=feature_order,
        fit_intercept=True,
        target=target,
        model_id=model_id,
        model_version=model_version,
        context=context,
    )
    full = fit_binary_glm(x, y, training=training, **common)
    predictions = [None] * len(y)
    lineage = []
    for tr, te in plans:
        ft = dict(
            training,
            known_at=max(metadata[i]["label_known_at"] for i in tr),
            data_fingerprint=_fingerprint([observations[i] for i in tr]),
        )
        fitted = fit_binary_glm(x[tr], y[tr], training=ft, **common)
        p = predict_pd(fitted, x[te], feature_order=feature_order, expected_target=target).value
        for i, v in zip(te, p):
            predictions[i] = float(v)
        lineage.append(
            dict(
                train=tr.tolist(),
                validation=te.tolist(),
                model_hash=_fingerprint(fitted.model),
                training_known_at=ft["known_at"],
            )
        )
    indices = [i for i, v in enumerate(predictions) if v is not None]
    p = np.array([predictions[i] for i in indices])
    yy = y[indices]
    loss = (yy - p) ** 2
    clusters = {}
    for i, row_loss in zip(indices, loss):
        clusters.setdefault(metadata[i]["entity_id"], []).append(float(row_loss))
    influence = np.array([sum(v) - len(v) * float(loss.mean()) for v in clusters.values()]) / len(
        loss
    )
    cluster_se = (
        float(np.sqrt(len(clusters) / (len(clusters) - 1) * np.sum(influence**2)))
        if len(clusters) > 1
        else None
    )
    diagnostics = dict(
        full.diagnostics,
        out_of_time_pd=predictions,
        warmup_indices=[i for i, v in enumerate(predictions) if v is None],
        folds=lineage,
        observed_evaluation_count=len(indices),
        out_of_time_brier=float(loss.mean()),
        brier_cluster_standard_error=cluster_se,
        uncertainty="ENTITY_CLUSTER_EMPIRICAL_INFLUENCE",
        evaluation_scope="OBSERVED_MATURE_MARKET_QUOTED_ISSUERS_ONLY",
        out_of_time_log_loss=float(-np.mean(yy * np.log(p) + (1 - yy) * np.log1p(-p))),
        feature_probability_measure="RISK_NEUTRAL",
        output_probability_measure="PHYSICAL",
        conversion_assumption="HISTORICALLY_ESTIMATED_CONDITIONAL_ASSOCIATION_NOT_UNIVERSAL_MEASURE_CHANGE",
    )
    model = deepcopy(full.model)
    model["method_id"] = "market_implied.fit_market_to_physical_pd_bridge"
    model["parameters"]["base_method_id"] = "classification.fit_binary_glm"
    model["preprocessing"]["market_bridge_contract"] = c
    model["diagnostics"] = diagnostics
    model = _model_artifact(model)
    return _fit_result(model, model["method_id"], diagnostics, context)


def _predict_market_bridge(model, X, *, feature_order, expected_target, context=None):
    model = _model_artifact(model)
    if (
        model["method_id"] != "market_implied.fit_market_to_physical_pd_bridge"
        or model["parameters"].get("base_method_id") != "classification.fit_binary_glm"
    ):
        _fail("INVALID_ARTIFACT", "Expected fitted historical physical bridge")
    c = model["preprocessing"].get("market_bridge_contract", {})
    if (
        c.get("feature_probability_measure") != "RISK_NEUTRAL"
        or model["target"]["probability_measure"] != "PHYSICAL"
    ):
        _fail("MEASURE_MISMATCH", "Market bridge measure contract changed")
    base = deepcopy(model)
    base["method_id"] = "classification.fit_binary_glm"
    base["preprocessing"] = {"transformation": "identity"}
    result = predict_pd(
        base, X, feature_order=feature_order, expected_target=expected_target, context=context
    )
    return result
