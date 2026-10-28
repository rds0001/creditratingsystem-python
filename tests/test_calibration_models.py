"""Independent binomial, normal, WLS and grouped likelihood calibration oracles."""

from copy import deepcopy
import json
import os
import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import beta, norm
from creditratingsystem._calibration_models import (
    calibrate_beta,
    calibrate_grade_pd,
    low_default_bound,
    fit_hierarchical_calibration,
    calibrate_lgd_level,
    estimate_margin_of_conservatism,
    population_reweighting,
    fit_stepwise,
    fit_regularization_path,
    resample_fit_uncertainty,
    fit_model_selection_protocol,
)
from creditratingsystem.calibration import apply_calibrator
from creditratingsystem.core import CreditRiskError, _fingerprint


def metadata():
    return {
        "target": {
            "event": "first_default",
            "horizon": 12,
            "time_unit": "months",
            "probability_measure": "PHYSICAL",
            "conditioning": "PIT",
        },
        "calibration": {
            "start": "2014-01-01",
            "end": "2014-12-31",
            "known_at": "2015-12-31",
            "split_id": "calibration",
            "prediction_origin": "HOLDOUT",
        },
        "base_model_version": "fixed-v1",
        "population_definition": "Independent synthetic algebra",
        "outcome_policy": "Mature binary event",
        "weight_type": "frequency",
        "data_origin": "SYNTHETIC",
    }


def protocol_fixture():
    X = []
    y = []
    rows = []
    for block in range(4):
        for level, count in zip([-2, -1, 0, 1, 2], [2, 5, 10, 15, 18]):
            for j in range(20):
                X.append([level, (-1) ** j])
                y.append(int(j < count))
                i = len(y) - 1
                year = 2010 + 2 * block
                rows.append(
                    {
                        "entity_id": f"e{i}",
                        "as_of": f"{year}-01-01",
                        "label_known_at": f"{year + 1}-01-01",
                        "role": "development"
                        if block < 2
                        else "calibration"
                        if block == 2
                        else "test",
                    }
                )
    plan = {
        "rows": rows,
        "folds": [{"train": list(range(100)), "validation": list(range(100, 200))}],
    }
    return {
        "X": X,
        "y": y,
        "feature_order": ["signal", "noise"],
        "split_plan": plan,
        "metadata": metadata(),
    }


def apply(a, p, groups=None):
    return apply_calibrator(
        a, p, target=metadata()["target"], base_model_version="fixed-v1", groups=groups
    ).value


def test_beta_identity_from_exact_binomial_cells():
    p = np.repeat([0.1, 0.2, 0.4, 0.7, 0.9], 2)
    y = np.tile([1, 0], 5)
    w = np.array([[1, 9], [2, 8], [4, 6], [7, 3], [9, 1]]).ravel()
    fit = calibrate_beta(p, y, w, metadata=metadata())
    np.testing.assert_allclose(
        [fit.value["parameters"][k] for k in ["intercept", "a", "b"]], [0, 1, 1], atol=1e-8
    )
    np.testing.assert_allclose(apply(fit, [0.01, 0.5, 0.99]), [0.01, 0.5, 0.99], atol=1e-10)
    restored = json.loads(json.dumps(fit.value))
    np.testing.assert_equal(apply(restored, [0.2]), apply(fit, [0.2]))
    restored["parameters"]["a"] += 1
    with pytest.raises(CreditRiskError, match="hash"):
        apply(restored, [0.2])


def test_beta_boundary_and_monotone_constraint():
    p = np.repeat([0, 0.2, 0.5, 0.8, 1], 2)
    y = np.tile([1, 0], 5)
    w = np.array([[9, 1], [7, 3], [5, 5], [3, 7], [1, 9]]).ravel()
    with pytest.raises(CreditRiskError):
        calibrate_beta(p, y, w, metadata=metadata())
    m = calibrate_beta(p, y, w, metadata=metadata(), boundary_policy="clip")
    assert m.value["parameters"]["a"] >= 0 and m.value["parameters"]["b"] >= 0
    assert np.all(np.diff(apply(m, np.linspace(0, 1, 11))) >= -1e-12)
    free = calibrate_beta(p, y, w, metadata=metadata(), boundary_policy="clip", monotonic=False)
    assert apply(free, [0.1])[0] > apply(free, [0.9])[0]


def test_grade_pooling_and_target_accounting():
    a = calibrate_grade_pd(
        [100, 100, 100, 20], [1, 4, 2, 20], grades=["A", "B", "C", "D"], default_grade="D"
    ).value
    assert [r["pd"] for r in a["grades"]] == pytest.approx([0.01, 0.03, 0.03, 1])
    assert [r["pool"] for r in a["grades"]] == [0, 1, 1, None]
    assert a["grades"][1]["upper"] == pytest.approx(beta.ppf(0.975, 7, 194))
    z = calibrate_grade_pd(
        [10, 20, 30], [0, 1, 5], prior_alpha=1, prior_beta=1, target_default_rate=0.2
    ).value
    assert z["achieved_mean_pd"] == pytest.approx(0.2, abs=1e-12)
    assert np.diff([r["pd"] for r in z["grades"]]).min() >= 0
    zero = calibrate_grade_pd([100], [0]).value["grades"][0]
    assert zero["pd"] == 0 and zero["upper"] > 0


def test_low_default_independent_and_correlated_limits():
    p = low_default_bound(0, 100).value["bounds"][0]["upper_pd"]
    assert p == pytest.approx(1 - 0.05**0.01, abs=1e-12)
    assert low_default_bound(3, 100).value["bounds"][0]["upper_pd"] == pytest.approx(
        beta.ppf(0.95, 4, 97), abs=1e-12
    )
    assert (
        low_default_bound(0, 100, dependence="gaussian_one_factor", correlation=0).value["bounds"][
            0
        ]["upper_pd"]
        == p
    )
    assert (
        low_default_bound(0, 100, dependence="gaussian_one_factor", correlation=0.1).value[
            "bounds"
        ][0]["upper_pd"]
        > p
    )
    tail = low_default_bound([0, 1, 2], [100, 50, 25], pooling="ordered_tail").value["bounds"]
    assert [r["count"] for r in tail] == [175, 75, 25]
    assert low_default_bound(5, 5).value["bounds"][0]["upper_pd"] == 1


def test_lgd_three_transformations_use_loss_domain():
    for mode, x, t in [
        ("additive", [-0.2, 0.5, 1.2], 0.7),
        ("multiplicative", [0.1, 0.5, 1.2], 0.8),
        ("logit", [0.1, 0.4, 0.8], 0.6),
    ]:
        a = calibrate_lgd_level(x, t, [1, 2, 3], metadata=metadata(), transformation=mode)
        p = apply(a, x)
        assert np.average(p, weights=[1, 2, 3]) == pytest.approx(t, abs=1e-11)
        assert np.all(np.diff(p) > 0)
    assert apply(calibrate_lgd_level([-1, 2], 1, metadata=metadata()), [-1, 2]).tolist() == [
        -0.5,
        2.5,
    ]


def test_hierarchical_partial_pooling_and_unknown_group_integral():
    p = [0.2] * 6
    y = [1, 0] * 3
    w = [1, 19, 8, 12, 16, 4]
    groups = ["A", "A", "B", "B", "C", "C"]
    fit = fit_hierarchical_calibration(p, y, groups, w, metadata=metadata())
    params = fit.value["parameters"]
    assert params["variance"] > 0
    effects = params["groups"]
    assert effects[0]["mode"] < effects[1]["mode"] < effects[2]["mode"]
    assert all(0 < r["conditional_variance"] <= params["variance"] for r in effects)
    from scipy.integrate import quad

    expected = quad(
        lambda z: expit(logit(0.3) + params["intercept"] + np.sqrt(params["variance"]) * z)
        * norm.pdf(z),
        -10,
        10,
        epsabs=1e-11,
    )[0]
    assert apply(fit, [0.3], ["NEW"])[0] == pytest.approx(expected, abs=2e-7)
    quiet = fit_hierarchical_calibration(
        [0.2] * 6, [1, 0] * 3, ["A", "A", "B", "B", "C", "C"], [4, 16] * 3, metadata=metadata()
    )
    assert quiet.value["parameters"]["variance"] < 1e-4


def test_moc_components_covariance_and_deduplication():
    c = [
        {"source_id": "a", "category": "data", "adverse_bias": 0.01, "standard_error": 0.02},
        {"source_id": "b", "category": "general", "adverse_bias": 0, "standard_error": 0.03},
    ]
    a = estimate_margin_of_conservatism(0.1, c, correlation=[[1, 0.5], [0.5, 1]]).value
    assert a["margin"] == pytest.approx(
        0.01 + norm.ppf(0.95) * np.sqrt(0.02**2 + 0.03**2 + 2 * 0.5 * 0.02 * 0.03)
    )
    assert estimate_margin_of_conservatism(0.1, c, already_included=["a", "b"]).value["margin"] == 0
    with pytest.raises(CreditRiskError):
        estimate_margin_of_conservatism(0.1, c + [c[0]])
    with pytest.raises(CreditRiskError):
        estimate_margin_of_conservatism(0.1, c, correlation=[[1, 2], [2, 1]])


def test_population_weights_support_cap_and_ess():
    a = population_reweighting(["a", "a", "a", "b"], {"a": 0.5, "b": 0.5}).value
    assert a["weights"] == pytest.approx([2 / 3] * 3 + [2])
    assert a["effective_sample_size"] == pytest.approx(3)
    assert population_reweighting(["a", "b"], {"a": 0.5, "b": 0.5}).value["weights"] == [1, 1]
    assert population_reweighting(
        [0.5, 0.5], {"sample_prior": 0.5, "target_prior": 0.5}, method="membership_odds"
    ).value["weights"] == [1, 1]
    with pytest.raises(CreditRiskError):
        population_reweighting(["a"], {"a": 0.5, "missing": 0.5})
    clipped = population_reweighting(["a", "a", "a", "b"], {"a": 0.5, "b": 0.5}, cap=1).value
    assert clipped["clipped_count"] == 1 and clipped["max_final_weight"] > 1


@pytest.mark.extended
@pytest.mark.skipif(
    os.getenv("CREDITRATINGSYSTEM_EXTENDED_TESTS", "").lower() != "true",
    reason="Opt-in exhaustive selection variant audit",
)
@pytest.mark.parametrize("direction", ["forward", "backward", "bidirectional"])
@pytest.mark.parametrize("criterion", ["AIC", "BIC", "LR"])
@pytest.mark.parametrize("estimator", ["OLS", "WLS", "GLM"])
def test_each_stepwise_direction_criterion_estimator(direction, criterion, estimator):
    fixture = protocol_fixture()
    x = np.array(fixture["X"])[:100]
    y = np.array(fixture["y"])[:100]
    if estimator != "GLM":
        y = 2 + 3 * x[:, 0] + 0.1 * x[:, 1] + np.sin(np.arange(100))
    w = 1 + np.arange(100) % 3 if estimator == "WLS" else None
    a = fit_stepwise(
        x,
        y,
        feature_order=["signal", "noise"],
        candidate_groups={"signal": ["signal"], "noise": ["noise"]},
        direction=direction,
        criterion=criterion.lower(),
        estimator=estimator.lower(),
        weights=w,
    ).value
    assert "signal" in a["selected_groups"]
    assert a["selected_model"]["row_count"] == 100
    m = a["selected_model"]
    cols = [["signal", "noise"].index(f) for f in m["feature_order"]]
    D = np.column_stack([np.ones(100), x[:, cols]])
    if estimator != "GLM":
        ww = np.ones(100) if w is None else w
        b = np.linalg.solve(D.T @ (ww[:, None] * D), D.T @ (ww * y))
        np.testing.assert_allclose([m["intercept"]] + m["coefficients"], b, atol=1e-10)
        ll = (
            -50 * (np.log(2 * np.pi) + 1 + np.log(np.sum(ww * (y - D @ b) ** 2) / 100))
            + 0.5 * np.log(ww).sum()
        )
        assert m["log_likelihood"] == pytest.approx(ll)
    if criterion != "LR":
        assert a["path"][-1]["criterion_value"] == pytest.approx(
            -2 * m["log_likelihood"]
            + (2 if criterion == "AIC" else np.log(100)) * m["parameter_count"]
        )


def test_stepwise_hierarchy_ties_and_exclusion():
    x = np.arange(20, dtype=float)
    X = np.column_stack([x, x, x * x])
    y = 2 * x + np.sin(x)
    a = fit_stepwise(
        X,
        y,
        feature_order=["a", "b", "square"],
        candidate_groups={"A": ["a"], "B": ["b"]},
        estimator="OLS",
        direction="forward",
    ).value
    assert a["selected_groups"] == ["A"]
    assert "square" not in a["selected_model"]["feature_order"]
    with pytest.raises(CreditRiskError):
        fit_stepwise(
            X,
            y,
            feature_order=["a", "b", "square"],
            candidate_groups={"A": ["a"], "S": ["square"]},
            mandatory_terms=["S"],
            hierarchy={"S": ["A"]},
            estimator="OLS",
        )


@pytest.mark.parametrize(
    "alpha",
    [
        0.0,
        pytest.param(
            0.5,
            marks=[
                pytest.mark.extended,
                pytest.mark.skipif(
                    os.getenv("CREDITRATINGSYSTEM_EXTENDED_TESTS", "").lower() != "true",
                    reason="Opt-in elastic-net variants",
                ),
            ],
        ),
        pytest.param(
            1.0,
            marks=[
                pytest.mark.extended,
                pytest.mark.skipif(
                    os.getenv("CREDITRATINGSYSTEM_EXTENDED_TESTS", "").lower() != "true",
                    reason="Opt-in elastic-net variants",
                ),
            ],
        ),
    ],
)
def test_regularization_folds_scale_train_only_and_holdout(alpha):
    f = protocol_fixture()
    args = {k: f[k] for k in ["feature_order", "split_plan"]}
    a = fit_regularization_path(
        f["X"], f["y"], penalty_grid=[0, 0.1, 1], elastic_net_alpha=alpha, **args
    ).value
    altered = deepcopy(f)
    altered["X"][350] = [1e12, -1e12]
    altered["y"][350] = 1 - altered["y"][350]
    b = fit_regularization_path(
        altered["X"], altered["y"], penalty_grid=[0, 0.1, 1], elastic_net_alpha=alpha, **args
    ).value
    assert a["selected_model"]["artifact_hash"] == b["selected_model"]["artifact_hash"]
    assert a["path"][0]["folds"][0]["training_center"] == [0, 0]
    bad = deepcopy(f["split_plan"])
    bad["rows"][100]["entity_id"] = bad["rows"][0]["entity_id"]
    with pytest.raises(CreditRiskError):
        fit_regularization_path(
            f["X"], f["y"], penalty_grid=[0.1], feature_order=f["feature_order"], split_plan=bad
        )
    bad = deepcopy(f["split_plan"])
    bad["rows"][0]["label_known_at"] = "2013-01-01"
    with pytest.raises(CreditRiskError):
        fit_regularization_path(
            f["X"], f["y"], penalty_grid=[0.1], feature_order=f["feature_order"], split_plan=bad
        )


def test_resampling_whole_units_failure_counts_and_repeated_selection():
    X = np.array([[0], [1], [2], [3], [4], [5]], float)
    y = np.array([0, 1, 1, 4, 3, 6], float)
    ix = [list(range(6)), [0, 1, 0, 1, 4, 5], [2, 3, 2, 3, 4, 5]]
    a = resample_fit_uncertainty(
        X,
        y,
        feature_order=["x"],
        resampling_indices=ix,
        resampling_units=["A", "A", "B", "B", "C", "C"],
        unit_kind="cluster",
        estimator="OLS",
    ).value
    assert a["effective_replicates"] == 3
    slopes = [
        np.linalg.lstsq(np.column_stack([np.ones(len(i)), X[i]]), y[i], rcond=None)[0][1]
        for i in ix
    ]
    assert a["coefficient_intervals"][1] == pytest.approx(np.quantile(slopes, [0.025, 0.975]))
    with pytest.raises(CreditRiskError):
        resample_fit_uncertainty(
            X,
            y,
            feature_order=["x"],
            resampling_indices=[[0, 1, 2]],
            resampling_units=["A", "A", "B", "B", "C", "C"],
            unit_kind="cluster",
            estimator="OLS",
        )
    z = resample_fit_uncertainty(
        X,
        [0, 0, 0, 1, 1, 1],
        feature_order=["x"],
        resampling_indices=[list(range(6)), [0, 1, 0, 1]],
        resampling_units=["A", "A", "B", "B", "C", "C"],
        unit_kind="time_block",
    ).value
    assert z["failed_replicates"] == 2 and z["coefficient_intervals"] is None
    b = resample_fit_uncertainty(
        X,
        y,
        feature_order=["x"],
        resampling_indices=ix,
        resampling_units=["A", "A", "B", "B", "C", "C"],
        unit_kind="cluster",
        estimator="OLS",
        selection_spec={"candidate_groups": {"x": ["x"]}},
    ).value
    assert b["selection_repeated"]


def test_complete_protocol_test_shuffle_invariance_and_failure_reporting():
    f = protocol_fixture()
    args = {k: f[k] for k in ["feature_order", "split_plan", "metadata"]}
    candidates = [
        {"id": "signal", "features": ["signal"], "penalty_grid": [0, 0.1]},
        {"id": "bad", "features": ["missing"]},
    ]
    a = fit_model_selection_protocol(
        f["X"], f["y"], candidates=candidates, search_budget=2, **args
    ).value
    altered = np.array(f["y"])
    altered[300:] = 1 - altered[300:]
    b = fit_model_selection_protocol(
        f["X"], altered, candidates=candidates, search_budget=2, **args
    ).value
    assert a["selected_model"]["artifact_hash"] == b["selected_model"]["artifact_hash"]
    assert a["calibration_artifact"]["artifact_hash"] == b["calibration_artifact"]["artifact_hash"]
    assert a["candidate_evidence"][1]["failure"]
    assert a["test_metrics"] is None


def test_fast_real_gaussian_and_binomial_selection():
    test_each_stepwise_direction_criterion_estimator("forward", "AIC", "GLM")
    test_each_stepwise_direction_criterion_estimator("backward", "BIC", "WLS")


def test_exposure_lgd_and_grade_target_weights():
    meta = metadata()
    meta["weight_type"] = "exposure"
    a = calibrate_lgd_level([0.1, 0.9], 0.6, [0.25, 0.75], metadata=meta)
    assert np.average(apply(a, [0.1, 0.9]), weights=[0.25, 0.75]) == pytest.approx(0.6)
    b = calibrate_grade_pd(
        [100, 100], [10, 30], target_weights=[0.25, 0.75], target_default_rate=0.4
    ).value
    assert b["achieved_mean_pd"] == pytest.approx(0.4)


def test_protocol_common_frozen_prediction_roundtrip_and_tamper(tmp_path):
    from creditratingsystem.classification import predict_pd
    from creditratingsystem.core import serialize_fitted_model, load_fitted_model

    f = protocol_fixture()
    args = {k: f[k] for k in ["feature_order", "split_plan", "metadata"]}
    a = fit_model_selection_protocol(
        f["X"],
        f["y"],
        candidates=[{"id": "signal", "features": ["signal"], "penalty_grid": [0, 0.1]}],
        search_budget=1,
        **args,
    ).value
    model = a["selected_frozen_model"]
    X = np.array([[-1.0], [1.0]])
    expected = expit(
        a["selected_model"]["intercept"]
        + ((X - np.array(a["selected_model"]["center"])) / np.array(a["selected_model"]["scale"]))
        @ np.array(a["selected_model"]["coefficients"])
    )
    result = predict_pd(model, X, feature_order=["signal"], expected_target=metadata()["target"])
    np.testing.assert_allclose(result.value, expected)
    path = tmp_path / "selected.json"
    serialize_fitted_model(model, path)
    loaded = load_fitted_model(path).value
    np.testing.assert_array_equal(
        predict_pd(loaded, X, feature_order=["signal"], expected_target=metadata()["target"]).value,
        result.value,
    )
    assert (
        model["training"]["known_at"] == "2013-01-01"
        and model["training"]["role"] == "development_only"
    )
    bad = deepcopy(model)
    bad["parameters"]["selection_model"]["intercept"] += 1
    with pytest.raises(CreditRiskError):
        predict_pd(bad, X, feature_order=["signal"], expected_target=metadata()["target"])
    with pytest.raises(CreditRiskError):
        predict_pd(model, X, feature_order=["wrong"], expected_target=metadata()["target"])
    with pytest.raises(CreditRiskError):
        predict_pd(
            model, X, feature_order=["signal"], expected_target=metadata()["target"], offset=[0, 0]
        )


def test_beta_exact_separation_and_rehashed_domain_failure():
    with pytest.raises(CreditRiskError) as error:
        calibrate_beta([0.1, 0.2, 0.7, 0.9], [0, 0, 1, 1], metadata=metadata())
    assert error.value.code == "SEPARATION"
    p = np.repeat([0.1, 0.2, 0.4, 0.7, 0.9], 2)
    fit = calibrate_beta(p, [1, 0] * 5, [1, 9, 2, 8, 4, 6, 7, 3, 9, 1], metadata=metadata()).value
    fit["parameters"]["a"] = -1
    fit.pop("artifact_hash")
    fit["artifact_hash"] = _fingerprint(fit)
    with pytest.raises(CreditRiskError) as error:
        apply(fit, [0.2])
    assert error.value.code == "INVALID_CALIBRATOR"
