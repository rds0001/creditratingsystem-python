from copy import deepcopy
import numpy as np
import pytest
from creditratingsystem.core import CreditRiskError
from creditratingsystem.macro import (
    fit_satellite_model,
    apply_satellite_model,
    simulate_macro_scenarios,
)
from test_macro import history, target, fitted, innovations


def satellite_fixture():
    x = np.sin(np.arange(26) * 0.71) + np.arange(26) * 0.03
    h = history(x[:, None])
    records = []
    for i in range(1, 26):
        records.append(
            dict(
                observation_id=f"o{i}",
                entity_id="portfolio",
                date=h["dates"][i],
                information_date=h["dates"][i],
                known_at=h["dates"][i],
                value=1 + 2 * x[i - 1] + 0.1 * np.sin(i * 1.13),
            )
        )
    tr = dict(
        start=h["dates"][1],
        end=h["dates"][20],
        known_at=h["dates"][20],
        split_id="satellite_split",
        data_origin="SYNTHETIC",
    )
    split = dict(
        split_id="satellite_split",
        mode="TEMPORAL",
        train_ids=[f"o{i}" for i in range(1, 21)],
        validation_ids=[f"o{i}" for i in range(21, 26)],
        training_cutoff=tr["known_at"],
    )
    spec = [
        dict(feature_id="x0_lag1", variable_id="x0", lag=1, unit="index", expected_sign="positive")
    ]
    args = dict(
        feature_spec=spec,
        estimator="OLS",
        link="identity",
        split_plan=split,
        training=tr,
        target=target(),
        target_unit="currency_units",
        model_id="satellite",
        model_version="1",
        residual_process="none",
    )
    return records, h, args


def test_satellite_training_only_lag_regression_and_zero_delta_baseline():
    records, h, args = satellite_fixture()
    fit = fit_satellite_model(records, h, **args)
    x = np.c_[np.ones(20), np.asarray(h["values"])[:20, 0]]
    y = np.array([r["value"] for r in records[:20]])
    oracle = np.linalg.solve(x.T @ x, x.T @ y)
    assert fit.model["parameters"]["coefficients"] == pytest.approx(oracle)
    changed = deepcopy(records)
    for r in changed[20:]:
        r["value"] = "deliberately unobserved holdout"
    refit = fit_satellite_model(changed, h, **args)
    assert fit.model == refit.model
    m = fitted()
    m["training"]["known_at"] = args["training"]["known_at"]
    m["preprocessing"]["training_end"] = args["training"]["end"]
    cube = simulate_macro_scenarios(m, innovations(np.zeros((2, 2, 1)))).value
    draws = dict(
        kind="STANDARD_NORMAL",
        scenario_ids=cube["scenario_ids"],
        time_grid=cube["time_grid"],
        values=[[0, 0], [0, 0]],
    )
    state = dict(
        as_of=cube["origin"],
        known_at=cube["information_cutoff"],
        mode="baseline_delta",
        macro_history=[[0]],
        baseline_link_path=[10, 11],
        reference_feature_path=[[0], [0]],
        residual_state=0,
    )
    result = apply_satellite_model(fit.model, cube, draws, state)
    assert np.asarray(result.value["values"])[:, :, 0] == pytest.approx(
        np.array([[10, 11], [10, 11]])
    )
    state["mode"] = "absolute"
    state["macro_history"] = [[3]]
    result = apply_satellite_model(fit.model, cube, draws, state)
    assert np.asarray(result.value["values"])[:, :, 0] == pytest.approx(
        np.tile([oracle[0] + 3 * oracle[1], oracle[0]], (2, 1))
    )


def test_satellite_release_and_split_leakage_guards():
    rows, h, args = satellite_fixture()
    h["release_dates"][0][0] = "2010-01-01"
    with pytest.raises(CreditRiskError, match="LEAKAGE"):
        fit_satellite_model(rows, h, **args)
    rows, h, args = satellite_fixture()
    args["split_plan"]["mode"] = "ENTITY_TIME"
    with pytest.raises(CreditRiskError, match="LEAKAGE"):
        fit_satellite_model(rows, h, **args)


@pytest.mark.parametrize("covariance", ["classical", "HC1", "HAC"])
def test_wls_covariance_independent_sandwich(covariance):
    rows, h, args = satellite_fixture()
    w = np.linspace(0.5, 2, 20)
    args.update(
        estimator="WLS",
        weights={f"o{i}": float(w[i - 1]) for i in range(1, 21)},
        covariance=covariance,
        hac_lags=2,
    )
    fit = fit_satellite_model(rows, h, **args)
    x = np.c_[np.ones(20), np.asarray(h["values"])[:20, 0]]
    y = np.array([r["value"] for r in rows[:20]])
    bread = np.linalg.inv(x.T @ (w[:, None] * x))
    b = bread @ x.T @ (w * y)
    e = y - x @ b
    scores = x * (w * e)[:, None]
    if covariance == "classical":
        expected = bread * np.sum(w * e * e) / 18
    else:
        meat = scores.T @ scores
        if covariance == "HC1":
            meat *= 20 / 18
        else:
            for lag in (1, 2):
                cross = scores[lag:].T @ scores[:-lag]
                meat += (1 - lag / 3) * (cross + cross.T)
        expected = bread @ meat @ bread
    assert fit.model["parameters"]["coefficients"] == pytest.approx(b)
    assert np.array(fit.model["parameters"]["covariance"]) == pytest.approx(expected, abs=1e-12)


def test_binary_satellite_real_fit_and_no_invented_residual_variance():
    rows, h, args = satellite_fixture()
    for i, r in enumerate(rows):
        r["value"] = int((i % 4) in (0, 2))
    args.update(estimator="GLM", link="logit", target_unit="probability")
    fit = fit_satellite_model(rows, h, **args)
    assert fit.model["parameters"]["residual_sigma"] == 0
    assert np.all(np.isfinite(fit.model["parameters"]["coefficients"]))


def test_satellite_rejects_future_fitted_macro_transform():
    rows, h, args = satellite_fixture()
    h["fit_window"] = dict(start=h["dates"][0], end=h["dates"][-1], known_at=h["dates"][-1])
    with pytest.raises(CreditRiskError, match="LEAKAGE"):
        fit_satellite_model(rows, h, **args)
