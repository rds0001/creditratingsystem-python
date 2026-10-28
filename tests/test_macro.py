"""Calendar/vintage, independent regression and joint Gaussian path oracles."""

from copy import deepcopy
import numpy as np
import pytest
from creditratingsystem.core import CreditRiskError
from creditratingsystem.macro import (
    transform_macro_history,
    fit_var,
    block_bootstrap_scenarios,
    simulate_macro_scenarios,
    condition_macro_paths,
    reverse_stress_scenario,
)


def history(values):
    values = np.asarray(values)
    n, k = values.shape
    dates = [f"{2000 + i // 12:04d}-{i % 12 + 1:02d}-01" for i in range(n)]
    return dict(
        dates=dates,
        values=values.tolist(),
        variable_ids=[f"x{j}" for j in range(k)],
        frequency="monthly",
        release_dates=[[d] * k for d in dates],
        units=["index"] * k,
    )


def training(h):
    return dict(
        start=h["dates"][0],
        end=h["dates"][-1],
        known_at=h["dates"][-1],
        split_id="macro_training",
        data_origin="SYNTHETIC",
    )


def target():
    return dict(
        event="macro_state",
        horizon=1,
        time_unit="months",
        probability_measure="PHYSICAL",
        conditioning="PIT",
    )


def fitted():
    h = history([[0], [1], [0], [-1], [0], [1], [0], [-1]])
    fit = fit_var(
        h,
        lags=1,
        training=training(h),
        target=target(),
        model_id="ar",
        model_version="1",
        covariance_estimator="mle",
    )
    # This independent frozen AR(1) reference is deliberately not claimed fitted.
    m = deepcopy(fit.model)
    m["parameters"].update(
        lag_coefficients=[[[0.5]]],
        intercept=[0],
        trend=[0],
        covariance=[[1]],
        initial_history=[[0]],
    )
    m["provenance"]["parameter_origin"] = "USER_SUPPLIED"
    return m


def innovations(values):
    v = np.asarray(values, dtype=float)
    return dict(
        kind="STANDARD_NORMAL",
        values=v.tolist(),
        factor_ids=[f"x{i}" for i in range(v.shape[2])],
        scenario_ids=[f"s{i}" for i in range(v.shape[0])],
        time_grid=(np.arange(1, v.shape[1] + 1) / 12).tolist(),
        weights=[1 / v.shape[0]] * v.shape[0],
    )


def test_vintage_fit_excludes_revisions_and_inverse_anchors():
    records = [
        dict(
            variable_id="gdp",
            date=f"2000-0{i + 1}-01",
            release_date=f"2000-0{i + 1}-01",
            value=float(100 + 10 * i),
        )
        for i in range(5)
    ]
    records += [dict(variable_id="gdp", date="2000-02-01", release_date="2000-05-01", value=500.0)]
    window = dict(start="2000-02-01", end="2000-03-01", known_at="2000-03-01")
    h = transform_macro_history(
        records,
        [dict(variable_id="gdp", transform="log_difference", standardize=False, unit="index")],
        frequency="monthly",
        fit_window=window,
        information_cutoff="2000-05-01",
    ).value
    assert np.asarray(h["training_values"])[:, 0] == pytest.approx(np.log([1.1, 120 / 110]))
    assert h["inverse_state"][0]["training_last_original_value"] == 120
    assert h["inverse_state"][0]["last_original_value"] == 140
    assert h["units"] == ["log_ratio"]
    reconstructed = 100 * np.exp(np.cumsum(np.asarray(h["values"])[:, 0]))
    assert reconstructed == pytest.approx([500, 120, 130, 140])
    bad = deepcopy(records)
    bad[0]["value"] = -1
    with pytest.raises(CreditRiskError, match="MACRO_LOG_DOMAIN"):
        transform_macro_history(
            bad,
            [dict(variable_id="gdp", transform="log", standardize=False, unit="index")],
            frequency="monthly",
            fit_window=window,
            information_cutoff="2000-05-01",
        )


def test_var_multivariate_matches_independent_normal_equations():
    x = np.zeros((50, 2))
    for i in range(1, len(x)):
        x[i] = (
            [0.2, -0.1]
            + np.array([[0.6, 0.1], [-0.2, 0.4]]) @ x[i - 1]
            + [np.sin(1.31 * i), np.cos(0.73 * i)]
        )
    h = history(x)
    fit = fit_var(
        h,
        lags=2,
        deterministic="constant_trend",
        training=training(h),
        target=target(),
        model_id="var",
        model_version="1",
    )
    d = np.c_[np.ones(48), np.arange(3, 51), x[1:-1], x[:-2]]
    oracle = np.linalg.solve(d.T @ d, d.T @ x[2:])
    assert fit.model["parameters"]["intercept"] == pytest.approx(oracle[0], abs=1e-9)
    assert fit.model["parameters"]["trend"] == pytest.approx(oracle[1], abs=1e-9)
    assert np.asarray(fit.model["parameters"]["lag_coefficients"]) == pytest.approx(
        np.array([oracle[2:4].T, oracle[4:6].T]), abs=1e-9
    )
    r = x[2:] - d @ oracle
    assert np.asarray(fit.model["parameters"]["covariance"]) == pytest.approx(
        r.T @ r / (48 - 6), abs=1e-10
    )


def test_explosive_fit_is_flagged_without_repair():
    x = (1.05 ** np.arange(20))[:, None]
    h = history(x)
    m = fit_var(
        h,
        lags=1,
        deterministic="none",
        training=training(h),
        target=target(),
        model_id="explosive",
        model_version="1",
    )
    assert m.diagnostics["spectral_radius"] == pytest.approx(1.05)
    assert not m.diagnostics["stable"] and m.diagnostics["eigenvalue_repair"] == "NONE"


def test_shared_path_recursion_and_gaussian_bridge_change_unconstrained_time():
    model = fitted()
    draws = innovations([[[1], [0]], [[0], [0]]])
    paths = np.asarray(simulate_macro_scenarios(model, draws).value["values"])[:, :, 0]
    assert paths == pytest.approx(np.array([[1, 0.5], [0, 0]]))
    conditional = condition_macro_paths(
        model, draws, [dict(time_index=1, factor_id="x0", value=2.0)]
    )
    values = np.asarray(conditional.value["values"])[:, :, 0]
    # Var(Y2)=1.25; Cov(Y1,Y2)=.5, so gain=.4. Overwrite would leave Y1 unchanged.
    assert values == pytest.approx(np.array([[1.6, 2], [0.8, 2]]))
    assert np.asarray(conditional.diagnostics["conditional_covariance"]) == pytest.approx(
        np.array([[0.8, 0], [0, 0]]), abs=1e-12
    )
    assert conditional.diagnostics["density_is_not_event_probability"]
    assert (
        condition_macro_paths(model, draws, []).value
        == simulate_macro_scenarios(model, draws).value
    )


@pytest.mark.parametrize("scheme", ["moving", "circular", "stationary"])
def test_joint_block_bootstrap_exact_indexes(scheme):
    draws = np.array([[[0.8, 0.9], [0.1, 0.9], [0.2, 0.1], [0.7, 0.9]]])
    v = block_bootstrap_scenarios(
        [[0, 10], [1, 11], [2, 12], [3, 13]],
        draws,
        factor_ids=["x0", "x1"],
        scenario_ids=["s0"],
        time_grid=[1 / 12, 2 / 12, 3 / 12, 4 / 12],
        frequency="monthly",
        block_length=2,
        scheme=scheme,
    )
    expected = [2, 3, 0, 1] if scheme == "moving" else [3, 0, 0, 1]
    assert v.diagnostics["selected_indices"] == [expected]
    cube = np.asarray(v.value["values"])[0]
    assert cube[:, 0] == pytest.approx(expected)
    assert cube[:, 1] - cube[:, 0] == pytest.approx([10] * 4)


def test_conditional_deterministic_support_and_reverse_linear_oracle():
    model = fitted()
    zero = deepcopy(model)
    zero["parameters"]["covariance"] = [[0]]
    with pytest.raises(CreditRiskError, match="INFEASIBLE_CONSTRAINT"):
        condition_macro_paths(
            zero, innovations([[[0]]]), [dict(time_index=0, factor_id="x0", value=1)]
        )
    response = dict(kind="linear_predictor", coefficients=[[2]], intercept=1, link="identity")
    result = reverse_stress_scenario(model, response, horizon=1, target_loss=5, shock_metric=[[1]])
    assert result.value["feasible"] and result.value["shocks"][0][0] == pytest.approx(2, abs=2e-6)
    assert result.value["objective"] == pytest.approx(2, abs=5e-6)
    failed = reverse_stress_scenario(
        model, response, horizon=1, target_loss=5, shock_metric=[[1]], shock_bounds=[[-1, 1]]
    )
    assert not failed.value["feasible"] and failed.value["achieved_loss"] <= 3 + 1e-10


def cointegrated_history():
    i = np.arange(100, dtype=float)
    common = np.cumsum(0.1 + np.sin(i * np.sqrt(2)) + 0.3 * np.sin(0.13 * i * i))
    stationary = np.zeros(100)
    for t in range(1, 100):
        stationary[t] = (
            0.3 * stationary[t - 1] + 0.4 * np.cos(t * np.sqrt(3)) + 0.1 * np.sin(0.19 * t * t)
        )
    return history(np.c_[common, common + stationary])


def test_vecm_johansen_level_conversion_and_rank_zero_difference_var():
    from creditratingsystem.macro import fit_vecm
    from statsmodels.tsa.vector_ar.vecm import VECM

    h = cointegrated_history()
    x = np.asarray(h["values"])
    fit = fit_vecm(
        h, rank=1, lags=2, training=training(h), target=target(), model_id="vecm", model_version="1"
    )
    assert np.asarray(fit.model["parameters"]["beta"])[:, 0] == pytest.approx([1, -1], abs=0.025)
    draws = innovations(np.zeros((1, 4, 2)))
    forecast = np.asarray(simulate_macro_scenarios(fit, draws).value["values"])[0]
    backend = VECM(x, k_ar_diff=1, coint_rank=1, deterministic="ci").fit()
    assert forecast == pytest.approx(backend.predict(steps=4), abs=1e-8)
    zero = fit_vecm(
        h,
        rank=0,
        lags=2,
        deterministic="constant_outside",
        training=training(h),
        target=target(),
        model_id="rank0",
        model_version="1",
    )
    diff = np.diff(x, axis=0)
    d = np.c_[diff[:-1], np.ones(len(diff) - 1)]
    b = np.linalg.lstsq(d, diff[1:], rcond=None)[0]
    assert np.asarray(zero.model["parameters"]["lag_coefficients"])[0] == pytest.approx(
        np.eye(2) + b[:2].T, abs=1e-8
    )
    assert np.asarray(zero.model["parameters"]["lag_coefficients"])[1] == pytest.approx(
        -b[:2].T, abs=1e-8
    )


@pytest.mark.extended
@pytest.mark.parametrize(
    "deterministic",
    [
        "none",
        "constant_inside",
        "constant_outside",
        "constant_outside_trend_inside",
        "constant_trend_outside",
    ],
)
@pytest.mark.parametrize("lags", [1, 2])
def test_vecm_deterministic_variants_forecast_against_backend(deterministic, lags):
    from creditratingsystem.macro import fit_vecm
    from statsmodels.tsa.vector_ar.vecm import VECM

    h = cointegrated_history()
    mapping = dict(
        none="n",
        constant_inside="ci",
        constant_outside="co",
        constant_outside_trend_inside="coli",
        constant_trend_outside="colo",
    )
    fit = fit_vecm(
        h,
        rank=1,
        lags=lags,
        deterministic=deterministic,
        training=training(h),
        target=target(),
        model_id="vecm",
        model_version="1",
    )
    got = np.asarray(
        simulate_macro_scenarios(fit, innovations(np.zeros((1, 3, 2)))).value["values"]
    )[0]
    expected = (
        VECM(
            np.asarray(h["values"]),
            k_ar_diff=lags - 1,
            coint_rank=1,
            deterministic=mapping[deterministic],
        )
        .fit()
        .predict(steps=3)
    )
    assert got == pytest.approx(expected, abs=2e-8)


def test_stationary_ttc_law_and_conditioning_uses_initial_uncertainty():
    m = fitted()
    z = innovations(np.zeros((2, 2, 1)))
    contract = dict(
        law="STATIONARY_GAUSSIAN",
        cycle_law_id="AR1_equilibrium",
        equilibrium_assumptions="Stable time-homogeneous Gaussian AR1",
        initial_state_draws=[[1], [-1]],
    )
    cube = simulate_macro_scenarios(m, z, sampling_contract=contract).value
    assert cube["sampling_contract"]["stationary_covariance"] == pytest.approx(np.array([[4 / 3]]))
    assert np.asarray(cube["values"])[:, :, 0] == pytest.approx(
        np.array([[1 / np.sqrt(3), 0.5 / np.sqrt(3)], [-1 / np.sqrt(3), -0.5 / np.sqrt(3)]])
    )
    conditioned = condition_macro_paths(
        m, z, [dict(time_index=1, factor_id="x0", value=2)], sampling_contract=contract
    )
    assert np.asarray(conditioned.diagnostics["conditional_mean"])[:, 0] == pytest.approx([1, 2])
    assert np.asarray(conditioned.diagnostics["conditional_covariance"]) == pytest.approx(
        np.array([[1, 0], [0, 0]]), abs=1e-12
    )
    assert conditioned.value["sampling_contract"]["conditioning_scope"] == "CONDITIONAL_STRESS"
    assert conditioned.value["sampling_contract"]["parent_conditioning_scope"] == "TTC_STATIONARY"
    m["parameters"]["lag_coefficients"] = [[[1]]]
    with pytest.raises(CreditRiskError, match="NONSTATIONARY_TTC"):
        simulate_macro_scenarios(m, z, sampling_contract=contract)


def test_declared_cycle_phase_mass_and_default_pit_are_explicit():
    m = fitted()
    z = innovations(np.zeros((2, 2, 1)))
    contract = dict(
        law="DECLARED_CYCLE_MIXTURE",
        cycle_law_id="bank_cycle_1",
        equilibrium_assumptions="Declared equal recession/expansion phase mass",
        phase_ids=["down", "up"],
        phase_probabilities={"down": 0.5, "up": 0.5},
    )
    cube = simulate_macro_scenarios(
        m, z, initial_state=[[[-2]], [[2]]], sampling_contract=contract
    ).value
    assert np.asarray(cube["values"])[:, :, 0] == pytest.approx(np.array([[-1, -0.5], [1, 0.5]]))
    assert not cube["sampling_contract"]["equilibrium_verified"]
    assert simulate_macro_scenarios(m, z).value["sampling_contract"]["conditioning_scope"] == "PIT"
    contract["phase_probabilities"] = {"down": 0.1, "up": 0.9}
    with pytest.raises(CreditRiskError, match="INVALID_WEIGHTS"):
        simulate_macro_scenarios(m, z, initial_state=[[[-2]], [[2]]], sampling_contract=contract)


def test_covariance_factor_is_unit_invariant_and_rejects_negative_variance():
    from creditratingsystem.macro import _sqrt_cov

    c = np.array([[1e-20, 0.5], [0.5, 1e20]])
    root, _ = _sqrt_cov(c)
    assert (root @ root.T) / np.sqrt(np.outer(np.diag(c), np.diag(c))) == pytest.approx(
        np.array([[1, 0.5], [0.5, 1]])
    )
    with pytest.raises(CreditRiskError, match="INVALID_COVARIANCE"):
        _sqrt_cov([[-1e-20]])
    m = fitted()
    m["parameters"]["covariance"] = [[1e-20]]
    z = innovations(np.zeros((2, 2, 1)))
    cond = condition_macro_paths(m, z, [dict(time_index=1, factor_id="x0", value=2e-10)])
    assert np.asarray(cond.diagnostics["conditional_mean"])[:, 0] / 1e-10 == pytest.approx([0.8, 2])


def test_hmm_filter_matches_enumeration_and_one_regime_matches_ols():
    from itertools import product
    from scipy.stats import norm
    from creditratingsystem._macro_models import _hmm_expectation
    from creditratingsystem.macro import fit_regime_switching

    y = np.array([[-1.0], [0.2], [1.0]])
    coef = np.array([[[-1]], [[1]]])
    cov = np.array([[[0.5]], [[0.8]]])
    p = np.array([[0.8, 0.2], [0.1, 0.9]])
    initial = np.array([0.6, 0.4])
    ll, gamma, _, filtered = _hmm_expectation(y, np.ones((3, 1)), coef, cov, p, initial)
    weights = []
    states = list(product(range(2), repeat=3))
    for state in states:
        w = initial[state[0]]
        for t in range(3):
            if t:
                w *= p[state[t - 1], state[t]]
            w *= norm.pdf(y[t, 0], coef[state[t], 0, 0], np.sqrt(cov[state[t], 0, 0]))
        weights.append(w)
    assert ll == pytest.approx(np.log(sum(weights)), abs=1e-12)
    posterior = np.array(
        [
            [sum(w for w, s in zip(weights, states) if s[t] == j) / sum(weights) for j in range(2)]
            for t in range(3)
        ]
    )
    assert gamma == pytest.approx(posterior, abs=1e-12)
    x = np.sin(np.arange(30) * 1.13) + 0.3 * np.cos(np.arange(30) * 0.47)
    h = history(x[:, None])
    start = dict(
        transition=[[1]], initial_probabilities=[1], coefficients=[[[0], [0]]], covariances=[[[1]]]
    )
    fit = fit_regime_switching(
        h,
        regime_count=1,
        lags=1,
        initialization=[start],
        training=training(h),
        target=target(),
        model_id="hmm1",
        model_version="1",
    )
    oracle = np.linalg.lstsq(np.c_[np.ones(29), x[:-1]], x[1:], rcond=None)[0]
    assert np.array(fit.model["parameters"]["coefficients"]).reshape(-1) == pytest.approx(oracle)
    assert fit.diagnostics["converged"]


def test_two_regime_fit_and_explicit_shared_regime_draws():
    from creditratingsystem.macro import fit_regime_switching

    i = np.arange(80)
    x = np.where((i // 10) % 2 == 0, -2.0, 2.0) + 0.25 * np.sin(i * 1.71)
    h = history(x[:, None])
    starts = [
        dict(
            transition=[[0.9, 0.1], [0.1, 0.9]],
            initial_probabilities=[0.5, 0.5],
            coefficients=[[[-1]], [[1]]],
            covariances=[[[1]], [[1]]],
        )
    ]
    fit = fit_regime_switching(
        h,
        regime_count=2,
        initialization=starts,
        training=training(h),
        target=target(),
        model_id="regimes",
        model_version="1",
    )
    assert fit.diagnostics["converged"]
    assert np.array(fit.diagnostics["filtered_probabilities"]).sum(axis=1) == pytest.approx(
        np.ones(80)
    )
    assert np.array(fit.model["parameters"]["coefficients"])[:, 0, 0] == pytest.approx(
        [-2, 2], abs=0.03
    )
    z = innovations(np.zeros((2, 3, 1)))
    u = [[0, 0, 0, 0], [0.99, 0.99, 0.99, 0.99]]
    sim = simulate_macro_scenarios(fit.model, z, regime_uniforms=u)
    assert sim.diagnostics["regime_paths"] == [[0, 0, 0], [1, 1, 1]]
    assert np.array(sim.value["values"])[:, :, 0] == pytest.approx(
        np.repeat(np.array(fit.model["parameters"]["coefficients"])[:, 0, 0, None], 3, axis=1)
    )


def test_dynamic_factor_kalman_likelihood_and_filter_match_joint_gaussian():
    from scipy.stats import multivariate_normal
    from creditratingsystem._macro_models import _factor_filter

    load = np.array([[1.0], [1.4]])
    mean = np.array([0.2, -0.3])
    ar = np.array([0.7, 0.2, -0.1])
    variance = np.array([0.4, 0.2, 0.3])
    design = np.c_[load, np.eye(2)]
    stationary = variance / (1 - ar**2)
    y = np.array([[0.4, -0.1], [np.nan, 0.2], [-0.3, 0.6], [0.1, np.nan]])
    observed = np.isfinite(y).reshape(-1)
    cov = np.block(
        [
            [design @ np.diag(stationary * ar ** abs(i - j)) @ design.T for j in range(4)]
            for i in range(4)
        ]
    )
    flat = y.reshape(-1)[observed]
    mu = np.tile(mean, 4)[observed]
    selected = cov[np.ix_(observed, observed)]
    result = _factor_filter(y, load, mean, ar, variance)
    assert result.llf == pytest.approx(
        multivariate_normal.logpdf(flat, mean=mu, cov=selected), abs=1e-10
    )
    cross = np.concatenate(
        [np.diag(stationary * ar ** (3 - t)) @ design.T for t in range(4)], axis=1
    )[:, observed]
    expected = cross @ np.linalg.solve(selected, flat - mu)
    expected_cov = np.diag(stationary) - cross @ np.linalg.solve(selected, cross.T)
    assert result.filtered_state[:, -1] == pytest.approx(expected, abs=1e-10)
    assert result.filtered_state_cov[:, :, -1] == pytest.approx(expected_cov, abs=1e-10)


def test_dynamic_factor_actual_fit_missingness_and_frozen_state_simulation():
    from creditratingsystem.macro import fit_dynamic_factor
    from creditratingsystem._macro_models import _factor_gaussian_moments

    i = np.arange(30)
    factor = np.sin(i * 0.53) + 0.3 * np.cos(i * 1.73)
    x = np.c_[factor + 0.3 * np.sin(i * 1.41), 1.4 * factor + 0.4 * np.cos(i * 0.93)]
    h = history(x)
    h["values"][8][0] = None
    initial = dict(
        loadings=[[1], [1.2]], mean=[0, 0], ar=[0.5, 0.1, 0.1], variances=[0.5, 0.2, 0.2]
    )
    fit = fit_dynamic_factor(
        h,
        factor_count=1,
        initialization=initial,
        max_iterations=80,
        training=training(h),
        target=target(),
        model_id="factor",
        model_version="1",
    )
    assert fit.diagnostics["log_likelihood"] > fit.diagnostics["initial_log_likelihood"] + 1
    assert fit.diagnostics["missing_entries"] == 1
    model = fit.model
    m = 3
    z = dict(
        kind="STANDARD_NORMAL",
        values=np.zeros((1, 2, m)).tolist(),
        factor_ids=model["parameters"]["state_ids"],
        scenario_ids=["s0"],
        time_grid=[1 / 12, 2 / 12],
        weights=[1],
    )
    contract = dict(law="FILTERED_GAUSSIAN", initial_state_draws=[[0] * m])
    cube = simulate_macro_scenarios(model, z, sampling_contract=contract).value
    mean, cov = _factor_gaussian_moments(model, 2)
    assert np.asarray(cube["values"]).reshape(-1) == pytest.approx(mean, abs=1e-10)
    assert np.min(np.linalg.eigvalsh(cov)) > -1e-8
    conditional = condition_macro_paths(
        model, z, [dict(time_index=1, factor_id="x0", value=0.5)], sampling_contract=contract
    )
    assert conditional.value["values"][0][1][0] == pytest.approx(0.5, abs=1e-8)


def test_latent_basis_rotation_preserves_observable_conditional_law():
    from creditratingsystem._macro_models import _factor_gaussian_moments

    m = fitted()
    m["method_id"] = "macro.fit_dynamic_factor"
    m["feature_order"] = ["x0", "x1"]
    m["preprocessing"]["units"] = ["index", "index"]
    p = dict(
        dynamics="dynamic_factor",
        state_ids=["f", "u0", "u1"],
        state_transition=np.diag([0.7, 0.2, -0.1]).tolist(),
        state_covariance=np.diag([0.4, 0.2, 0.3]).tolist(),
        observation_matrix=[[1, 1, 0], [1.4, 0, 1]],
        mean=[0.2, -0.3],
        filtered_state_end=[0.1, -0.2, 0.3],
        filtered_covariance_end=[[0.2, 0.01, 0], [0.01, 0.1, 0], [0, 0, 0.15]],
    )
    m["parameters"] = p
    original = _factor_gaussian_moments(m, 3)
    angle = 0.63
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    rotated = deepcopy(m)
    q = rotated["parameters"]
    q["state_transition"] = (rotation @ np.array(p["state_transition"]) @ rotation.T).tolist()
    q["state_covariance"] = (rotation @ np.array(p["state_covariance"]) @ rotation.T).tolist()
    q["observation_matrix"] = (np.array(p["observation_matrix"]) @ rotation.T).tolist()
    q["filtered_state_end"] = (rotation @ p["filtered_state_end"]).tolist()
    q["filtered_covariance_end"] = (
        rotation @ np.array(p["filtered_covariance_end"]) @ rotation.T
    ).tolist()
    transformed = _factor_gaussian_moments(rotated, 3)
    assert transformed[0] == pytest.approx(original[0], abs=1e-12)
    assert transformed[1] == pytest.approx(original[1], abs=1e-12)


def test_frozen_artifact_tamper_and_controlled_dynamics_shape(tmp_path):
    import json
    from creditratingsystem.core import serialize_fitted_model, load_fitted_model

    m = fitted()
    path = tmp_path / "macro.json"
    serialize_fitted_model(m, str(path))
    loaded = load_fitted_model(str(path)).value
    z = innovations(np.zeros((1, 2, 1)))
    assert simulate_macro_scenarios(loaded, z).value == simulate_macro_scenarios(m, z).value
    data = json.loads(path.read_text())
    data["model"]["parameters"]["intercept"] = [99]
    path.write_text(json.dumps(data))
    with pytest.raises(CreditRiskError, match="INTEGRITY_ERROR"):
        load_fitted_model(str(path))
    m["parameters"]["dynamics"] = "regime"
    with pytest.raises(CreditRiskError, match="INVALID_MODEL"):
        simulate_macro_scenarios(m, z)
