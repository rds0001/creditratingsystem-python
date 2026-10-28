"""Analytical selection/missing-outcome oracles; genuine small likelihood fits."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from creditratingsystem.core import CreditRiskError
from creditratingsystem.selection_bias import (
    build_selection_dataset,
    fit_selection_propensity,
    selection_ipw_weights,
    fuzzy_reject_augmentation,
    parcel_rejected_applications,
    fit_joint_selection_model,
    estimate_population_risk_aipw,
    reject_inference_bounds,
    reject_inference_sensitivity,
    prepare_reject_evaluation,
    _bvn,
    _joint_objective,
    _predict_joint_pd,
)


def target(event="default", conditioning="POTENTIAL_OWN_CONTRACT"):
    return dict(
        event=event,
        horizon=12,
        time_unit="months",
        probability_measure="PHYSICAL",
        conditioning=conditioning,
    )


def training():
    return dict(
        known_at="2025-12-31", split_id="train", data_fingerprint="fixture", data_origin="SYNTHETIC"
    )


def row_metadata(n):
    return [
        dict(
            entity_id=f"e{i}",
            as_of="2020-01-01" if i < n // 2 else "2022-01-01",
            label_known_at="2021-01-01" if i < n // 2 else "2023-01-01",
            features_known_at="2019-12-31" if i < n // 2 else "2021-12-31",
        )
        for i in range(n)
    ]


def lineage(n):
    return dict(
        rows=row_metadata(n),
        folds=[dict(train=list(range(n // 2)), validation=list(range(n // 2, n)))],
        nuisance_prediction_origin="OUT_OF_FOLD",
    )


def joint_data():
    # Eight independent binomial cells, no separation, explicit missing reject labels.
    x = []
    z = []
    s = []
    y = []
    for xx in [-1.0, 1.0]:
        for zz in [-1.0, 1.0]:
            selected = 8 if zz < 0 else 12
            bad = 2 if xx < 0 else 6
            for i in range(20):
                x.append([xx])
                z.append([zz])
                s.append(int(i < selected))
                y.append(int(i < bad) if i < selected else None)
    return np.array(x), np.array(z), s, y


def joint_fit(rho_grid=(0,), selected_rho=0):
    x, z, s, y = joint_data()
    return fit_joint_selection_model(
        x,
        z,
        s,
        y,
        outcome_features=["x"],
        selection_features=["instrument"],
        target=target(),
        training=training(),
        model_id="joint",
        model_version="1",
        rho_grid=rho_grid,
        selected_rho=selected_rho,
    )


def test_stages_preserve_counterfactual_missingness_and_nonuptake():
    apps = [
        dict(application_id=str(i), entity_id=str(i), submitted_at="2020-01-01") for i in range(4)
    ]
    decisions = [
        dict(
            application_id=str(i),
            as_of="2020-01-02",
            known_at="2020-01-02",
            decision="rejected" if i == 0 else "offered",
            policy_version="p1",
        )
        for i in range(4)
    ]
    books = [
        dict(
            application_id=str(i),
            as_of="2020-01-03",
            known_at="2020-01-03",
            taken_up=i > 1,
            funded=i > 1,
        )
        for i in range(1, 4)
    ]
    outcomes = [
        dict(
            application_id="3",
            as_of="2021-01-03",
            known_at="2021-01-04",
            outcome_scope="own_contract",
            status="observed",
            outcome=1,
            target=target(),
        )
    ]
    v = build_selection_dataset(
        apps,
        decisions,
        books,
        outcomes,
        as_of="2021-02-01",
        target_population="all applicants",
        target=target(),
    ).value
    assert [r["stratum"] for r in v["rows"]] == [
        "rejected",
        "offered_not_taken_up",
        "funded_immature",
        "funded_observed",
    ]
    assert [r["outcome"] for r in v["rows"]] == [None, None, None, 1]
    outcomes[0]["application_id"] = "0"
    with pytest.raises(CreditRiskError, match="UNOBSERVED_OUTCOME"):
        build_selection_dataset(
            apps,
            decisions,
            books,
            outcomes,
            as_of="2021-02-01",
            target_population="all",
            target=target(),
        )


def test_sequential_ipw_hand_mass_ess_and_changed_estimand():
    r = selection_ipw_weights(
        [1, 1, 0], [[0.5, 0.5], [1, 0.5], [0.1, 0.2]], stabilization=[0.5, 0.5]
    ).value
    assert r["weights"] == [1, 0.5, 0]
    assert r["effective_sample_size"] == pytest.approx(1.8)
    trimmed = selection_ipw_weights([1, 1, 0], [0.25, 0.5, 0], trim=[0.1, 1]).value
    assert trimmed["excluded_target_mass"] == 1 and trimmed["estimand_changed"]
    with pytest.raises(CreditRiskError, match="POSITIVITY_FAILURE"):
        selection_ipw_weights([1, 0], [0.5, 0])
    assert selection_ipw_weights([1, 1], [1, 1]).value["weights"] == [1, 1]


def test_fuzzy_pair_mass_not_validation():
    rows = fuzzy_reject_augmentation(
        [1, None], [1, 0], [0.2, 0.3], application_ids=["a", "b"], augmentation_weight=2
    ).value["rows"]
    assert [r["weight"] for r in rows] == [1, 1.4, 0.6]
    assert rows[1]["validation_eligible"] is False and rows[2]["label_origin"] == "HYPOTHETICAL"
    with pytest.raises(CreditRiskError, match="UNOBSERVED_OUTCOME"):
        fuzzy_reject_augmentation([1, 0], [1, 0], [0.2, 0.3], application_ids=["a", "b"])


def test_parcels_log_odds_exact_and_shared_draws():
    v = parcel_rejected_applications(
        ["a", "b"],
        [2, 5],
        [10, 10],
        [3, 0],
        log_odds_shift=np.log(2),
        uniforms=[[0.1, 0.4, 0.9], []],
    ).value
    assert v["assumed_reject_bad_rate"] == pytest.approx([1 / 3, 2 / 3])
    assert v["hypothetical_labels"] == [[1, 0, 0], []]
    assert v["hypothetical_default_mass"] == pytest.approx([1, 0])


def test_sharp_weighted_bounds_with_stratum_restrictions():
    r = reject_inference_bounds([1, 0, None, None], [1, 1, 0, 0], target_weights=[2, 1, 3, 4]).value
    assert (r["lower"], r["upper"]) == (0.2, 0.9)
    r = reject_inference_bounds(
        [1, 0, None, None],
        [1, 1, 0, 0],
        target_weights=[2, 1, 3, 4],
        strata=["a", "a", "a", "b"],
        risk_bounds={"a": [0.2, 0.4], "b": [0.5, 0.6]},
    ).value
    assert r["lower"] == pytest.approx(0.46) and r["upper"] == pytest.approx(0.56)
    assert reject_inference_bounds([1, 0], [1, 1]).value["lower"] == 0.5


def test_sensitivity_preserves_actual_and_stable_boundary():
    r = reject_inference_sensitivity(
        [0.9, 0.2, 0, 1],
        [1, 0, 0, 0],
        [0, None, None, None],
        log_odds_shifts=[-1000, 0, np.log(4), 1000],
        threshold=0.35,
    ).value
    assert [s["population_pd"] for s in r["surface"]] == pytest.approx([0.25, 0.3, 0.375, 0.5])
    assert r["crossover_brackets"] == [[0, np.log(4)]] and not r["validation_eligible"]


def test_aipw_hand_formula_cluster_variance_and_all_observed_mean():
    n = 8
    lin = lineage(n)
    w = [0] * 4 + [1] * 4
    result = estimate_population_risk_aipw(
        [0, 1, 0, 1, 1, None, 0, None],
        [1, 1, 1, 1, 1, 0, 1, 0],
        [0.25] * 8,
        [0.5] * 8,
        fold_lineage=lin,
        target_weights=w,
        clusters=["warm"] * 4 + ["a", "a", "b", "b"],
    ).value
    assert result["augmented_contributions"][4:] == [1.75, 0.25, -0.25, 0.25]
    assert result["population_pd"] == 0.5
    assert result["standard_error"] == 0.5
    r = estimate_population_risk_aipw(
        [0, 1, 0, 1, 1, 0, 0, 1], [1] * 8, [0.2] * 8, [1] * 8, fold_lineage=lin, target_weights=w
    ).value
    assert r["population_pd"] == 0.5
    lin["rows"][4]["entity_id"] = "e0"
    with pytest.raises(CreditRiskError, match="LEAKAGE"):
        estimate_population_risk_aipw(
            [0] * 8, [1] * 8, [0.2] * 8, [1] * 8, fold_lineage=lin, target_weights=w
        )


def test_intercept_selection_rate_forward_prediction():
    n = 16
    s = [0, 0, 0, 1] * 4
    fit = fit_selection_propensity(
        np.empty((n, 0)),
        s,
        feature_order=[],
        mechanism="funding",
        folds=lineage(n)["folds"],
        row_metadata=row_metadata(n),
        target=target("funding", "eligible_offers"),
        training=training(),
        model_id="prop",
        model_version="1",
    )
    assert fit.diagnostics["out_of_fold_propensity"][:8] == [None] * 8
    assert fit.diagnostics["out_of_fold_propensity"][8:] == pytest.approx([0.25] * 8)
    assert fit.model["parameters"]["coefficients"] == pytest.approx([np.log(1 / 3)])


def test_deterministic_policy_has_no_finite_overlap_fit():
    x = np.array([[-1.0], [1.0]] * 8)
    s = [0, 1] * 8
    with pytest.raises(CreditRiskError, match="SEPARATION"):
        fit_selection_propensity(
            x,
            s,
            feature_order=["policy_score"],
            mechanism="decision",
            folds=lineage(16)["folds"],
            row_metadata=row_metadata(16),
            target=target("decision", "eligible"),
            training=training(),
            model_id="p",
            model_version="1",
        )


def test_bivariate_cdf_arcsine_and_conditional_integral_oracles():
    for rho in [-0.95, -0.4, 0, 0.6, 0.95]:
        assert float(_bvn(0, 0, rho)) == pytest.approx(
            0.25 + np.arcsin(rho) / (2 * np.pi), abs=2e-13
        )
        for a, b in [(-1, 0.3), (0.4, -0.7), (2, 1)]:
            ref = quad(
                lambda t: norm.pdf(t) * norm.cdf((b - rho * t) / np.sqrt(1 - rho * rho)),
                -np.inf,
                a,
                epsabs=1e-12,
            )[0]
            assert float(_bvn(a, b, rho)) == pytest.approx(ref, abs=3e-12)


def test_joint_score_derivative_and_rho_zero_separate_probits():
    x, z, s, y = joint_data()
    d = np.c_[np.ones(len(x)), x]
    zz = np.c_[np.ones(len(z)), z]
    yy = np.array([0 if v is None else v for v in y])
    s = np.array(s)
    theta = np.array([-0.4, 0.5, 0.2, 0.3, 0.25])
    val, grad = _joint_objective(theta, d, zz, s, yy, None)
    h = 1e-5
    numeric = [
        (
            _joint_objective(theta + np.eye(5)[i] * h, d, zz, s, yy, None)[0]
            - _joint_objective(theta - np.eye(5)[i] * h, d, zz, s, yy, None)[0]
        )
        / (2 * h)
        for i in range(5)
    ]
    assert grad == pytest.approx(numeric, abs=1e-7)
    fit = joint_fit()
    p = fit.model["parameters"]
    assert p["outcome_coefficients"] == pytest.approx(
        [(norm.ppf(0.2) + norm.ppf(0.6)) / 2, (norm.ppf(0.6) - norm.ppf(0.2)) / 2], abs=2e-7
    )
    assert p["selection_coefficients"] == pytest.approx([0, norm.ppf(0.6)], abs=2e-7)
    result = _predict_joint_pd(
        fit, np.c_[x, z], feature_order=["x", "instrument"], expected_target=target()
    ).value
    assert result[:40] == pytest.approx([0.2] * 40, abs=1e-7)
    assert result[40:] == pytest.approx([0.6] * 40, abs=1e-7)


def test_evaluation_pseudo_and_external_are_not_own_loan_truth():
    candidate = [
        dict(
            model_id="m",
            model_hash="frozen",
            training_known_at="2020-01-01",
            training_entities=["trained"],
        )
    ]
    rows = [
        dict(
            application_id=str(i),
            entity_id=str(i),
            as_of="2021-01-01",
            known_at="2022-01-01",
            label_origin=origin,
            outcome=0,
            target=target(),
            funded=True,
            mature=True,
        )
        for i, origin in enumerate(["PSEUDO", "EXTERNAL_CONTRACT", "OBSERVED_OWN_CONTRACT"])
    ]
    r = prepare_reject_evaluation(candidate, rows, as_of="2023-01-01", target=target()).value
    assert [v["validation_eligible"] for v in r["rows"]] == [False, False, True]
    rows[2]["entity_id"] = "trained"
    assert not prepare_reject_evaluation(
        candidate, rows, as_of="2023-01-01", target=target()
    ).value["validation_identified"]
    rows = rows[:1]
    rows[0].update(label_origin="SYNTHETIC_TRUTH", data_origin="SYNTHETIC")
    assert (
        prepare_reject_evaluation(candidate, rows, as_of="2023-01-01", target=target()).provenance[
            "data_origin"
        ]
        == "SYNTHETIC"
    )


@pytest.mark.extended
def test_fixed_rho_profiles_are_scenarios_not_validation():
    fit = joint_fit((-0.4, 0, 0.4), 0)
    assert len(fit.diagnostics["profiles"]) == 3
    assert fit.diagnostics["mode"] == "FIXED_RHO_SENSITIVITY"
    assert fit.diagnostics["selected_rho"] == 0
    assert fit.diagnostics["profiles"][1]["model_population_pd"] == pytest.approx(0.4)
    assert all(
        p["risk_scope"] == "MODEL_ASSUMED_POTENTIAL_OWN_CONTRACT"
        for p in fit.diagnostics["profiles"]
    )


@pytest.mark.extended
def test_estimated_rho_recovers_independently_integrated_cell_counts():
    # Counts rounded from independent conditional-normal integration with
    # beta=(-.4,.5), gamma=(.1,.8), rho=.35, 100 trials in each of 9 cells.
    counts = [
        (24, 8),
        (54, 14),
        (82, 17),
        (24, 13),
        (54, 24),
        (82, 31),
        (24, 17),
        (54, 35),
        (82, 48),
    ]
    x = []
    z = []
    s = []
    y = []
    for (xx, zz), (selected, bad) in zip(
        [(a, b) for a in [-1.0, 0.0, 1.0] for b in [-1.0, 0.0, 1.0]], counts
    ):
        for i in range(100):
            x.append([xx])
            z.append([zz])
            s.append(int(i < selected))
            y.append(int(i < bad) if i < selected else None)
    fit = fit_joint_selection_model(
        x,
        z,
        s,
        y,
        outcome_features=["x"],
        selection_features=["instrument"],
        target=target(),
        training=training(),
        model_id="joint",
        model_version="1",
        exclusion={
            "features": ["instrument"],
            "justification": "Independent synthetic selection-only instrument.",
        },
    )
    assert fit.model["parameters"]["rho"] == pytest.approx(0.35, abs=0.03)
    assert fit.model["parameters"]["outcome_coefficients"] == pytest.approx([-0.4, 0.5], abs=0.012)
    assert fit.model["parameters"]["selection_coefficients"] == pytest.approx([0.1, 0.8], abs=0.012)
    assert not fit.diagnostics["weak_identification"]
    assert fit.diagnostics["mode"] == "ESTIMATED_RHO_ASSUMED_EXCLUSION"


def test_leakage_and_false_evidence_guards():
    meta = row_metadata(16)
    meta[8]["features_known_at"] = "2023-01-01"
    with pytest.raises(CreditRiskError, match="LEAKAGE"):
        fit_selection_propensity(
            np.empty((16, 0)),
            [0, 0, 0, 1] * 4,
            feature_order=[],
            mechanism="funding",
            folds=lineage(16)["folds"],
            row_metadata=meta,
            target=target("funding", "eligible"),
            training=training(),
            model_id="p",
            model_version="1",
        )
    x, z, s, y = joint_data()
    with pytest.raises(CreditRiskError, match="IDENTIFICATION_REQUIRED"):
        fit_joint_selection_model(
            x,
            z,
            s,
            y,
            outcome_features=["x"],
            selection_features=["z"],
            target=target(),
            training=training(),
            model_id="m",
            model_version="1",
        )
    with pytest.raises(CreditRiskError, match="INVALID_GRID"):
        joint_fit((0, 0.4), None)
    with pytest.raises(CreditRiskError, match="LEAKAGE"):
        estimate_population_risk_aipw([0] * 8, [1] * 8, [0.2] * 8, [1] * 8, fold_lineage=lineage(8))
    candidate = [dict(model_id="m", model_hash="hash", training_known_at="2020-01-01")]
    with pytest.raises(CreditRiskError, match="INVALID_SPLIT"):
        prepare_reject_evaluation(candidate, [], as_of="2023-01-01", target=target())


def test_frozen_joint_roundtrip(tmp_path):
    from creditratingsystem.core import serialize_fitted_model, load_fitted_model

    fit = joint_fit()
    x = np.array([[-1, 1], [1, -1]])
    path = tmp_path / "joint.json"
    serialize_fitted_model(fit, path)
    loaded = load_fitted_model(path).value
    assert _predict_joint_pd(
        loaded, x, feature_order=["x", "instrument"], expected_target=target()
    ).value == pytest.approx([0.2, 0.6], abs=1e-7)
    import json

    envelope = json.loads(path.read_text())
    envelope["model"]["parameters"]["rho"] = 0.5
    path.write_text(json.dumps(envelope))
    with pytest.raises(CreditRiskError, match="INTEGRITY_ERROR"):
        load_fitted_model(path)


def test_double_robust_mean_in_exact_probability_mass_population():
    # Enumerate (S=1,Y=1), (S=1,Y=0), (S=0) probabilities in two
    # strata. Their true risks are .2 and .6, equally weighted.
    outcomes = [0] * 6 + [1, 0, None, 1, 0, None]
    observed = [1] * 6 + [1, 1, 0, 1, 1, 0]
    weights = [0] * 6 + [0.1, 0.4, 0.5, 0.15, 0.1, 0.75]
    e = [0.5] * 6 + [0.5] * 3 + [0.25] * 3

    def estimate(m, propensity):
        return estimate_population_risk_aipw(
            outcomes, observed, m, propensity, fold_lineage=lineage(12), target_weights=weights
        ).value["population_pd"]

    assert estimate([0.8] * 12, e) == pytest.approx(0.4)
    assert estimate([0.2] * 6 + [0.2] * 3 + [0.6] * 3, [0.8] * 12) == pytest.approx(0.4)
    assert estimate([0.8] * 12, [0.8] * 12) == pytest.approx(0.58125)
    assert selection_ipw_weights([1, 1], [1, 1], stabilization=[1]).value["weights"] == [1, 1]
