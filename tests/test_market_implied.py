from copy import deepcopy
from datetime import date, timedelta
import math
import numpy as np
import pytest
from scipy.integrate import quad
from creditratingsystem._market_pricing import (
    cds_expected_cashflow_legs,
    risky_bond_expected_cashflows,
)
from creditratingsystem.market_implied import bootstrap_cds_hazard_curve


def curve(rate=0.02, hazard=False):
    c = dict(
        as_of="2025-01-01",
        known_at="2025-01-01",
        currency="EUR",
        time_basis="ACT/365F",
        extrapolation="FLAT_LAST_RATE",
        knots=[1.0, 3.0],
        rates=[rate, rate],
    )
    if hazard:
        c["probability_measure"] = "RISK_NEUTRAL"
    return c


def day(t):
    return (date(2025, 1, 1) + timedelta(days=t)).isoformat()


def cds(days=365):
    return dict(
        as_of=day(0),
        currency="EUR",
        probability_measure="RISK_NEUTRAL",
        default_settlement="AT_DEFAULT",
        recovery=0.4,
        coupon=0.012,
        protection_start=day(0),
        maturity=day(days),
        settlement_date=day(0),
        accrued_premium=0.0,
        upfront=0.0,
        premium_mode="CONTINUOUS",
        instrument_type="CDS",
        issuer_id="A",
        seniority="SENIOR",
    )


def bond():
    return dict(
        as_of=day(0),
        currency="EUR",
        probability_measure="RISK_NEUTRAL",
        default_settlement="AT_DEFAULT",
        recovery=0.4,
        liquidity_spread=0.0,
        recovery_convention="PAR",
        cashflow_assumption="CONTRACTUAL_NONOPTIONAL",
        outstanding_principal=100.0,
        accrued_interest=0.0,
        settlement_date=day(0),
        schedule=[dict(payment_date=day(365), coupon_amount=5.0, principal_amount=100.0)],
        instrument_type="BOND",
        issuer_id="A",
        seniority="SENIOR",
    )


def test_flat_continuous_cds_and_zero_hazard():
    v = cds_expected_cashflow_legs(curve(0.02, True), curve(0.03), cds()).value
    assert v["par_spread"] == pytest.approx(0.012)
    assert v["upfront_residual_pv"] == pytest.approx(0, abs=1e-15)
    assert v["protection_pv"] == pytest.approx(0.6 * 0.02 * (-math.expm1(-0.05)) / 0.05)
    assert (
        cds_expected_cashflow_legs(curve(0, True), curve(0.03), cds()).value["protection_pv"] == 0
    )


def test_scheduled_accrual_independent_quadrature():
    h = curve(0.02, True)
    h["knots"] = [0.3, 1.0]
    h["rates"] = [0.02, 0.05]
    d = curve(0.03)
    c = cds()
    c.update(
        premium_mode="SCHEDULED",
        day_count="ACT/360",
        accrual_on_default=True,
        schedule=[
            dict(accrual_start=day(0), accrual_end=day(180), payment_date=day(182)),
            dict(accrual_start=day(180), accrual_end=day(365), payment_date=day(367)),
        ],
    )
    v = cds_expected_cashflow_legs(h, d, c).value

    def surv(t):
        return math.exp(-0.02 * min(t, 0.3) - 0.05 * max(0, t - 0.3))

    expected = sum(
        quad(
            lambda t: math.exp(-0.03 * t)
            * surv(t)
            * (0.02 if t < 0.3 else 0.05)
            * (t - a)
            * 365
            / 360,
            a,
            b,
            points=[x for x in [0.3] if a < x < b],
            epsabs=1e-13,
        )[0]
        for a, b in [(0, 180 / 365), (180 / 365, 1.0)]
    )
    assert v["accrual_on_default_annuity"] == pytest.approx(expected, abs=1e-13)
    expected_premium = 180 / 360 * math.exp(-0.03 * 182 / 365) * surv(
        180 / 365
    ) + 185 / 360 * math.exp(-0.03 * 367 / 365) * surv(1)
    assert v["premium_annuity"] == pytest.approx(expected_premium)


@pytest.mark.parametrize("convention", ["PAR", "TREASURY", "MARKET_VALUE"])
def test_distinct_recovery_algebra(convention):
    b = bond()
    b["recovery_convention"] = convention
    v = risky_bond_expected_cashflows(curve(0.02, True), curve(0.03), b).value
    expected = {
        "PAR": 105 * math.exp(-0.05) + 40 * 0.02 * (-math.expm1(-0.05)) / 0.05,
        "TREASURY": 105 * math.exp(-0.03) * (0.4 + 0.6 * math.exp(-0.02)),
        "MARKET_VALUE": 105 * math.exp(-0.03 - 0.6 * 0.02),
    }[convention]
    assert v["dirty_price"] == pytest.approx(expected)
    assert v["dirty_price"] == pytest.approx(v["coupon_pv"] + v["principal_pv"] + v["recovery_pv"])
    zero = risky_bond_expected_cashflows(curve(0, True), curve(0.03), b).value
    assert zero["dirty_price"] == pytest.approx(105 * math.exp(-0.03))


def test_bootstrap_recovers_piecewise_hazards_and_infeasible_quotes():
    h = curve(0.02, True)
    h["rates"] = [0.02, 0.05]
    rows = []
    for i, days in enumerate([365, 1095]):
        ins = cds(days)
        value = cds_expected_cashflow_legs(h, curve(0.03), ins).value["par_spread"]
        rows.append(
            dict(
                quote_id=str(i),
                eligible=True,
                observed_at=day(0),
                known_at=day(0),
                instrument=ins,
                normalized_unit="FRACTION",
                quote_type="CDS_SPREAD",
                value=value,
            )
        )
    contract = dict(recovery=0.4, maximum_hazard=2.0, price_tolerance=1e-11)
    fit = bootstrap_cds_hazard_curve(rows, curve(0.03), contract).value
    assert fit["feasible"]
    np.testing.assert_allclose(fit["hazard_curve"]["rates"], [0.02, 0.05], atol=1e-12)
    bad = deepcopy(rows)
    bad[1]["value"] = 0.0
    result = bootstrap_cds_hazard_curve(bad, curve(0.03), contract).value
    assert not result["feasible"]
    assert result["solver"][-1]["status"] == "NO_NONNEGATIVE_BRACKET"


def market_fits():
    from creditratingsystem.market_implied import calibrate_bond_hazard_curve

    rows = []
    h = curve(0.02, True)
    h["rates"] = [0.02, 0.05]
    for i, days in enumerate([365, 1095]):
        ins = bond()
        ins["schedule"] = [dict(payment_date=day(days), coupon_amount=5.0, principal_amount=100.0)]
        value = risky_bond_expected_cashflows(h, curve(0.03), ins).value["clean_price_per_100"]
        rows.append(
            dict(
                quote_id=str(i),
                eligible=True,
                observed_at=day(0),
                known_at=day(0),
                instrument=ins,
                normalized_unit="PER_100_PAR",
                quote_type="BOND_CLEAN_PRICE",
                value=value,
                price_standard_error=0.1,
            )
        )
    contract = dict(
        hazard_knots=[1.0, 3.0],
        initial_hazards=[0.03, 0.03],
        maximum_hazard=2.0,
        recovery=0.4,
        liquidity_spread=0.0,
        weight_interpretation="KNOWN_GAUSSIAN_PRICE_STANDARD_ERRORS",
        max_evaluations=200,
    )
    fit = calibrate_bond_hazard_curve(rows, curve(0.03), contract).value
    return rows, contract, fit


def test_bond_calibration_recovers_prices_and_local_information():
    _, _, fit = market_fits()
    np.testing.assert_allclose(fit["hazard_curve"]["rates"], [0.02, 0.05], atol=1e-9)
    assert fit["identifiability"]["full_column_rank"]
    assert len(fit["parameter_intervals"]) == 2
    assert fit["objective"] < 1e-14


def test_recovery_hazard_nonidentification_and_independent_evidence():
    from creditratingsystem.market_implied import diagnose_market_credit_identifiability

    ins = cds()
    row = dict(
        quote_id="q",
        eligible=True,
        observed_at=day(0),
        known_at=day(0),
        instrument=ins,
        normalized_unit="FRACTION",
        quote_type="CDS_SPREAD",
        value=0.012,
    )
    fit = bootstrap_cds_hazard_curve(
        [row], curve(0.03), dict(recovery=0.4, maximum_hazard=2.0, price_tolerance=1e-12)
    ).value
    alternatives = [dict(scenario_id="low", recovery=0.2), dict(scenario_id="high", recovery=0.6)]
    contract = dict(
        equivalence_tolerance=1e-8, quote_standard_errors=[0.0001], recovery_bounds=[0.0, 0.9]
    )
    v = diagnose_market_credit_identifiability(fit, alternatives, contract).value
    assert v["joint_identifiability"]["rank"] == 1
    assert len(v["equivalent_scenario_ids"]) == 2
    np.testing.assert_allclose([r["hazards"][0] for r in v["profiles"]], [0.015, 0.03], atol=1e-12)
    contract["independent_recovery_observation"] = dict(
        value=0.4, standard_error=0.01, evidence_id="independent-recovery-study", known_at=day(0)
    )
    w = diagnose_market_credit_identifiability(fit, alternatives, contract).value
    assert w["joint_identifiability"]["full_column_rank"]
    assert not w["equivalent_scenario_ids"]


def test_generated_world_basis_zero_and_seniority_mismatch_marked():
    from creditratingsystem.market_implied import compare_cds_bond_credit_basis

    _, _, bf = market_fits()
    rows = []
    for i, days in enumerate([365, 1095]):
        ins = cds(days)
        value = cds_expected_cashflow_legs(bf["hazard_curve"], curve(0.03), ins).value["par_spread"]
        rows.append(
            dict(
                quote_id=str(i),
                eligible=True,
                observed_at=day(0),
                known_at=day(0),
                instrument=ins,
                normalized_unit="FRACTION",
                quote_type="CDS_SPREAD",
                value=value,
            )
        )
    cf = bootstrap_cds_hazard_curve(
        rows, curve(0.03), dict(recovery=0.4, maximum_hazard=2.0, price_tolerance=1e-12)
    ).value
    contract = dict(
        recovery=0.4,
        bond_liquidity_spread=0.0,
        funding_treatment="NO_ADDITIONAL_FUNDING_WEDGE",
        maximum_quote_age_days=5,
        comparison_cds_contract=cds(1095),
    )
    v = compare_cds_bond_credit_basis(cf, bf, contract).value
    assert v["comparable"] and abs(v["basis"]) < 1e-12
    from creditratingsystem.market_implied import _seal

    bad = deepcopy(bf)
    bad.pop("calibration_hash")
    bad["seniority"] = "SUBORDINATED"
    bad = _seal(bad)
    out = compare_cds_bond_credit_basis(cf, bad, contract).value
    assert (
        not out["comparable"] and out["basis"] is None and "SENIORITY_MISMATCH" in out["findings"]
    )


def test_snapshot_units_future_stale_and_crossed_quotes():
    from creditratingsystem.market_implied import build_credit_quote_snapshot

    ins = cds()
    ins["instrument_id"] = "c"
    quotes = [
        dict(
            quote_id="spread",
            instrument_id="c",
            quote_type="CDS_SPREAD",
            unit="BASIS_POINTS",
            value=120.0,
            observed_at=day(0),
            known_at=day(0),
        ),
        dict(
            quote_id="upfront",
            instrument_id="c",
            quote_type="CDS_UPFRONT",
            unit="PERCENT_NOTIONAL",
            value=2.0,
            observed_at=day(0),
            known_at=day(0),
        ),
        dict(
            quote_id="future",
            instrument_id="c",
            quote_type="CDS_SPREAD",
            unit="FRACTION",
            value=0.02,
            observed_at=day(1),
            known_at=day(1),
        ),
    ]
    contract = dict(
        as_of=day(0),
        known_at=day(0),
        currency="EUR",
        maximum_age_days=5,
        stale_policy="EXCLUDE",
        bid_ask_policy="OPTIONAL",
    )
    v = build_credit_quote_snapshot(quotes, [ins], curve(0.03), contract).value
    indexed = {r["quote_id"]: r for r in v["quotes"]}
    assert indexed["spread"]["value"] == pytest.approx(0.012)
    assert indexed["upfront"]["value"] == pytest.approx(0.02)
    assert not indexed["future"]["eligible"]


def bridge_fixture():
    rows = []
    for cohort, asof in enumerate(["2018-01-01", "2020-01-01", "2022-01-01"]):
        end = str(int(asof[:4]) + 1) + asof[4:]
        for i in range(20):
            y = int(i % 5 == 0 or i % 7 == 0)
            x = (i % 4) / 10 + 0.05
            rows.append(
                dict(
                    observation_id=str(len(rows)),
                    entity_id="e" + str(len(rows)),
                    as_of=asof,
                    label_known_at=end,
                    complete_through=end,
                    market_curve_as_of=asof,
                    market_curve_known_at=asof,
                    features_known_at=asof,
                    market_curve_hash="historical-" + str(len(rows)),
                    outcome_origin="OBSERVED_ISSUER_DEFAULT",
                    initial_state="PERFORMING",
                    outcome=y,
                    default_date=asof[:4] + "-06-01" if y else None,
                    features={"q_pd": x},
                )
            )
    target = dict(
        event="first_default",
        horizon=1.0,
        time_unit="years",
        probability_measure="PHYSICAL",
        conditioning="initially_performing",
    )
    c = dict(
        market_feature_contracts={
            "q_pd": dict(
                probability_measure="RISK_NEUTRAL",
                horizon=1.0,
                time_unit="years",
                definition="One year cumulative Q first-default probability",
            )
        },
        target=target,
        feature_probability_measure="RISK_NEUTRAL",
        outcome_origin="OBSERVED_ISSUER_DEFAULT",
        target_population="quoted issuers",
        population_selection="MARKET_QUOTED_ISSUERS",
        transformation="IDENTITY_PREDECLARED_FEATURES",
        known_at="2023-01-01",
        split_id="forward-entity",
        link="logit",
        data_origin="SYNTHETIC",
    )
    folds = [
        dict(train=list(range(20)), validation=list(range(20, 40))),
        dict(train=list(range(40)), validation=list(range(40, 60))),
    ]
    return rows, c, folds


def test_historical_bridge_real_glm_and_leakage_guards():
    from creditratingsystem.market_implied import fit_market_to_physical_pd_bridge
    from creditratingsystem._market_bridge import _predict_market_bridge
    from creditratingsystem.classification import fit_binary_glm, predict_pd

    rows, c, folds = bridge_fixture()
    fit = fit_market_to_physical_pd_bridge(
        rows, ["q_pd"], c, folds, model_id="bridge", model_version="1"
    )
    assert fit.model["target"]["probability_measure"] == "PHYSICAL"
    assert fit.diagnostics["out_of_time_pd"][:20] == [None] * 20
    assert fit.diagnostics["observed_evaluation_count"] == 40
    direct = fit_binary_glm(
        [[r["features"]["q_pd"]] for r in rows],
        [r["outcome"] for r in rows],
        link="logit",
        feature_order=["q_pd"],
        fit_intercept=True,
        target=c["target"],
        training=fit.model["training"],
        model_id="direct",
        model_version="1",
    )
    expected = predict_pd(
        direct, [[0.1], [0.3]], feature_order=["q_pd"], expected_target=c["target"]
    ).value
    got = _predict_market_bridge(
        fit.model, [[0.1], [0.3]], feature_order=["q_pd"], expected_target=c["target"]
    ).value
    np.testing.assert_allclose(got, expected, atol=1e-12)
    bad = deepcopy(rows)
    bad[0]["market_curve_known_at"] = "2019-01-01"
    with pytest.raises(Exception, match="LOOKAHEAD_BIAS"):
        fit_market_to_physical_pd_bridge(
            bad, ["q_pd"], c, folds, model_id="bridge", model_version="1"
        )
    bad = deepcopy(rows)
    bad[20]["entity_id"] = bad[0]["entity_id"]
    with pytest.raises(Exception, match="LEAKAGE"):
        fit_market_to_physical_pd_bridge(
            bad, ["q_pd"], c, folds, model_id="bridge", model_version="1"
        )


def test_full_curve_sensitivity_accounts_for_previous_bootstrap_knots():
    from creditratingsystem.market_implied import bootstrap_cds_hazard_curve

    h = curve(0.02, True)
    h["rates"] = [0.02, 0.05]
    rows = []
    for i, days in enumerate([365, 1095]):
        ins = cds(days)
        rows.append(
            dict(
                quote_id=str(i),
                eligible=True,
                observed_at=day(0),
                known_at=day(0),
                instrument=ins,
                normalized_unit="FRACTION",
                quote_type="CDS_SPREAD",
                value=cds_expected_cashflow_legs(h, curve(0.03), ins).value["par_spread"],
            )
        )
    c = dict(recovery=0.4, maximum_hazard=2.0, price_tolerance=1e-12)
    fit = bootstrap_cds_hazard_curve(rows, curve(0.03), c).value
    bumped = deepcopy(rows)
    bumped[0]["value"] += 1e-7
    refit = bootstrap_cds_hazard_curve(bumped, curve(0.03), c).value
    difference = (np.array(refit["hazard_curve"]["rates"]) - fit["hazard_curve"]["rates"]) / 1e-7
    np.testing.assert_allclose(
        difference, np.array(fit["curve_sensitivities"]["hazard_per_quote"])[:, 0], atol=2e-6
    )
    assert difference[1] < 0


def test_boundary_and_rank_deficiency_do_not_claim_normal_identification():
    from creditratingsystem.market_implied import calibrate_bond_hazard_curve

    rows, c, _ = market_fits()
    fit = calibrate_bond_hazard_curve(rows[:1], curve(0.03), c).value
    assert not fit["identifiability"]["full_column_rank"]
    assert fit["parameter_intervals"] == [] and fit["covariance"] is None
    ins = cds()
    row = dict(
        quote_id="z",
        eligible=True,
        observed_at=day(0),
        known_at=day(0),
        instrument=ins,
        normalized_unit="FRACTION",
        quote_type="CDS_SPREAD",
        value=0.0,
    )
    zero = bootstrap_cds_hazard_curve(
        [row], curve(0.03), dict(recovery=1.0, maximum_hazard=2.0, price_tolerance=1e-12)
    ).value
    assert zero["feasible"]
    assert zero["curve_sensitivities"]["identifiability"]["rank"] == 0
    assert zero["curve_sensitivities"]["hazard_per_quote"] is None


def test_amortizing_bond_recovery_uses_remaining_face_and_liquidity_separate():
    b = bond()
    b["schedule"] = [
        dict(payment_date=day(365), coupon_amount=4.0, principal_amount=40.0),
        dict(payment_date=day(730), coupon_amount=2.4, principal_amount=60.0),
    ]
    b["liquidity_spread"] = 0.01
    v = risky_bond_expected_cashflows(curve(0.02, True), curve(0.03), b).value
    i = (-math.expm1(-0.06)) / 0.06
    expected = (
        44 * math.exp(-0.06)
        + 62.4 * math.exp(-0.12)
        + 0.4 * 0.02 * i * (100 + 60 * math.exp(-0.06))
    )
    assert v["dirty_price"] == pytest.approx(expected)
    assert v["credit_price_effect"] < 0 and v["liquidity_price_effect"] < 0
    b["recovery"] = 0
    assert risky_bond_expected_cashflows(curve(0.02, True), curve(0.03), b).value[
        "dirty_price"
    ] == pytest.approx(44 * math.exp(-0.06) + 62.4 * math.exp(-0.12))


def test_snapshot_conflicting_vintage_excluded_and_past_discount_vintage_usable():
    from creditratingsystem.market_implied import build_credit_quote_snapshot

    ins = cds()
    ins["instrument_id"] = "c"
    q = dict(
        instrument_id="c",
        quote_type="CDS_SPREAD",
        unit="BASIS_POINTS",
        observed_at=day(0),
        known_at=day(0),
    )
    c = dict(
        as_of=day(0),
        known_at=day(0),
        currency="EUR",
        maximum_age_days=5,
        stale_policy="EXCLUDE",
        bid_ask_policy="OPTIONAL",
    )
    result = build_credit_quote_snapshot(
        [dict(q, quote_id="a", value=100.0), dict(q, quote_id="b", value=110.0)],
        [ins],
        curve(0.03),
        c,
    ).value
    assert all(
        not r["eligible"] and "CONFLICTING_SAME_VINTAGE_QUOTES" in r["findings"]
        for r in result["quotes"]
    )
    d = curve(0.03)
    d["known_at"] = "2024-12-31"
    row = dict(
        quote_id="q",
        eligible=True,
        observed_at=day(0),
        known_at=day(0),
        instrument=cds(),
        normalized_unit="FRACTION",
        quote_type="CDS_SPREAD",
        value=0.012,
    )
    fitted = bootstrap_cds_hazard_curve(
        [row], d, dict(recovery=0.4, maximum_hazard=2.0, price_tolerance=1e-12)
    ).value
    assert fitted["known_at"] == day(0) and fitted["hazard_curve"]["known_at"] == day(0)


@pytest.mark.extended
@pytest.mark.parametrize("convention", ["PAR", "TREASURY", "MARKET_VALUE"])
def test_extended_bond_inverse_calibration_recovery_liquidity_grid(convention):
    from creditratingsystem.market_implied import calibrate_bond_hazard_curve

    for recovery in [0.0, 0.2, 0.7]:
        for liquidity in [0.0, 0.005]:
            rows, c, _ = market_fits()
            c.update(recovery=recovery, liquidity_spread=liquidity)
            h = curve(0.02, True)
            h["rates"] = [0.02, 0.05]
            for row in rows:
                row["instrument"].update(
                    recovery=recovery, liquidity_spread=liquidity, recovery_convention=convention
                )
                row["value"] = risky_bond_expected_cashflows(
                    h, curve(0.03), row["instrument"]
                ).value["clean_price_per_100"]
            fit = calibrate_bond_hazard_curve(rows, curve(0.03), c).value
            np.testing.assert_allclose(fit["hazard_curve"]["rates"], [0.02, 0.05], atol=2e-7)


def test_public_bridge_dispatch_serialization_and_offset_rejection(tmp_path):
    from creditratingsystem.market_implied import fit_market_to_physical_pd_bridge
    from creditratingsystem.classification import predict_pd
    from creditratingsystem.core import serialize_fitted_model, load_fitted_model

    rows, c, folds = bridge_fixture()
    fitted = fit_market_to_physical_pd_bridge(
        rows, ["q_pd"], c, folds, model_id="bridge", model_version="1"
    )
    expected = predict_pd(
        fitted, [[0.1], [0.3]], feature_order=["q_pd"], expected_target=c["target"]
    ).value
    path = str(tmp_path / "bridge.json")
    serialize_fitted_model(fitted, path)
    frozen = load_fitted_model(path).value
    np.testing.assert_allclose(
        predict_pd(
            frozen, [[0.1], [0.3]], feature_order=["q_pd"], expected_target=c["target"]
        ).value,
        expected,
        atol=1e-15,
    )
    with pytest.raises(Exception, match="OFFSET_MISMATCH"):
        predict_pd(
            frozen,
            [[0.1], [0.3]],
            feature_order=["q_pd"],
            expected_target=c["target"],
            offset=[0.0, 0.0],
        )


def test_bond_calibration_independent_zero_coupon_log_price_inverse():
    from creditratingsystem.market_implied import calibrate_bond_hazard_curve

    b = bond()
    b.update(
        recovery=0.0,
        schedule=[dict(payment_date=day(730), coupon_amount=0.0, principal_amount=100.0)],
    )
    observed = 100 * math.exp(-0.123 * 2)
    row = dict(
        quote_id="closed-form",
        eligible=True,
        observed_at=day(0),
        known_at=day(0),
        instrument=b,
        normalized_unit="PER_100_PAR",
        quote_type="BOND_CLEAN_PRICE",
        value=observed,
        price_standard_error=0.1,
    )
    c = dict(
        hazard_knots=[2.0],
        initial_hazards=[0.03],
        maximum_hazard=2.0,
        recovery=0.0,
        liquidity_spread=0.0,
        weight_interpretation="KNOWN_GAUSSIAN_PRICE_STANDARD_ERRORS",
        max_evaluations=100,
    )
    fitted = calibrate_bond_hazard_curve([row], curve(0.0), c).value
    assert fitted["hazard_curve"]["rates"][0] == pytest.approx(
        -math.log(observed / 100) / 2, abs=1e-10
    )
    expected_information = (2 * observed / 0.1) ** 2
    assert fitted["covariance"][0][0] == pytest.approx(1 / expected_information, rel=1e-7)


def test_market_context_uses_canonical_calculation_dates():
    from creditratingsystem.core import build_calculation_context

    bad = build_calculation_context("2025-01-02", "2025-01-02", probability_measure="RISK_NEUTRAL")
    with pytest.raises(Exception, match="CONTEXT_MISMATCH"):
        cds_expected_cashflow_legs(curve(0.02, True), curve(0.03), cds(), context=bad)
    old = build_calculation_context("2025-01-01", "2024-12-31", probability_measure="RISK_NEUTRAL")
    with pytest.raises(Exception, match="LOOKAHEAD_BIAS"):
        risky_bond_expected_cashflows(curve(0.02, True), curve(0.03), bond(), context=old)
