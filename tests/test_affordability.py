"""Independent dimensional examples and threshold boundary cases."""
import pytest

from creditratingsystem.affordability import (
    debt_to_income, debt_service_to_income, residual_income, assess_affordability,
)
from creditratingsystem.core import CreditRiskError


def policy():
    return {"schema_version": "1.0", "policy_id": "illustrative", "version": "1",
            "required_evidence": ["income_verified"],
            "rules": [{"rule_id": "dsti", "metric": "dsti", "operator": "<=", "threshold": .3},
                      {"rule_id": "buffer", "metric": "residual", "operator": ">", "threshold": 0}]}


def test_stock_and_service_are_different_quantities():
    dti = debt_to_income(60000, 2500, income_frequency="monthly", currency="EUR", ratio_period="annual")
    dsti = debt_service_to_income(900, 3000, period="monthly", currency="EUR")
    assert dti.value["ratio"] == 2
    assert dsti.value["ratio"] == .3
    assert debt_to_income(60000, 2500, income_frequency="monthly", currency="EUR", ratio_period="income_period").value["ratio"] == 24


def test_nonpositive_income_is_not_a_safe_zero():
    result = debt_service_to_income(100, [0, -10, 200], period="monthly", currency="EUR")
    assert result.value["ratio"] == [None, None, .5]
    assert result.value["applicable"] == [False, False, True]


def test_residual_keeps_negative_periods():
    result = residual_income([3000, 500], 1000, 900, 200, period="monthly", currency="EUR")
    assert result.value == {"residual": [900, -1600], "negative_buffer": [False, True], "minimum": -1600}


def test_exact_unrounded_threshold_and_missing_evidence():
    assert assess_affordability({"dsti": .3, "residual": 1}, policy(), {"income_verified": True}).value["assessment"] == "AFFORDABLE"
    result = assess_affordability({"dsti": .3, "residual": 0}, policy(), {"income_verified": True})
    assert result.value["assessment"] == "NOT_AFFORDABLE"
    assert result.value["breaches"] == ["buffer"]
    assert result.value["rule_trace"][1]["headroom"] == 0
    missing = assess_affordability({"dsti": None, "residual": -1}, policy(), {"income_verified": False})
    assert missing.value["assessment"] == "INSUFFICIENT_EVIDENCE"
    assert missing.value["breaches"] == ["buffer"]
    assert missing.value["missing_metrics"] == ["dsti"]


def test_no_recycling_or_negative_debt():
    with pytest.raises(CreditRiskError) as error:
        residual_income([1, 2], [1, 2, 3], 1, 1, period="monthly", currency="EUR")
    assert error.value.code == "INCOMPATIBLE_SHAPE"
    with pytest.raises(CreditRiskError) as error:
        debt_service_to_income(-100, 200, period="monthly", currency="EUR")
    assert error.value.code == "INVALID_AMOUNT"
from copy import deepcopy
from creditratingsystem.affordability import (
    normalize_income_period, estimate_recurring_expenses, combined_loan_to_value,
    stress_affordability,
)

CALENDAR = {"weekly": 52, "monthly": 12, "annual": 1, "thirteenth_salary": 1}


def extended_income_case():
    recurring = {"economic_id": "salary", "amount": 1000, "frequency": "weekly",
                 "basis": "gross", "currency": "EUR", "kind": "recurring", "tax_amount": 200}
    return dict(components=[recurring, dict(recurring),
        {"economic_id": "bonus", "amount": 1200, "frequency": "annual", "basis": "net", "currency": "EUR", "kind": "variable"},
        {"economic_id": "loss", "amount": -100, "frequency": "monthly", "basis": "net", "currency": "EUR", "kind": "variable"},
        {"economic_id": "transfer", "amount": 5000, "frequency": "monthly", "basis": "net", "currency": "EUR", "kind": "internal_transfer"}],
        target_period="monthly", target_basis="net", currency="EUR", payments_per_year=CALENDAR, variable_income_factor=.5)


def extended_expense_case():
    return dict(expenses=[
        {"economic_id": "insurance", "category": "insurance", "amount": 1200, "frequency": "annual", "currency": "EUR", "kind": "essential"},
        {"economic_id": "insurance", "category": "insurance", "amount": 110, "frequency": "monthly", "currency": "EUR", "kind": "essential"},
        {"economic_id": "loan", "category": "loan", "amount": 500, "frequency": "monthly", "currency": "EUR", "kind": "debt_service"},
        {"economic_id": "loan", "category": "loan", "amount": 500, "frequency": "monthly", "currency": "EUR", "kind": "debt_service"},
        {"economic_id": "own", "category": "transfer", "amount": 900, "frequency": "monthly", "currency": "EUR", "kind": "internal_transfer"}],
        target_period="monthly", currency="EUR", payments_per_year=CALENDAR,
        household_profile={"profile_id": "two_adults", "required_categories": ["insurance", "food"]},
        benchmarks=[{"profile_id": "two_adults", "category": "food", "amount": 300, "frequency": "monthly", "currency": "EUR", "kind": "essential"}])


def extended_cltv_case():
    return dict(claims=[{"claim_id": "senior", "amount": 80, "currency": "EUR"},
                        {"claim_id": "junior", "amount": 20, "currency": "EUR"}],
        collateral=[{"collateral_id": "house", "value": 125, "currency": "EUR", "valuation_date": "2024-01-01", "freshness": "CURRENT"}],
        allocations=[{"claim_id": "senior", "collateral_id": "house", "amount": 80, "priority": 1},
                     {"claim_id": "junior", "collateral_id": "house", "amount": 20, "priority": 2}], currency="EUR")


def extended_stress_case():
    baseline = {"as_of": "2024-01-31", "currency": "EUR", "liquid_buffer": 500,
        "cltv_scope": "all_contracts_secured_by_pool",
        "incomes": [{"economic_id": "salary", "monthly_amount": 1500, "basis": "net", "currency": "EUR"}],
        "expenses": [{"economic_id": "living", "monthly_amount": 500, "currency": "EUR"}],
        "contracts": [{"economic_id": "loan", "principal": 1200, "remaining_months": 3,
                       "annual_rate": 0, "repayment": "annuity", "balloon_amount": 600,
                       "rate_type": "reset", "reset_dates": ["2024-03-31"],
                       "rate_floor": 0, "rate_cap": .24, "currency": "EUR"}],
        "collateral": [{"economic_id": "car", "value": 1500, "currency": "EUR"}],
        "thresholds": {"minimum_residual": 0, "maximum_dsti": .4, "maximum_cltv": .9}}
    scenario = {"schema_version": "1.0", "scenario_id": "joint-stress", "periods": [
        {"date": d, "income_factors": {"salary": f}, "expense_factors": {"living": 1.1},
         "collateral_factors": {"car": .8}, "rate_shifts": {"loan": .12}, "fx_to_reporting": {"EUR": 1}}
        for d, f in [("2024-02-29", 1), ("2024-03-31", .5), ("2024-04-30", 0)]]}
    return dict(baseline=baseline, scenario=scenario)


def test_normalized_income_tax_bridge_and_economic_deduplication():
    v = normalize_income_period(**extended_income_case()).value
    assert v["total_income"] == pytest.approx(800*52/12 + 1200/12*.5 - 100)
    assert v["duplicate_ids"] == ["salary"]
    assert v["components"][2]["normalized_amount"] == -100
    c = extended_income_case(); c["components"][1]["amount"] = 999
    with pytest.raises(CreditRiskError, match="CONFLICTING_EVIDENCE"):
        normalize_income_period(**c)
    c = extended_income_case(); c["components"] = [c["components"][0]]; del c["components"][0]["tax_amount"]
    assert normalize_income_period(**c).value["total_income"] is None


def test_recurring_expenses_reconcile_floor_and_separate_debt_service():
    v = estimate_recurring_expenses(**extended_expense_case()).value
    assert v["recurring_expenses"] == 410
    assert v["debt_service"] == 500
    assert v["excluded_transfers"] == ["own"]
    c = extended_expense_case(); c["benchmarks"] = []
    v = estimate_recurring_expenses(**c).value
    assert v["recurring_expenses"] is None
    assert v["missing_categories"] == ["food"]


def test_combined_ltv_static_rank_coverage_and_zero_value():
    c = extended_cltv_case()
    v = combined_loan_to_value(**c).value
    assert v["cltv"] == .8
    c["collateral"][0]["value"] = 90
    v = combined_loan_to_value(**c).value
    assert [r["covered_amount"] for r in v["priority_coverage"]] == [80, 10]
    c["collateral"][0]["value"] = 0
    assert combined_loan_to_value(**c).value["cltv"] is None
    c["allocations"][0]["amount"] = 81
    with pytest.raises(CreditRiskError, match="INVALID_ALLOCATION"):
        combined_loan_to_value(**c)


def test_lien_same_priority_shares_value_proportionally():
    c = extended_cltv_case(); c["collateral"][0]["value"] = 50; c["allocations"][1]["priority"] = 1
    rows = combined_loan_to_value(**c).value["priority_coverage"]
    assert {r["claim_id"]: r["covered_amount"] for r in rows} == {"senior": 40, "junior": 10}


def test_stress_reset_balloon_income_expenses_and_liquidity_oracle():
    c = extended_stress_case(); before = deepcopy(c)
    v = stress_affordability(**c).value
    p1, p2, p3 = v["periods"]
    assert p1["debt_service"] == 200  # (1200-600)/3 at zero fixed coupon
    assert p1["residual"] == 750  # 1500 - 550 - 200
    # At reset: 1000 = P/1.01 + (P+600)/1.01^2.
    payment = (1000*1.01**2-600)/(2.01)
    assert p2["debt_service"] == pytest.approx(payment)
    assert p3["debt_service"] == pytest.approx(payment+600)
    assert p3["closing_debt"] == 0
    assert p3["dsti"] is None
    assert v["first_liquidity_exhaustion"] == "2024-04-30"
    assert c == before


def test_fixed_coupon_and_interest_only_foreign_currency_balloon():
    c = extended_stress_case(); loan = c["baseline"]["contracts"][0]
    loan.update(rate_type="fixed", reset_dates=[], repayment="interest_only", balloon_amount=1200, annual_rate=.12, currency="USD")
    for period in c["scenario"]["periods"]: period["fx_to_reporting"]["USD"] = 2
    v = stress_affordability(**c).value
    assert [p["debt_service"] for p in v["periods"]] == [24, 24, 2424]
    assert all(p["contracts"][0]["annual_rate"] == .12 for p in v["periods"])


def test_stress_explicit_mappings_and_no_debt_double_count():
    c = extended_stress_case(); del c["scenario"]["periods"][0]["income_factors"]["salary"]
    with pytest.raises(CreditRiskError, match="MISSING_SCENARIO_MAPPING"):
        stress_affordability(**c)
    c = extended_stress_case(); c["baseline"]["expenses"][0]["economic_id"] = "loan"
    with pytest.raises(CreditRiskError, match="DOUBLE_COUNTING"):
        stress_affordability(**c)
    c = extended_stress_case(); c["baseline"]["contracts"][0]["reset_dates"] = ["2024-03-15"]
    with pytest.raises(CreditRiskError, match="INVALID_DATE"):
        stress_affordability(**c)


def test_unknown_debt_service_and_zero_collateral_are_not_safe_zeroes():
    c = extended_expense_case()
    for row in c["expenses"]:
        if row["economic_id"] == "loan": row["amount"] = None
    value = estimate_recurring_expenses(**c).value
    assert value["debt_service"] is None
    assert value["known_debt_service"] == 0
    c = extended_stress_case()
    c["scenario"]["periods"][0]["collateral_factors"]["car"] = 0
    first = stress_affordability(**c).value["periods"][0]
    assert first["cltv"] is None
    assert "ZERO_COLLATERAL_VALUE" in first["breaches"]


def test_collateral_partition_reuses_neither_claim_nor_asset():
    c = extended_cltv_case()
    c["collateral"].append(dict(c["collateral"][0], collateral_id="second", value=25))
    c["allocations"][0]["amount"] = 60
    c["allocations"].append(dict(c["allocations"][0], collateral_id="second", amount=20))
    assert combined_loan_to_value(**c).value["cltv"] == pytest.approx(100/150)
    c["allocations"][2]["amount"] = 80
    with pytest.raises(CreditRiskError, match="INVALID_ALLOCATION"):
        combined_loan_to_value(**c)
