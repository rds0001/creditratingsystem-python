"""Hand-defined policy decisions and exact boundary/conflict oracles."""
from copy import deepcopy
import math
import pytest

from creditratingsystem.core import CreditRiskError, _fingerprint
from creditratingsystem.decisioning import (
    compile_credit_decision_policy, apply_credit_decision_policy, decompose_policy_decision,
)


def policy_spec():
    return {
        "schema_version": "1.0", "policy_id": "retail-example", "version": "1",
        "approval_reference": "synthetic-policy-record", "known_at": "2024-01-01",
        "effective_from": "2024-01-01", "effective_until": "2025-01-01",
        "default_action": "REFER", "missing_action": "BLOCKED",
        "target": {"event": "default", "horizon": 1, "time_unit": "years",
                   "probability_measure": "PHYSICAL", "conditioning": "PIT"},
        "fields": {"risk.pd": {"type": "number"},
                   "affordability.status": {"type": "category", "values": ["AFFORDABLE", "NOT_AFFORDABLE", "INSUFFICIENT_EVIDENCE"]},
                   "eligibility.status": {"type": "category", "values": ["ELIGIBLE", "INELIGIBLE"]}},
        "rules": [
            {"rule_id": "eligibility", "priority": 0, "action": "DECLINE", "reason_code": "ELIGIBILITY",
             "when": [{"field": "eligibility.status", "op": "eq", "value": "INELIGIBLE"}]},
            {"rule_id": "affordability", "priority": 1, "action": "DECLINE", "reason_code": "UNAFFORDABLE",
             "when": [{"field": "affordability.status", "op": "eq", "value": "NOT_AFFORDABLE"}]},
            {"rule_id": "high-risk", "priority": 2, "action": "DECLINE", "reason_code": "RISK",
             "when": [{"field": "risk.pd", "op": "gt", "value": .1}]},
            {"rule_id": "low-risk", "priority": 3, "action": "APPROVE", "reason_code": "LOW_RISK",
             "when": [{"field": "risk.pd", "op": "le", "value": .02}]},
        ],
    }


def inputs():
    timing = {"as_of": "2024-06-01", "known_at": "2024-06-02"}
    return dict(risk=dict(timing, pd=.01, target=policy_spec()["target"], freshness="CURRENT"),
                affordability=dict(timing, status="AFFORDABLE"), eligibility=dict(timing, status="ELIGIBLE"),
                operations=dict(timing, status="CLEAR"), as_of="2024-06-02", known_at="2024-06-02")


def apply(spec=None, data=None):
    return apply_credit_decision_policy(compile_credit_decision_policy(spec or policy_spec()), **(data or inputs())).value


@pytest.mark.parametrize("pd,action", [(0, "APPROVE"), (.02, "APPROVE"),
    (math.nextafter(.02, 1), "REFER"), (.1, "REFER"), (math.nextafter(.1, 1), "DECLINE")])
def test_exact_cutoff_edges(pd, action):
    data = inputs()
    data["risk"]["pd"] = pd
    assert apply(data=data)["action"] == action


def test_separate_affordability_eligibility_and_rule_explanations():
    data = inputs()
    data["affordability"]["status"] = "NOT_AFFORDABLE"
    data["eligibility"]["status"] = "INELIGIBLE"
    result = apply(data=data)
    assert result["action"] == "DECLINE"
    assert data["risk"]["pd"] == .01
    reasons = decompose_policy_decision(result, {"ELIGIBILITY": "Outside declared eligibility",
        "UNAFFORDABLE": "Declared debt service exceeds capacity", "LOW_RISK": "Below risk cutoff"}).value
    assert reasons["reasons"][0]["sources"] == ["eligibility"]
    assert reasons["reasons"][0]["selected"] is True
    assert reasons["reasons"][1]["selected"] is False


@pytest.mark.parametrize("source,status", [("affordability", "INSUFFICIENT_EVIDENCE"), ("operations", "PENDING")])
def test_nonwaivable_approval_gates(source, status):
    data = inputs()
    data[source]["status"] = status
    result = apply(data=data)
    assert result["action"] == "REFER"
    assert source + ".APPROVAL_GATE" in result["findings"]


@pytest.mark.parametrize("change,finding", [("missing", "risk.pd.MISSING"),
    ("stale", "risk.NOT_CURRENT"), ("future", "risk.FUTURE_INPUT"),
    ("target", "risk.TARGET_MISMATCH"), ("expired", "POLICY_NOT_EFFECTIVE")])
def test_incomplete_stale_future_incompatible_information_blocks(change, finding):
    data = inputs()
    if change == "missing": del data["risk"]["pd"]
    if change == "stale": data["risk"]["freshness"] = "STALE"
    if change == "future": data["risk"]["known_at"] = "2024-06-03"
    if change == "target": data["risk"]["target"]["conditioning"] = "TTC"
    if change == "expired": data.update(as_of="2025-01-01", known_at="2025-01-01")
    result = apply(data=data)
    assert result["action"] == "BLOCKED"
    assert finding in result["findings"]


def test_overlap_detection_is_semantic_not_identical_predicate_matching():
    spec = policy_spec()
    a = {"rule_id": "a", "priority": 0, "action": "APPROVE", "reason_code": "A",
         "when": [{"field": "risk.pd", "op": "le", "value": .05}]}
    b = {"rule_id": "b", "priority": 0, "action": "DECLINE", "reason_code": "B",
         "when": [{"field": "risk.pd", "op": "ge", "value": .05}]}
    spec["rules"] = [a, b]
    with pytest.raises(CreditRiskError, match="CONFLICTING_RULES"):
        compile_credit_decision_policy(spec)
    b["when"][0]["op"] = "gt"
    compile_credit_decision_policy(spec)
    b["when"] = [{"field": "risk.pd", "op": "gt", "value": .05},
                 {"field": "risk.pd", "op": "lt", "value": math.nextafter(.05, 1)}]
    with pytest.raises(CreditRiskError, match="UNREACHABLE_RULE"):
        compile_credit_decision_policy(spec)


def test_priority_exception_and_shadow_diagnostics():
    spec = policy_spec()
    spec["rules"].append({"rule_id": "fallback", "priority": 4, "action": "REFER", "reason_code": "FALLBACK", "when": []})
    spec["rules"].append({"rule_id": "shadow", "priority": 5, "action": "REFER", "reason_code": "SHADOW", "when": []})
    findings = compile_credit_decision_policy(spec).diagnostics["rule_findings"]
    assert {"rule_id": "shadow", "finding": "SHADOWED_RULE", "by": "fallback"} in findings


def test_frozen_replay_no_input_mutation_and_tamper_detection():
    spec, data = policy_spec(), inputs()
    original = deepcopy((spec, data))
    p = compile_credit_decision_policy(spec)
    first = apply_credit_decision_policy(p, **data).value
    assert first == apply_credit_decision_policy(p, **data).value
    assert (spec, data) == original
    p.value["specification"]["rules"][0]["action"] = "APPROVE"
    with pytest.raises(CreditRiskError, match="ARTIFACT_HASH_MISMATCH"):
        apply_credit_decision_policy(p, **data)
    first["action"] = "APPROVE"
    first["rule_trace"] = "invalid"
    payload = {k: v for k, v in first.items() if k != "sha256"}
    first["sha256"] = _fingerprint(payload)
    with pytest.raises(CreditRiskError, match="INVALID_ARTIFACT"):
        decompose_policy_decision(first, {})


def test_invalid_chronology_and_target_unit():
    data = inputs()
    data["risk"]["known_at"] = "2024-05-31"
    with pytest.raises(CreditRiskError, match="INVALID_DATE"):
        apply(data=data)
    spec = policy_spec()
    spec["target"]["time_unit"] = "bananas"
    with pytest.raises(CreditRiskError, match="INVALID_POLICY"):
        compile_credit_decision_policy(spec)
