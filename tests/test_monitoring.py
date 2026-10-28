from copy import deepcopy
import pytest
from creditratingsystem.core import CreditRiskError
from creditratingsystem.rating_system import register_rating_model_version, select_effective_model
from creditratingsystem.monitoring import assign_model_role, rollback_model_assignment, classify_monitoring_findings


def registry():
    result=None
    for version in ("1","2"):
        model=dict(schema_version="1.0",model_id="glm",model_version=version,method_id="classification.fit_binary_glm",
            feature_order=["x"],parameters=dict(link="logit",fit_intercept=True,coefficients=[0,1],offset_required=False),
            preprocessing=dict(transformation="identity"),target=dict(event="default",horizon=1,time_unit="years",probability_measure="PHYSICAL",conditioning="PIT"),
            training=dict(data_fingerprint="synthetic-counts",split_id="train",known_at="2020-01-01"),diagnostics={},
            provenance=dict(data_origin="SYNTHETIC",parameter_origin="USER_SUPPLIED"))
        result=register_rating_model_version(result,dict(model=model,dependencies={},configuration={}),
            dict(model_id="glm",model_version=version,registered_at="2020-01-01"))
    return result


def event(reg,version="1",event_id="initial",effective="2021-01-01",known="2020-12-01"):
    return dict(event_id=event_id,profile_id="retail",purpose="application",role="champion",slot_id="champion",
        model_id="glm",model_version=version,model_hash=next(e["model_hash"] for e in reg.value["entries"] if e["model_version"]==version),
        effective_from=effective,known_at=known,evaluation_reference="synthetic-evidence",reason="explicit local example",
        required_feature_order=["x"])


def test_champion_switch_historical_knowledge_and_rollback():
    reg=registry(); first=assign_model_role(None,reg,event(reg));original=deepcopy(first.value)
    second=assign_model_role(first,reg,event(reg,"2","promotion","2022-01-01","2021-12-01"))
    assert first.value==original and len(second.value["events"])==2
    prior=select_effective_model(reg,second,"retail","application","2021-12-31","2021-12-31").value
    current=select_effective_model(reg,second,"retail","application","2022-01-01","2022-01-01").value
    assert prior["selected"]["model_version"]=="1" and current["selected"]["model_version"]=="2"
    # A newly learnt, backdated local assignment cannot alter an earlier knowledge replay.
    revised=assign_model_role(second,reg,event(reg,"1","later-information","2022-06-01","2022-07-01"))
    assert select_effective_model(reg,revised,"retail","application","2022-06-15","2022-06-15").value["selected"]["model_version"]=="2"
    assert select_effective_model(reg,revised,"retail","application","2022-06-15","2022-07-01").value["selected"]["model_version"]=="1"
    rolled=rollback_model_assignment(second,reg,"initial",dict(event_id="rollback",effective_from="2023-01-01",known_at="2022-12-01",
        reason="synthetic finding",evaluation_reference="re-evaluation",required_feature_order=["x"]))
    assert len(rolled.value["events"])==3
    assert select_effective_model(reg,rolled,"retail","application","2023-01-01","2023-01-01").value["selected"]["model_version"]=="1"
    assert select_effective_model(reg,rolled,"retail","application","2022-03-01","2022-03-01").value["selected"]["model_version"]=="2"


def test_conflicting_events_idempotence_and_incompatible_rollback():
    reg=registry(); e=event(reg); h=assign_model_role(None,reg,e)
    assert assign_model_role(h,reg,e).value==h.value
    bad=dict(e,model_version="2")
    with pytest.raises(CreditRiskError,match="ASSIGNMENT_CONFLICT"): assign_model_role(h,reg,bad)
    bad=dict(event(reg,"2"),event_id="duplicate-boundary")
    with pytest.raises(CreditRiskError,match="ASSIGNMENT_CONFLICT"): assign_model_role(h,reg,bad)
    with pytest.raises(CreditRiskError,match="FEATURE_MISMATCH"):
        rollback_model_assignment(h,reg,"initial",dict(event_id="rollback",effective_from="2023-01-01",known_at="2022-12-01",
            reason="review",evaluation_reference="test",required_feature_order=["different"]))


def policy():
    return dict(schema_version="1.0",policy_id="example-monitor",version="1",periods=["Q1","Q2","Q3"],series_ids=["retail"],
        rules=[dict(rule_id="bias",metric="calibration_gap",operator="gt",threshold=.02,consecutive_periods=2,
                    severity="WARNING",action_code="REVIEW_CALIBRATION")])


def panel(values):
    return [dict(series_id="retail",metric="calibration_gap",period=p,status="OK",value=v)
            for p,v in zip(["Q1","Q2","Q3"],values)]


def test_persistence_exact_boundary_and_missing_period_break():
    r=classify_monitoring_findings(panel([.02,.03,.04]),policy()).value
    assert [x["assessment"] for x in r["trace"]]==["WITHIN_POLICY","WATCH","WARNING"]
    assert r["status"]=="WARNING"
    p=panel([.03,None,.04]);p[1]["status"]="MISSING_OUTCOME"
    r=classify_monitoring_findings(p,policy()).value
    assert r["status"]=="WATCH" and r["trace"][1]["assessment"]=="MISSING_OUTCOME"
    assert r["trace"][2]["streak"]==1


def test_missing_whole_series_and_technical_failure_never_green():
    p=policy();p["series_ids"].append("mortgage")
    r=classify_monitoring_findings(panel([0,0,0]),p).value
    assert r["status"]=="INSUFFICIENT_INFORMATION" and len(r["trace"])==6
    assert classify_monitoring_findings([],policy()).value["status"]=="INSUFFICIENT_INFORMATION"
    rows=panel([0,0,None]);rows[-1]["status"]="TECHNICAL_ERROR"
    assert classify_monitoring_findings(rows,policy()).value["status"]=="TECHNICAL_ERROR"
