"""Independent dated cohort, cash allocation and matrix probability oracles."""
from copy import deepcopy
import math
import pytest
import numpy as np
from creditratingsystem.core import CreditRiskError, _fingerprint
from creditratingsystem.behavioral import (
    compute_days_past_due, assign_delinquency_state, build_account_month_panel,
    build_behavioral_targets, estimate_retail_roll_rates, project_retail_roll_rates,
    predict_behavioral_risk, build_vintage_performance,
)


def profile():
    return {"schema_version":"1.0","profile_id":"synthetic-states","immaterial_state":"CURRENT",
        "materiality":{"absolute":1,"relative":.01,"logic":"and"},
        "buckets":[{"state":"CURRENT","lower":0,"upper":30},{"state":"EARLY","lower":30,"upper":60},
                   {"state":"LATE","lower":60,"upper":None}],
        "event_priority":[{"event":"chargeoff","state":"WRITTEN_OFF","at_risk":False,"absorbing":True},
                          {"event":"utp","state":"DEFAULT","at_risk":False,"absorbing":True}],
        "dpd_default_rule":{"minimum_dpd":90,"state":"DEFAULT","material_required":True}}


def dpd_case():
    return {"obligations":[{"account_id":"a","obligation_id":"jan","amount":100,"currency":"EUR","due_at":"2024-01-01","known_at":"2023-12-20"},
                           {"account_id":"a","obligation_id":"feb","amount":100,"currency":"EUR","due_at":"2024-02-01","known_at":"2023-12-20"}],
        "payments":[{"account_id":"a","payment_id":"p","amount":150,"currency":"EUR","paid_at":"2024-01-15","known_at":"2024-01-16","kind":"payment"},
                    {"account_id":"a","payment_id":"r","amount":100,"currency":"EUR","paid_at":"2024-02-10","known_at":"2024-02-12","kind":"reversal","reverses_payment_id":"p"}],
        "as_of":"2024-02-29","known_at":"2024-02-29","currency":"EUR"}


def cohort():
    accounts=[]; statements=[]; events=[]; coverage=[]
    for i in range(100):
        aid=f"a{i:03d}"
        accounts.append(dict(account_id=aid,customer_id=f"c{i:03d}",product_id="card",opened_at="2024-01-01",known_at="2024-01-01",currency="EUR",original_balance=100))
        coverage.append(dict(account_id=aid,observed_through="2024-03-31",known_at="2024-03-31"))
        for t in ("2024-01-31","2024-02-29","2024-03-31"):
            statements.append(dict(account_id=aid,statement_id=aid+t,observed_at=t,known_at=t,balance=0 if i==0 else 100,
                credit_limit=1000,dpd=30 if 30<=i<35 and t!="2024-01-31" else 0,past_due_amount=10 if 30<=i<35 else 0,currency="EUR"))
        if i<10 or 10<=i<30:
            events.append(dict(account_id=aid,event_id=aid+"event",event_type="DEFAULT" if i<10 else "CLOSED",occurred_at="2024-02-15",known_at="2024-02-16"))
    panel=build_account_month_panel(accounts,statements,events,observation_dates=["2024-01-31","2024-02-29","2024-03-31"]).value["accounts"]
    for r in panel:
        if r["features"]["dpd"]==30:
            r["state"]="ARREARS"
            payload={k:v for k,v in r.items() if k!="snapshot_hash"};r["snapshot_hash"]=_fingerprint(payload)
    return accounts,statements,events,coverage,panel


def model():
    target=dict(event="first_default",horizon=12,time_unit="months",probability_measure="PHYSICAL",conditioning="PIT")
    return dict(schema_version="1.0",model_id="synthetic-behavioral",model_version="1",method_id="classification.fit_binary_glm",
        feature_order=["balance"],parameters=dict(link="logit",fit_intercept=True,coefficients=[math.log(1/9),0],offset_required=False),
        preprocessing=dict(transformation="identity"),target=target,
        training=dict(data_fingerprint="synthetic-reference",split_id="train-only",known_at="2023-12-01"),diagnostics={},
        provenance=dict(data_origin="SYNTHETIC",parameter_origin="USER_SUPPLIED"),
        behavioral_contract=dict(risk_unit="account",event_policy="first_default",product_ids=["card"],allowed_states=["ACTIVE"],maximum_source_age_days=31))


def test_fifo_reversal_is_dated_not_retrospectively_erased():
    c=dpd_case();v=compute_days_past_due(**c).value
    assert v["dpd"]==59 and v["past_due_amount"]==150
    assert [r["unpaid"] for r in v["obligations"]]==[50,100]
    c.update(as_of="2024-01-31",known_at="2024-01-31")
    v=compute_days_past_due(**c).value
    assert v["dpd"]==0 and v["unallocated_payments"]==50
    c.update(as_of="2024-02-11",known_at="2024-02-11")
    assert compute_days_past_due(**c).value["past_due_amount"]==50


def test_explicit_allocation_and_due_today():
    c=dpd_case();c["payments"]=c["payments"][:1];c["payments"][0]["obligation_id"]="feb"
    c.update(as_of="2024-02-01",known_at="2024-02-01",allocation_policy="explicit")
    v=compute_days_past_due(**c).value
    assert v["dpd"]==31 and v["past_due_amount"]==100 and v["unallocated_payments"]==50
    c["payments"][0]["amount"]=250;c["payments"][0]["obligation_id"]="jan"
    v=compute_days_past_due(**c).value
    assert v["dpd"]==0 and v["past_due_amount"]==0 and v["unpaid_due_amount"]==100


def test_reversal_cannot_exceed_known_payment():
    c=dpd_case();c["payments"][1]["amount"]=151
    with pytest.raises(CreditRiskError,match="BEHAVIORAL_REVERSAL"):compute_days_past_due(**c)


@pytest.mark.parametrize("dpd,state",[(0,"CURRENT"),(29,"CURRENT"),(30,"EARLY"),(59,"EARLY"),(60,"LATE"),(90,"DEFAULT")])
def test_explicit_delinquency_boundaries(dpd,state):
    assert assign_delinquency_state(dpd,10,100,{},profile()).value["state"]==state


def test_utp_independent_of_dpd_and_immaterial_no_automatic_default():
    assert assign_delinquency_state(0,0,100,{"utp":True},profile()).value["state"]=="DEFAULT"
    assert assign_delinquency_state(100,.5,100,{},profile()).value["state"]=="CURRENT"
    assert assign_delinquency_state(0,0,100,{"utp":True,"chargeoff":True},profile()).value["state"]=="WRITTEN_OFF"


def test_panel_zero_drawn_active_and_closed_persistence():
    a,s,e,c,p=cohort()
    zero=p[0]
    assert zero["features"]["balance"]==0 and zero["at_risk"]
    feb=[r for r in p if r["as_of"]=="2024-02-29"]
    assert sum(r["at_risk"] for r in feb)==70
    assert sum(r["lifecycle_state"]=="CLOSED" for r in feb)==20
    s.append(dict(s[0],statement_id="late-revision",known_at="2024-02-01",balance=999))
    jan=build_account_month_panel(a,s,e,observation_dates=["2024-01-31"]).value["accounts"]
    assert jan[0]["features"]["balance"]==0


def test_target_100_account_cohort_competing_exits_and_maturity():
    a,s,e,c,p=cohort();jan=[r for r in p if r["as_of"]=="2024-01-31"]
    kwargs=dict(panel=jan,events=e,horizon_months=1,target_event="DEFAULT",competing_events=["CLOSED"],exit_treatment="competing",evaluation_cutoff="2024-03-31",coverage=c)
    labels=build_behavioral_targets(**kwargs).value
    assert sum(r["label"]==1 for r in labels)==10
    assert sum(r["label"]==0 for r in labels)==90
    assert sum(r["status"]=="COMPETING_EVENT" for r in labels)==20
    kwargs["exit_treatment"]="censor"
    labels=build_behavioral_targets(**kwargs).value
    assert sum(r["label"] is None for r in labels)==20
    kwargs["coverage"]=[]
    labels=build_behavioral_targets(**kwargs).value
    assert sum(r["label"] is None for r in labels)==90
    assert labels[-1]["followup_days"] is None


def test_roll_rates_reconcile_counts_and_keep_zero_balance_risk():
    a,s,e,c,p=cohort();states=["ACTIVE","ARREARS","DEFAULT","CLOSED"]
    kwargs=dict(panel=p,states=states,from_date="2024-01-31",to_date="2024-02-29",group_by=["product_id"],weighting="count",missing_successor="error")
    v=estimate_retail_roll_rates(**kwargs).value["groups"][0]
    assert v["origin_count"]==[100,0,0,0]
    assert v["transition_matrix"][0]==[.65,.05,.1,.2]
    assert v["transition_matrix"][1]==[None]*4
    kwargs["weighting"]="opening_balance"
    v=estimate_retail_roll_rates(**kwargs).value["groups"][0]
    assert v["origin_count"][0]==100 and v["origin_mass"][0]==9900
    assert v["transition_matrix"][0][2]==pytest.approx(9/99)


def test_missing_successor_is_censored_not_closed_or_zero():
    a,s,e,c,p=cohort();p=[r for r in p if not(r["account_id"]=="a099" and r["as_of"]=="2024-02-29")]
    k=dict(panel=p,states=["ACTIVE","ARREARS","DEFAULT","CLOSED"],from_date="2024-01-31",to_date="2024-02-29",group_by=[],weighting="count",missing_successor="error")
    with pytest.raises(CreditRiskError,match="MISSING_SUCCESSOR"):estimate_retail_roll_rates(**k)
    k["missing_successor"]="censor";v=estimate_retail_roll_rates(**k).value["groups"][0]
    assert v["origin_count"][0]==100 and v["censored_count"][0]==1
    assert sum(v["counts"][0])+v["censored_count"][0]==100


def test_projection_matrix_power_and_separate_maturing_exposure():
    matrix=[[.8,.1,.1],[0,1,0],[0,0,1]]
    v=project_retail_roll_rates([1,0,0],[matrix,matrix],states=["P","D","C"],default_states=["D"],closure_states=["C"],exposure_by_period=[[100,100,0],[0,0,0]]).value
    np.testing.assert_allclose(v["periods"][1]["state_probabilities"],[.64,.18,.18],atol=1e-15)
    assert v["periods"][1]["marginal_default"]==pytest.approx(.08)
    assert v["periods"][1]["expected_balance"]==0
    assert v["periods"][0]["expected_balance"]==90


def test_behavioral_prediction_nonzero_pd_on_active_zero_balance():
    a,s,e,c,p=cohort();m=model()
    risk=predict_behavioral_risk(p[0],m,m["target"]).value
    assert risk["pd"]==pytest.approx(.1)
    bad=deepcopy(p[0]);bad["features"]["balance"]=999
    with pytest.raises(CreditRiskError,match="INTEGRITY_ERROR"):predict_behavioral_risk(bad,m,m["target"])
    default=next(r for r in p if r["account_id"]=="a000" and r["as_of"]=="2024-02-29")
    m["behavioral_contract"]["allowed_states"].append("DEFAULT")
    with pytest.raises(CreditRiskError,match="BEHAVIORAL_RISK_SET"):predict_behavioral_risk(default,m,m["target"])


def test_vintage_cohort_triangle_100_accounts_and_young_missing():
    a,s,e,c,p=cohort()
    a.append(dict(a[0],account_id="young",customer_id="young",opened_at="2024-03-01",known_at="2024-03-01"))
    v=build_vintage_performance(a,p,e,months_on_book=[0,1,2],evaluation_cutoff="2024-03-31",coverage=c,default_event="DEFAULT",delinquency_states=["ARREARS"],exit_events=["CLOSED"]).value
    feb=next(x for x in v if x["cohort"]=="2024-01" and x["months_on_book"]==1)
    assert feb["cumulative_default_rate"]==.1 and feb["incidence_rate"]==.1
    assert feb["delinquency_stock_rate"]==.05 and feb["exits"]==20
    mar=next(x for x in v if x["cohort"]=="2024-01" and x["months_on_book"]==2)
    assert mar["cumulative_default_rate"]==.1 and mar["incidence_rate"]==0
    assert all(x["cumulative_default_rate"] is None for x in v if x["cohort"]=="2024-03")


def test_cure_does_not_restore_first_default_risk_set_or_erase_vintage():
    a,s,e,c,p=cohort()
    e.append(dict(account_id="a000",event_id="cure",event_type="CURED",occurred_at="2024-03-15",known_at="2024-03-16"))
    new=build_account_month_panel(a,s,e,observation_dates=["2024-03-31"]).value["accounts"]
    assert new[0]["lifecycle_state"]=="CURED" and new[0]["ever_default"] and not new[0]["at_risk"]
    v=build_vintage_performance(a,new,e,months_on_book=[2],evaluation_cutoff="2024-03-31",coverage=c,default_event="DEFAULT",delinquency_states=["DEFAULT"],exit_events=["CLOSED"]).value[0]
    assert v["cumulative_default_rate"]==.1 and v["delinquency_stock_rate"]==.09


def test_future_fitted_model_is_not_a_historical_prediction():
    a,s,e,c,p=cohort();m=model();m["training"]["known_at"]="2024-02-01"
    with pytest.raises(CreditRiskError,match="BEHAVIORAL_LEAKAGE"):
        predict_behavioral_risk(p[0],m,m["target"])


def test_same_day_competing_event_is_not_resolved_by_identifier():
    a,s,e,c,p=cohort()
    e.append(dict(account_id="a000",event_id="zzzexit",event_type="CLOSED",occurred_at="2024-02-15",known_at="2024-02-16"))
    with pytest.raises(CreditRiskError,match="BEHAVIORAL_AMBIGUITY"):
        build_vintage_performance(a,p,e,months_on_book=[1],evaluation_cutoff="2024-03-31",coverage=c,default_event="DEFAULT",delinquency_states=[],exit_events=["CLOSED"])
