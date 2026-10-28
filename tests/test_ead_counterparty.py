"""Hand-auditable signed netting, units/settlement ledger and frozen closeout."""
from copy import deepcopy
import pytest
from creditratingsystem.core import CreditRiskError
from creditratingsystem.ead import aggregate_netting_set_exposure,simulate_collateral_margin,closeout_exposure_at_default
DATES=['2024-01-01','2024-01-02','2024-01-03']
def netting_case(values=(100,120,150),separate=False):
    rows=[dict(path_id='p',date=date,trade_id=trade,currency='EUR',value=v)for date,value in zip(DATES,values)for trade,v in [('a',value),('b',-60)]]
    base=dict(counterparty_id='cp',legally_enforceable=True,evidence_reference='user-evidence',known_at=DATES[0],effective_from=DATES[0],effective_until=None)
    agreements=[dict(base,netting_set_id='n',trade_ids=['a','b'])]if not separate else[dict(base,netting_set_id=t,trade_ids=[t])for t in ['a','b']]
    return dict(trade_value_paths=rows,netting_agreements=agreements,fx_paths=[dict(path_id='p',date=d,currency='EUR',rate=1)for d in DATES],as_of=DATES[0],reporting_currency='EUR',valuation_convention='EX_DIVIDEND_BEFORE_MARGIN')
def margin_case(net,lag=False,haircut=0):
    policy=dict(netting_set_id='n',known_at=DATES[0],collateral_currency='EUR',initial_units=0,threshold_received=0,threshold_posted=0,minimum_transfer_amount=0,ownership='TITLE_TRANSFER',received_enforceable=True,posted_returnability='UNSECURED_CLAIM',posted_claim_netting_enforceable=True,reuse='ALLOWED')
    calls=[dict(call_id=str(i),path_id='p',netting_set_id='n',valuation_date=d,call_date=d,settlement_date=DATES[min(i+1,2)]if lag else d,dispute_delay_days=0)for i,d in enumerate(DATES)]
    market=[dict(path_id='p',date=d,netting_set_id='n',unit_price=1,fx_rate=1,currency='EUR',received_haircut=haircut,posted_haircut=haircut)for d in DATES]
    return dict(netting_paths=net,margin_policies=[policy],call_calendar=calls,collateral_market_paths=market)
def test_signed_offsets_only_within_sets_and_artifact_integrity():
    n=aggregate_netting_set_exposure(**netting_case());assert [r['positive_exposure']for r in n.value['paths']]==[40,60,90]
    n=aggregate_netting_set_exposure(**netting_case(separate=True));assert sum(r['positive_exposure']for r in n.value['paths']if r['date']==DATES[0])==100
    bad=netting_case();bad['netting_agreements'].append(dict(bad['netting_agreements'][0],netting_set_id='overlap'))
    with pytest.raises(CreditRiskError,match='NETTING_OVERLAP'):aggregate_netting_set_exposure(**bad)
    n=aggregate_netting_set_exposure(**netting_case());bad=deepcopy(n.value);bad['paths'][0]['signed_net_value']=999
    with pytest.raises(CreditRiskError,match='INTEGRITY_ERROR'):simulate_collateral_margin(**margin_case(bad))
def test_immediate_margin_and_lagged_unsecured_exposure():
    n=aggregate_netting_set_exposure(**netting_case());m=simulate_collateral_margin(**margin_case(n))
    assert [r['closing_units']for r in m.value['paths']]==[40,60,90];assert [r['residual_exposure']for r in m.value['paths']]==[0,0,0]
    m=simulate_collateral_margin(**margin_case(n,lag=True));assert [r['closing_units']for r in m.value['paths']]==[0,40,90];assert [r['residual_exposure']for r in m.value['paths']]==[40,20,0]
def test_pending_units_prevent_duplicate_margin_calls():
    n=aggregate_netting_set_exposure(**netting_case(values=(100,100,100)));a=margin_case(n)
    for c in a['call_calendar']:c['settlement_date']=DATES[-1]
    m=simulate_collateral_margin(**a);assert [c['transfer_units']for c in m.value['call_ledger']]==[40,0,0];assert [r['closing_units']for r in m.value['paths']]==[0,0,40]
def test_haircut_posted_claim_and_bankruptcy_remote_ownership():
    n=aggregate_netting_set_exposure(**netting_case(values=(-40,-40,-40)));a=margin_case(n,haircut=.2);m=simulate_collateral_margin(**a)
    assert m.value['paths'][0]['closing_units']==-125;assert m.value['paths'][0]['posted_unsecured_claim']==125;assert m.value['paths'][0]['residual_exposure']==25
    a['margin_policies'][0]['posted_claim_netting_enforceable']=False
    assert simulate_collateral_margin(**a).value['paths'][0]['residual_exposure']==125
    a['margin_policies'][0].update(ownership='SECURITY_INTEREST',posted_returnability='BANKRUPTCY_REMOTE');m=simulate_collateral_margin(**a);assert m.value['paths'][0]['residual_exposure']==0
    a['margin_policies'][0]['ownership']='TITLE_TRANSFER'
    with pytest.raises(CreditRiskError,match='COLLATERAL_OWNERSHIP'):simulate_collateral_margin(**a)
def test_stale_valuation_mta_dispute_and_future_information_rejection():
    n=aggregate_netting_set_exposure(**netting_case());a=margin_case(n);a['call_calendar']=a['call_calendar'][:1];a['call_calendar'][0]['dispute_delay_days']=1
    m=simulate_collateral_margin(**a);assert [r['residual_exposure']for r in m.value['paths']]==[40,20,50]
    a=margin_case(n);a['margin_policies'][0]['minimum_transfer_amount']=50;m=simulate_collateral_margin(**a);assert [r['closing_units']for r in m.value['paths']]==[0,60,60]
    a=margin_case(n);a['call_calendar'][0]['valuation_date']=DATES[1]
    with pytest.raises(CreditRiskError,match='COLLATERAL_DATE'):simulate_collateral_margin(**a)
def test_closeout_freezes_units_revalues_and_adds_excluded_cashflows_once():
    n=aggregate_netting_set_exposure(**netting_case());a=margin_case(n);a['collateral_market_paths'][-1]['unit_price']=.5;m=simulate_collateral_margin(**a)
    events=[dict(path_id='p',netting_set_id='n',margin_observation_date=DATES[0],default_date=DATES[0],closeout_date=DATES[-1])]
    flows=[dict(cashflow_id='unpaid',path_id='p',netting_set_id='n',date=DATES[1],amount=10,currency='EUR',included_in_closeout_value=False)]
    kwargs=dict(freeze_timing='AFTER_DATE_EVENTS',cashflow_convention='UNSETTLED_EXCLUDED_FROM_CLOSEOUT_VALUE')
    c=closeout_exposure_at_default(n,m,events,flows,**kwargs).value['closeouts'][0]
    assert c['frozen_collateral_units']==40;assert c['recognized_received']==20;assert c['closeout_exposure']==80 #90+10-20; later margin180 ignored
    kwargs['freeze_timing']='BEFORE_DATE_EVENTS';assert closeout_exposure_at_default(n,m,events,flows,**kwargs).value['closeouts'][0]['closeout_exposure']==100
    flows[0]['included_in_closeout_value']=True
    with pytest.raises(CreditRiskError,match='CLOSEOUT_CASHFLOW'):closeout_exposure_at_default(n,m,events,flows,**kwargs)
    events[0]['closeout_date']='2024-01-04'
    with pytest.raises(CreditRiskError,match='INSUFFICIENT_PATH_EXTENSION'):closeout_exposure_at_default(n,m,events,[],**kwargs)

def test_non_nettable_singletons_and_declared_mpor_observation_anchor():
    a=netting_case(separate=True)
    for x in a['netting_agreements']:x['legally_enforceable']=False
    assert aggregate_netting_set_exposure(**a).value['paths'][0]['positive_exposure']==100
    a=netting_case();a['netting_agreements'][0]['legally_enforceable']=False
    with pytest.raises(CreditRiskError,match='NETTING_ELIGIBILITY'):aggregate_netting_set_exposure(**a)
    n=aggregate_netting_set_exposure(**netting_case());m=simulate_collateral_margin(**margin_case(n))
    events=[dict(path_id='p',netting_set_id='n',margin_observation_date=DATES[0],default_date=DATES[1],closeout_date=DATES[2])]
    c=closeout_exposure_at_default(n,m,events,[],freeze_timing='BEFORE_DATE_EVENTS',cashflow_convention='UNSETTLED_EXCLUDED_FROM_CLOSEOUT_VALUE').value['closeouts'][0]
    assert c['margin_period_days']==2 and c['default_to_closeout_days']==1
