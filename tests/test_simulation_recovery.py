from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._simulation_recovery import simulate_recovery_waterfall


def fixture():
    events=[dict(scenario_id='s',default_event_id='d',subject_id='o',default_date='2024-01-01')]
    claims=[dict(scenario_id='s',default_event_id='d',subject_id='o',claim_id=cid,pool_id='p',currency='EUR',ead_at_default=ead,legal_balance=ead,priority=rank,deficit_cost_share=share)for cid,ead,rank,share in [('a',60,1,.6),('b',40,2,.4)]]
    paths=[dict(scenario_id='s',default_event_id='d',pool_id='p',currency='EUR',realization_id='r',collateral_id='asset',date='2025-01-01',gross_value=100,realization_fraction=.8,cost=10,discount_factor=.9,discount_anchor_date='2024-01-01')]
    policy=dict(policy_id='p',version='1',currency='EUR',allocation='LEGAL_CAP',cost_rule='PROCEEDS_FIRST_EXPLICIT_DEFICIT_SHARES',completion='ULTIMATE_COMPLETE')
    return events,claims,paths,policy


def test_priority_discount_and_ead_share_one_default():
    r=simulate_recovery_waterfall(*fixture()).value
    a,b=r['claim_results']
    assert (a['gross_recovery'],b['gross_recovery'])==(60,10)
    assert (a['economic_loss'],b['economic_loss'])==pytest.approx((6,31))
    assert (a['lgd'],b['lgd'])==pytest.approx((.1,.775))
    p=r['pool_results'][0]
    assert p['realized_proceeds']+p['creditor_deficit_funding']==p['costs']+p['creditor_gross_recovery']+p['unallocated_surplus']


def test_cost_deficit_once_and_raw_lgd_above_one():
    e,c,p,policy=fixture();p[0].update(gross_value=0,cost=20)
    r=simulate_recovery_waterfall(e,c,p,policy).value
    assert [x['economic_loss']for x in r['claim_results']]==pytest.approx([70.8,47.2])
    assert [x['lgd']for x in r['claim_results']]==pytest.approx([1.18,1.18])
    assert r['pool_results'][0]['creditor_deficit_funding']==20


def test_equal_rank_pro_rata_cumulative_caps_and_surplus():
    e,c,p,policy=fixture();c[1]['priority']=1
    extra=deepcopy(p[0]);extra.update(realization_id='r2',collateral_id='asset2',date='2026-01-01',gross_value=100,realization_fraction=1,cost=0,discount_factor=.8);p.append(extra)
    r=simulate_recovery_waterfall(e,c,p,policy).value
    assert [x['gross_recovery']for x in r['claim_results']]==pytest.approx([60,40])
    assert [x['gross_allocation']for x in r['allocation_ledger']]==pytest.approx([42,28,18,12])
    assert r['pool_results'][0]['unallocated_surplus']==pytest.approx(70)


def test_explicit_uncapped_entitlement_negative_lgd_zero_ead():
    e,c,p,policy=fixture();policy['allocation']='UNCAPPED_PRO_RATA';c[1]['priority']=1;c[0]['ead_at_default']=0;p[0].update(gross_value=200,realization_fraction=1,cost=0,discount_factor=1)
    r=simulate_recovery_waterfall(e,c,p,policy).value
    assert r['claim_results'][0]['lgd']is None
    assert r['claim_results'][1]['lgd']==-1


def test_duplicate_collateral_wrong_event_and_open_workout_rejected():
    e,c,p,policy=fixture();p.append(deepcopy(p[0]));p[1]['realization_id']='r2'
    with pytest.raises(CreditRiskError,match='RECOVERY_DUPLICATE'):simulate_recovery_waterfall(e,c,p,policy)
    e,c,p,policy=fixture();p[0]['discount_anchor_date']='2023-01-01'
    with pytest.raises(CreditRiskError,match='RECOVERY_EVENT'):simulate_recovery_waterfall(e,c,p,policy)
    e,c,p,policy=fixture();policy['completion']='OPEN'
    with pytest.raises(CreditRiskError,match='RECOVERY_POLICY'):simulate_recovery_waterfall(e,c,p,policy)
    e,c,p,policy=fixture()
    with pytest.raises(CreditRiskError,match='RECOVERY_COMPLETENESS'):simulate_recovery_waterfall(e,c,[],policy)
