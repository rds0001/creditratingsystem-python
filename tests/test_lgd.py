"""Recovery valuation and LGD independent cashflow oracles."""
from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.lgd import discount_recoveries,workout_lgd


def recovery_case():
    dates=['2024-02-01','2024-03-01']
    return dict(cashflows=[dict(cashflow_id='r1',cashflow_date=dates[0],known_at=dates[0],currency='EUR',kind='RECOVERY',amount=80),
        dict(cashflow_id='c1',cashflow_date=dates[1],known_at=dates[1],currency='USD',kind='DIRECT_COST',amount=5),
        dict(cashflow_id='c2',cashflow_date=dates[1],known_at=dates[1],currency='EUR',kind='INDIRECT_COST',amount=4)],
        discount_curve=dict(anchor_date='2024-01-01',known_at='2024-01-01',discount_basis='USER_SPECIFIED_ECONOMIC_FACTORS',points=[dict(cashflow_date=d,discount_factor=f)for d,f in zip(dates,[.9,.8])]),
        fx_contract=dict(reporting_currency='EUR',known_at='2024-03-01',conversion_basis='cashflow_date',rates=[dict(currency='EUR',date=d,rate=1)for d in dates]+[dict(currency='USD',date=dates[1],rate=2)]),
        default_date='2024-01-01',evaluation_cutoff='2024-03-01',reporting_currency='EUR')


def test_exact_discount_fx_cost_sign_and_workout_binding():
    r=discount_recoveries(**recovery_case())
    assert [v['discounted_contribution']for v in r.value['contributions']]==pytest.approx([72,-8,-3.2])
    assert r.value['discounted_net_recoveries']==pytest.approx(60.8)
    w=workout_lgd(100,r,episode_status='CLOSED',default_date='2024-01-01',evaluation_cutoff='2024-03-01',currency='EUR')
    assert w.value['economic_lgd']==pytest.approx(.392)
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):
        workout_lgd(100,r,episode_status='CLOSED',default_date='2024-01-01',evaluation_cutoff='2024-03-02',currency='EUR')


def test_raw_open_workout_and_undefined_zero_denominator():
    kwargs=dict(episode_status='CLOSED',default_date='2024-01-01',evaluation_cutoff='2024-03-01',currency='EUR')
    assert workout_lgd(100,120,**kwargs).value['economic_lgd']==-.2
    assert workout_lgd(100,-20,**kwargs).value['economic_lgd']==1.2
    kwargs['episode_status']='OPEN';v=workout_lgd(100,80,**kwargs).value
    assert v['economic_lgd'] is None and v['economic_lgd_to_date']==.2 and not v['complete']
    v=workout_lgd(0,80,**kwargs).value
    assert v['economic_loss']==-80 and v['economic_lgd_to_date'] is None and v['denominator_status']=='ZERO_EAD'
    kwargs['episode_status']='CLOSED'
    assert workout_lgd(1e-320,0,**kwargs).value['economic_lgd']==1


def test_zero_discount_rate_nominal_identity_and_cutoff_exclusions():
    a=recovery_case()
    for r in a['discount_curve']['points']:r['discount_factor']=1
    v=discount_recoveries(**a).value
    assert v['discounted_net_recoveries']==66
    a=recovery_case();a['cashflows'][0]['known_at']='2024-04-01'
    v=discount_recoveries(**a).value
    assert v['discounted_net_recoveries']==pytest.approx(-11.2)
    assert v['excluded_cashflows']==[dict(cashflow_id='r1',reason='NOT_YET_KNOWN')]
    a['cashflows'].append(dict(a['cashflows'][0],cashflow_id='future',cashflow_date='2024-06-01',known_at='2024-06-01'))
    assert discount_recoveries(**a).value['excluded_cashflows'][-1]['reason']=='FUTURE_CASHFLOW'


def test_no_silent_discount_interpolation_duplicate_or_future_curve():
    a=recovery_case();a['discount_curve']['points'].pop()
    with pytest.raises(CreditRiskError,match='MISSING_DISCOUNT'):discount_recoveries(**a)
    a=recovery_case();a['cashflows'].append(deepcopy(a['cashflows'][0]))
    with pytest.raises(CreditRiskError,match='DATA_DUPLICATE'):discount_recoveries(**a)
    a=recovery_case();a['discount_curve']['known_at']='2024-04-01'
    with pytest.raises(CreditRiskError,match='RECOVERY_CONTRACT'):discount_recoveries(**a)
