from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._simulation_counterparties import simulate_contract_counterparties


def counterparty_fixture():
    dates=['2021-01-01','2021-02-01','2021-03-01','2021-04-01']
    paths=[dict(scenario_id='s',counterparty_id='tenant',period_start=a,period_end=b,conditional_pd=q,uniform=0,probability_measure='PHYSICAL')for a,b,q in zip(dates,dates[1:],[0,1,1])]
    base=dict(contract_id='a',counterparty_id='tenant',currency='EUR',default_payment_timing='BEFORE_DUE',cure_probability=0,cure_delay_days=2,replacement_enabled=True,replacement_delay_days=10,replacement_payment_fraction=.4,cashflows=[dict(date=d,amount=100)for d in dates])
    contracts=[base,dict(base,contract_id='b',cure_probability=1,cure_delay_days=5,replacement_enabled=False,replacement_delay_days=0,replacement_payment_fraction=0),dict(base,contract_id='c',replacement_enabled=False,replacement_delay_days=0,replacement_payment_fraction=0)]
    draws=[dict(scenario_id='s',contract_id=c['contract_id'],uniform=.5)for c in contracts]
    return dict(counterparty_paths=paths,contracts=contracts,cure_uniforms=draws,currency='EUR')


def test_one_shared_default_interrupted_cash_cure_and_replacement():
    value=simulate_contract_counterparties(**counterparty_fixture()).value
    assert len(value['counterparty_events'])==1
    assert value['counterparty_events'][0]['first_default_date']=='2021-03-01'
    assert [r['received_cash']for r in value['cashflows']]==[100,100,0,40,100,100,0,100,100,100,0,0]
    assert value['contract_events'][0]['replacement_date']=='2021-03-13'
    assert value['contract_events'][1]['cure_date']=='2021-03-06'
    assert all(r['counterparty_default_date']=='2021-03-01'for r in value['contract_events'])
    assert sum(r['cash_change']for r in value['cashflows'])==-460
    reverse=counterparty_fixture()
    for key in ['counterparty_paths','contracts','cure_uniforms']:reverse[key].reverse()
    assert simulate_contract_counterparties(**reverse).value==value


def test_zero_conditional_probability_never_defaults_even_at_zero_draw():
    f=counterparty_fixture()
    for r in f['counterparty_paths']:r['conditional_pd']=0
    value=simulate_contract_counterparties(**f).value
    assert not value['counterparty_events'][0]['defaulted']
    assert all(r['received_cash']==100 for r in value['cashflows'])
    assert all(not r['contract_cured']for r in value['contract_events'])


def test_payment_grid_probability_and_innovation_alignment_rejected():
    f=counterparty_fixture();f['contracts'][0]['cashflows'][0]['date']='2021-01-15'
    with pytest.raises(CreditRiskError,match='SIMULATION_CALENDAR'):simulate_contract_counterparties(**f)
    f=counterparty_fixture();f['counterparty_paths'][0]['uniform']=1
    with pytest.raises(CreditRiskError,match='SIMULATION_PROBABILITY'):simulate_contract_counterparties(**f)
    f=counterparty_fixture();f['cure_uniforms'].pop()
    with pytest.raises(CreditRiskError,match='SIMULATION_COUNTERPARTY'):simulate_contract_counterparties(**f)
    f=counterparty_fixture();f['cure_uniforms'].append(f['cure_uniforms'][0])
    with pytest.raises(CreditRiskError,match='SIMULATION_DUPLICATE'):simulate_contract_counterparties(**f)


def test_public_context_probability_measure_compatibility():
    from creditratingsystem import build_calculation_context
    ctx=build_calculation_context(currency='EUR',probability_measure='PHYSICAL')
    assert simulate_contract_counterparties(**counterparty_fixture(),context=ctx).value['counterparty_events'][0]['defaulted']
    q=build_calculation_context(currency='EUR',probability_measure='RISK_NEUTRAL')
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):simulate_contract_counterparties(**counterparty_fixture(),context=q)
