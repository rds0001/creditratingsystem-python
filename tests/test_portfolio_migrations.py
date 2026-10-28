import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._portfolio_simulation import simulate_portfolio_migrations


def fixture():
    contract=dict(contract_id='m',version='1',group_ids=['g'],states=['P','D'],default_state='D',allow_cure=False,probability_measure='PHYSICAL',time_grid=[1,2],time_unit='years',matrix_mode='HOMOGENEOUS',innovation_law='CONDITIONALLY_UNIFORM_GIVEN_PRIOR_STATES')
    uniforms=[[[a],[b]]for a in [.1,.3,.5,.7,.9]for b in [.1,.3,.5,.7,.9]]
    draws=dict(scenario_ids=[str(i)for i in range(25)],group_ids=['g'],values=uniforms,weights=[1]*25)
    return dict(g='P'),[[.8,.2],[0,1]],draws,contract


def test_two_state_markov_power_and_exact_enumerated_paths():
    r=simulate_portfolio_migrations(*fixture()).value
    assert r['terminal_empirical'][0]['probabilities']['D']==pytest.approx(.36)
    assert r['terminal_markov_probabilities'][0]['probabilities']['D']==pytest.approx(.36)
    assert sum(p[-1][0]=='D'for p in r['grade_paths'])==9
    assert sum(row[0]==1 for row in r['first_default_times'])==5
    assert sum(row[0]==2 for row in r['first_default_times'])==4


def test_identity_absorption_and_explicit_cure_keeps_first_event():
    args=list(fixture());args[1]=[[1,0],[0,1]]
    r=simulate_portfolio_migrations(*args).value
    assert all(path==[['P'],['P'],['P']]for path in r['grade_paths'])
    args[0]={'g':'D'};args[1]=[[1,0],[1,0]]
    with pytest.raises(CreditRiskError,match='PORTFOLIO_MIGRATION'):simulate_portfolio_migrations(*args)
    args[3]['allow_cure']=True;r=simulate_portfolio_migrations(*args).value
    assert all(path==[['D'],['P'],['P']]for path in r['grade_paths'])
    assert all(row==[0]for row in r['first_default_times'])


def test_marked_values_do_not_relabel_physical_migration_probabilities():
    v=dict(valuation_id='v',version='1',currency='EUR',valuation_basis='SUPPLIED_FROZEN_MARKED_VALUES',group_ids=['g'],states=['P','D'],initial_values=[100],terminal_values_by_grade=[[98,40]],cash_distributions=[[5]]*25)
    r=simulate_portfolio_migrations(*fixture(),valuation_contract=v).value
    assert r['contract']['probability_measure']=='PHYSICAL'
    assert sorted(set(r['marked_values']['portfolio_marked_loss_paths']))==[-3,55]
    assert sum(r['marked_values']['portfolio_marked_loss_paths'])/25==pytest.approx(17.88)
