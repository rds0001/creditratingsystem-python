from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.core import _fingerprint
from creditratingsystem.links import build_entity_graph
from creditratingsystem._portfolio_simulation import prepare_credit_portfolio,simulate_credit_portfolio


def fixture(shared=False):
    entities=[dict(entity_id=x)for x in ['a','b']]
    graph=build_entity_graph(entities,[],dict(policy_id='g',version='1',edge_types=['CROSS_DEFAULT']),as_of='2024-01-01',known_at='2024-01-01')
    policy=dict(policy_id='d',version='1',edge_types=['CROSS_DEFAULT'],direction_policy='mutual_only',transitive=True)
    rows=[dict(position_id='p'+x,claim_id='c'+x,facility_id='f'+x,entity_id='a'if shared else x,currency='EUR',ead=100,facility_ead=100,sector='s',region='r',exposure_basis='GROSS',collateral_treatment='NONE')for x in ['a','b']]
    contract=dict(portfolio_id='p',version='1',currency='EUR',as_of='2024-01-01',known_at='2024-01-01',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='initially_performing',event='first_default')
    portfolio=prepare_credit_portfolio(rows,graph,policy,contract)
    groups=portfolio.value['default_group_ids'];dependence=dict(kind='SUPPLIED_DEPENDENT_UNIFORMS',group_ids=groups)
    u=[[.25,.25],[.25,.75],[.75,.25],[.75,.75]]if not shared else[[.25],[.25],[.75],[.75]]
    draws=dict(scenario_ids=['s1','s2','s3','s4'],weights=[.25]*4,group_ids=groups,sampling_design='EXACT_STATES',group_uniforms=u)
    paths=dict(scenario_ids=draws['scenario_ids'],position_ids=['pa','pb'],driver_hash=_fingerprint(dict(dependence_contract=dependence,shared_innovations=draws)),ead=[[100,100]]*4,lgd=[[.5,.5]]*4,guarantee_treatment='NOT_YET_APPLIED')
    return portfolio,{g:.5 for g in groups},dependence,draws,paths


def test_independent_bernoulli_exact_four_state_oracle():
    r=simulate_credit_portfolio(*fixture(),levels=[.75]).value
    assert sorted(r['portfolio_loss_paths'])==[0,50,50,100]
    assert r['distribution']['expected_loss']==50
    assert r['distribution']['tails'][0]['value_at_risk']==50
    assert r['distribution']['tails'][0]['expected_shortfall']==100


def test_two_facilities_use_same_obligor_default_event():
    r=simulate_credit_portfolio(*fixture(shared=True),levels=[.75]).value
    assert r['portfolio_loss_paths']==[100,100,0,0]
    assert all(row[0]==row[1]for row in r['position_defaults'])
    assert r['distribution']['expected_loss']==50


def test_gaussian_common_factor_and_variance_rejection():
    args=list(fixture());groups=args[0].value['default_group_ids'];args[2]=dict(kind='GAUSSIAN_FACTORS',group_ids=groups,factor_ids=['common'],factor_correlation=[[1]],loadings=[[1],[1]])
    args[3].pop('group_uniforms');args[3].update(factor_normals=[[-1],[-1],[1],[1]],group_normals=[[0,0]]*4)
    args[4]['driver_hash']=_fingerprint(dict(dependence_contract=args[2],shared_innovations=args[3]))
    r=simulate_credit_portfolio(*args).value
    assert r['portfolio_loss_paths']==[100,100,0,0]
    assert r['systematic_asset_variance']==[1,1]
    args[2]['loadings']=[[1.1],[1]]
    with pytest.raises(CreditRiskError,match='PORTFOLIO_VARIANCE'):simulate_credit_portfolio(*args)


def test_frozen_shared_driver_and_portfolio_seals():
    args=list(fixture());args[4]['driver_hash']='wrong'
    with pytest.raises(CreditRiskError,match='PORTFOLIO_PATHS'):simulate_credit_portfolio(*args)
    args=list(fixture());p=deepcopy(args[0].value);p['positions'][0]['ead']=200;args[0]=p
    with pytest.raises(CreditRiskError,match='LINK_INTEGRITY'):simulate_credit_portfolio(*args)


def test_guarantee_terminal_cap_and_pro_rata_loss_are_explicit():
    from creditratingsystem.links import allocate_guarantee_coverage
    args=list(fixture());old=args[0].value;graph=build_entity_graph([dict(entity_id=x)for x in ['a','b']],[],dict(policy_id='g',version='1',edge_types=['CROSS_DEFAULT']),as_of='2024-01-01',known_at='2024-01-01')
    policy=dict(policy_id='d',version='1',edge_types=['CROSS_DEFAULT'],direction_policy='mutual_only',transitive=True)
    claims=[dict(claim_id='ca',entity_id='a',amount=100,currency='EUR',due_at='2025-01-01',known_at='2024-01-01')]
    guarantee=dict(guarantee_id='guarantee',guarantor_id='b',currency='EUR',cap=80,cap_used=0,coverage_fraction=1,covered_claim_ids=['ca'],valid_from='2024-01-01',valid_until='2025-01-02',known_at='2024-01-01',legally_effective=True,evidence_reference='contract')
    allocation=allocate_guarantee_coverage([guarantee],claims,dict(policy_id='alloc',version='1',guarantee_priority=['guarantee'],claim_priority=['ca'],cap_convention='non_reinstating_total'),as_of='2024-01-01',known_at='2024-01-01',currency='EUR')
    contract=deepcopy(old['contract']);contract['guarantee_loss_rule']='FIRST_LOSS_CAP_IF_GUARANTOR_SURVIVES_HORIZON'
    args[0]=prepare_credit_portfolio(old['positions'],graph,policy,contract,guarantee_allocations=[allocation])
    r=simulate_credit_portfolio(*args).value
    assert sorted(r['guarantee_payments'][0]['payments'])==[0,0,0,50]
    assert r['distribution']['expected_loss']==pytest.approx(37.5)
    contract['guarantee_loss_rule']='PRO_RATA_EAD_CAPPED_IF_GUARANTOR_SURVIVES_HORIZON'
    args[0]=prepare_credit_portfolio(old['positions'],graph,policy,contract,guarantee_allocations=[allocation])
    r=simulate_credit_portfolio(*args).value
    assert sorted(r['guarantee_payments'][0]['payments'])==[0,0,0,40]
    assert r['distribution']['expected_loss']==40
    with pytest.raises(CreditRiskError,match='PORTFOLIO_GUARANTEE'):prepare_credit_portfolio(old['positions'],graph,policy,contract,guarantee_allocations=[allocation,allocation])


def test_stress_reuses_coordinates_and_changes_dependency_not_marginal_pd():
    from creditratingsystem._portfolio_simulation import stress_portfolio_factor_structure
    args=list(fixture());groups=args[0].value['default_group_ids'];args[2]=dict(kind='GAUSSIAN_FACTORS',group_ids=groups,factor_ids=['common'],factor_correlation=[[1]],loadings=[[0],[0]])
    args[3].pop('group_uniforms');args[3].update(factor_normals=[[-1],[-1],[1],[1]],group_normals=[[-1,-1],[-1,1],[1,-1],[1,1]])
    args[4]['driver_hash']=_fingerprint(dict(dependence_contract=args[2],shared_innovations=args[3]));dep=deepcopy(args[2]);dep['loadings']=[[1],[1]];paths=deepcopy(args[4]);paths['driver_hash']=_fingerprint(dict(dependence_contract=dep,shared_innovations=args[3]))
    case=dict(case_id='common',dependence_contract=dep,group_pds=args[1],conditional_loss_paths=paths,recovery_response='FIXED_PATHS')
    r=stress_portfolio_factor_structure(*args,[case],levels=[.5]).value
    assert r['base']['distribution']['expected_loss']==50
    assert r['stress_cases'][0]['stressed']['distribution']['expected_loss']==50
    assert r['base']['distribution']['tails'][0]['expected_shortfall']==75
    assert r['stress_cases'][0]['stressed']['distribution']['tails'][0]['expected_shortfall']==100
