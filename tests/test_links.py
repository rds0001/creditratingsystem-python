from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.links import build_entity_graph,resolve_parent_links,assign_default_groups,apply_parent_support,allocate_guarantee_coverage

POLICY=dict(policy_id='relations',version='1',edge_types=['OWNERSHIP','CONTROL','CROSS_DEFAULT'])
CONTROL=dict(policy_id='control',version='1',control_edge_types=['CONTROL'],ownership_threshold=.5,ownership_inclusive=False)
TARGET=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
def edge(i,a,b,kind='OWNERSHIP',**kw):return dict(edge_id=i,source_id=a,target_id=b,edge_type=kind,valid_from='2024-01-01',valid_until=None,known_at='2024-01-01',evidence_reference='test-contract',**({'ownership_fraction':.6}if kind=='OWNERSHIP'else{}),**kw)
def graph(edges,as_of='2024-04-01',known_at='2024-04-01'):return build_entity_graph([dict(entity_id=x)for x in 'ABCD'],edges,POLICY,as_of=as_of,known_at=known_at)

def test_parent_path_time_ambiguity_and_cycles():
    e1=edge('ab','A','B');e1['valid_until']='2024-07-01';e2=edge('db','D','B');e2.update(valid_from='2024-07-01',known_at='2024-06-01')
    g=graph([e1,e2,edge('bc','B','C')]);r=resolve_parent_links(g,CONTROL).value['parents']
    assert r[2]['path']==['C','B','A'] and r[2]['ultimate_parent']=='A'
    later=graph([e1,e2,edge('bc','B','C')],as_of='2024-07-01',known_at='2024-07-01')
    assert resolve_parent_links(later,CONTROL).value['parents'][2]['ultimate_parent']=='D'
    cycle=graph([edge('ab','A','B'),edge('ba','B','A')]);assert cycle.diagnostics['cycles']=={'OWNERSHIP':[['A','B']]}
    assert resolve_parent_links(cycle,CONTROL).value['parents'][0]['status']=='CONTROL_CYCLE'
    ambiguity=graph([edge('ab','A','B'),edge('db','D','B')]);assert resolve_parent_links(ambiguity,CONTROL).value['parents'][1]['status']=='AMBIGUOUS_PARENT'
    bad=deepcopy(g.value);bad['edges'][0]['source_id']='D'
    with pytest.raises(CreditRiskError,match='LINK_INTEGRITY'):resolve_parent_links(bad,CONTROL)


def test_default_equivalence_requires_explicit_direction_and_transitivity():
    g=graph([edge('ab','A','B','CROSS_DEFAULT'),edge('ba','B','A','CROSS_DEFAULT'),edge('bc','B','C','CROSS_DEFAULT'),edge('cb','C','B','CROSS_DEFAULT')])
    policy=dict(policy_id='defaults',version='1',edge_types=['CROSS_DEFAULT'],direction_policy='mutual_only',transitive=True)
    r=assign_default_groups(g,policy,[dict(facility_id='f',entity_id='A')]).value
    assert sorted(len(v['members'])for v in r['groups'])==[1,3]
    r=assign_default_groups(g,dict(policy,transitive=False),[]).value
    assert r['conflicts']==[dict(members=['A','B','C'],status='NONTRANSITIVE_OVERLAP')]
    assert [v['group_id']for v in r['assignments'][:3]]==[None]*3
    ownership=graph([edge('ab','A','B')]);assert len(assign_default_groups(ownership,policy,[]).value['groups'])==4
    unilateral=graph([edge('ab','A','B','CROSS_DEFAULT')]);assert len(assign_default_groups(unilateral,policy,[]).value['groups'])==4


def support_terms():
    state=dict(available_capacity=100,willingness_given_state=.5,success_given_willing_and_funded=1,support_permitted=True,evidence_reference='conditional-support')
    return dict(effect_id='support-A-B',assumption_reference='conditional-tree',currency='EUR',total_capacity=200,reserved_elsewhere=50,allocated_capacity=100,support_required=80,states=dict(parent_alive=state,joint_default=dict(state,available_capacity=0,support_permitted=False)))

def test_conditional_support_budget_and_supporter_failure():
    child=dict(entity_id='B',pd=.1,stage='STANDALONE',target=TARGET,applied_effects=[]);parent=dict(child,entity_id='A',pd=.2)
    contract=support_terms();result=apply_parent_support(child,parent,contract,joint_default_probability=.02)
    assert result.value['pd']==pytest.approx(.06)
    assert result.value['stage']=='PRE_TRANSFER'
    contract['states']['parent_alive']['available_capacity']=70
    assert apply_parent_support(child,parent,contract,joint_default_probability=.02).value['pd']==.1
    contract=support_terms();contract['states']['joint_default']=deepcopy(contract['states']['parent_alive'])
    assert apply_parent_support(child,parent,contract,joint_default_probability=.02).value['pd']==pytest.approx(.05)
    contract['allocated_capacity']=160
    with pytest.raises(CreditRiskError,match='SUPPORT_CAPACITY'):apply_parent_support(child,parent,contract,joint_default_probability=.02)


def test_shared_guarantee_cap_priority_and_maturity():
    claims=[dict(claim_id=x,entity_id=x,amount=80,currency='EUR',due_at='2024-12-01',known_at='2024-01-01')for x in ['A','B']]
    guarantee=dict(guarantee_id='g',guarantor_id='P',currency='EUR',cap=100,cap_used=0,coverage_fraction=1,covered_claim_ids=['A','B'],valid_from='2024-01-01',valid_until='2025-01-01',known_at='2024-01-01',legally_effective=True,evidence_reference='contract')
    policy=dict(policy_id='allocation',version='1',guarantee_priority=['g'],claim_priority=['A','B'],cap_convention='non_reinstating_total')
    kw=dict(as_of='2024-01-01',known_at='2024-01-01',currency='EUR')
    result=allocate_guarantee_coverage([guarantee],claims,policy,**kw)
    assert [r['covered_amount']for r in result.value['allocations']]==[80,20]
    assert result.value['uncovered_claims']==[dict(claim_id='A',amount=0),dict(claim_id='B',amount=60)]
    assert allocate_guarantee_coverage([guarantee],claims[::-1],policy,**kw).value==result.value
    assert allocate_guarantee_coverage([dict(guarantee,valid_until='2024-11-01')],claims,policy,**kw).value['allocations']==[]

from creditratingsystem.links import apply_guarantee_risk,apply_rating_constraints,apply_override,solve_contagion_network

def test_payment_guarantee_preserves_pd_and_conditions_on_guarantor_default():
    claims=[dict(claim_id='c',entity_id='B',amount=100,currency='EUR',due_at='2024-12-01',known_at='2024-01-01')]
    guarantee=dict(guarantee_id='g',guarantor_id='P',currency='EUR',cap=60,cap_used=0,coverage_fraction=1,covered_claim_ids=['c'],valid_from='2024-01-01',valid_until='2025-01-01',known_at='2024-01-01',legally_effective=True,evidence_reference='contract')
    policy=dict(policy_id='allocation',version='1',guarantee_priority=['g'],claim_priority=['c'],cap_convention='non_reinstating_total')
    allocation=allocate_guarantee_coverage([guarantee],claims,policy,as_of='2024-01-01',known_at='2024-01-01',currency='EUR')
    borrower=dict(entity_id='B',pd=.1,stage='PRE_TRANSFER',target=dict(TARGET,exposure_id='c'),applied_effects=[]);guarantor=dict(borrower,entity_id='P',pd=.2,target=TARGET)
    contract=dict(effect_id='guarantee',currency='EUR',assumption_reference='four-state-table',stage='PRE_GUARANTEE',valuation_date='2024-12-01',states=dict(borrower_only=dict(loss_before_guarantee=80,payout_fraction=1,evidence_reference='payout'),joint_default=dict(loss_before_guarantee=100,payout_fraction=0,evidence_reference='zero-payout')))
    r=apply_guarantee_risk(borrower,guarantor,allocation,'c',contract,joint_default_probability=.02).value
    assert r['pd']==.1 and r['covered_amount']==60
    assert r['expected_loss_before']==pytest.approx(8.4)
    assert r['expected_loss_after']==pytest.approx(3.6)
    with pytest.raises(CreditRiskError,match='LINK_DOUBLE_COUNT'):apply_guarantee_risk(r,guarantor,allocation,'c',contract,joint_default_probability=.02)
    with pytest.raises(CreditRiskError,match='LINK_DOUBLE_COUNT'):apply_guarantee_risk(r,guarantor,allocation,'c',dict(contract,effect_id='renamed-effect'),joint_default_probability=.02)


def rating_scale():
    return dict(schema_version='1.0',scale_id='test',version='1',boundary_convention='left_closed',grades=[dict(grade=g,lower=a,upper=b,representative_pd=p,is_default=False)for g,a,b,p in [('A',0,.01,.005),('B',.01,.1,.05),('C',.1,1,.5)]])

def test_nonbinding_constraints_identity_conflicts_and_notches():
    base=dict(pd=.023,grade='B');rule=dict(rule_id='bound',kind='grade_bounds',best_allowed_grade='A',worst_allowed_grade='B',evidence_reference='policy')
    assert apply_rating_constraints(base,rating_scale(),[rule],representative_pd_policy='representative_on_binding').value['pd']==.023
    rule['worst_allowed_grade']='A';r=apply_rating_constraints(base,rating_scale(),[rule],representative_pd_policy='preserve_pd').value
    assert r['grade']=='A' and r['pd']==.023 and not r['pd_grade_consistent']
    assert apply_rating_constraints(base,rating_scale(),[rule],representative_pd_policy='representative_on_binding').value['pd']==.005
    notch=dict(rule_id='notch',kind='notch_bounds',reference_grade='C',max_improvement=0,max_deterioration=0,evidence_reference='policy')
    with pytest.raises(CreditRiskError,match='RATING_CONFLICT'):apply_rating_constraints(base,rating_scale(),[rule,notch],representative_pd_policy='preserve_pd')


def test_override_immutable_history_expiry_duplicates_and_parameter_paths():
    base=dict(pd=.02,grade='B',parameters=dict(lgd=.4),stage='CALIBRATED')
    policy=dict(policy_id='override',version='1',require_dual_control=True,targets=dict(pd=dict(path=['pd'],kind='probability'),lgd=dict(path=['parameters','lgd'],kind='numeric',minimum=0,maximum=2)))
    request=dict(override_id='o1',target='pd',value=.03,reason='documented new evidence',author='analyst',approved_by='reviewer',recorded_at='2024-01-01',valid_from='2024-01-01',valid_until='2025-01-01')
    kw=dict(as_of='2024-04-01',known_at='2024-04-01');r=apply_override(base,request,policy,**kw)
    assert r.value['original']==base and base['pd']==.02 and r.value['current']['pd']==.03
    with pytest.raises(CreditRiskError,match='OVERRIDE_DUPLICATE'):apply_override(r,request,policy,**kw)
    new=dict(request,override_id='o2',target='lgd',value=.6);r2=apply_override(r,new,policy,**kw)
    assert r2.value['current']['parameters']['lgd']==.6 and len(r2.value['history'])==2
    assert r2.value['history'][1]['previous_hash']==r.value['override_hash']
    expired=apply_override(base,request,policy,as_of='2025-01-01',known_at='2025-01-01')
    assert expired.value['status']=='EXPIRED'and expired.value['current']==base
    with pytest.raises(CreditRiskError,match='OVERRIDE_REPLAY'):apply_override(r,new,policy,as_of='2025-01-01',known_at='2025-01-01')
    bad=deepcopy(r.value);bad['current']['pd']=.9
    with pytest.raises(CreditRiskError,match='LINK_INTEGRITY'):apply_override(bad,new,policy,**kw)


def test_clearing_independent_linear_solution_and_closed_cycle_nonuniqueness():
    assumptions=dict(model='eisenberg_noe',seniority='single_pari_passu',bankruptcy_costs='none',outside_creditor='sink');kw=dict(entity_ids=['A','B'],currency='EUR',model_assumptions=assumptions)
    result=solve_contagion_network([[0,50],[50,0]],[20,10],[50,50],asset_scenarios=[dict(scenario_id='more-assets',external_assets=[40,10])],**kw).value
    assert result['base']['clearing_payments']==pytest.approx([100/3,80/3])
    assert result['sensitivity'][0]['result']['clearing_payments']==pytest.approx([60,40])
    assert result['base']['equity']==pytest.approx([0,0],abs=1e-12)
    assert result['base']['default_set']==['A','B']
    cycle=solve_contagion_network([[0,1],[1,0]],[0,0],[0,0],**kw).value['base']
    assert cycle['clearing_payments']==[1,1]and cycle['diagnostics']['uniqueness']=='MULTIPLE_CLEARINGS'
    none=solve_contagion_network([[0,0],[0,0]],[2,3],[0,0],**kw).value['base']
    assert none['clearing_payments']==[0,0]and none['equity']==[2,3]
    nearly=solve_contagion_network([[0,1],[1,0]],[1e-8,0],[0,0],max_iterations=1,**kw).value['base']
    assert nearly['diagnostics']['uniqueness']=='NOT_ESTABLISHED'


@pytest.mark.parametrize("q", [.1, .2, .7])
def test_certain_borrower_frechet_boundary_remains_exact(q):
    claim=dict(claim_id='c',entity_id='B',amount=100,currency='EUR',due_at='2024-12-01',known_at='2024-01-01')
    guarantee=dict(guarantee_id='g',guarantor_id='P',currency='EUR',cap=60,cap_used=0,coverage_fraction=1,covered_claim_ids=['c'],valid_from='2024-01-01',valid_until='2025-01-01',known_at='2024-01-01',legally_effective=True,evidence_reference='contract')
    policy=dict(policy_id='allocation',version='1',guarantee_priority=['g'],claim_priority=['c'],cap_convention='non_reinstating_total')
    allocation=allocate_guarantee_coverage([guarantee],[claim],policy,as_of='2024-01-01',known_at='2024-01-01',currency='EUR')
    borrower=dict(entity_id='B',pd=1,stage='PRE_TRANSFER',target=TARGET,applied_effects=[])
    guarantor=dict(borrower,entity_id='P',pd=q)
    contract=dict(effect_id='g',currency='EUR',assumption_reference='explicit-boundary',stage='PRE_GUARANTEE',valuation_date='2024-12-01',states=dict(borrower_only=dict(loss_before_guarantee=80,payout_fraction=1,evidence_reference='payout'),joint_default=dict(loss_before_guarantee=100,payout_fraction=0,evidence_reference='no-payout')))
    result=apply_guarantee_risk(borrower,guarantor,allocation,'c',contract,joint_default_probability=q).value
    assert result['pd']==1
    assert result['expected_loss_after']==pytest.approx(20*(1-q)+100*q)
    for bad in [q-1e-7,q+1e-7]:
        with pytest.raises(CreditRiskError,match='INVALID_JOINT_PROBABILITY'):
            apply_guarantee_risk(borrower,guarantor,allocation,'c',contract,joint_default_probability=bad)
