import pytest
from creditratingsystem._explainability_counterfactual import find_feasible_credit_counterfactual


def fixture():
    m=dict(model_id='linear',model_version='1',feature_order=['x','y'],output_scale='score',deterministic_frozen=True,target=dict(event='score',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))
    c=dict(contract_id='c',version='1',bounds=dict(x=[0,3],y=[0,3]),actionable_features=['x','y'],immutable_features=[],integer_features=['x','y'],one_hot_groups=[],linear_constraints=[],derived_relations=[])
    costs=dict(cost_id='distance',version='1',metric='WEIGHTED_L1',feature_weights=dict(x=1,y=2),feature_scales=dict(x=1,y=1));target=dict(target_id='t',version='1',kind='MODEL_OUTPUT',operator='<=',value=0)
    rows=[[x,y]for x in range(4)for y in range(4)];ids=[f'{x}-{y}'for x,y in rows]
    return (lambda z:4-z[:,0]-2*z[:,1]),[0,0],rows,m,c,costs,target,ids


def test_complete_linear_grid_exact_cost_and_tie_witness():
    *args,ids=fixture();r=find_feasible_credit_counterfactual(*args,candidate_ids=ids).value
    assert r['status']=='FEASIBLE_SOLUTION';assert r['domain_complete'];assert r['best_candidate']['candidate_id']=='0-2';assert r['best_candidate']['cost']==4
    assert len(r['evaluated_candidates'])==16
    assert min(c['cost']for c in r['successful_candidates'])==4


def test_immutable_feature_and_no_feasible_domain():
    *args,ids=fixture();args[4]['immutable_features']=['y'];args[4]['actionable_features']=['x'];r=find_feasible_credit_counterfactual(*args,candidate_ids=ids).value
    assert r['status']=='NO_FEASIBLE_SOLUTION';assert r['best_candidate']is None
    assert all('IMMUTABLE:y'in row['findings']for row in r['evaluated_candidates']if row['record'][1]!=0)
    assert all(row['prediction']is None for row in r['evaluated_candidates']if not row['feasible'])


def test_budget_remains_explicit_even_after_finding_a_witness():
    *args,ids=fixture();r=find_feasible_credit_counterfactual(*args,candidate_ids=ids,max_candidates=9).value
    assert r['status']=='SEARCH_LIMIT';assert r['best_candidate']['cost']==4;assert len(r['unevaluated_candidate_ids'])==7
    assert r['optimality_scope']=='EVALUATED_SUBSET_ONLY'


def test_euclidean_and_change_count_and_closed_target_interval():
    *args,ids=fixture();args[5]['metric']='WEIGHTED_L2';r=find_feasible_credit_counterfactual(*args,candidate_ids=ids).value
    assert r['best_candidate']['candidate_id']=='2-1';assert r['best_candidate']['cost']==pytest.approx(6**.5)
    args[5]['metric']='WEIGHTED_CHANGE_COUNT';args[6].update(operator='INTERVAL',value=[-1,0]);r=find_feasible_credit_counterfactual(*args,candidate_ids=ids).value
    assert r['best_candidate']['candidate_id']=='0-2';assert r['best_candidate']['cost']==2


def test_policy_target_uses_full_approval_gates():
    from test_decisioning import policy_spec,inputs
    from creditratingsystem.decisioning import compile_credit_decision_policy,apply_credit_decision_policy
    from creditratingsystem.core import _fingerprint
    _,_,_,m,c,costs,_,_=fixture();m['feature_order']=['pd'];m['output_scale']='probability';m['target']=policy_spec()['target'];c.update(bounds=dict(pd=[0,1]),actionable_features=['pd'],integer_features=[]);costs.update(feature_weights=dict(pd=1),feature_scales=dict(pd=1))
    policy=compile_credit_decision_policy(policy_spec());data=inputs();contract=dict(policy_hash=policy.value['sha256'],source_mode='REPLAY_FULL_POLICY',fixed_source_hashes={k:_fingerprint(data[k])for k in ['affordability','eligibility','operations']})
    def replay(row,pred):
        d=inputs();d['risk']['pd']=pred
        return dict(decision=apply_credit_decision_policy(policy,**d),explained_input_hash=_fingerprint(dict(record=row,prediction=pred)))
    target=dict(target_id='approval',version='1',kind='POLICY_ACTION',actions=['APPROVE'])
    r=find_feasible_credit_counterfactual(lambda z:z[:,0],[.2],[[.01],[.05],[.2]],m,c,costs,target,candidate_ids=['low','medium','high'],policy_evaluator=replay,policy_contract=contract).value
    assert r['best_candidate']['candidate_id']=='low';assert r['best_candidate']['policy_decision']['action']=='APPROVE'
    assert r['best_candidate']['cost']==pytest.approx(.19)
