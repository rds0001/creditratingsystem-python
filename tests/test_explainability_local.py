import numpy as np
from creditratingsystem._explainability_local import compute_credit_ice


def model(features):
    return dict(model_id='local',model_version='1',feature_order=features,output_scale='score',deterministic_frozen=True,target=dict(event='default_score',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))


def constraints(features):
    return dict(contract_id='feasible',version='1',bounds={f:[-10,100]for f in features},actionable_features=features,immutable_features=[],integer_features=[],one_hot_groups=[],linear_constraints=[],derived_relations=[])


def test_u_shape_keeps_two_crossings_and_turn():
    r=compute_credit_ice(lambda z:z[:,0]**2,[0],[[-2],[-1],[0],[1],[2]],model(['x']),constraints(['x']),varying_features=['x'],thresholds=[dict(threshold_id='risk',value=1,operator='<=')]).value
    assert r['predictions']==[4,1,0,1,4]
    assert r['turn_indices']==[2]
    assert [s['threshold_crossings']for s in r['segments']]==[['risk'],[],[],['risk']]
    assert r['nonmonotone_on_evaluated_path']


def test_age_tenor_and_immutable_inputs_rejected_before_prediction():
    calls=[]
    def predict(z):calls.extend(z.tolist());return z[:,1]
    f=['age','tenor'];c=constraints(f);c['immutable_features']=['age'];c['actionable_features']=['tenor'];c['bounds']['age']=[18,100];c['bounds']['tenor']=[0,50];c['integer_features']=['age','tenor'];c['linear_constraints']=[dict(constraint_id='maturity_age',coefficients=dict(age=1,tenor=1),operator='<=',rhs=75)]
    r=compute_credit_ice(predict,[60,5],[[60,10],[60,20],[61,5]],model(f),c,varying_features=f).value
    assert r['predictions']==[10,None,None]
    assert r['feasibility_findings']==[[],['LINEAR:maturity_age'],['IMMUTABLE:age']]
    assert calls==[[60,10],[60,5]]
    assert all(s['status']=='NON_ESTIMABLE'for s in r['segments'])


def test_declared_income_ratio_and_one_hot_consistency():
    f=['debt','income','ratio','a','b'];c=constraints(f);c['derived_relations']=[dict(constraint_id='dti',output_feature='ratio',input_features=['debt','income'],operation='ratio')];c['one_hot_groups']=[['a','b']]
    r=compute_credit_ice(lambda z:z[:,2],[10,20,.5,1,0],[[10,40,.25,1,0],[10,0,.5,1,0],[10,20,.5,1,1]],model(f),c,varying_features=f).value
    assert r['predictions']==[.25,None,None]
    assert r['feasibility_findings']==[[],['DERIVED:dti'],['ONE_HOT:0']]


def test_grade_and_full_policy_crossings_preserve_frozen_inputs():
    import pytest
    from test_decisioning import policy_spec,inputs
    from creditratingsystem import CreditRiskError
    from creditratingsystem.core import _fingerprint
    from creditratingsystem.decisioning import compile_credit_decision_policy,apply_credit_decision_policy
    f=['pd'];m=model(f);m['output_scale']='probability';m['target']=policy_spec()['target'];c=constraints(f);c['bounds']['pd']=[0,1]
    scale=dict(schema_version='1.0',scale_id='ice',version='1',boundary_convention='left_closed',grades=[dict(grade='A',lower=0,upper=.1,representative_pd=.05,is_default=False),dict(grade='B',lower=.1,upper=1,representative_pd=.2,is_default=False),dict(grade='D',lower=1,upper=1,representative_pd=1,is_default=True)])
    policy=compile_credit_decision_policy(policy_spec());data=inputs();contract=dict(policy_hash=policy.value['sha256'],source_mode='REPLAY_FULL_POLICY',fixed_source_hashes={k:_fingerprint(data[k])for k in ['affordability','eligibility','operations']})
    def replay(row,pred):
        d=inputs();d['risk']['pd']=pred
        return dict(decision=apply_credit_decision_policy(policy,**d),explained_input_hash=_fingerprint(dict(record=row,prediction=pred)))
    r=compute_credit_ice(lambda z:z[:,0],[.01],[[.01],[.05],[.2]],m,c,varying_features=f,rating_scale=scale,policy_evaluator=replay,policy_contract=contract).value
    assert r['grades']==['A','A','B']
    assert [d['action']for d in r['policy_decisions']]==['APPROVE','REFER','DECLINE']
    assert [s['grade_change']for s in r['segments']]==[False,True]
    assert all(s['policy_action_change']for s in r['segments'])
    def bad(row,pred):
        value=replay(row,pred);value['explained_input_hash']='wrong';return value
    with pytest.raises(CreditRiskError,match='LOCAL_POLICY'):
        compute_credit_ice(lambda z:z[:,0],[.01],[[.01]],m,c,varying_features=f,policy_evaluator=bad,policy_contract=contract)
