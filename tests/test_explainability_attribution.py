from copy import deepcopy
import numpy as np
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.explainability import explain_linear_credit_prediction, estimate_credit_shapley_values


def contract(scale='score'):
    return dict(model_id='frozen',model_version='1',feature_order=['x','y'],output_scale=scale,deterministic_frozen=True,target=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))


def test_linear_logit_grouped_and_offset_identity_is_on_log_odds():
    c=contract('probability');c.update(coefficients=[2,-3],intercept=1,link='logit',transformed_feature_lineage={'x':['raw_x'],'y':['raw_y']})
    r=explain_linear_credit_prediction(c,[[4,2]],[1,1],feature_groups={'all':['x','y']},offset=[.5],baseline_offset=.25).value
    assert r['baseline_linear_predictor']==.25
    assert r['feature_contributions']==[[6,-3]]
    assert r['group_contributions']==[[3]]
    assert r['offset_contributions']==[.25]
    assert r['linear_predictor']==[3.5]
    assert r['reconstruction_residual']==[0]
    assert r['response'][0]==pytest.approx(1/(1+np.exp(-3.5)))
    assert r['contribution_scale']=='LINEAR_PREDICTOR'


def test_identity_probability_rejects_invalid_baseline_and_response():
    c=contract('probability');c.update(coefficients=[1,0],intercept=0,link='identity',transformed_feature_lineage={'x':['x'],'y':['y']})
    for x,b in [([[2,0]],[0,0]),([[.5,0]],[2,0])]:
        with pytest.raises(CreditRiskError,match='EXPLANATION_PREDICTION'):
            explain_linear_credit_prediction(c,x,b,feature_groups={'x':['x'],'y':['y']})


def test_exact_shapley_interaction_and_complete_permutation_average():
    c=contract();background=dict(values=[[0,0]],weights=[1]);mask=dict(masking='INTERVENTIONAL_EMPIRICAL',reference_id='zero',reference_version='1')
    predictor=lambda z: z[:,0]*z[:,1]
    args=(predictor,[2,3],background,c)
    kw=dict(feature_groups={'x':['x'],'y':['y']},masking_contract=mask)
    r=estimate_credit_shapley_values(*args,**kw).value
    assert r['values']==[3,3];assert r['baseline']==0;assert r['prediction']==6
    mask['permutation_design']='FIXED_PERMUTATIONS'
    q=estimate_credit_shapley_values(*args,**kw,method='permutation',permutations=[['x','y'],['y','x']]).value
    assert q['values']==[3,3];assert q['permutation_contributions']==[[0,6],[6,0]]
    assert q['permutation_standard_error'] is None
    mask['permutation_design']='IID_RANDOM_PERMUTATIONS'
    q=estimate_credit_shapley_values(*args,**kw,method='permutation',permutations=[['x','y'],['y','x']]).value
    assert q['permutation_standard_error']==pytest.approx([3,3])


def test_conditional_game_preserves_dependence_and_observed_coalitions():
    c=contract();groups={'x':['x'],'y':['y']}
    # Perfectly correlated equal-probability (0,0)/(1,1), explain f=x at (1,1).
    mask=dict(masking='SUPPLIED_CONDITIONAL_DISTRIBUTIONS',reference_id='joint',reference_version='1',coalition_backgrounds={'EMPTY':dict(values=[[0,0],[1,1]],weights=[.5,.5]),'0':dict(values=[[1,1]],weights=[1]),'1':dict(values=[[1,1]],weights=[1]),'0,1':dict(values=[[1,1]],weights=[1])})
    r=estimate_credit_shapley_values(lambda z:z[:,0],[1,1],None,c,feature_groups=groups,masking_contract=mask).value
    assert r['baseline']==.5;assert r['values']==[.25,.25]
    broken=deepcopy(mask);broken['coalition_backgrounds']['1']['values']=[[1,0]]
    with pytest.raises(CreditRiskError,match='EXPLANATION_MASK'):
        estimate_credit_shapley_values(lambda z:z[:,0],[1,1],None,c,feature_groups=groups,masking_contract=broken)
