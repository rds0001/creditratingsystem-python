"""Frozen generic inference, typed input and direct forward-equation oracles."""
from copy import deepcopy
import math
import pytest
from creditratingsystem import predict_pd, serialize_fitted_model, load_fitted_model
from creditratingsystem.classification import fit_naive_bayes_pd
from creditratingsystem.core import _fingerprint


def cases():
    target={'event':'first_default','horizon':1,'time_unit':'years','probability_measure':'PHYSICAL','conditioning':'performing'}
    base={'schema_version':'1.0','model_id':'portable','model_version':'1','method_id':'classification.fit_binary_glm','feature_order':['x'],'parameters':{'link':'logit','fit_intercept':True,'coefficients':[0,1],'offset_required':False},'preprocessing':{'transformation':'identity'},'target':target,'training':{'data_fingerprint':'hand-forward-equations','split_id':'train','known_at':'2023-12-01'},'diagnostics':{},'provenance':{'data_origin':'SYNTHETIC','parameter_origin':'USER_SUPPLIED'}}
    result=[{'id':'glm','model':base,'X':[[0],[1]],'expected':[.5,1/(1+math.exp(-1))]}]
    mlp=deepcopy(base);mlp.update(method_id='classification.fit_neural_pd',preprocessing={'transformation':'standardize','center':[0],'scale':[1]},parameters={'kind':'neural','architecture':{'kind':'mlp','hidden_units':[1],'activations':['tanh']},'weights':{'layer0_weight':[[1]],'layer0_bias':[0],'layer1_weight':[[2]],'layer1_bias':[0]}})
    result.append({'id':'mlp','model':mlp,'X':[[0],[1]],'expected':[.5,1/(1+math.exp(-2*math.tanh(1)))]})
    for variant in ['gru','lstm']:
        model=deepcopy(base);g=3 if variant=='gru' else 4
        weights={'input_weight':[[0,0,1]+([0] if g==4 else [])],'hidden_weight':[[0]*g],'input_bias':[0]*g,'hidden_bias':[0]*g,'output_weight':[[1]],'output_bias':[0]}
        model.update(method_id='classification.fit_sequence_pd',preprocessing={'transformation':'standardize','center':[0],'scale':[1]},parameters={'kind':'sequence','architecture':{'kind':'recurrent','variant':variant,'hidden_size':1,'include_time_gaps':False,'time_gap_scale':1.,'initial_state':'ZERO_PER_SEQUENCE','gate_order':['r','z','n'] if g==3 else ['i','f','g','o'],'gru_reset':'AFTER_HIDDEN_AFFINE'},'weights':weights})
        x=[{'values':[[1],[2],[999]],'observation_times':[0,1,2],'known_times':[0,1,2],'mask':[1,1,1],'forecast_time':1.5}]
        h=.25*math.tanh(1)+.5*math.tanh(2)
        if variant=='lstm':h=.5*math.tanh(h)
        result.append({'id':variant,'model':model,'X':x,'expected':[1/(1+math.exp(-h))]})
    bayes=deepcopy(base);bayes['method_id']='classification.fit_bayesian_pd';bayes['diagnostics']={'convergence_status':'SUPPLIED_POSTERIOR_ORACLE_NOT_ESTIMATION'}
    bayes['parameters']={'kind':'bayesian','link':'logit','groups':{'levels':['a','b']},'posterior_draws':[[[0,0,-1,1] for _ in range(8)] for _ in range(2)],'credible_level':.95}
    result.append({'id':'grouped','model':bayes,'X':{'X':[[0],[0]],'groups':['a','b']},'expected':[1/(1+math.exp(1)),1/(1+math.exp(-1))]})
    fitted=fit_naive_bayes_pd([['a'],['a'],['b'],['b']],[0,0,1,1],feature_order=['x'],feature_types=['categorical'],variant='categorical',alpha=1,missing_policy='category',unknown_policy='reserved',target=target,training=base['training'],model_id='categorical',model_version='1')
    result.append({'id':'categorical','model':fitted.model,'X':[['a'],['unseen'],[None]],'expected':[.25,.5,.5]})
    return result


@pytest.mark.parametrize('case',cases(),ids=lambda c:c['id'])
def test_public_frozen_inference_forward_oracle_and_lineage(case,tmp_path):
    m=case['model'];path=tmp_path/(case['id']+'.json');serialize_fitted_model(m,str(path))
    loaded=load_fitted_model(str(path)).value
    pred=predict_pd(loaded,case['X'],feature_order=m['feature_order'],expected_target=m['target'],context={'known_at':'2024-01-01'})
    assert pred.value==pytest.approx(case['expected'],abs=2e-15)
    assert pred.diagnostics['model_hash']==_fingerprint(m)
    assert pred.diagnostics['training_known_at']=='2023-12-01'
    assert pred.diagnostics['prediction_count']==len(case['expected'])
    assert pred.diagnostics['calibration_applied']=='NONE_IMPLICIT'
    with pytest.raises(ValueError,match='LOOKAHEAD_BIAS'):predict_pd(loaded,case['X'],feature_order=m['feature_order'],expected_target=m['target'],context={'known_at':'2023-11-30'})
    with pytest.raises(ValueError,match='TARGET_MISMATCH'):predict_pd(loaded,case['X'],feature_order=m['feature_order'],expected_target=m['target'],context={'probability_measure':'RISK_NEUTRAL'})


def test_method_kind_spoof_and_unknown_method_rejected():
    c=cases()[1];m=deepcopy(c['model']);m['method_id']='classification.fit_sequence_pd'
    with pytest.raises(ValueError,match='INVALID_MODEL'):predict_pd(m,c['X'],feature_order=['x'],expected_target=m['target'])
    m['method_id']='untrusted.some_function'
    with pytest.raises(ValueError,match='UNSUPPORTED_MODEL'):predict_pd(m,c['X'],feature_order=['x'],expected_target=m['target'])
