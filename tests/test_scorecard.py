import math
import numpy as np
import pytest
from scipy.special import expit
from creditratingsystem.scorecard import fit_scorecard,score_scorecard,scale_score_points,apply_qualitative_adjustments,scorecard_reason_codes
from creditratingsystem.core import CreditRiskError
from creditratingsystem.classification import predict_pd
from creditratingsystem.rating_system import register_rating_model_version, rate_subject_batch, replay_assessment_history
TARGET=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
TRAINING=dict(data_fingerprint='two-group-oracle',split_id='train',known_at='2025-01-01',data_origin='SYNTHETIC')

def fit():
 return fit_scorecard([[0],[0],[1],[1]],[0,1,0,1],feature_order=['x'],fit_spec={'neutral_profile':[0],'feature_kinds':['expert']},target=TARGET,training=TRAINING,model_id='scorecard',model_version='1',weights=[8,2,4,6])

def test_scorecard_fit_independent_group_mle_and_replay():
 model=fit();result=score_scorecard(model,[[0],[1]],feature_order=['x'],expected_target=TARGET).value
 np.testing.assert_allclose(result['pd'],[.2,.6],atol=1e-8)
 np.testing.assert_allclose(result['log_odds'],result['intercept']+result['feature_contributions'].sum(axis=1))
 assert model.model['parameters']['coefficients'][0]==pytest.approx(math.log(.2/.8),abs=1e-7)
 assert model.model['parameters']['coefficients'][1]==pytest.approx(math.log(.6/.4)-math.log(.2/.8),abs=1e-7)
 assert result['reference_contributions'][0]==0

def test_public_pd_dispatch_uses_frozen_scorecard_contract():
 model=fit()
 np.testing.assert_allclose(predict_pd(model,[[0],[1]],feature_order=['x'],expected_target=TARGET).value,[.2,.6],atol=1e-8)
 with pytest.raises(CreditRiskError,match='OFFSET_MISMATCH'):
  predict_pd(model,[[0]],feature_order=['x'],expected_target=TARGET,offset=[0])
 with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):
  predict_pd(model,[[0]],feature_order=['x'],expected_target=dict(TARGET,horizon=2))

def test_scorecard_frozen_batch_and_historical_replay():
 fitted=fit()
 registry=register_rating_model_version(None,dict(model=fitted.model,dependencies={},configuration={}),
     dict(model_id='scorecard',model_version='1',registered_at='2025-01-01'))
 request=dict(row_id='r',known_at='2025-01-02',model_id='scorecard',model_version='1',
     model_hash=registry.value['entries'][0]['model_hash'],input=dict(X=[[0],[1]],feature_order=['x']),
     recipe=dict(method='pd_model',version='1.0',parameters=dict(expected_target=TARGET)))
 result=rate_subject_batch([request],registry,'batch','2025-01-02').value
 assert result['rows'][0]['status']=='OK'
 np.testing.assert_allclose(result['rows'][0]['value'],[.2,.6],atol=1e-8)
 assert replay_assessment_history([request],registry,result['rows'],'2025-01-02').value['status']=='REPRODUCED'
 request['recipe']['method']='binary_glm'
 assert rate_subject_batch([request],registry,'wrong-recipe','2025-01-02').value['rows'][0]['error_code']=='RATING_RECIPE'

@pytest.mark.parametrize('odds_convention,base_eta',[('default_over_nondefault',math.log(50)),('nondefault_over_default',-math.log(50))])
def test_score_scale_inverse_and_doubling(odds_convention,base_eta):
 result=scale_score_points([base_eta,base_eta+math.log(2)],base_score=600,base_odds=50,points_to_double_odds=20,odds_convention=odds_convention,score_direction='higher_is_safer').value
 np.testing.assert_allclose(result['points'],[600,580])
 inverse=result['inverse'];np.testing.assert_allclose(expit((result['points']-inverse['score_intercept'])/inverse['score_slope_per_default_log_odds']),expit([base_eta,base_eta+math.log(2)]))

def test_qualitative_neutral_contrast_units_and_missing():
 coeff={'management':{'weak':.8,'neutral':.2,'strong':-.2}}
 result=apply_qualitative_adjustments({'management':'weak'},coeff,{'management':'neutral'},base_value=-2,adjustment_scale='log_odds').value
 assert result['total_adjustment']==pytest.approx(.6)
 assert result['adjusted_pd']==pytest.approx(expit(-1.4))
 assert apply_qualitative_adjustments({'management':'neutral'},coeff,{'management':'neutral'},base_value=-2,adjustment_scale='log_odds').value['total_adjustment']==0
 with pytest.raises(CreditRiskError,match='MISSING_INPUT'):apply_qualitative_adjustments({},coeff,{'management':'neutral'},base_value=-2,adjustment_scale='log_odds')
 with pytest.raises(CreditRiskError,match='INCOMPATIBLE_UNIT'):apply_qualitative_adjustments({'management':'weak'},coeff,{'management':'neutral'},base_value=3,adjustment_scale='rating_notches')

def test_reasons_add_up_and_ties_deterministic():
 result=scorecard_reason_codes({'b':2,'a':2,'c':-1},{'a':0,'b':0,'c':0},{'adverse_direction':'positive','max_reasons':1,'reason_codes':{'a':'A','b':'B','c':'C'}}).value
 assert result['reasons'][0]['feature']=='a'
 assert result['total_difference']==3
 assert result['unreported_difference']==1
 assert sum(r['difference'] for r in result['all_differences'])==3

def test_separation_and_target_mismatch_are_explicit():
 with pytest.raises(CreditRiskError,match='SEPARATION'):
  fit_scorecard([[0],[0],[1],[1]],[0,0,1,1],feature_order=['x'],fit_spec={'neutral_profile':[0],'feature_kinds':['expert']},target=TARGET,training=TRAINING,model_id='separated',model_version='1')
 with pytest.raises(CreditRiskError):
  score_scorecard(fit(),[[0]],feature_order=['x'],expected_target={**TARGET,'horizon':2})
