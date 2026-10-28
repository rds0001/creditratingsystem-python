from copy import deepcopy
import numpy as np
import pytest
from scipy.special import expit,logit
from creditratingsystem.hybrid import combine_pd_signals,fit_stacked_pd,_predict_stacked_pd,_hy_horizon_end
from creditratingsystem._classification_models import fit_penalized_glm
from creditratingsystem.classification import predict_pd
T=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='performing')
M=dict(target=T,training=dict(data_fingerprint='synthetic-forward-oof',split_id='development',known_at='2021-01-01',data_origin='SYNTHETIC'),model_id='stack',model_version='1')

def fixture():
 rows=[dict(row_id='r'+str(i),entity_id='e'+str(i),group_id='g'+str(i),role='development'if i<12 else'test',as_of='2020-01-01'if i<4 else'2020-03-01'if i<12 else'2021-06-01',label_known_at='2020-02-01'if i<4 else'2020-04-01'if i<12 else None,complete_through='2020-01-31'if i<4 else'2020-03-31'if i<12 else None,prediction_known_at='2020-03-01'if 4<=i<12 else None)for i in range(14)]
 train=['r'+str(i)for i in range(4)];val=['r'+str(i)for i in range(4,12)];lineage={b:dict(model_id=b,model_version='fold1',training_fingerprint='synthetic-base-train',reference='frozen fold fit manifest',target=T,prediction_origin='HELD_OUT',fit_row_ids=train,preprocessing_row_ids=train,selection_row_ids=[],known_at='2020-02-15')for b in ['scorecard','structural']}
 plan=dict(horizon_date_rule='calendar_clamp',rows=rows,folds=[dict(fold_id='forward1',train_row_ids=train,validation_row_ids=val,fit_known_at='2020-02-15',base_lineage=lineage)])
 x=[[None,None]]*4+np.column_stack(([.1,.2,.25,.35,.5,.6,.7,.9],[.4,.3,.5,.4,.7,.65,.8,.75])).tolist()+[[None,None]]*2
 y=[0,0,1,1,0,1,0,0,1,0,1,1,None,None];spec=dict(input_scale='probability',penalty='ridge',lambda_path=[.1],selected_lambda=.1,fit_intercept=True,scale=True,tolerance=1e-10)
 return x,y,plan,spec

def fit(x,y,plan,spec):return fit_stacked_pd(x,y,base_model_ids=['scorecard','structural'],split_plan=plan,meta_model_spec=spec,**M)

@pytest.mark.parametrize('scale',['linear','log_odds','mixture'])
def test_signal_combination_actual_variants_and_hand_values(scale):
 signals=[dict(signal_id=s,model_id=s,model_version='1',row_ids=['a','b'],pd=p,target=T,as_of='2024-01-01',known_at='2024-01-01')for s,p in [('score',[.1,.8]),('structure',[.4,.2])]]
 interpretation={'linear':'estimator_pool','log_odds':'log_odds_pool','mixture':'latent_regime_mixture'};contract=dict(interpretation=interpretation[scale],reference='explicit pooling exercise');w=[1,3]
 if scale=='mixture':contract.update(regime_ids=['score','structure'],conditional_regime_probabilities=True);w=[[1,3],[3,1]]
 result=combine_pd_signals(signals,w,combination_scale=scale,combination_contract=contract).value
 expected=[.325,.35]if scale=='linear'else [.325,.65]if scale=='mixture'else expit(.25*logit([.1,.8])+.75*logit([.4,.2]))
 assert result['pd']==pytest.approx(expected)
 if scale=='mixture':assert np.array(result['mixture_within_regime_variance'])+result['mixture_between_regime_variance']==pytest.approx(np.array(expected)*(1-np.array(expected)))
 if scale=='log_odds':
  signals[0]['pd']=[0,1];signals[1]['pd']=[.4,.2]
  assert combine_pd_signals(signals,w,combination_scale=scale,combination_contract=contract).value['pd']==[0,1]
  signals[1]['pd']=[1,.2]
  with pytest.raises(ValueError,match='BOUNDARY_CONFLICT'):combine_pd_signals(signals,w,combination_scale=scale,combination_contract=contract)
 signals[1]['target']=dict(T,horizon=31)
 with pytest.raises(ValueError,match='ALIGNMENT'):combine_pd_signals(signals,w,combination_scale=scale,combination_contract=contract)

def test_stacking_only_oof_training_against_independent_direct_likelihood_fit():
 x,y,plan,spec=fixture();model=fit(x,y,plan,spec);new=[[.2,.3],[.8,.7]];actual=predict_pd(model,new,feature_order=['scorecard','structural'],expected_target=T).value
 direct=fit_penalized_glm(np.array(x[4:12]),y[4:12],feature_order=['scorecard','structural'],**{k:v for k,v in spec.items()if k!='input_scale'},**M)
 assert actual==pytest.approx(predict_pd(direct,new,feature_order=['scorecard','structural'],expected_target=T).value,abs=1e-12)
 y[-2:]=['unobserved hypothetical label',1];x[-2:]=[[-9,9],[9,-9]];changed=fit(x,y,plan,spec)
 assert changed.model['parameters']==model.model['parameters'] and changed.diagnostics['meta_data_fingerprint']==model.diagnostics['meta_data_fingerprint']
 assert model.diagnostics['meta_row_ids']==['r'+str(i)for i in range(4,12)]

def test_constant_bases_fit_intercept_and_detect_prediction_departure():
 x,y,plan,spec=fixture();x[4:12]=[[.2,.3]]*8;y[4:12]=[0,0,0,0,0,0,1,1];model=fit(x,y,plan,spec)
 predicted=predict_pd(model,[[.2,.3],[.5,.3]],feature_order=['scorecard','structural'],expected_target=T)
 assert predicted.value==pytest.approx([.25,.25],abs=1e-7)
 assert predicted.diagnostics['constant_base_departure_rows']==1 and model.model['parameters']['active_indices']==[]

def test_oof_temporal_group_maturity_and_lineage_leakage_rejected():
 x,y,plan,spec=fixture();bad=deepcopy(plan);bad['folds'][0]['base_lineage']['scorecard']['preprocessing_row_ids']=bad['folds'][0]['base_lineage']['scorecard']['preprocessing_row_ids']+['r4']
 with pytest.raises(ValueError,match='LEAKAGE'):fit(x,y,bad,spec)
 bad=deepcopy(plan);bad['rows'][4]['group_id']='g0'
 with pytest.raises(ValueError,match='LEAKAGE'):fit(x,y,bad,spec)
 bad=deepcopy(plan);bad['rows'][4]['complete_through']='2020-03-15'
 with pytest.raises(ValueError,match='MATURITY'):fit(x,y,bad,spec)
 bad=deepcopy(plan);bad['rows'][4]['prediction_known_at']='2020-03-02'
 with pytest.raises(ValueError,match='LEAKAGE'):fit(x,y,bad,spec)
 assert _hy_horizon_end('2024-01-31',dict(T,horizon=1,time_unit='months')).isoformat()=='2024-02-29'

def test_public_stacking_context_and_offset_rejection():
 from creditratingsystem.core import CreditRiskError,build_calculation_context
 x,y,plan,spec=fixture();model=fit(x,y,plan,spec);kw=dict(feature_order=['scorecard','structural'],expected_target=T)
 with pytest.raises(CreditRiskError):predict_pd(model,[[.2,.3]],offset=[0.],**kw)
 with pytest.raises(CreditRiskError):predict_pd(model,[[.2,.3]],context=build_calculation_context(horizon=2,time_unit='years'),**kw)

def test_pooling_invariants_and_weight_domain():
 from creditratingsystem.core import CreditRiskError
 signals=[dict(signal_id=s,model_id=s,model_version='1',row_ids=['a','b'],pd=[.1,.8],target=T,as_of='2024-01-01',known_at='2024-01-01')for s in ['first','second']]
 for scale in ['linear','log_odds','mixture']:
  contract=dict(interpretation={'linear':'estimator_pool','log_odds':'log_odds_pool','mixture':'latent_regime_mixture'}[scale],reference='hand invariants')
  if scale=='mixture':contract.update(regime_ids=['first','second'],conditional_regime_probabilities=True)
  kw=dict(combination_scale=scale,combination_contract=contract)
  weights=[[1,3],[3,1]]if scale=='mixture'else[1,3]
  assert combine_pd_signals(signals,weights,**kw).value['pd']==pytest.approx([.1,.8])
  different=deepcopy(signals);different[1]['pd']=[.9,.2]
  weights=[[1,0],[1,0]]if scale=='mixture'else[1,0]
  assert combine_pd_signals(different,weights,**kw).value['pd']==pytest.approx([.1,.8])
  for bad in [[[-1,2],[1,1]],[[0,0],[1,1]]]if scale=='mixture'else[[-1,2],[0,0]]:
   with pytest.raises(CreditRiskError):combine_pd_signals(signals,bad,**kw)
