from copy import deepcopy
import pytest
from creditratingsystem._validation_selection import selection_bias_sensitivity
from creditratingsystem.selection_bias import selection_ipw_weights,reject_inference_bounds,reject_inference_sensitivity
from creditratingsystem.core import CreditRiskError

def fixture():
 target=dict(event='default',horizon=1,time_unit='months',probability_measure='PHYSICAL',conditioning='application')
 d=dict(model_id='comparison',model_version='1',parameter_version='1',data_fingerprint='applications',population_id='through-door',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-02-02',target=target)
 rows=[]
 for i in range(4):
  rows.append(dict(prediction_id=str(i),application_id=str(i),entity_id=str(i),cluster_id=str(i),segment='retail',as_of='2024-01-01',forecast_known_at='2024-01-01',horizon_end='2024-02-01',known_at='2024-02-02',complete_through='2024-02-01',model_id='comparison',model_version='1',parameter_version='1',funded=i<2,label_origin='OBSERVED_OWN_CONTRACT'if i<2 else'PSEUDO',outcome=i if i<2 else 1,target=target,weight=1))
 candidates=[dict(model_id=m,model_hash='frozen-'+m,training_entities=[],training_known_at='2023-12-01',predictions=[dict(application_id=str(i),pd=p,known_at='2024-01-01')for i,p in enumerate(values)])for m,values in [('a',[.2,.8,.6,.6]),('b',[.4,.6,.6,.6])]]
 scenarios=[]
 for kind,inputs,fn in [('IPW',dict(observed=[1,1,0,0],propensities=[.5,.5,.5,.5],target_weights=[1,1,1,1]),selection_ipw_weights),('BOUNDS',dict(outcomes=[0,1,None,None],observed=[1,1,0,0],target_weights=[1,1,1,1]),reject_inference_bounds),('MNAR_SHIFT',dict(outcomes=[0,1,None,None],observed=[1,1,0,0],target_weights=[1,1,1,1],baseline_pd=[.2,.8,.6,.6],log_odds_shifts=[0]),reject_inference_sensitivity)]:
  scenarios.append(dict(result_id=kind,kind=kind,model_id='a',population_id='through-door',target=target,application_ids=['0','1','2','3'],inputs=inputs,result=fn(**inputs).value))
 return dict(candidates=candidates,evaluation_rows=rows,selection_results=scenarios,design=d,reference_model_id='a')

def test_factual_pseudo_separation_and_independent_arithmetic():
 r=selection_bias_sensitivity(**fixture()).value;m=r['metrics'];assert m['funded_count']==2 and m['mature_factual_count']==2
 assert m['evidence_scopes']['FUNDED_OBSERVED_TEST']['a']['brier']==pytest.approx(.04)
 assert m['paired_candidate_minus_reference']['FUNDED_OBSERVED_TEST|b|brier']==pytest.approx(.12)
 s=r['diagnostics']['transport_scenarios'];assert s[0]['recomputed_result']['effective_sample_size']==2
 assert s[1]['recomputed_result']['lower']==.25 and s[1]['recomputed_result']['upper']==.75
 assert s[2]['recomputed_result']['surface'][0]['population_pd']==.55
 assert all(x['verified']for x in s) and r['decision']['status']=='NOT_ASSESSED'

def test_zero_support_trim_and_changed_result():
 kw=fixture();s=kw['selection_results'][0];s['inputs']['propensities']=[.5,.5,0,0];s['result']=None
 r=selection_bias_sensitivity(**kw).value;assert r['diagnostics']['transport_scenarios'][0]['status']=='NOT_IDENTIFIED_ZERO_SUPPORT'
 s['inputs']['trim']=[.1,1];s['result']=selection_ipw_weights(**s['inputs']).value;r=selection_bias_sensitivity(**kw).value
 assert r['diagnostics']['transport_scenarios'][0]['recomputed_result']['excluded_target_mass']==2
 s['result']['effective_sample_size']=3;assert selection_bias_sensitivity(**kw).value['decision']['status']=='BLOCKED'

def test_synthetic_scope_incomplete_negative_and_actual_funding():
 kw=fixture();kw['selection_results']=[];kw['evaluation_rows'][2].update(label_origin='SYNTHETIC_TRUTH',data_origin='SYNTHETIC');kw['evaluation_rows'][0]['complete_through']='2024-01-20'
 r=selection_bias_sensitivity(**kw).value;assert r['metrics']['evidence_scopes']['SYNTHETIC_ENGINEERING_ONLY']['a']['n']==1
 assert r['metrics']['mature_factual_count']==1 and r['coverage']['exclusion_counts']['INCOMPLETE_NEGATIVE_COVERAGE']==1
 kw=fixture();kw['evaluation_rows'][0]['funded']=False
 with pytest.raises(CreditRiskError):selection_bias_sensitivity(**kw)

def test_paired_bootstrap_and_no_pseudolabel_transport():
 kw=fixture();kw.update(uncertainty='cluster_bootstrap',cluster_resamples=[['0','1','0','1'],['0','1','2','3']]);r=selection_bias_sensitivity(**kw).value
 assert r['uncertainty']['intervals']['FUNDED_OBSERVED_TEST|b|brier']['lower']==pytest.approx(.12)
 kw['selection_results'][1]['inputs']['outcomes'][2]=1
 with pytest.raises(CreditRiskError):selection_bias_sensitivity(**kw)

def aipw_fixture():
 from creditratingsystem.selection_bias import estimate_population_risk_aipw
 kw=fixture();kw['evaluation_rows']=kw['evaluation_rows'][:3];rows=kw['evaluation_rows'];rows[0].update(as_of='2022-01-01',forecast_known_at='2022-01-01',horizon_end='2022-02-01',known_at='2022-02-02',weight=0,label_origin='PSEUDO',funded=False);rows[1]['outcome']=0;rows[2].update(outcome=1,label_origin='OBSERVED_OWN_CONTRACT',funded=True)
 for c in kw['candidates']:
  c['predictions']=c['predictions'][:3];c['training_known_at']='2021-12-01';c['predictions'][0]['known_at']='2022-01-01'
 inputs=dict(observed=[0,1,1],outcomes=[None,0,1],target_weights=[0,1,1],outcome_predictions=[.2,.2,.2],propensities=[.5,.5,.5],fold_lineage=dict(nuisance_prediction_origin='OUT_OF_FOLD',folds=[dict(train=[0],validation=[1,2])],rows=[dict(entity_id=r['entity_id'],as_of=r['as_of'],label_known_at=r['known_at'],features_known_at=r['as_of'])for r in rows]))
 kw['selection_results']=[dict(result_id='aipw',kind='AIPW',model_id='a',population_id='through-door',target=kw['design']['target'],application_ids=['0','1','2'],inputs=inputs,result=estimate_population_risk_aipw(**inputs).value)]
 return kw

def test_frozen_oof_aipw_arithmetic():
 r=selection_bias_sensitivity(**aipw_fixture()).value;s=r['diagnostics']['transport_scenarios'][0]
 assert s['verified']and s['recomputed_result']['population_pd']==pytest.approx(.8)
 assert s['recomputed_result']['standard_error']==pytest.approx(1)
