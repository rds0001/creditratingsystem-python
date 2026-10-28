from copy import deepcopy
import pytest
from creditratingsystem._validation_policy import decision_policy_backtest
from creditratingsystem._decisioning_evaluation import evaluate_decision_grid
from creditratingsystem.core import _fingerprint,CreditRiskError
from test_decisioning_evaluation import fixture as grid_fixture

def fixture():
 kw=grid_fixture()
 for r in kw['candidate_inputs']:r['offer']['contract_terms']=dict(limit=100,annual_rate=.1,tenor_days=365,currency='EUR')
 kw['observed_outcomes'][0]['booked_offer_hash']=_fingerprint(kw['candidate_inputs'][0]['offer'])
 grid=evaluate_decision_grid(**kw).value
 d=dict(model_id='policy-validation',model_version='1',parameter_version='1',data_fingerprint=kw['evaluation_profile']['data_fingerprint'],population_id='applications',split_id='validation',prediction_origin='HOLDOUT',evaluation_cutoff=kw['evaluation_cutoff'],currency='EUR',target=kw['evaluation_profile']['target'])
 rows=[]
 for c,t in zip(kw['cases'],grid['candidates'][0]['decisions']):
  approved=t['action']=='APPROVE';rows.append(dict(prediction_id=c['case_id'],entity_id=c['case_id'],cluster_id=c['case_id'],segment='retail',case_id=c['case_id'],weight=1,as_of=c['decision_date'],forecast_known_at=c['decision_date'],horizon_end='2024-01-01',model_id=d['model_id'],model_version='1',parameter_version='1',historical_candidate_id='a',historical_decision=t['decision'],offered=approved,taken_up=approved,offered_limit=100 if approved else None,booked_amount=80 if approved else None,stages_known_at='2023-01-01',application_cost=2,cost_known_at='2023-01-02',cost_complete=True))
 return dict(evaluated_grid=grid,grid_inputs=kw,historical_cases=rows,design=d,reference_candidate_id='a')

def test_actual_booking_risk_and_paired_scenario_arithmetic():
 r=decision_policy_backtest(**fixture()).value
 assert r['metrics']['factual']['funded_rate']==.5 and r['metrics']['factual']['observed_pd']==1
 assert r['metrics']['factual']['mean_application_cost']==2
 assert r['metrics']['paired_candidate_minus_reference']['b|approval_rate']==.5
 assert r['metrics']['paired_candidate_minus_reference']['b|expected_npv_per_application']==2
 assert r['violation_counts']['CANDIDATE_SCENARIO_CONSTRAINT_VIOLATED']==1
 assert r['counterfactual']['full_policy_causal_value']is None

def test_booked_limit_and_offer_terms_cannot_be_reused_causally():
 kw=fixture();kw['historical_cases'][0]['booked_amount']=120;r=decision_policy_backtest(**kw).value
 assert r['violation_counts']['BOOKED_LIMIT_EXCEEDED']==1
 kw=fixture();kw['grid_inputs']['candidate_inputs'][2]['offer']['contract_terms']['limit']=200;kw['evaluated_grid']=evaluate_decision_grid(**kw['grid_inputs']).value
 r=decision_policy_backtest(**kw).value;assert r['candidate_observed_scopes']['b']['observed_pd']is None
 assert r['candidate_observed_scopes']['b']['case_status'][0]['status']=='DIFFERENT_BOOKED_OFFER'

def test_unfunded_outcome_and_future_policy_no_false_performance():
 kw=fixture();kw['grid_inputs']['observed_outcomes'][1]['defaulted']=False
 with pytest.raises(CreditRiskError):decision_policy_backtest(**kw)
 kw=fixture();kw['historical_cases'][0]['stages_known_at']='2024-02-01';r=decision_policy_backtest(**kw).value
 assert r['metrics']['factual']['funded_rate']==0 and r['metrics']['factual']['visible_stage_weight']==1
 # A correctly hashed mutation still fails comparison with actual original inputs.
 kw=fixture();kw['evaluated_grid']['candidates'][0]['metrics']['approval_rate']=1;g=kw['evaluated_grid'];g['sha256']=_fingerprint({k:v for k,v in g.items()if k!='sha256'})
 assert decision_policy_backtest(**kw).value['violation_counts']['FROZEN_GRID_RECONCILIATION']==1

def test_actual_shared_cluster_uncertainty():
 kw=fixture();kw.update(uncertainty='cluster_bootstrap',cluster_resamples=[['one','one'],['two','two']]);r=decision_policy_backtest(**kw).value
 ci=r['uncertainty']['intervals']['b|expected_npv_per_application'];assert ci['lower']==pytest.approx(.1)and ci['upper']==pytest.approx(3.9)

def gated_fixture():
 from creditratingsystem.decisioning import compile_credit_decision_policy
 kw=fixture();g=kw['grid_inputs']
 for c in g['policy_candidates']:
  s=deepcopy(c['policy']['specification']);s['schema_version']='1.1';s['fields']['operations.legal_status']=dict(type='category',values=['VERIFIED','UNVERIFIED']);s['approval_contract']=dict(domain='RETAIL_CREDIT',affordability=dict(mode='REQUIRED'),required_gates=[dict(field='operations.legal_status',value='VERIFIED')]);c['policy']=compile_credit_decision_policy(s).value
 g['evaluation_profile']['candidate_hashes']={p['candidate_id']:_fingerprint(p)for p in g['policy_candidates']}
 for r in g['candidate_inputs']:r['operations']['legal_status']='VERIFIED'
 kw['evaluated_grid']=evaluate_decision_grid(**g).value
 for r,t in zip(kw['historical_cases'],kw['evaluated_grid']['candidates'][0]['decisions']):r['historical_decision']=t['decision']
 return kw

def test_schema11_gate_and_historical_policy_dates():
 kw=gated_fixture();r=decision_policy_backtest(**kw).value
 assert not any(v['code']=='HISTORICAL_DECISION_RECONCILIATION'for v in r['violations'])
 assert kw['historical_cases'][0]['historical_decision']['approval_domain']=='RETAIL_CREDIT'
 kw['grid_inputs']['candidate_inputs'][0]['operations']['legal_status']='UNVERIFIED';kw['evaluated_grid']=evaluate_decision_grid(**kw['grid_inputs']).value;kw['historical_cases'][0]['historical_decision']=kw['evaluated_grid']['candidates'][0]['decisions'][0]['decision']
 assert decision_policy_backtest(**kw).value['violation_counts']['OFFER_WITHOUT_POLICY_APPROVAL']==1

def expired_fixture():
 from creditratingsystem.decisioning import compile_credit_decision_policy
 kw=fixture();g=kw['grid_inputs'];p=g['policy_candidates'][0];s=deepcopy(p['policy']['specification']);s.update(known_at='2022-01-01',effective_from='2022-01-01',effective_until='2023-01-01');p['policy']=compile_credit_decision_policy(s).value;g['evaluation_profile']['candidate_hashes']={p['candidate_id']:_fingerprint(p)for p in g['policy_candidates']};kw['evaluated_grid']=evaluate_decision_grid(**g).value
 for r,t in zip(kw['historical_cases'],kw['evaluated_grid']['candidates'][0]['decisions']):r['historical_decision']=t['decision']
 return kw

def test_expired_history_future_fact_exclusion_and_cluster_consistency():
 r=decision_policy_backtest(**expired_fixture()).value
 assert r['violation_counts']['HISTORICAL_POLICY_NOT_EFFECTIVE']==2
 kw=fixture();kw['historical_cases'][0].update(stages_known_at='2025-01-01',booked_amount=120)
 assert 'BOOKED_LIMIT_EXCEEDED'not in decision_policy_backtest(**kw).value['violation_counts']
 kw=fixture();kw['historical_cases'][1]['entity_id']='one'
 with pytest.raises(CreditRiskError,match='VALIDATION_DEPENDENCE'):decision_policy_backtest(**kw)

def test_no_violations_is_explicit_empty_object():
 kw=fixture();kw['grid_inputs']['evaluation_profile']['constraints'][0]['threshold']=2;kw['evaluated_grid']=evaluate_decision_grid(**kw['grid_inputs']).value
 r=decision_policy_backtest(**kw).value;assert r['violation_counts']=={}and r['decision']['status']=='NOT_ASSESSED'


def first_draw_fixture():
 from creditratingsystem.decisioning import compile_credit_decision_policy
 kw=fixture();g=kw['grid_inputs'];t=dict(g['evaluation_profile']['target'],event='first_default',performance_origin='first_draw_at')
 g['evaluation_profile'].update(target=t,target_contract_version='1.1')
 for candidate in g['policy_candidates']:
  spec=deepcopy(candidate['policy']['specification']);spec['target']=t;candidate['policy']=compile_credit_decision_policy(spec).value
 g['evaluation_profile']['candidate_hashes']={p['candidate_id']:_fingerprint(p)for p in g['policy_candidates']}
 for row in g['candidate_inputs']:row['risk']['target']=t;row['offer']['target']=t
 g['observed_outcomes'][0]['booked_offer_hash']=_fingerprint(g['candidate_inputs'][0]['offer'])
 kw['evaluated_grid']=evaluate_decision_grid(**g).value
 for row,decision in zip(kw['historical_cases'],kw['evaluated_grid']['candidates'][0]['decisions']):row['historical_decision']=decision['decision']
 kw['design'].update(target={k:v for k,v in t.items()if k!='performance_origin'},policy_target_contract_version='1.1',performance_origin='first_draw_at')
 return kw

def test_versioned_first_draw_target_binding_and_legacy_default():
 from creditratingsystem._decisioning_evaluation import select_constrained_policy
 legacy=fixture();assert 'target_contract_version'not in legacy['grid_inputs']['evaluation_profile']
 assert decision_policy_backtest(**legacy).value['metrics']['factual']['observed_pd']==1
 kw=first_draw_fixture();original=deepcopy(kw['grid_inputs']['policy_candidates'])
 result=decision_policy_backtest(**kw).value
 assert result['metrics']['factual']['observed_pd']==1 and result['metrics']['factual']['funded_rate']==.5
 assert kw['grid_inputs']['policy_candidates']==original
 assert kw['evaluated_grid']['profile']['target']['event']=='first_default'
 assert select_constrained_policy(kw['evaluated_grid']).value['selected_candidate_id']=='a'
 for field,value in [('policy_target_contract_version','1.0'),('policy_target_contract_version','9'),('performance_origin','application_at')]:
  bad=deepcopy(kw);bad['design'][field]=value
  with pytest.raises(CreditRiskError):decision_policy_backtest(**bad)
 for version,event,origin in [('1.0','first_default','first_draw_at'),('9','first_default','first_draw_at'),('1.1','delinquency','first_draw_at'),('1.1','first_default','application_at')]:
  bad=deepcopy(kw['grid_inputs']);bad['evaluation_profile']['target_contract_version']=version;bad['evaluation_profile']['target'].update(event=event,performance_origin=origin)
  with pytest.raises(CreditRiskError):evaluate_decision_grid(**bad)
