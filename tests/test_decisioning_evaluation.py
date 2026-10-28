from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.core import _fingerprint
from creditratingsystem.decisioning import compile_credit_decision_policy
from creditratingsystem._decisioning_evaluation import evaluate_decision_grid,select_constrained_policy,assess_policy_robustness
D='2023-01-01';E='2024-01-02'
def target():return dict(event='default',horizon=365,time_unit='days',probability_measure='PHYSICAL',conditioning='initially_performing')
def policy(cutoff=.05):return compile_credit_decision_policy(dict(schema_version='1.0',policy_id='p'+str(cutoff),version='1',approval_reference='bank-reference',known_at=D,effective_from=D,effective_until='2024-01-01',default_action='DECLINE',missing_action='BLOCKED',target=target(),fields={'risk.pd':{'type':'number'}},rules=[dict(rule_id='approve',priority=1,action='APPROVE',reason_code='PD_CUTOFF',when=[dict(field='risk.pd',op='le',value=cutoff)])])).value
def sources(pd):
 timing=dict(as_of=D,known_at=D)
 return dict(risk=dict(timing,pd=pd,target=target(),freshness='CURRENT'),affordability=dict(timing,status='AFFORDABLE'),eligibility=dict(timing,status='ELIGIBLE'),operations=dict(timing,status='CLEAR'))
def offer(id,profit=10):return dict(offer_id=id,version='1',known_at=D,currency='EUR',target=target(),risk_conditioning='CONDITIONAL_ON_THIS_ACCEPTED_OFFER',assumption_reference='scenario',support_status='SUPPORTED',takeup_probability=.5,conditional_booked_expected_loss=2,conditional_booked_npv=profit,ead=100,offer_cost=1)
def fixture():
 policies=[dict(candidate_id='a',policy=policy(.05),offer_rule_reference='rule-a'),dict(candidate_id='b',policy=policy(.2),offer_rule_reference='rule-b')]
 cases=[dict(case_id=id,decision_date=D,weight=1,split_id='validation',prediction_reference='held-out',excluded_from_model_training=True,excluded_from_calibration=True)for id in ['one','two']]
 inputs=[dict(candidate_id=p['candidate_id'],case_id=c['case_id'],**sources(pd),offer=offer(c['case_id']),nonapproval_costs={'REFER':2,'DECLINE':0,'BLOCKED':0})for p in policies for c,pd in zip(cases,[.01,.1])]
 outcomes=[dict(case_id='one',funded=True,complete=True,booked_offer_hash=_fingerprint(offer('one')),funded_at=D,funding_known_at=D,outcome_end='2024-01-01',known_at=E,defaulted=True,economic_loss=50,net_profit=-40),dict(case_id='two',funded=False,complete=False,booked_offer_hash=None,funded_at=None,funding_known_at=None,outcome_end=None,known_at=None,defaulted=None,economic_loss=None,net_profit=None)]
 profile=dict(profile_id='selection',version='1',known_at=D,currency='EUR',registered_at=D,outcome_access_date=E,evaluation_role='VALIDATION',prediction_design='HELD_OUT',split_id='validation',data_fingerprint='bank-declared-data',target=target(),candidate_hashes={p['candidate_id']:_fingerprint(p)for p in policies},objective='expected_npv_per_application',constraints=[dict(constraint_id='loss',metric='expected_loss_per_application',operator='le',threshold=.75)],tie_break='candidate_id',tolerance=1e-12,loss_bounds=[0,100],profit_bounds=[-100,100])
 return dict(policy_candidates=policies,cases=cases,candidate_inputs=inputs,observed_outcomes=outcomes,evaluation_profile=profile,evaluation_cutoff=E)

def test_policy_grid_scenario_vs_observed_funded_only():
 r=evaluate_decision_grid(**fixture()).value;a,b=r['candidates']
 assert a['metrics']['approval_rate']==.5 and b['metrics']['approval_rate']==1
 assert a['metrics']['expected_loss_per_application']==.5 and b['metrics']['expected_loss_per_application']==1
 assert a['metrics']['expected_npv_per_application']==2 and b['metrics']['expected_npv_per_application']==4
 assert a['status']=='FEASIBLE' and b['status']=='INFEASIBLE'
 assert b['observed_outcomes']['observed_pd']==1
 assert b['observed_outcomes']['mature_matched_weight']==1
 assert b['observed_outcomes']['case_status'][1]['status']=='NOT_FUNDED'
 assert b['observed_outcomes']['full_candidate_causal_value']is None
 selected=select_constrained_policy(r).value
 assert selected['selected_candidate_id']=='a' and selected['objective_value']==2

def test_missing_funded_outcomes_bounds_not_nondefaults():
 f=fixture();o=f['observed_outcomes'][0];o.update(complete=False,known_at=None,defaulted=None,economic_loss=None,net_profit=None)
 r=evaluate_decision_grid(**f).value['candidates'][0]['observed_outcomes']
 assert r['observed_pd']is None and r['matched_funded_mean_loss_bounds']==[0,100]
 assert r['matched_funded_mean_profit_bounds']==[-100,100]
 assert r['missing_outcome_weight']==1
 f['observed_outcomes'][1]['defaulted']=False
 with pytest.raises(CreditRiskError,match='DECISION_OUTCOME'):evaluate_decision_grid(**f)

def test_unidentified_offer_and_selection_infeasible():
 f=fixture()
 for row in f['candidate_inputs']:row['offer']['support_status']='UNSUPPORTED'
 r=evaluate_decision_grid(**f).value
 assert all(x['status']=='UNIDENTIFIED'for x in r['candidates'])
 assert all(x['metrics']['expected_npv_per_application']is None for x in r['candidates'])
 assert select_constrained_policy(r).value['status']=='INFEASIBLE'

def test_selection_no_test_or_training_or_posthoc_candidate():
 f=fixture();f['evaluation_profile']['evaluation_role']='TEST';r=evaluate_decision_grid(**f).value
 with pytest.raises(CreditRiskError,match='DECISION_SPLIT'):select_constrained_policy(r)
 f=fixture();f['cases'][0]['excluded_from_model_training']=False
 with pytest.raises(CreditRiskError,match='DECISION_SPLIT'):evaluate_decision_grid(**f)
 f=fixture();f['policy_candidates'][0]['offer_rule_reference']='changed'
 with pytest.raises(CreditRiskError,match='DECISION_SELECTION_CONTRACT'):evaluate_decision_grid(**f)
 f=fixture();f['evaluation_profile']['registered_at']='2024-01-03'
 with pytest.raises(CreditRiskError,match='DECISION_SELECTION_CONTRACT'):evaluate_decision_grid(**f)

def test_dominance_and_deterministic_ties():
 f=fixture();f['evaluation_profile']['constraints'][0]['threshold']=2
 r=evaluate_decision_grid(**f).value
 assert select_constrained_policy(r).value['selected_candidate_id']=='b'
 # Same policy choices/utility, but b lower loss: domination resolves objective tie.
 f['policy_candidates'][0]['policy']=deepcopy(f['policy_candidates'][1]['policy'])
 f['evaluation_profile']['candidate_hashes']={p['candidate_id']:_fingerprint(p)for p in f['policy_candidates']}
 for row in f['candidate_inputs']:
  if row['candidate_id']=='b':row['offer']['conditional_booked_expected_loss']=1
 r=select_constrained_policy(evaluate_decision_grid(**f)).value
 assert r['selected_candidate_id']=='b' and r['dominance']==[{'candidate_id':'a','dominated_by':'b'}]

def robust_fixture():
 rows=[dict(case_id=id,decision_date=D,weight=1,**sources(pd),offer_terms=dict(offer_id=id,version='1',amount=100,annual_rate=.1),utility_reference='scenario-values',utility_by_action={'APPROVE':10,'REFER':-1,'DECLINE':0,'BLOCKED':-2},feasible_by_action={a:True for a in ['APPROVE','REFER','DECLINE','BLOCKED']})for id,pd in [('near',.049),('far',.01)]]
 low=deepcopy(rows);high=deepcopy(rows);high[0]['risk']['pd']=.051
 profile=dict(profile_id='robust',version='1',known_at=D,currency='EUR',target=target(),scenario_design='BOUNDED_GRID',generation_reference='two-shared-shocks',interpretation='PERTURBATION_NOT_CAUSAL',tolerance=1e-12)
 return dict(policy=policy(),baseline_cases=rows,perturbation_scenarios=[dict(scenario_id='low',weight=.5,perturbation_reference='low',cases=low),dict(scenario_id='high',weight=.5,perturbation_reference='high',cases=high)],robustness_profile=profile,as_of=D)

def test_fixed_policy_shared_perturbation_flip_and_regret():
 f=robust_fixture();r=assess_policy_robustness(**f).value
 assert r['weighted_action_flip_rate']==.25 and r['worst_case_mean_regret']==5
 high=r['scenarios'][0]
 assert high['cases'][1]['action_flipped'] and high['cases'][1]['changed_input_sources']==['risk']
 assert not high['cases'][0]['action_flipped']
 assert r['weighted_known_infeasible_action_rate']==0

def test_robustness_unknown_utilities_and_feasibility_separate():
 f=robust_fixture();f['perturbation_scenarios'][0]['cases'][0]['utility_by_action']['APPROVE']=None
 f['perturbation_scenarios'][1]['cases'][1]['feasible_by_action']['APPROVE']=False
 r=assess_policy_robustness(**f).value
 assert r['worst_case_mean_regret']is None and 'utility_regret'in r['unidentified_dimensions']
 assert r['weighted_known_infeasible_action_rate']==.25
 f=robust_fixture();f['perturbation_scenarios'][0]['weight']=1
 with pytest.raises(CreditRiskError,match='DECISION_ROBUSTNESS_CONTRACT'):assess_policy_robustness(**f)

def test_funding_knowledge_cutoff_and_selected_policy_binding():
 f=fixture();f['observed_outcomes'][0]['funding_known_at']='2024-01-03'
 r=evaluate_decision_grid(**f).value
 assert r['candidates'][0]['observed_outcomes']['matched_funded_weight']==0
 assert r['candidates'][0]['observed_outcomes']['case_status'][0]['status']=='FUTURE_FUNDING_EVIDENCE'
 r=evaluate_decision_grid(**fixture()).value
 r['candidates'][0]['policy']=policy(.3)
 r['sha256']=_fingerprint({k:v for k,v in r.items()if k!='sha256'})
 with pytest.raises(CreditRiskError,match='DECISION_SELECTION_CONTRACT'):select_constrained_policy(r)
