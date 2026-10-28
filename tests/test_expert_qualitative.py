from copy import deepcopy
import pytest
from creditratingsystem._expert_qualitative import compile_qualitative_rubric,evaluate_qualitative_evidence
T=dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')
C=dict(rubric_id='synthetic',version='1',population_id='SME',reference='synthetic qualitative exercise',known_at='2024-01-01',ordered_grades=['A','B','C','D'],score_direction='higher_is_worse',target=T)
P=dict(score_floor=0,score_cap=80,minimum_evidence_coverage=0,unresolved_interaction_policy='apply_adverse')
D=dict(valid_from='2024-01-01',valid_until='2025-01-01',known_at='2024-01-01')

def inputs():
 common=dict(weight=1,required_evidence_types=['audit'],minimum_independent_sources=1,missing_policy='worst',not_applicable_policy='refer')
 criteria=[dict(common,criterion_id='management',kind='categorical',categories=['strong','weak'],minimum_independent_sources=2,levels=[dict(level_id='strong',values=['strong'],points=0),dict(level_id='weak',values=['weak'],points=100)]),dict(common,criterion_id='debt',kind='numeric',risk_order='increasing',weight=2,levels=[dict(level_id='low',lower=0,upper=2,points=20),dict(level_id='stress',lower=2,upper=5,points=70),dict(level_id='high',lower=5,upper=10,points=100)]),dict(common,criterion_id='permit',kind='categorical',categories=['valid','expired'],levels=[dict(level_id='valid',values=['valid'],points=0),dict(level_id='expired',values=['expired'],points=100)])]
 interactions=[dict(interaction_id='joint',conditions=dict(management=['weak'],debt=['stress','high']),points_delta=15,reference='documented adverse combination')]
 limits=[dict(limit_id='management-limit',conditions=dict(management=['weak']),best_allowed_grade='B',worst_allowed_grade='D',reference='rating policy')]
 ev=[dict(D,evidence_id='e'+str(i),evidence_type='audit',source_id='source'+str(i),independence_group='group'+str(i),content_fingerprint='content'+str(i),status='VERIFIED')for i in range(4)]
 obs=[dict(D,observation_id='o1',criterion_id='management',value='weak',not_applicable=False,evidence_ids=['e0','e1'],observed_at='2024-01-01'),dict(D,observation_id='o2',criterion_id='debt',value=2,not_applicable=False,evidence_ids=['e2'],observed_at='2024-01-01'),dict(D,observation_id='o3',criterion_id='permit',value='valid',not_applicable=False,evidence_ids=['e3'],observed_at='2024-01-01')]
 return criteria,interactions,limits,obs,ev

def compile_inputs(criteria,interactions,limits,policy=P):return compile_qualitative_rubric(criteria,rubric_contract=C,score_policy=policy,interactions=interactions,rating_limits=limits)
def evaluate(r,obs,ev,**kwargs):return evaluate_qualitative_evidence(r,obs,ev,as_of='2024-04-01',known_at='2024-04-01',**kwargs).value

def test_qualitative_hand_score_interaction_rating_limit_and_exact_boundary():
 criteria,interactions,limits,obs,ev=inputs();before=deepcopy(criteria);r=compile_inputs(criteria,interactions,limits);result=evaluate(r,obs,ev)
 assert result['score']==75 and result['score_interval']==[75,75] and result['status']=='READY'
 assert result['permitted_grade_interval']==['B','D'] and result['evidence_coverage']==1
 assert criteria==before
 obs[1]['value']=10;assert evaluate(r,obs,ev)['score']==80
 obs[1]['value']=10.1;assert evaluate(r,obs,ev)['status']=='REFER'

def test_missing_verified_independence_and_temporal_evidence_never_bonus():
 criteria,interactions,limits,obs,ev=inputs();r=compile_inputs(criteria,interactions,limits);base=evaluate(r,obs,ev)['score']
 assert evaluate(r,obs[:-1],ev)['score']>=base
 ev[3]['known_at']='2024-06-01';result=evaluate(r,obs,ev)
 assert result['score']>=base and result['evidence_coverage']==.75 and result['excluded_records'][0]['reason']=='NOT_KNOWN'
 ev[1]['independence_group']=ev[0]['independence_group'];result=evaluate(r,obs,ev)
 assert result['status']=='REFER' and result['evidence_coverage']==.5 # unknown rating limit refers
 future=dict(obs[0],observation_id='future',value='strong',known_at='2024-08-01',observed_at='2024-08-01',valid_from='2024-08-01')
 assert evaluate(r,obs+[future],ev)['score']==result['score']

def test_conflicting_facts_require_dated_resolution_and_preserve_originals():
 criteria,interactions,limits,obs,ev=inputs();r=compile_inputs(criteria,interactions,limits);obs.append(dict(obs[0],observation_id='o4',value='strong'))
 result=evaluate(r,obs,ev);assert result['status']=='REFER' and result['score'] is None
 conflict=dict(D,conflict_id='c1',observation_ids=['o1','o4'],status='RESOLVED',resolution_observation_id='o1',reference='reviewed original records')
 fixed=evaluate(r,obs,ev,conflicts=[conflict]);assert fixed['score']==75 and fixed['status']=='READY'
 conflict['known_at']='2024-05-01';assert evaluate(r,obs,ev,conflicts=[conflict])['status']=='REFER'
 assert obs[-1]['value']=='strong'

def test_compile_rejects_overlap_and_unjustified_missing_and_nonapplicable_policies():
 criteria,interactions,limits,obs,ev=inputs();bad=deepcopy(criteria);bad[1]['levels'][0]['upper']=3
 with pytest.raises(ValueError,match='COVERAGE'):compile_inputs(bad,interactions,limits)
 bad=deepcopy(criteria);bad[2]['missing_policy']='exclude'
 with pytest.raises(ValueError,match='MISSING'):compile_inputs(bad,interactions,limits)
 criteria[2].update(not_applicable_policy='fixed',not_applicable_points=50);r=compile_inputs(criteria,interactions,limits);obs[2].update(not_applicable=True,value=None)
 assert evaluate(r,obs,ev)['score']==80
 ev[3]['status']='UNVERIFIED';result=evaluate(r,obs,ev);assert next(x for x in result['criterion_results']if x['criterion_id']=='permit')['points']==100
 criteria[2]['missing_policy']='refer';r=compile_inputs(criteria,interactions,limits);assert evaluate(r,obs,ev)['score'] is None
