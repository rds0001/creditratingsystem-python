"""Hand adjacent merges, immutable boundary replay and local cohort ratios."""
from copy import deepcopy
import math
import pytest
from creditratingsystem._retail_pools import fit_retail_risk_pools,assign_retail_risk_pool,estimate_pool_risk_parameters
from creditratingsystem.core import CreditRiskError
STRATA=dict(product='card',borrower_type='person',facility_type='revolving',delinquency_state='current')
def target():return dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='not_defaulted_at_index',default_unit='facility',default_definition='explicit local default policy',currency='EUR')
def spec():return dict(spec_id='retail',version='1',known_at='2023-01-01',stratum_fields=list(STRATA),score_field='score',segments=[dict(segment_id='card_current',values=STRATA,cutpoints=[25,50,75],score_min=0,score_max=100,monotonicity=dict(pd='increasing',lgd='none',ead='none'))],minimum_cases=2,minimum_defaults=0,minimum_lgd_count=0,minimum_ead_count=0,maximum_bin_mean_spread=dict(pd=None,lgd=None,ead=None))
def training_rows():
    rows=[]
    for score,labels in [(10,[False,False]),(30,[False,True]),(60,[False,False,False,True]),(80,[True,True])]:
        for default in labels:
            id=str(len(rows));rows.append(dict(observation_id=id,default_unit_id=id,role='TRAIN',index_date='2023-01-01',feature_observed_at='2023-01-01',feature_known_at='2023-01-01',label_window_end='2023-01-31',label_known_at='2023-02-01',defaulted=default,default_date='2023-01-10'if default else None,lgd_known_at='2023-02-01'if default else None,ead_known_at='2023-01-10'if default else None,lgd_status='CLOSED'if default else 'NOT_DEFAULTED',lgd=.4 if default else None,ead=100 if default else None,score=score,**STRATA))
    return rows

def fit(policy=None,rows=None):return fit_retail_risk_pools(training_rows()if rows is None else rows,spec()if policy is None else policy,target=target(),training=dict(role='TRAIN',known_at='2023-02-01',split_id='train',data_fingerprint='local-source',data_origin='SYNTHETIC'),model_id='retail',model_version='1')
def snapshot(id,score):return dict(observation_id=id,index_date='2024-01-01',feature_observed_at='2024-01-01',feature_known_at='2024-01-01',score=score,**STRATA)
def assign(rows,model=None,policy='unassigned'):return assign_retail_risk_pool(rows,fit()if model is None else model,as_of='2024-01-01',information_cutoff='2024-01-01',expected_target=target(),uncovered_policy=policy)
def profile():return dict(profile_id='local',version='1',known_at='2024-01-01',pd_weighting='reference_exposure',lgd_weighting='ead',ead_weighting='episode',ccf_weighting='undrawn',confidence=.95,independence_assumption='independent_default_units',minimum_pool_cases=2)
def outcomes():return [dict(observation_id=str(i),default_unit_id=str(i),index_date='2024-01-01',label_window_end='2024-01-31',label_known_at='2024-01-31',outcome_status='MATURE',reference_exposure=w,reference_known_at='2024-01-01',defaulted=i<3,default_date='2024-01-10'if i<3 else None)for i,w in enumerate([100,300,200,400,100,100])]
def workout(id,ead,recovery,status):return dict(episode_id='episode'+id,observation_id=id,default_date='2024-01-10',default_known_at='2024-01-10',ead_at_default=ead,episode_status=status,valuation_known_at='2024-02-01',reference_date='2024-01-01',reference_known_at='2024-01-01',reference_drawn=50,reference_limit=150,recovery_cashflows=[dict(cashflow_id='cash',cashflow_date='2024-01-20',known_at='2024-01-20',currency='EUR',kind='RECOVERY',amount=recovery)],discount_curve=dict(anchor_date='2024-01-10',known_at='2024-01-10',discount_basis='explicit unit factors',points=[dict(cashflow_date='2024-01-20',discount_factor=1)]),fx_contract=dict(reporting_currency='EUR',known_at='2024-01-20',conversion_basis='cashflow_date',rates=[dict(currency='EUR',date='2024-01-20',rate=1)]))
def estimation_args():return dict(pool_assignments=assign([snapshot(str(i),10 if i<4 else 80)for i in range(6)]),cohort_outcomes=outcomes(),workouts=[workout('0',100,40,'CLOSED'),workout('1',300,240,'CLOSED'),workout('2',200,100,'OPEN')],estimation_profile=profile(),evaluation_cutoff='2024-02-01',calibration_id='cal',calibration_version='1')
def test_monotone_hand_merge_preserves_counts_and_separate_parameters():
    m=fit().model;s=m['parameters']['segments'][0]
    assert s['status']=='APPLICABLE' and [(p['first_bin'],p['last_bin'])for p in s['pools']]==[(0,0),(1,2),(3,3)]
    assert [p['statistics']['means']['pd']for p in s['pools']]==[0,2/6,1]
    assert sum(p['statistics']['cases']for p in s['pools'])==10 and sum(p['statistics']['defaults']for p in s['pools'])==4
    assert s['pools'][1]['initial_bin_mean_spread']['pd']==.25
    assert m['parameters']['merge_path'][0]['reason']=='MONOTONICITY:pd'
    assert not m['diagnostics']['global_feasibility_claim']

def test_support_merges_and_blocked_homogeneity_unassignable():
    p=spec();p['minimum_cases']=3;m=fit(p).model;s=m['parameters']['segments'][0];assert s['status']=='APPLICABLE' and len(s['pools'])==1
    # Local closest-neighbor merging first combines bins1+2; supporting the
    # remaining endpoint cells then merges all10. No global-optimality claim.
    assert [(m['left_first'],m['right_last'])for m in m['parameters']['merge_path']]==[(1,2),(0,2),(0,3)]
    p=spec();p['maximum_bin_mean_spread']['pd']=.1;m=fit(p)
    assert m.model['parameters']['segments'][0]['status']=='INSUFFICIENT_SUPPORT'
    assert assign([snapshot('a',10)],m).value['assignments'][0]['status']=='INSUFFICIENT_SUPPORT'
    p=spec();p['minimum_cases']=11;assert fit(p).model['parameters']['segments'][0]['status']=='INSUFFICIENT_SUPPORT'

def test_exact_cutpoints_unseen_missing_and_integrity():
    r=assign([snapshot(str(i),s)for i,s in enumerate([0,25,50,75,100])]).value['assignments'];assert [x['score_bin']for x in r]==[0,1,2,3,3]
    assert r[1]['pool_id']==r[2]['pool_id'] and r[2]['pool_id']!=r[3]['pool_id']
    unknown=snapshot('a',10);unknown['product']='unseen';assert assign([unknown]).value['assignments'][0]['status']=='UNSEEN_STRATUM'
    assert assign([snapshot('a',None)]).value['assignments'][0]['status']=='MISSING_FEATURE'
    assert assign([snapshot('a',101)]).value['assignments'][0]['status']=='OUTSIDE_SCORE_DOMAIN'
    with pytest.raises(CreditRiskError):assign([unknown],policy='error')
    model=deepcopy(fit().model);model['parameters']['segments'][0]['cutpoints'][0]=26
    with pytest.raises(CreditRiskError):assign([snapshot('a',10)],model)

def test_training_only_maturity_feature_chronology_and_default_units():
    rows=training_rows();rows[0]['role']='VALIDATION'
    with pytest.raises(CreditRiskError):fit(rows=rows)
    rows=training_rows();rows[0]['label_known_at']='2023-02-02'
    with pytest.raises(CreditRiskError):fit(rows=rows)
    rows=training_rows();rows[0]['feature_known_at']='2023-01-02'
    with pytest.raises(CreditRiskError):fit(rows=rows)
    rows=training_rows();rows[3]['lgd_known_at']='2023-02-02'
    with pytest.raises(CreditRiskError):fit(rows=rows)
    rows=training_rows();rows[1]['default_unit_id']=rows[0]['default_unit_id']
    with pytest.raises(CreditRiskError):fit(rows=rows)

def test_local_pd_lgd_ead_denominators_open_workout_and_zero_default_bound():
    r=estimate_pool_risk_parameters(**estimation_args());table=r.value['pool_risk_table'];first=table[0];last=table[-1]
    assert first['pd']['estimate']==.6 and first['pd']['denominator']==1000 and first['pd']['numerator']==600
    assert first['unweighted_pd_exact_interval']['estimate']==.75
    assert first['closed_workout_lgd']['estimate']==pytest.approx(.3) and first['closed_workout_lgd']['denominator']==400
    assert first['default_conditioned_ead']['estimate']==pytest.approx(200)
    assert first['default_conditioned_ccf']['estimate']==1.5 and first['default_conditioned_ccf']['denominator']==300
    assert first['open_workouts']==1 and first['closed_workout_lgd']['count']==2
    assert last['unweighted_pd_one_sided_upper']==pytest.approx(1-math.sqrt(.05))
    assert r.value['aggregate_mature_default_units']==6 and r.value['aggregate_defaults']==3
    assert r.provenance['data_origin']=='SYNTHETIC' and table[1]['status']=='INSUFFICIENT_SUPPORT'

def test_censored_outcomes_and_missing_workouts_not_imputed():
    args=estimation_args();args['cohort_outcomes'][0]['outcome_status']='CENSORED';r=estimate_pool_risk_parameters(**args)
    assert r.value['aggregate_mature_default_units']==5 and r.value['aggregate_defaults']==2
    assert r.diagnostics['excluded_workouts'][0]['reason']=='COHORT_NOT_ELIGIBLE'
    args=estimation_args();args['workouts']=[];r=estimate_pool_risk_parameters(**args).value['pool_risk_table'][0]
    assert r['missing_default_workouts']==3 and r['closed_workout_lgd']['estimate']is None and r['default_conditioned_ead']['estimate']is None

def test_estimation_context_target_and_assignment_binding():
    args=estimation_args();args['context']=dict(currency='USD')
    with pytest.raises(CreditRiskError):estimate_pool_risk_parameters(**args)
    args=estimation_args();raw=deepcopy(args['pool_assignments'].value);raw['assignments'][0]['pool_id']='other';args['pool_assignments']=raw
    with pytest.raises(CreditRiskError):estimate_pool_risk_parameters(**args)
