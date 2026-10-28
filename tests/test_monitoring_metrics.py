import copy
import itertools
import math
import numpy as np
import pytest
from creditratingsystem._monitoring_metrics import (
 build_monitoring_cohorts,monitor_rating_cohort,detect_credit_feature_drift,
 detect_pd_calibration_drift,compare_challenger_cohorts,diagnose_fairness_slices)

TARGET={'event':'default','probability_measure':'PHYSICAL','default_definition':'90DPD','unit':'obligor','horizon_days':30,'default_grade':'D'}
def sample(p=(.1,.2,.7,.8),cutoff='2025-02-15',coverage=True):
    predictions=[dict(forecast_id='f'+str(i),case_id='c'+str(i),entity_id='e'+str(i),cluster_id='u'+str(i),cohort='2025-01',segment='retail',model_id='m',model_version='1',forecast_at='2025-01-01',known_at='2025-01-01',pd=v,target=TARGET,origin_grade='A',decision='approve' if v<.5 else 'decline',override=False,prediction_origin='OUT_OF_TIME') for i,v in enumerate(p)]
    events=[dict(event_id='d'+str(i),entity_id='e'+str(i),event_at='2025-01-10',known_at='2025-01-11',event='default',default_definition='90DPD') for i in range(len(p)) if i%2]
    covers=[dict(entity_id='e'+str(i),complete_through='2025-01-31',known_at='2025-02-01') for i in range(len(p))] if coverage else []
    return predictions,events,covers

def cohort(p=(.1,.2,.7,.8),**kw):
    cutoff=kw.pop('cutoff','2025-02-15');return build_monitoring_cohorts(*sample(p,**kw),target_contract=TARGET,evaluation_cutoff=cutoff)

def draws(n=4):
    ids=['u'+str(i) for i in range(n)]
    return [ids,ids[::-1],[ids[0]]*n,[ids[-1]]*n]

def test_maturity_coverage_and_early_known_default():
    c=cohort(cutoff='2025-01-20');assert [r['target'] for r in c.value['rows']]==[None,1,None,1]
    assert not any(r['eligible'] for r in c.value['rows'])
    c=cohort(coverage=False);assert [r['eligible'] for r in c.value['rows']]==[False,True,False,True]
    assert c.value['rows'][0]['status']=='MISSING_OUTCOME'

def test_events_known_time_censoring_grade_provenance_and_hash():
    p,e,c=sample();p[0]['outcome_grade']='LEAKED';e += [dict(event_id='rating',entity_id='e0',event_at='2025-01-15',known_at='2025-01-16',event='rating',grade='B',default_definition='90DPD'),dict(event_id='future',entity_id='e0',event_at='2025-01-20',known_at='2025-03-01',event='rating',grade='C',default_definition='90DPD'),dict(event_id='exit',entity_id='e2',event_at='2025-01-15',known_at='2025-01-16',event='exit',default_definition='90DPD')]
    b=build_monitoring_cohorts(p,e,c,target_contract=TARGET,evaluation_cutoff='2025-02-15');r=b.value['rows'];assert r[0]['outcome_grade']=='B';assert r[1]['outcome_grade']=='D';assert r[2]['target'] is None and r[2]['status']=='CENSORED';assert b.diagnostics['ignored_not_yet_known_records']==1
    changed=copy.deepcopy(b.value);changed['rows'][0]['pd']=.3
    with pytest.raises(ValueError,match='COHORT_HASH_MISMATCH'):monitor_rating_cohort(changed,group_by=['cohort'],reference_window={'start':'2025-01-01','end':'2025-01-31'},minimum_count=1,uncertainty='none')

def test_monitor_weighted_arithmetic_and_boundary_exact_interval():
    b=cohort();m=monitor_rating_cohort(b,group_by=['cohort','model_version'],reference_window={'start':'2024-01-01','end':'2024-12-31'},minimum_count=5,uncertainty='iid_homogeneous_binomial').value[0]
    assert m['metrics']['auc']==.75;assert m['metrics']['brier']==pytest.approx((.01+.64+.49+.04)/4);assert m['reference_metrics']['n']==0;assert m['status']=='INSUFFICIENT_INFORMATION'
    p,e,c=sample();b=build_monitoring_cohorts(p,[],c,target_contract=TARGET,evaluation_cutoff='2025-02-15');m=monitor_rating_cohort(b,group_by=['cohort'],reference_window={'start':'2025-01-01','end':'2025-01-31'},minimum_count=1,uncertainty='iid_homogeneous_binomial').value[0]
    assert m['metrics']['auc'] is None;assert m['intervals']['observed_rate']['upper']==pytest.approx(1-.025**.25)

def test_repeated_cluster_requires_cluster_uncertainty():
    p,e,c=sample();p[1]['cluster_id']='u0';b=build_monitoring_cohorts(p,e,c,target_contract=TARGET,evaluation_cutoff='2025-02-15');assert b.diagnostics['repeated_cluster_rows']==1
    with pytest.raises(ValueError,match='INCOMPATIBLE_ASSUMPTIONS'):monitor_rating_cohort(b,group_by=['cohort'],reference_window={'start':'2025-01-01','end':'2025-01-31'},minimum_count=1,uncertainty='iid_homogeneous_binomial')
    result=monitor_rating_cohort(b,group_by=['cohort'],reference_window={'start':'2025-01-01','end':'2025-01-31'},minimum_count=1,uncertainty='independent_cluster_bootstrap',cluster_resamples=[['u0','u2','u3'],['u0','u0','u3']]);assert result.value[0]['intervals']['brier']['identified_resamples']==2

@pytest.mark.parametrize('method,expected',[('ks',1),('wasserstein',2)])
def test_drift_exact_permutation_distance(method,expected):
    permutations=list(itertools.permutations(range(4)))
    v=detect_credit_feature_drift([{'x':0},{'x':1}],[{'x':2},{'x':3}],feature_order=['x'],method=method,permutations=permutations,multiple_testing='holm',dependence_assumption='exchangeable_rows').value[0]
    assert v['statistic']==pytest.approx(expected);assert v['p_value']==pytest.approx(9/25);assert v['adjusted_p_value']==v['p_value']

def test_mmd_embedding_oracle_and_multiple_testing():
    v=detect_credit_feature_drift([{'x':0}],[{'x':2}],feature_order=['x'],method='mmd_rbf',permutations=[[0,1],[1,0]],multiple_testing='none',dependence_assumption='exchangeable_rows',bandwidth=1).value[0]
    assert v['statistic']==pytest.approx(2*(1-math.exp(-2)));assert v['p_value']==1
    v=detect_credit_feature_drift([{'x':0,'z':1},{'x':1,'z':0}],[{'x':2,'z':1},{'x':3,'z':0}],feature_order=['x','z'],method='ks',permutations=list(itertools.permutations(range(4))),multiple_testing='holm',dependence_assumption='exchangeable_rows').value
    assert [r['adjusted_p_value'] for r in v]==pytest.approx([18/25,1])

def test_psi_frozen_unknown_and_missing_categories():
    v=detect_credit_feature_drift([{'x':'a'},{'x':'b'}],[{'x':'c'},{'x':None}],feature_order=['x'],method='psi',permutations=None,multiple_testing='not_applicable',dependence_assumption='descriptive',frozen_bins={'x':{'kind':'categorical','levels':['a','b']}}).value[0]
    assert v['new_category_rows']==1 and v['current_missing']==1
    assert v['statistic']==pytest.approx(math.log(3));assert v['p_value'] is None

def test_cluster_permutation_integrity():
    with pytest.raises(ValueError,match='INVALID_PERMUTATIONS'):detect_credit_feature_drift([{'x':0},{'x':1}],[{'x':2},{'x':3}],feature_order=['x'],method='ks',permutations=[[0,1,2,3],[0,2,1,3]],multiple_testing='none',dependence_assumption='exchangeable_clusters',permutation_units=['a','a','b','b'])

def sequential(method):
    residuals=[{'time':'2025-02-0'+str(i+1),'value':1.,'mature':True} for i in range(3)]
    null_spec=dict(model_id='m',model_version='1',calibration_id='pretest',residual_scale='observed_minus_expected_rate',dependence='prespecified_dependent',probability_measure='PHYSICAL',calibrated_before='2025-01-31',validation_paths_independent=True)
    return detect_pd_calibration_drift(residuals,method=method,parameters={'smoothing':.5} if method=='ewma' else {'allowance':.2},null_paths=np.zeros((19,3)),null_spec=null_spec,alpha=.05,validation_null_paths=np.zeros((20,3)))

def test_sequential_closed_form_and_false_alarm_uncertainty():
    e=sequential('ewma');assert e.value['statistic_path']==pytest.approx([.5,.75,.875]);assert e.value['threshold']==0;assert e.value['first_crossing_index']==0;assert e.value['validation_false_alarm']['interval']['upper']==pytest.approx(1-.025**(1/20))
    c=sequential('cusum');assert c.value['statistic_path']==pytest.approx([.8,1.6,2.4]);assert c.diagnostics['order_statistic_rank']==19

PROTOCOL={'evaluation_role':'OUT_OF_TIME_TEST','multiple_challenger_policy':'single_prespecified','protocol_id':'holdout-v1'}
def test_paired_identical_and_metric_tradeoff():
    a=cohort();v=compare_challenger_cohorts(a,a,cluster_resamples=draws(),selection_protocol=PROTOCOL).value
    assert all(x==0 for x in v['differences'].values());assert v['intervals']['brier']['degenerate'];assert v['intervals']['auc']['identified_resamples']==2
    b=cohort((.49,.5,.51,.52));v=compare_challenger_cohorts(a,b,cluster_resamples=draws(),selection_protocol=PROTOCOL).value
    assert v['differences']['auc']==0;assert v['differences']['brier']<0;assert v['recommendation']=='REQUIRES_MULTIDIMENSIONAL_REVIEW'

def test_paired_coverage_disagreements_are_explicit():
    a=cohort();b=cohort(coverage=False)
    v=compare_challenger_cohorts(a,b,cluster_resamples=[['u1','u3'],['u1','u1']],selection_protocol=PROTOCOL).value
    assert v['eligibility_disagreements']==['c0','c2'];assert v['paired_count']==2

def test_fairness_denominators_small_samples_and_paired_gaps():
    labels=[{'case_id':'c'+str(i),'group':'A' if i<2 else 'B'} for i in range(4)]
    v=diagnose_fairness_slices(cohort(),group_labels=labels,reference_group='A',use_basis='documented approved analysis',minimum_count=3,risk_threshold=.5,cluster_resamples=draws()).value
    assert v[0]['metrics']['selection_rate']==1 and v[1]['metrics']['selection_rate']==0
    assert v[0]['metrics']['true_positive_rate']==0 and v[1]['metrics']['true_positive_rate']==1
    assert v[1]['reference_gaps']['false_positive_rate']==1;assert v[0]['status']=='INSUFFICIENT_INFORMATION'
    v=diagnose_fairness_slices(cohort(cutoff='2025-01-20'),group_labels=labels,reference_group='A',use_basis='approved',minimum_count=1,risk_threshold=.5,cluster_resamples=draws()).value
    assert v[0]['metrics']['selection_rate']==1;assert v[0]['metrics']['true_positive_rate'] is None;assert v[0]['mature_observed_count']==0

def test_order_statistic_null_calibration_and_fail_closed_inputs():
    residuals=[{'time':'2025-02-01','value':9.,'mature':True}]
    spec={'model_id':'m','model_version':'1','calibration_id':'v','residual_scale':'rate','dependence':'independent','probability_measure':'PHYSICAL','calibrated_before':'2025-01-01'}
    v=detect_pd_calibration_drift(residuals,method='cusum',parameters={'allowance':0},null_paths=np.arange(9.)[:,None],null_spec=spec,alpha=.2)
    assert v.value['threshold']==7 and v.diagnostics['order_statistic_rank']==8
    with pytest.raises(ValueError,match='INSUFFICIENT_NULL_SIMULATIONS'):detect_pd_calibration_drift(residuals,method='cusum',parameters={'allowance':0},null_paths=np.zeros((2,1)),null_spec=spec,alpha=.05)
    residuals[0]['mature']=False
    with pytest.raises(ValueError,match='INVALID_RESIDUALS'):detect_pd_calibration_drift(residuals,method='cusum',parameters={'allowance':0},null_paths=np.zeros((19,1)),null_spec=spec,alpha=.05)
