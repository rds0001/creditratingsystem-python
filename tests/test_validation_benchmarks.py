from copy import deepcopy
import pytest
from creditratingsystem._validation_benchmarks import benchmark_predictions
from creditratingsystem.monitoring import build_monitoring_cohorts
from creditratingsystem.core import CreditRiskError


def fixture(reverse=False,identical=False,drop=False):
 target=dict(event='default',probability_measure='PHYSICAL',default_definition='contract-default-v1',unit='obligor',horizon_days=30,default_grade='D');events=[dict(event_id='d'+str(i),entity_id='e'+str(i),event='default',event_at='2020-01-15',known_at='2020-01-16',default_definition=target['default_definition'])for i in [1,3]];coverage=[dict(entity_id='e'+str(i),complete_through='2020-01-31',known_at='2020-02-01')for i in range(4)];entries={}
 for model,pd in [('reference',[.1,.8,.3,.6]),('candidate',[.1,.8,.3,.6]if identical else[.2,.7,.4,.5])]:
  rows=[dict(forecast_id=model+str(i),case_id='c'+str(i),entity_id='e'+str(i),cluster_id='g'+str(i),cohort='2020-01',segment='all',model_id=model,model_version='1',forecast_at='2020-01-01',known_at='2020-01-01',pd=p,weight=1,target=target,prediction_origin='HOLDOUT')for i,p in enumerate(pd)]
  if drop and model=='candidate':rows=rows[1:]
  if reverse:rows=rows[::-1]
  bundle=build_monitoring_cohorts(rows,events,coverage,target_contract=target,evaluation_cutoff='2020-03-01')
  entries[model]=dict(cohort=bundle.value,model_version='1',parameter_version='1',model_fingerprint='frozen-'+model,training_known_at='2019-01-01')
 design=dict(model_id='reference',model_version='1',parameter_version='1',data_fingerprint='paired-cohorts',population_id='performing',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2020-03-01',target=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='performing'))
 return entries,design


def test_paired_actual_differences_intervals_and_identity_alignment():
 cohorts,d=fixture();r=benchmark_predictions(cohorts,reference_model_id='reference',metrics=['brier','auc'],matching_policy='strict',multiple_comparisons='bonferroni',design=d,uncertainty='cluster_bootstrap',cluster_resamples=[['g0']*4,['g3']*4],confidence_level=.8).value
 assert r['metrics']['paired_differences']['candidate']['brier']['difference']==pytest.approx(.06);assert r['metrics']['paired_differences']['candidate']['auc']['difference']==0
 band=r['uncertainty']['intervals']['candidate/brier'];assert band['lower']==pytest.approx(.033);assert band['upper']==pytest.approx(.087);assert r['uncertainty']['intervals']['candidate/auc']['reason']=='NON_ESTIMABLE_RESAMPLE'
 cohorts2,_=fixture(reverse=True);r2=benchmark_predictions(cohorts2,reference_model_id='reference',metrics=['brier','auc'],matching_policy='strict',multiple_comparisons='none',design=d,uncertainty='none').value;assert r2['metrics']==r['metrics'];assert r2['diagnostics']['paired_case_ids']==r['diagnostics']['paired_case_ids']


def test_identical_predictions_zero_differences_missing_cases_and_frozen_parameters():
 cohorts,d=fixture(identical=True);r=benchmark_predictions(cohorts,reference_model_id='reference',metrics=['brier','log_loss','auc','gini'],matching_policy='strict',multiple_comparisons='none',design=d,uncertainty='none').value
 assert all(v['difference']==0 for v in r['metrics']['paired_differences']['candidate'].values())
 cohorts,d=fixture(drop=True)
 with pytest.raises(CreditRiskError):benchmark_predictions(cohorts,reference_model_id='reference',metrics=['brier'],matching_policy='strict',multiple_comparisons='none',design=d,uncertainty='none')
 r=benchmark_predictions(cohorts,reference_model_id='reference',metrics=['brier'],matching_policy='intersection',multiple_comparisons='none',design=d,uncertainty='none').value;assert r['coverage']['included_count']==3 and r['coverage']['excluded_count']==1
 cohorts['candidate']['training_known_at']='2020-01-02'
 with pytest.raises(CreditRiskError):benchmark_predictions(cohorts,reference_model_id='reference',metrics=['brier'],matching_policy='intersection',multiple_comparisons='none',design=d,uncertainty='none')

from creditratingsystem._validation_benchmarks import benchmark_ratings


def rating_fixture():
 _,d=fixture();d['evaluation_cutoff']='2020-04-01'
 dimensions=dict(rating_type='issuer',support_concept='standalone',currency='EUR',rating_philosophy='TTC')
 rows=[dict(prediction_id='p'+str(i),entity_id='e'+str(i),cluster_id='g'+str(i),segment='all',as_of='2020-02-01',forecast_known_at='2020-02-01',horizon_end='2020-03-02',model_id='reference',model_version='1',parameter_version='1',rating_unit_id='u'+str(i),scale_id='internal',scale_version='1',rating_status='ACTIVE',grade=g,**dimensions)for i,g in enumerate(['A','B','C','A'])]
 external=[dict(benchmark_id='b'+str(i),entity_id='e'+str(i),rating_unit_id='u'+str(i),provider_id='provider',scale_id='external',scale_version='1',status='ACTIVE',grade=g,rating_at='2020-01-01',known_at='2020-01-01',target=d['target'],**dimensions)for i,g in enumerate(['x','x','w','y'])]
 mapping=dict(mapping_id='m',version='1',reference='frozen',known_at='2019-01-01',common_grades=['1','2','3'],internal=dict(scale_id='internal',scale_version='1',grades=['A','B','C']),benchmark=dict(scale_id='external',scale_version='1',grades=['x','y','z','w']),internal_to_common=dict(A='1',B='2',C='3'),benchmark_to_common=dict(x='1',y='2',z='2',w='3'))
 policy=dict(benchmark_origin='EXTERNAL_JUDGEMENT',method='latest_known',watch_policy='include',kappa_weighting='quadratic',provider_id='provider',max_age_days=40)
 return rows,external,dict(scale_mapping=mapping,matching_policy=policy,design=d,uncertainty='none')


def test_rating_independent_confusion_kappa_midrank_and_degenerate_draws():
 rows,external,kwargs=rating_fixture();r=benchmark_ratings(rows,external,**kwargs).value;m=r['metrics']
 assert m['exact_agreement']==.5 and m['within_one_notch']==1
 assert m['mean_signed_notches']==0 and m['mean_absolute_notches']==.5
 assert m['observed_weighted_agreement']==.875
 assert m['expected_weighted_agreement']==.65625
 assert m['kappa']==pytest.approx(7/11)
 assert m['spearman_rank_correlation']==pytest.approx(.5)
 kwargs.update(uncertainty='cluster_bootstrap',cluster_resamples=[['g0']*4,['g3']*4])
 r=benchmark_ratings(rows,external,**kwargs).value
 assert r['uncertainty']['intervals']['kappa']['reason']=='NON_ESTIMABLE_RESAMPLE'


def test_rating_dated_withdrawal_no_fallback_and_mapping_controls():
 rows,external,kwargs=rating_fixture();withdraw=deepcopy(external[0]);withdraw.update(benchmark_id='withdraw',status='WITHDRAWN',grade=None,rating_at='2020-01-15',known_at='2020-01-15')
 r=benchmark_ratings(rows,external+[withdraw],**kwargs).value
 assert r['coverage']['excluded_count']==1 and r['coverage']['exclusion_counts']=={'WITHDRAWN_BENCHMARK':1}
 withdraw['known_at']='2020-03-01';assert benchmark_ratings(rows,external+[withdraw],**kwargs).value['coverage']['included_count']==4
 kwargs['matching_policy']['max_age_days']=20
 assert benchmark_ratings(rows,external,**kwargs).value['applicability']=='NOT_APPLICABLE'
 kwargs['matching_policy']['max_age_days']=40
 duplicate=deepcopy(external[0]);duplicate['benchmark_id']='duplicate'
 with pytest.raises(CreditRiskError):benchmark_ratings(rows,external+[duplicate],**kwargs)
 kwargs['scale_mapping']['internal_to_common']['A']='3'
 with pytest.raises(CreditRiskError):benchmark_ratings(rows,external,**kwargs)
