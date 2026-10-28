from copy import deepcopy
import pytest
from creditratingsystem._validation_retail_pools import retail_pool_diagnostics
from creditratingsystem._retail_pools import assign_retail_risk_pool
from creditratingsystem.core import CreditRiskError
from test_retail_pools import fit,snapshot,outcomes,workout,profile,target

def fixture():
 model=fit().model;args=dict(observations=[snapshot(str(i),10 if i<4 else 80)for i in range(6)],as_of='2024-01-01',information_cutoff='2024-01-01',expected_target=target(),uncovered_policy='unassigned');assignment=assign_retail_risk_pool(fitted_model=model,**args).value;cohort=outcomes()
 for o in cohort:o['complete_through']=o['label_window_end']
 d=dict(model_id='pool-risk',model_version='1',parameter_version='frozen1',data_fingerprint='retail-holdout',population_id='retail',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-02-01',currency='EUR',target={k:target()[k]for k in ('event','horizon','time_unit','probability_measure','conditioning')})
 rows=[dict(prediction_id=str(i),observation_id=str(i),entity_id=str(i),cluster_id='h'+str(i//2),segment='cards',as_of='2024-01-01',forecast_known_at='2024-01-01',horizon_end='2024-01-31',model_id='pool-risk',model_version='1',parameter_version='frozen1',excluded_from_training=True,excluded_from_calibration=True,reference_exposure=o['reference_exposure'],pd=.5 if i<4 else .1,lgd=.4,ead=150)for i,o in enumerate(cohort)]
 links=[dict(observation_id=str(i),facility_id=str(i),obligor_id='b'+str(i),household_id='h'+str(i//2),risk_group_id='h'+str(i//2),diagnostic_group=str(i%2),known_at='2023-12-01')for i in range(6)]
 return dict(fitted_model=model,pool_assignments=assignment,assignment_inputs=args,predictions=rows,cohort_outcomes=cohort,workouts=[workout('0',100,40,'CLOSED'),workout('1',300,240,'CLOSED'),workout('2',200,100,'OPEN')],links=links,estimation_profile=profile(),design=d)

def test_distinct_parameter_denominators_and_constant_pool_ranking():
 r=retail_pool_diagnostics(**fixture()).value;p=r['metrics']['pools'][0]
 assert p['pd_count_weighted']['observed_rate']==.75 and p['pd_exposure_weighted']['observed_rate']==.6
 assert p['lgd_error']['bias']==pytest.approx(.1)and p['lgd_error']['n']==2
 assert p['ead_error']['bias']==pytest.approx(-50)and p['ead_error']['n']==3
 assert p['within_pool_discrimination']['auc']is None and p['residual_group_spread']==.5
 assert r['concentration']['household_id']['exposure_hhi']==pytest.approx(7/18)
 assert not r['dependence']['independent_binomial_inference_used']and not r['dependence']['household_default_propagation']

def test_incomplete_negative_missing_workout_and_default_unit_duplicates():
 kw=fixture();kw['cohort_outcomes'][3]['complete_through']='2024-01-15';kw['workouts']=[];r=retail_pool_diagnostics(**kw).value
 assert r['coverage']['mature_eligible_count']==5 and r['metrics']['pools'][0]['lgd_error']['bias']is None
 kw=fixture();kw['links'][1]['facility_id']='0'
 with pytest.raises(CreditRiskError):retail_pool_diagnostics(**kw)

def test_connected_bootstrap_and_dated_migration():
 kw=fixture();kw.update(uncertainty='cluster_bootstrap',cluster_resamples=[['h0','h1','h2'],['h1','h0','h2']]);r=retail_pool_diagnostics(**kw).value
 key=kw['pool_assignments']['pool_ids'][0]+'|residual_group_spread';assert r['uncertainty']['intervals'][key]['lower']==.5
 kw['predictions'][1]['cluster_id']='wrong'
 with pytest.raises(CreditRiskError):retail_pool_diagnostics(**kw)
 kw=fixture();args=deepcopy(kw['assignment_inputs']);args.update(as_of='2023-12-01',information_cutoff='2023-12-01')
 for o in args['observations']:o.update(index_date='2023-12-01',feature_observed_at='2023-12-01',feature_known_at='2023-12-01',score=80)
 kw['previous_assignments']=assign_retail_risk_pool(fitted_model=kw['fitted_model'],**args).value
 for link in kw['links']:link['previous_observation_id']=link['observation_id']
 r=retail_pool_diagnostics(**kw).value;assert sum(x['changed']for x in r['migration'])==4
