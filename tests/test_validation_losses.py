from copy import deepcopy
import numpy as np
import pytest
from creditratingsystem._validation_losses import lgd_backtest,ead_backtest
from creditratingsystem.core import CreditRiskError


def fixture():
 design=dict(model_id='frozen',model_version='1',parameter_version='1',data_fingerprint='synthetic-workouts',population_id='defaulted-loans',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2024-01-01',target=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='default'),currency='EUR')
 rows=[dict(prediction_id=str(i),entity_id='e'+str(i),cluster_id='g'+str(i),segment='all',as_of='2020-01-01',forecast_known_at='2020-01-01',horizon_end='2020-01-31',model_id='frozen',model_version='1',parameter_version='1',default_date='2020-01-15',outcome_known_at='2021-01-01',workout_closed_at='2020-12-31',workout_status='CLOSED',predicted_lgd=p,realized_lgd=y,ead_at_default=e)for i,(p,y,e)in enumerate([(.4,.2,100),(.6,.8,300),(.1,None,100)])]
 rows[2].update(workout_status='OPEN',workout_closed_at=None,realized_lgd=None)
 return rows,design


def test_lgd_closed_only_arithmetic_open_not_zero_and_actual_cluster_interval():
 rows,d=fixture();r=lgd_backtest(rows,design=d,workout_policy=dict(method='closed_only',reference='closed descriptive cohort'),uncertainty='cluster_bootstrap',cluster_resamples=[['g0','g0','g2'],['g1','g1','g2'],['g0','g1','g2']],confidence_level=.8).value
 m=r['metrics']['overall'];assert m['count_weighted']['bias']==pytest.approx(0);assert m['ead_weighted']['bias']==pytest.approx(-.1);assert m['ead_weighted']['mae']==pytest.approx(.2);assert m['ead_weighted']['rmse']==pytest.approx(.2);assert sum(m['ead_weighted']['bias_contributions'])==pytest.approx(-.1)
 assert r['coverage']['included_count']==2 and r['coverage']['exclusion_counts']=={'OPEN_WORKOUT':1};band=r['uncertainty']['intervals']['count_weighted.bias'];assert band['lower']==pytest.approx(-.16);assert band['upper']==pytest.approx(.16)
 rows[2]['realized_lgd']=-10000
 assert lgd_backtest(rows,design=d,workout_policy=dict(method='closed_only',reference='closed'),uncertainty='none').value['metrics']==r['metrics']


def test_lgd_ipcw_support_maturity_and_unidentified_bootstrap():
 rows,d=fixture();policy=dict(method='ipcw',reference='supplied train-only completion model',identification='closure_independent_of_final_lgd_given_recorded_covariates',model_fingerprint='frozen-selection',model_known_at='2019-01-01',minimum_observation_probability=.05)
 for row,prob in zip(rows,[.5,1,.5]):row['observation_probability']=prob
 r=lgd_backtest(rows,design=d,workout_policy=policy,uncertainty='cluster_bootstrap',cluster_resamples=[['g0','g1','g2'],['g2','g2','g2']]).value
 assert r['metrics']['overall']['count_weighted']['bias']==pytest.approx(.2/3);assert r['uncertainty']['intervals']['count_weighted.bias']['lower']is None
 d['evaluation_cutoff']='2020-01-20';r=lgd_backtest(rows,design=d,workout_policy=policy,uncertainty='none').value;assert r['applicability']=='NOT_APPLICABLE';assert r['coverage']['exclusion_counts']=={'IMMATURE_HORIZON':3}
 policy['model_known_at']='2021-01-01'
 with pytest.raises(CreditRiskError):lgd_backtest(rows,design=d,workout_policy=policy,uncertainty='none')


def test_ead_ccf_raw_boundaries_and_bias_reconstruction():
 rows,d=fixture()
 for row,values in zip(rows,[(100,120,50,150),(90,100,100,100),(110,120,120,100)]):
  p,y,drawn,limit=values;row.update(predicted_ead=p,realized_ead=y,drawn_at_reference=drawn,limit_at_reference=limit,realized_ead_date='2020-01-15',currency='EUR')
 result=ead_backtest(rows,design=d,weighting_policy=dict(method='reference_limit',reference='frozen facility limits'),uncertainty='none').value
 m=result['metrics']['ead']['overall']['selected_weighting'];assert m['bias']==pytest.approx(-100/7);assert sum(m['bias_contributions'])==pytest.approx(m['bias']);assert result['diagnostics']['ccf_undefined_count']==2
 denom=result['diagnostics']['denominators'];assert denom[0]['observed_ccf']==pytest.approx(.7);assert denom[0]['predicted_ccf']==pytest.approx(.5);assert denom[1]['observed_ccf']is None and denom[2]['observed_ccf']is None
 assert result['metrics']['ccf']['overall']['unweighted']['bias']==pytest.approx(-.2)
 rows[0]['realized_ead_date']='2020-01-16'
 with pytest.raises(CreditRiskError):ead_backtest(rows,design=d,weighting_policy=dict(method='equal',reference='count'),uncertainty='none')


def test_empty_and_wrong_horizon_have_explicit_applicability():
 rows,d=fixture();r=lgd_backtest([],design=d,workout_policy=dict(method='closed_only',reference='empty'),uncertainty='none').value
 assert r['applicability']=='NOT_APPLICABLE'and r['metrics']['overall']['ead_weighted']['bias']is None
 rows[0]['horizon_end']='2020-02-01'
 with pytest.raises(CreditRiskError):lgd_backtest(rows,design=d,workout_policy=dict(method='closed_only',reference='fixed'),uncertainty='none')
