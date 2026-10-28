from copy import deepcopy
import math
import numpy as np
import pytest
from creditratingsystem._validation_migration import rating_migration_backtest
from creditratingsystem.core import CreditRiskError


def fixture():
 d=dict(model_id='migration',model_version='1',parameter_version='1',data_fingerprint='synthetic-transitions',population_id='rated',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2021-01-01',target=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='origin_state'))
 rows=[dict(prediction_id=str(i),entity_id='e'+str(i),cluster_id='g'+str(i),segment='all',as_of='2020-01-01',forecast_known_at='2020-01-01',horizon_end='2020-01-31',model_id='migration',model_version='1',parameter_version='1',outcome_known_at='2020-02-01',scale_id='grades',scale_version='1',origin_state=a,destination_state=b,observation_status='OBSERVED',state_observed_at='2020-01-31')for i,(a,b)in enumerate([('A','A'),('A','D'),('B','A'),('D','D')])]
 P=[[.7,.2,.1,0],[.1,.7,.2,0],[0,0,1,0],[0,0,0,1]];contract=dict(scale_id='grades',scale_version='1',reference='frozen grades/default/exit',known_at='2019-01-01',default_states=['D'],cure_states=[],exit_states=['X'],absorbing_states=['D','X']);return rows,P,contract,d


def test_multiclass_scores_expected_counts_and_empty_origin():
 rows,P,c,d=fixture();r=rating_migration_backtest(rows,P,states=['A','B','D','X'],state_contract=c,design=d,uncertainty='none').value
 assert r['metrics']['overall']['brier']==pytest.approx(.705);assert r['metrics']['overall']['log_loss']==pytest.approx((-math.log(.7)-2*math.log(.1))/4)
 a=r['metrics']['by_origin'][0];assert a['residuals']==pytest.approx([-.4,-.4,.8,0]);assert a['observed_counts']==[1,0,1,0];assert a['expected_counts']==pytest.approx([1.4,.4,.2,0]);empty=r['metrics']['by_origin'][-1];assert empty['applicability']=='NOT_APPLICABLE'and empty['observed_probabilities']is None and empty['residuals']is None
 r2=rating_migration_backtest(rows,np.repeat(np.array(P)[None],4,axis=0),states=['A','B','D','X'],state_contract=c,design=d,uncertainty='none').value;assert r2==r


def test_censoring_absorption_and_unsupported_bootstrap_are_not_perfection():
 rows,P,c,d=fixture();rows[0]['observation_status']='CENSORED';rows[0]['destination_state']=None;rows[3]['destination_state']='A'
 r=rating_migration_backtest(rows,P,states=['A','B','D','X'],state_contract=c,design=d,uncertainty='cluster_bootstrap',cluster_resamples=[['g0']*4,['g1','g1','g2','g2']]).value
 assert r['coverage']['excluded_count']==2;assert r['coverage']['exclusion_counts']=={'CENSORED_ENDPOINT':1,'OBSERVED_ABSORBING_STATE_VIOLATION':1};assert r['uncertainty']['intervals']['brier']['reason']=='NON_ESTIMABLE_RESAMPLE'
 P[2]=[.1,0,.9,0]
 with pytest.raises(CreditRiskError):rating_migration_backtest(rows,P,states=['A','B','D','X'],state_contract=c,design=d,uncertainty='none')


def test_infinite_logloss_and_no_implicit_cure_or_exit():
 rows,P,c,d=fixture();P[0]=[1,0,0,0];r=rating_migration_backtest(rows,P,states=['A','B','D','X'],state_contract=c,design=d,uncertainty='none').value
 assert r['metrics']['overall']['log_loss_status']=='INFINITE'and r['metrics']['overall']['log_loss']is None
 rows[1]['state_observed_at']='2020-01-15';r=rating_migration_backtest(rows,P,states=['A','B','D','X'],state_contract=c,design=d,uncertainty='none').value;assert r['coverage']['included_count']==4
 c['absorbing_states']=['X'];r=rating_migration_backtest(rows,P,states=['A','B','D','X'],state_contract=c,design=d,uncertainty='none').value;assert r['coverage']['exclusion_counts']=={'ENDPOINT_STATE_NOT_OBSERVED':1}
