from copy import deepcopy
import numpy as np
import pytest
from creditratingsystem._validation_survival import survival_backtest
from creditratingsystem.validation import brier_score
from creditratingsystem.core import CreditRiskError


def fixture():
 d=dict(model_id='survival',model_version='1',parameter_version='1',data_fingerprint='synthetic-times',population_id='performing',split_id='holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2020-01-10',days_per_time_unit=1,target=dict(event='default',horizon=2,time_unit='days',probability_measure='PHYSICAL',conditioning='performing'))
 rows=[dict(prediction_id=str(i),entity_id='e'+str(i),cluster_id='g'+str(i),segment='all',as_of='2020-01-01',forecast_known_at='2020-01-01',horizon_end='2020-01-03',model_id='survival',model_version='1',parameter_version='1',outcome_known_at='2020-01-05',observed_time=t,event=e)for i,(t,e)in enumerate([(1,True),(3,True),(3,False),(3,False)])]
 p=[[.7,.8],[.1,.2],[.1,.3],[.2,.4]];return rows,p,d


def test_complete_followup_binary_brier_oracle_auc_and_trapezoid():
 rows,p,d=fixture();r=survival_backtest(p,rows,horizons=[1,2],censoring=dict(method='complete_followup',reference='complete observed sample'),design=d,uncertainty='none').value
 assert [v['brier']for v in r['metrics']['by_horizon']]==pytest.approx([.0375,.0825]);assert r['metrics']['by_horizon'][1]['brier']==pytest.approx(brier_score([1,0,0,0],np.array(p)[:,1]).value['brier']);assert r['metrics']['integrated_brier']==pytest.approx(.06);assert r['metrics']['concordance']['value']==1


def test_ipcw_left_right_limits_fixed_denominator_and_cluster_pair_exclusion():
 rows,p,d=fixture();rows[2]['observed_time']=2;censoring=dict(method='frozen_ipcw',reference='known censor distribution',independence_assumption='censoring_independent_of_event_features_and_across_clusters',tie_rule='event_before_censoring',model_fingerprint='synthetic-G',model_known_at='2019-01-01',times=[1],survival=[.5],support_end=3,minimum_survival=.1)
 r=survival_backtest(p,rows,horizons=[1,2],censoring=censoring,design=d,uncertainty='none').value
 assert [v['brier']for v in r['metrics']['by_horizon']]==pytest.approx([.0525,.11]);assert r['metrics']['concordance']['pair_mass']==6;assert r['metrics']['by_horizon'][1]['censored_by_horizon']==1
 p[1]=[.7,.8];rows[3]['cluster_id']='g0';r=survival_backtest(p,rows,horizons=[1,2],censoring=censoring,design=d,uncertainty='none').value
 assert r['metrics']['by_horizon'][0]['auc']==pytest.approx(.75);assert r['metrics']['concordance']['value']==pytest.approx(.75)
 censoring['survival']=[0.];r=survival_backtest(p,rows,horizons=[1,2],censoring=censoring,design=d,uncertainty='none').value;assert r['applicability']=='NOT_APPLICABLE';assert all(v['brier']is None for v in r['metrics']['by_horizon'])


def test_actual_cluster_bootstrap_unknown_followup_and_no_fabricated_zero():
 rows,p,d=fixture();kw=dict(horizons=[1,2],censoring=dict(method='complete_followup',reference='complete'),design=d)
 r=survival_backtest(p,rows,uncertainty='cluster_bootstrap',cluster_resamples=[['g0']*4,['g1','g2','g3','g3']],**kw).value
 assert r['uncertainty']['intervals']['auc_0']['reason']=='NON_ESTIMABLE_RESAMPLE';assert r['uncertainty']['intervals']['brier_0']['lower']is not None
 rows[2]['outcome_known_at']='2021-01-01';r=survival_backtest(p,rows,uncertainty='none',**kw).value;assert r['coverage']['included_count']==3
 rows[3]['observed_time']=2
 with pytest.raises(CreditRiskError,match='strictly beyond'):survival_backtest(p,rows,uncertainty='none',**kw)
 rows,p,d=fixture();p[0]=[.9,.8]
 with pytest.raises(CreditRiskError):survival_backtest(p,rows,uncertainty='none',**kw)
 r=survival_backtest([],[],uncertainty='none',**kw).value
 assert r['applicability']=='NOT_APPLICABLE'
