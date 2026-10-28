import math
import numpy as np
import pytest
from creditratingsystem.core import CreditRiskError,serialize_fitted_model,load_fitted_model
from creditratingsystem.ead import fit_ead_model,predict_ead
TARGET=dict(event='ead',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='given_default',currency='EUR')
TRAIN=dict(data_fingerprint='independent-oracle',split_id='train',known_at='2024-02-01',data_origin='SYNTHETIC')
def episodes(values):return [dict(episode_id=str(i),reference_date='2024-01-01',reference_known_at='2024-01-01',default_date='2024-01-31',default_known_at='2024-01-31',features_observed_at='2024-01-01',features_known_at='2024-01-01',reference_drawn=40,reference_limit=100,ead_at_default=v,currency='EUR')for i,v in enumerate(values)]
def fit(values,mode='ccf',zero_model='none',rows=None):return fit_ead_model(episodes(values)if rows is None else rows,np.empty((len(values),0)),feature_order=[],target_mode=mode,model_spec=dict(mean_model='identity',covariance='HC0')if mode=='ccf'else dict(positive_distribution='lognormal',zero_model=zero_model),target=TARGET,training=TRAIN,model_id='ead',model_version='1',currency='EUR')
def predict(f,**kwargs):return predict_ead(f,np.empty((1,0)),feature_order=[],expected_target=TARGET,reference_dates=['2024-02-01'],default_dates=['2024-03-02'],currency='EUR',**kwargs)
def test_raw_ccf_mean_covariance_and_reconstruction(tmp_path):
    f=fit([10,70,130]);assert f.model['parameters']['regression']['coefficients']==pytest.approx([.5]);assert f.model['parameters']['regression']['covariance'][0][0]==pytest.approx(2/9)
    path=tmp_path/'ead.json';serialize_fitted_model(f,path);p=predict(load_fitted_model(path),reference_drawn=[40],reference_limit=[100]);assert p.value['conditional_ead']==pytest.approx([70]);assert p.provenance['data_origin']=='SYNTHETIC'
    with pytest.raises(CreditRiskError,match='DISTRIBUTION_NOT_IDENTIFIED'):predict(f,reference_drawn=[40],reference_limit=[100],quantile_levels=[.5])
def test_direct_lognormal_and_zero_hurdle_hand_moments():
    f=fit([1,math.exp(2)],'direct_ead');p=predict(f,quantile_levels=[.5]);assert p.value['conditional_ead']==pytest.approx([math.exp(1.5)]);assert p.value['quantiles'][0]==pytest.approx([math.e])
    f=fit([0,0,1,math.exp(2)],'direct_ead','logit');p=predict(f,quantile_levels=[.25,.5,.75]);assert p.value['conditional_ead']==pytest.approx([.5*math.exp(1.5)]);assert p.value['quantiles'][0]==pytest.approx([0,0,math.e])
    f=fit([0,0],'direct_ead','all_zero');assert predict(f,quantile_levels=[.5]).value['quantiles']==[[0.]]
def test_denominator_exclusion_future_feature_and_negative_reconstruction():
    rows=episodes([10,70,130,100]);rows[-1].update(reference_drawn=100,reference_limit=100);f=fit([10,70,130,100],rows=rows);assert f.diagnostics['excluded_rows']==[dict(episode_id='3',reason='NO_POSITIVE_UNDRAWN')]
    rows=episodes([10,70,130]);rows[0]['features_known_at']='2024-01-02'
    with pytest.raises(CreditRiskError,match='EXPOSURE_DATE'):fit([10,70,130],rows=rows)
    f=fit([10,10,10])
    with pytest.raises(CreditRiskError,match='NEGATIVE_EAD_PREDICTION'):predict(f,reference_drawn=[0],reference_limit=[100])
def test_schedule_exact_date_prediction():
    from creditratingsystem.ead import exposure_schedule
    s=exposure_schedule(dict(facility_id='f',currency='EUR',as_of='2024-01-01',drawn_principal=100,credit_limit=100,accrued_interest=0),[dict(event_id='repay',event_date='2024-01-31',known_at='2024-01-01',sequence=0,kind='PRINCIPAL_REPAYMENT',amount=100)],[0],period_end_dates=['2024-01-31'],day_count='ACT/365F',default_timing='AFTER_PERIOD_END_EVENTS',allow_overlimit=False,currency='EUR')
    p=predict_ead(s,None,feature_order=[],expected_target=None,reference_dates=['2024-01-01'],default_dates=['2024-01-31'],currency='EUR');assert p.value['conditional_ead']==[0]

def test_unidentified_positive_dispersion_and_forecast_date_leakage():
    with pytest.raises(CreditRiskError,match='NON_IDENTIFIABLE'):fit([10,10],'direct_ead')
    f=fit([10,70,130])
    with pytest.raises(CreditRiskError,match='EXPOSURE_DATE'):
        predict_ead(f,np.empty((1,0)),feature_order=[],expected_target=TARGET,reference_dates=['2024-01-01'],default_dates=['2024-01-31'],currency='EUR',reference_drawn=[40],reference_limit=[100])
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):predict(f,reference_drawn=[40],reference_limit=[100],context=dict(currency='USD'))


def test_ead_event_cannot_be_relabelled_as_pd():
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):
        fit_ead_model(episodes([10,70,130]),np.empty((3,0)),feature_order=[],target_mode='ccf',model_spec=dict(mean_model='identity',covariance='HC0'),target=dict(TARGET,event='default_probability'),training=TRAIN,model_id='bad',model_version='1',currency='EUR')
