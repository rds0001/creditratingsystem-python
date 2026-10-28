"""Conditional LGD estimators: independent likelihood and covariance oracles."""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import digamma
from creditratingsystem.core import CreditRiskError
from creditratingsystem.lgd import fit_fractional_lgd,fit_beta_lgd,_beta_statistics
TARGET=dict(event='economic_loss',horizon=5,time_unit='years',probability_measure='PHYSICAL',conditioning='given_default')
TRAINING=dict(data_fingerprint='hand-designed',split_id='train',known_at='2025-12-31',data_origin='SYNTHETIC',workout_population='closed_defaulted_episodes')
def options():return dict(feature_order=[],target=TARGET,training=TRAINING,model_id='lgd',model_version='1')
def test_fractional_endpoint_mean_and_robust_covariance():
    y=np.array([0,.2,.8,1]);r=fit_fractional_lgd(np.empty((4,0)),y,link='logit',covariance='HC0',**options())
    assert r.model['parameters']['coefficients']==pytest.approx([0])
    assert r.model['parameters']['covariance'][0][0]==pytest.approx(np.sum((y-.5)**2))
    r1=fit_fractional_lgd(np.empty((4,0)),y,link='logit',covariance='HC1',**options())
    assert r1.model['parameters']['covariance'][0][0]==pytest.approx(np.sum((y-.5)**2)*4/3)
def test_fractional_constant_interior_and_separation():
    r=fit_fractional_lgd(np.empty((3,0)),[.5]*3,link='logit',covariance='HC0',**options())
    assert r.model['parameters']['covariance']==[[0.]]
    with pytest.raises(CreditRiskError,match='SEPARATION'):fit_fractional_lgd(np.empty((3,0)),[1]*3,link='logit',covariance='HC0',**options())
def test_fractional_group_oracle_frequency_replication():
    opt=options();opt['feature_order']=['group'];x=np.array([[0],[0],[1],[1]]);y=np.array([0,.5,.5,1]);w=np.array([2,2,2,2])
    r=fit_fractional_lgd(x,y,weights=w,link='logit',covariance='HC0',**opt)
    np.testing.assert_allclose(r.model['parameters']['coefficients'],[-np.log(3),2*np.log(3)],atol=1e-8)
    rr=fit_fractional_lgd(np.repeat(x,w,axis=0),np.repeat(y,w),link='logit',covariance='HC0',**opt)
    np.testing.assert_allclose(r.model['parameters']['covariance'],rr.model['parameters']['covariance'],atol=1e-12)
def precision_oracle(y):return brentq(lambda p:digamma(p)-digamma(p/2)+np.mean(np.log(y)),.01,1000)
def test_beta_symmetric_precision_scalar_root():
    y=np.array([.1,.2,.8,.9]);r=fit_beta_lgd(np.empty((4,0)),y,mean_link='logit',precision_link='log',**options())
    assert r.model['parameters']['mean_coefficients']==pytest.approx([0],abs=1e-7)
    assert np.exp(r.model['parameters']['precision_coefficients'][0])==pytest.approx(precision_oracle(y),rel=1e-7)
def test_beta_variable_precision_two_independent_roots():
    y=np.array([.1,.2,.8,.9,.4,.45,.55,.6]);z=np.array([[0]]*4+[[1]]*4)
    r=fit_beta_lgd(np.empty((8,0)),y,precision_design=z,precision_feature_order=['group'],mean_link='logit',precision_link='log',**options())
    p=np.log([precision_oracle(y[:4]),precision_oracle(y[4:])]);np.testing.assert_allclose(r.model['parameters']['precision_coefficients'],[p[0],p[1]-p[0]],atol=1e-6)
def test_beta_analytic_information_against_score_difference():
    d=np.array([[1,-1],[1,0],[1,1],[1,2]],float);z=np.array([[1,0],[1,1],[1,0],[1,1]],float);y=np.array([.2,.4,.6,.9]);w=np.array([1,2,1,3]);theta=np.array([-.3,.2,1,.1]);h=1e-5
    info=_beta_statistics(theta,d,z,y,w)[2]
    numerical=np.column_stack([ -(_beta_statistics(theta+np.eye(4)[j]*h,d,z,y,w)[1]-_beta_statistics(theta-np.eye(4)[j]*h,d,z,y,w)[1])/(2*h)for j in range(4)])
    np.testing.assert_allclose(info,numerical,rtol=1e-8,atol=1e-8)
def test_beta_endpoint_and_zero_dispersion_rejected():
    for y,code in [([0,.2,.8,1],'INVALID_LGD_TARGET'),([.5]*4,'NON_IDENTIFIABLE')]:
        with pytest.raises(CreditRiskError,match=code):fit_beta_lgd(np.empty((4,0)),y,mean_link='logit',precision_link='log',**options())

def test_frozen_beta_quantiles_and_fractional_no_distribution(tmp_path):
    from creditratingsystem.lgd import predict_lgd
    from creditratingsystem.core import serialize_fitted_model,load_fitted_model
    from scipy.stats import beta
    y=np.array([.1,.2,.8,.9]);r=fit_beta_lgd(np.empty((4,0)),y,mean_link='logit',precision_link='log',**options())
    path=tmp_path/'lgd.json';serialize_fitted_model(r,path);loaded=load_fitted_model(path)
    p=predict_lgd(loaded,np.empty((1,0)),feature_order=[],expected_target=TARGET,default_condition='given_default',quantile_levels=[.1,.5,.9])
    phi=precision_oracle(y);np.testing.assert_allclose(p.value['quantiles'],[beta.ppf([.1,.5,.9],phi/2,phi/2)],atol=1e-8)
    f=fit_fractional_lgd(np.empty((4,0)),y,link='logit',covariance='HC0',**options())
    with pytest.raises(CreditRiskError,match='DISTRIBUTION_NOT_IDENTIFIED'):predict_lgd(f,np.empty((1,0)),feature_order=[],expected_target=TARGET,default_condition='given_default',quantile_levels=[.5])

def test_structural_endpoint_mixture_hand_probabilities_and_quantiles():
    from creditratingsystem.lgd import fit_zero_one_inflated_lgd,predict_lgd
    y=[0,0,1,1,.1,.2,.8,.9]
    r=fit_zero_one_inflated_lgd(np.empty((8,0)),y,component_spec=dict(boundary_origin='structural',gate_link='logit'),**options())
    p=predict_lgd(r,np.empty((1,0)),feature_order=[],expected_target=TARGET,default_condition='given_default',quantile_levels=[.1,.25,.5,.75,.9]).value
    assert p['zero_probability']==pytest.approx([.25]);assert p['one_probability']==pytest.approx([.25]);assert p['interior_probability']==pytest.approx([.5]);assert p['mean_lgd']==pytest.approx([.5]);np.testing.assert_allclose(p['quantiles'],[[0,0,.5,1,1]],atol=1e-8)

def test_model_contracts_target_design_and_structural_support():
    from creditratingsystem.lgd import fit_zero_one_inflated_lgd,predict_lgd
    from copy import deepcopy
    opts=options();opts['target']=dict(TARGET,conditioning='performing')
    with pytest.raises(CreditRiskError,match='INCOMPATIBLE_TARGET'):fit_fractional_lgd(np.empty((4,0)),[0,.2,.8,1],link='logit',covariance='HC0',**opts)
    with pytest.raises(CreditRiskError,match='INSUFFICIENT_COMPONENT_SUPPORT'):fit_zero_one_inflated_lgd(np.empty((4,0)),[0,.2,.8,.9],component_spec=dict(boundary_origin='structural',gate_link='logit'),**options())
    r=fit_beta_lgd(np.empty((4,0)),[.1,.2,.8,.9],mean_link='logit',precision_link='log',**options())
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):predict_lgd(r,np.empty((1,0)),feature_order=[],expected_target=dict(TARGET,horizon=1),default_condition='given_default')
    broken=deepcopy(r.model);broken['parameters']['precision_coefficients']=[0,1]
    with pytest.raises(CreditRiskError,match='INVALID_MODEL'):predict_lgd(broken,np.empty((1,0)),feature_order=[],expected_target=TARGET,default_condition='given_default')
    with pytest.raises(CreditRiskError,match='INVALID_QUANTILE'):predict_lgd(r,np.empty((1,0)),feature_order=[],expected_target=TARGET,default_condition='given_default',quantile_levels=[.9,.1])

def test_frozen_context_and_training_provenance():
    from creditratingsystem.lgd import predict_lgd
    r=fit_fractional_lgd(np.empty((4,0)),[0,.2,.8,1],link='logit',covariance='HC0',**options())
    a=dict(feature_order=[],expected_target=TARGET,default_condition='given_default')
    for ctx in [dict(probability_measure='RISK_NEUTRAL'),dict(horizon=1),dict(conditioning='performing'),dict(model_version='2')]:
        with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'):predict_lgd(r,np.empty((1,0)),context=ctx,**a)
    p=predict_lgd(r,np.empty((1,0)),**a)
    assert p.provenance['data_origin']=='SYNTHETIC' and p.provenance['parameter_origin']=='ESTIMATED'
