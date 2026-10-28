import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from creditratingsystem.structural import (merton_equity_value,infer_merton_assets,distance_to_default,merton_default_probability,first_passage_pd,simulate_structural_paths,calibrate_distance_pd_mapping)
from creditratingsystem.core import CreditRiskError

META={'target':dict(event='default',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='observed_DD'),'calibration':dict(start='2020-01-01',end='2021-01-01',known_at='2022-01-01',split_id='calibration',prediction_origin='HOLDOUT'),'base_model_version':'1','population_definition':'test firms','outcome_policy':'complete one-year outcomes','weight_type':'frequency','data_origin':'SYNTHETIC'}

def test_equity_independent_payoff_integration_and_reference():
    result=merton_equity_value(100,100,.2,.05,1,currency='EUR').value
    # Direct discounted payoff integral, independent of closed form.
    oracle=quad(lambda z:max(100*math.exp(.03+.2*z)-100,0)*norm.pdf(z)*math.exp(-.05),-.15,10,epsabs=1e-11)[0]
    assert result['equity_value']==pytest.approx(oracle,abs=2e-12)
    assert result['equity_value']==pytest.approx(10.450583572185565,abs=2e-12)
    assert result['delta']==pytest.approx(.6368306511756191)
    assert merton_equity_value(100,0,.2,.05,1,.03,currency='EUR').value['equity_value']==pytest.approx(100*math.exp(-.03))
    assert merton_equity_value(100,100,0,0,1,currency='EUR').value['delta']==.5

@pytest.mark.parametrize('a,d,s,r,t,q',[(100,100,.2,.05,1,0),(130,80,.5,-.01,2,.03),(100,0,.2,.05,1,.02),(40,100,.8,.03,1,0)])
def test_inference_reconstructs_assets(a,d,s,r,t,q):
    option=merton_equity_value(a,d,s,r,t,q,currency='EUR').value
    inferred=infer_merton_assets(option['equity_value'],a*option['delta']*s/option['equity_value'],d,r,t,asset_bounds=[1,10000],volatility_bounds=[.001,4],currency='EUR',payout=q)
    assert inferred.value['asset_value']==pytest.approx(a,rel=1e-8)
    assert inferred.value['asset_volatility']==pytest.approx(s,rel=1e-8)

def test_nonconvergence_not_false_solution():
    with pytest.raises(CreditRiskError,match='NONCONVERGENCE'):
        infer_merton_assets(50,.2,100,.05,1,asset_bounds=[1,2],volatility_bounds=[.001,.01],currency='EUR')

def test_measure_event_and_deterministic_boundaries():
    physical=merton_default_probability(100,80,.2,.1,1,measure='PHYSICAL')
    risk_neutral=merton_default_probability(100,80,.2,.03,1,measure='RISK_NEUTRAL')
    assert physical.value<risk_neutral.value
    dd=distance_to_default(100,80,.2,.1,1).value['distance_to_default']
    assert physical.value==pytest.approx(norm.cdf(-dd))
    assert distance_to_default(100,100,0,0,0).value['boundary_status']=='DETERMINISTIC_AT_BARRIER'
    assert merton_default_probability(100,100,0,0,0,measure='PHYSICAL').value==1
    assert merton_default_probability(100,0,0,0,1,measure='PHYSICAL').value==0
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'): merton_default_probability(100,80,.2,.03,1,measure='PHYSICAL',context={'probability_measure':'RISK_NEUTRAL'})

@pytest.mark.parametrize('drift',[-.2,.02,.3])
def test_first_passage_independent_density_integral(drift):
    # Inverse Gaussian first-hitting density: x/(s sqrt(2 pi t^3))*exp(-(x+m*t)^2/(2s²t)).
    x=math.log(100/80);s=.2;m=drift-.5*s*s
    oracle=quad(lambda t:x/(s*math.sqrt(2*math.pi*t**3))*math.exp(-(x+m*t)**2/(2*s*s*t)),0,1,epsabs=1e-11)[0]
    pd=first_passage_pd(100,dict(kind='constant',initial_level=80),s,drift,1,measure='PHYSICAL').value
    assert pd==pytest.approx(oracle,abs=1e-11)
    assert pd>=merton_default_probability(100,80,s,drift,1,measure='PHYSICAL').value

def test_first_passage_reflection_exponential_and_start():
    oracle=math.erfc(math.log(100/80)/(.2*math.sqrt(2)))
    assert first_passage_pd(100,dict(kind='constant',initial_level=80),.2,.02,1,measure='PHYSICAL').value==pytest.approx(oracle)
    moving=first_passage_pd(100,dict(kind='exponential',initial_level=80,growth_rate=.03),.2,.05,1,measure='PHYSICAL').value
    assert moving==pytest.approx(oracle)
    assert first_passage_pd(80,dict(kind='constant',initial_level=80),.2,.02,0,measure='PHYSICAL').value==1
    assert first_passage_pd(100,dict(kind='constant',initial_level=80),0,-1,1,measure='PHYSICAL').value==1
    assert first_passage_pd(100,dict(kind='constant',initial_level=0),.2,.1,1,measure='PHYSICAL').value==0

def test_shared_paths_and_jumps_analytic():
    dynamics=dict(asset_value=100,volatility=.2,drift=.02)
    z=np.array([[0.,1.],[-1.,0.]])
    result=simulate_structural_paths(dynamics,[80,80,80],[0,.5,1],z,measure='PHYSICAL')
    expected=100*np.exp(np.column_stack([np.zeros(2),np.cumsum(.2*math.sqrt(.5)*z,axis=1)]))
    np.testing.assert_allclose(result.value['asset_paths'],expected,rtol=1e-14)
    none=simulate_structural_paths(dynamics,[80,80,80],[0,.5,1],z,measure='PHYSICAL',jump_counts=np.zeros_like(z),jump_innovations=np.ones_like(z))
    np.testing.assert_array_equal(none.value['asset_paths'],result.value['asset_paths'])
    jump=dict(asset_value=100,volatility=0,drift=0,jump_intensity=1,jump_log_mean=math.log(.5),jump_log_sd=0)
    result=simulate_structural_paths(jump,[80,80],[0,1],[[0]],measure='PHYSICAL',jump_counts=[[1]],jump_innovations=[[0]])
    assert result.value['asset_paths'][0,1]==pytest.approx(50*math.exp(.5))
    assert result.diagnostics['barrier_miss_between_grid_points']

def test_bridge_sees_unobserved_crossing():
    dynamics=dict(asset_value=100,volatility=.2,drift=.02)
    plain=simulate_structural_paths(dynamics,[90,90],[0,1],[[0]],measure='PHYSICAL')
    bridge=simulate_structural_paths(dynamics,[90,90],[0,1],[[0]],measure='PHYSICAL',bridge_uniforms=[[.1]])
    assert plain.value['survival_curve'][-1]==1
    assert bridge.value['survival_curve'][-1]==0
    assert bridge.value['default_time_upper_bounds']==[1.]

def test_bridge_mc_against_continuous_oracle():
    # Fixed local generator; no cross-language same-seed claim.
    rng=np.random.default_rng(913);n=30000
    result=simulate_structural_paths(dict(asset_value=100,volatility=.2,drift=.02),[80,80],[0,1],rng.standard_normal((n,1)),measure='PHYSICAL',bridge_uniforms=rng.uniform(size=(n,1)))
    oracle=math.erfc(math.log(100/80)/(.2*math.sqrt(2)))
    assert abs(1-result.value['survival_curve'][-1]-oracle)<5*math.sqrt(oracle*(1-oracle)/n)

def test_empirical_distance_mapping_and_bootstrap():
    x=[0,1,2,3];y=[1,0,1,0]
    result=calibrate_distance_pd_mapping(x,y,metadata=META,bootstrap_indices=[[0,1,2,3],[0,0,3,3]],out_of_bounds='clip')
    np.testing.assert_allclose(result.value['pd'],[1,.5,.5,0])
    assert np.all(np.diff(result.value['pd'])<=0)
    assert result.value['lower'] is not None
    assert result.diagnostics['bootstrap_samples']==2
    assert not result.diagnostics['out_of_sample_validation_performed']
    with pytest.raises(CreditRiskError,match='INVALID_BOOTSTRAP'):
        calibrate_distance_pd_mapping(x,y,metadata=META,bootstrap_indices=[[1,2,3,4],[0,1,2,3]])
