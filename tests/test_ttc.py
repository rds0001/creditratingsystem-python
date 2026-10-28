import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm
from creditratingsystem._ttc import pit_ttc_transform
from creditratingsystem.core import CreditRiskError

F = dict(family='standard_normal', factor_id='cycle', reference_period='synthetic', model_version='1')
def transform(p,z,r,direction='ttc_to_pit', **kw):
    return pit_ttc_transform(p,z,r,direction=direction,factor_distribution=F,horizon=1,time_unit='years',probability_measure='PHYSICAL',**kw)

@pytest.mark.parametrize('p,r',[(.01,.2),(.2,.7),(.9,.01)])
def test_integrated_conditional_pd_is_ttc(p,r):
    # Independent law-of-total-probability oracle, numerical integration over z.
    integral, error = quad(lambda z: transform(p,z,r).value*norm.pdf(z), -10,10,epsabs=1e-12)
    assert abs(integral-p) < 1e-10
    assert error < 1e-8
    assert transform(p,0,r).value != pytest.approx(p,abs=1e-8)

def test_inverse_sign_boundary_and_zero_correlation():
    p=np.array([.001,.02,.5,.9]); z=np.array([-1.,0.,1.,2.])
    pit=transform(p,z,.2).value
    np.testing.assert_allclose(transform(pit,z,.2,'pit_to_ttc').value,p,rtol=1e-12)
    assert transform(.02,-1,.2).value > transform(.02,1,.2).value
    np.testing.assert_array_equal(transform([0,1],1,.2).value,[0,1])
    np.testing.assert_array_equal(transform(p,1,0).value,p)
    assert transform(.5,0,.5).value == .5

@pytest.mark.parametrize('p,z,r,code',[(.1,0,1,'INVALID_CORRELATION'),(.1,0,-.1,'INVALID_CORRELATION'),([.1,.2],[0,1,2],.2,'INCOMPATIBLE_SHAPE'),(.1,np.nan,.2,'NON_FINITE_INPUT')])
def test_errors(p,z,r,code):
    with pytest.raises(CreditRiskError) as caught: transform(p,z,r)
    assert caught.value.code==code

def test_target_and_numerical_boundary():
    with pytest.raises(CreditRiskError,match='TARGET_MISMATCH'): transform(.1,0,.2,context={'horizon':2})
    result=transform(.1,100,.99)
    assert result.diagnostics['inverse_information_loss']
