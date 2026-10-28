import numpy as np
import pytest
from scipy.stats import norm
from creditratingsystem import CreditRiskError
from creditratingsystem._simulation_risk import importance_sample_defaults


def contract(design='IID'):
    return dict(contract_id='is',version='1',evaluator_id='threshold',evaluator_version='1',base_law='INDEPENDENT_STANDARD_NORMAL',proposal='MEAN_SHIFT',all_randomness_in_innovations=True,deterministic_given_innovations=True,sampling_design=design,subject_id='o',currency='EUR',probability_measure='PHYSICAL',event='first_default',conditioning='initially_performing',horizon=1,time_unit='years')


def evaluator(x):
    d=x[:,0]>0
    return dict(defaulted=d,economic_loss=d*100.)


def test_zero_shift_is_ordinary_monte_carlo():
    r=importance_sample_defaults([[-1.],[1.],[2.]],[0.],evaluator,contract()).value
    assert r['pd_estimate']==pytest.approx(2/3)
    assert r['expected_loss_estimate']==pytest.approx(200/3)
    assert r['likelihood_ratios']==[1,1,1]
    assert r['pd_mc_standard_error']==pytest.approx(1/3)


def test_every_shifted_coordinate_enters_full_density_ratio():
    z=np.array([[-1.,.2],[1.,-.4],[.2,.7]]);shift=np.array([.5,-.8]);x=z+shift
    ratios=np.prod(norm.pdf(x),axis=1)/np.prod(norm.pdf(z),axis=1)
    r=importance_sample_defaults(z,shift,evaluator,contract()).value
    assert r['likelihood_ratios']==pytest.approx(ratios)
    assert r['pd_estimate']==pytest.approx(np.mean(ratios*(x[:,0]>0)))
    assert r['expected_loss_estimate']==pytest.approx(100*r['pd_estimate'])


def test_analytic_gaussian_rare_tail_without_fake_qmc_standard_error():
    n=2048;z=norm.ppf((np.arange(n)+.5)/n)[:,None]
    def tail(x):
        d=x[:,0]>3
        return dict(defaulted=d,economic_loss=70.*d)
    r=importance_sample_defaults(z,[3.],tail,contract('DEPENDENT_OR_QMC')).value
    assert r['pd_estimate']==pytest.approx(norm.sf(3),abs=1e-8)
    assert r['pd_mc_standard_error']is None
    assert r['expected_loss_estimate']==pytest.approx(70*r['pd_estimate'])


def test_zero_proposal_defaults_has_no_target_binomial_bound():
    r=importance_sample_defaults([[-4.],[-5.]],[1.],evaluator,contract()).value
    assert r['zero_proposal_defaults']and r['pd_estimate']==0
    assert r['target_zero_default_bound']is None
    assert r['pd_mc_standard_error']==0


def test_no_clipping_of_unbiased_estimate_and_controlled_bad_inputs():
    r=importance_sample_defaults([[-1.]],[-1.],lambda x:dict(defaulted=[True],economic_loss=[10.]),contract()).value
    assert r['pd_estimate']<1
    r=importance_sample_defaults([[-2.]],[1.],lambda x:dict(defaulted=[True],economic_loss=[10.]),contract()).value
    assert r['pd_estimate']>1 and not r['pd_estimate_in_unit_interval']
    for z,shift in [([[1,2]],[1]),([[1]],[1000])]:
        with pytest.raises(CreditRiskError):importance_sample_defaults(z,shift,evaluator,contract())
    with pytest.raises(CreditRiskError):importance_sample_defaults([[-1]],[0],lambda x:dict(defaulted=[False],economic_loss=[1]),contract())
