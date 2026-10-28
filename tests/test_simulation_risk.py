import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem._simulation_risk import estimate_simulation_risk


def sampling(design='IID_UNWEIGHTED'):
    return dict(contract_id='mc',version='1',subject_id='o',currency='EUR',probability_measure='PHYSICAL',horizon=1,time_unit='years',conditioning='initially_performing',sampling_design=design,event='first_default')


def trial(i,defaulted=False,ead=0,loss=0,weight=1,initial=False,complete=True):
    return dict(scenario_id=str(i),subject_id='o',initial_default=initial,defaulted=defaulted,horizon_complete=complete,weight=weight,ead_at_default=ead,economic_loss=loss)


def test_zero_default_exact_bound_and_initial_default_exclusion():
    rows=[trial(i)for i in range(10)]+[trial('old',True,100,50,initial=True)]
    r=estimate_simulation_risk(rows,sampling()).value
    assert r['pd']==0 and r['risk_set_trials']==10
    assert r['pd_interval']==pytest.approx([0,1-.025**.1])
    assert r['pd_mc_standard_error']==0 and r['pd_boundary_sample']
    assert r['conditional_mean_lgd']is None
    assert r['excluded']==[dict(scenario_id='old',reason='INITIAL_DEFAULT')]


def test_joint_loss_mean_not_product_of_separate_conditional_means():
    rows=[trial('a',True,100,20),trial('b',True,200,160),trial('c')]
    r=estimate_simulation_risk(rows,sampling()).value
    assert r['pd']==pytest.approx(2/3) and r['expected_loss']==pytest.approx(60)
    lower,upper=r['pd_interval']
    assert 3*lower**2-2*lower**3==pytest.approx(.025)
    assert upper==pytest.approx(.975**(1/3))
    assert r['conditional_mean_ead']==150 and r['conditional_mean_lgd']==.5
    assert r['ead_weighted_lgd']==.6
    assert r['pd']*r['conditional_mean_ead']*r['conditional_mean_lgd']==pytest.approx(50)
    assert r['pd']*r['conditional_mean_ead']*r['ead_weighted_lgd']==pytest.approx(r['expected_loss'])


def test_weighted_and_dependent_samples_do_not_invent_binomial_trials():
    rows=[trial('a',True,100,20,.1),trial('b',True,200,160,.2),trial('c',weight=.7)]
    for design in ['FIXED_WEIGHTED_SCENARIOS','DEPENDENT_OR_QMC']:
        r=estimate_simulation_risk(rows,sampling(design)).value
        assert r['pd']==pytest.approx(.3) and r['expected_loss']==pytest.approx(34)
        assert r['pd_interval']is None and r['loss_mc_standard_error']is None
        assert r['effective_weight_count']==pytest.approx(1/.54)
    with pytest.raises(CreditRiskError,match='SIMULATION_SAMPLING'):estimate_simulation_risk(rows,sampling())


def test_zero_ead_raw_lgd_and_empty_risk_set_are_explicit():
    r=estimate_simulation_risk([trial('a',True,0,10)],sampling()).value
    assert r['expected_loss']==10 and r['conditional_mean_lgd']is None and r['ead_weighted_lgd']is None
    r=estimate_simulation_risk([trial('a',True,100,150)],sampling()).value
    assert r['conditional_mean_lgd']==1.5
    r=estimate_simulation_risk([],sampling()).value
    assert r['status']=='EMPTY_RISK_SET'and r['pd']is None


def test_incomplete_horizon_and_repeated_subjects_are_not_extra_trials():
    with pytest.raises(CreditRiskError,match='SIMULATION_CENSORING'):estimate_simulation_risk([trial('a',complete=False)],sampling())
    with pytest.raises(CreditRiskError,match='SIMULATION_DUPLICATE'):estimate_simulation_risk([trial('a'),trial('a')],sampling())
    row=trial('a');row['subject_id']='other'
    with pytest.raises(CreditRiskError,match='SIMULATION_SAMPLING'):estimate_simulation_risk([row],sampling())


def test_weighted_moments_use_correctly_accumulated_binary64_products():
    from decimal import Decimal, localcontext
    from itertools import permutations
    # Exact decimal accumulation of already rounded binary64 products is an
    # independent oracle; BLAS reductions lose the small net recovery here.
    weights=[1.,1.,1.]
    losses=[1e16,1.,-1e16]
    with localcontext() as ctx:
        ctx.prec=100
        expected=float(sum(Decimal.from_float((w/3)*v)for w,v in zip(weights,losses)))
    for ordering in permutations(range(3)):
        rows=[trial(i,True,1e16,losses[i],weights[i])for i in ordering]
        result=estimate_simulation_risk(rows,sampling('FIXED_WEIGHTED_SCENARIOS')).value
        assert result['expected_loss']==expected
        assert result['conditional_mean_loss']==expected
        assert result['effective_weight_count']==3.
