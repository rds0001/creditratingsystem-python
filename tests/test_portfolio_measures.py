import math
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.portfolio import credit_loss_distribution,allocate_expected_shortfall,incremental_credit_risk,credit_concentration_measures,asymptotic_factor_loss


def test_two_atoms_fractional_es_mass_and_boundary_limits():
    r=credit_loss_distribution([0,100],[.9,.1],scenario_ids=['a','b'],levels=[0,.8,.95,1],currency='EUR').value
    assert r['expected_loss']==10
    assert [t['value_at_risk']for t in r['tails']]==[0,0,100,100]
    assert [t['expected_shortfall']for t in r['tails']]==pytest.approx([10,50,100,100])
    assert r['tails'][1]['normalized_tail_weights']==pytest.approx([.5,.5])
    assert r['standard_deviation']==pytest.approx(30)


def test_signed_position_allocation_reconciles_under_boundary_ties():
    r=allocate_expected_shortfall([[-10,10],[120,-20]],[.9,.1],scenario_ids=['a','b'],position_ids=['loan','hedge'],level=.8,currency='EUR',groups=dict(loan='book',hedge='book')).value
    assert [p['contribution']for p in r['contributions']]==pytest.approx([55,-5])
    assert r['expected_shortfall']==pytest.approx(50)
    assert r['groups'][0]['contribution']==pytest.approx(50)
    assert abs(r['reconciliation_residual'])<1e-12


def test_paired_zero_change_and_positive_homogeneity():
    base=[0,100]
    zero=incremental_credit_risk(base,[0,0],[.9,.1],scenario_ids=['a','b'],levels=[.8],currency='EUR',change_id='none').value
    double=incremental_credit_risk(base,base,[.9,.1],scenario_ids=['a','b'],levels=[.8],currency='EUR',change_id='double').value
    assert zero['expected_loss_change']==0 and zero['tail_changes'][0]['expected_shortfall_change']==0
    assert double['expected_loss_change']==10 and double['tail_changes'][0]['expected_shortfall_change']==pytest.approx(50)


def test_concentration_and_connected_group_aggregation():
    rows=[dict(position_id=str(i),ead=1,group='connected'if i<2 else str(i))for i in range(4)]
    r=credit_concentration_measures(rows,weight_field='ead',group_fields=['group'],weight_unit='EUR').value
    assert r['dimensions'][0]['hhi']==.25 and r['dimensions'][0]['effective_number']==4
    assert r['dimensions'][1]['hhi']==.375
    assert r['dimensions'][0]['entropy_effective_number']==4
    for row in rows:row['ead']=0
    r=credit_concentration_measures(rows,weight_field='ead',group_fields=['group'],weight_unit='EUR').value
    assert r['status']=='ZERO_WEIGHT_DENOMINATOR'and r['dimensions'][0]['hhi']is None


def test_asrf_zero_correlation_and_symmetric_gaussian_oracle():
    r=asymptotic_factor_loss(.2,.4,0,[-2,0,2]).value
    assert r['conditional_loss_rate']==pytest.approx([.08]*3)
    r=asymptotic_factor_loss(.5,.4,.5,[-1,0,1]).value
    expected=[.5*(1+math.erf(-y/math.sqrt(2)))for y in [-1,0,1]]
    assert r['conditional_pd']==pytest.approx(expected)
    assert r['unconditional_expected_loss_rate']==.2
    assert asymptotic_factor_loss(0,.4,.9,[0]).value['conditional_pd']==[0]


def test_zero_mass_outlier_never_becomes_var_or_es():
    r=credit_loss_distribution([-100,0,100],[0,1,0],scenario_ids=['a','b','c'],levels=[0,.95,1],currency='EUR').value
    assert all(t['expected_shortfall']==0 and t['value_at_risk']==0 for t in r['tails'])
    with pytest.raises(CreditRiskError):credit_loss_distribution([1],[0],scenario_ids=['a'],currency='EUR')
