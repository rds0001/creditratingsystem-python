"""Independent finite-state arithmetic and dated evidence changes."""
from copy import deepcopy
from fractions import Fraction
from importlib.resources import files
import importlib.util
import json
from pathlib import Path
import pytest
import creditratingsystem as cr

spec=importlib.util.spec_from_file_location('transfer_review',Path(__file__).parents[1]/'examples/transfer_review.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)

def data():return json.loads(files('creditratingsystem.datasets').joinpath('transfer_review.json').read_text())

def test_independent_event_partition_and_replay():
    result=example.run();p=Fraction(1,10);q=Fraction(2,25)
    for case,j in [('independent',p*q),('nested',q),('disjoint',Fraction(0))]:
        states=dict(neither=1-p-q+j,default_only=p-j,transfer_only=q-j,joint=j)
        expected=sum(states[k]*v for k,v in dict(neither=0,default_only=60,transfer_only=20,joint=70).items())
        row=result['original']['scenarios'][case]
        assert row['risk']['exclusive_probabilities']==pytest.approx({k:float(v)for k,v in states.items()})
        assert row['loss']['expected_loss']==pytest.approx(float(expected))
    assert result['original_hash']==result['replay_hash']
    assert result['missing_route']['scenarios']is None

def test_frechet_stress_and_same_currency_conversion():
    d=data();d['pre_transfer_pd']=.7;d['event_probability']=.75;d['event_to_default_probability']=1
    result=example.assess(d)['scenarios']
    assert result['disjoint']['risk']['combined_pd']==pytest.approx(1)
    assert result['disjoint']['loss']['expected_loss']==pytest.approx(51)
    assert result['nested']['risk']['combined_pd']==pytest.approx(.7)
    assert result['nested']['loss']['expected_loss']==pytest.approx(48)
    d=data();d['event_policy']['events'][0]['event_type']='CONVERSION_RESTRICTION'
    assert example.assess(d)['transfer']['probability']==0

def test_mitigation_cannot_be_applied_twice():
    d=data();result=example.assess(d);risk=result['scenarios']['independent']['risk']
    with pytest.raises(cr.CreditRiskError,match='TRANSFER_DOUBLE_COUNT'):
        cr.combine_default_transfer(dict(risk,pd=risk['combined_pd']),result['transfer'],dependence='independent')
    with pytest.raises(cr.CreditRiskError,match='TRANSFER_DOUBLE_COUNT'):
        cr.transfer_loss_components(risk,d['conditional_losses'],currency='EUR',applied_loss_effects=['prearranged-liquidity'])
