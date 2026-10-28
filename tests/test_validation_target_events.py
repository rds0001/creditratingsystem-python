from copy import deepcopy
import pytest
from creditratingsystem._expert_common import _ex_target
from creditratingsystem._validation_common import _vb_design
from creditratingsystem import content_fingerprint
from creditratingsystem.core import CreditRiskError


def design():
    return dict(model_id='fixed',model_version='1',parameter_version='1',data_fingerprint='cohort',population_id='funded',split_id='heldout',prediction_origin='HOLDOUT',evaluation_cutoff='2025-01-02',target=dict(event='default',horizon=365,time_unit='days',probability_measure='PHYSICAL',conditioning='given_funding'))


def test_validation_event_preserved_expert_default_unchanged():
    d=design();assert _vb_design(d)==d and _ex_target(d['target'])==d['target']
    original_hash=content_fingerprint(d).value;first=deepcopy(d);first['target']['event']='first_default'
    assert _vb_design(first)==first
    assert content_fingerprint(_vb_design(first)).value!=original_hash
    with pytest.raises(CreditRiskError):_ex_target(first['target'])
    for bad in ['delinquency','prepayment','unknown']:
        z=deepcopy(d);z['target']['event']=bad
        with pytest.raises(CreditRiskError):_vb_design(z)


def test_existing_measure_horizon_and_context_guards():
    for field,value in [('probability_measure','RISK_NEUTRAL'),('horizon',0),('conditioning','')]:
        d=design();d['target']['event']='first_default';d['target'][field]=value
        with pytest.raises(CreditRiskError):_vb_design(d)
    with pytest.raises(CreditRiskError):_vb_design(design(),context=dict(horizon=30))
