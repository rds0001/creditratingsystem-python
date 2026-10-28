"""Explicit finite probability partitions and dated payment-route cases."""
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.transfer import (assess_transfer_applicability,
    transfer_event_default_probability,combine_default_transfer,transfer_loss_components)

TARGET=dict(exposure_id='facility',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing')


def risks(p=.1,event=.2,failure=.5,success=0):
    transfer=transfer_event_default_probability(event,failure,dict(effect_id='liquidity',
        success_probability_given_event_and_unmitigated_failure=success,conditioning='event_and_unmitigated_failure',
        assumption_reference='synthetic-tree'),event_id='freeze',target=TARGET)
    return dict(stage='PRE_TRANSFER',pd=p,target=TARGET,applied_effects=[]),transfer


@pytest.mark.parametrize('failure,success,q',[(0,0,0),(1,0,.2),(.5,0,.1),(.5,.6,.04),(.5,1,0)])
def test_conditional_mapping_and_mitigation(failure,success,q):
    _,transfer=risks(failure=failure,success=success)
    assert transfer.value['probability']==pytest.approx(q)
    assert transfer.value['unmitigated_default_probability']==pytest.approx(transfer.value['probability']+transfer.value['mitigated_default_mass'])


def test_independent_and_nested_event_union():
    base,transfer=risks()
    independent=combine_default_transfer(base,transfer,dependence='independent')
    assert independent.value['combined_pd']==pytest.approx(.19)
    assert independent.value['exclusive_probabilities']==pytest.approx(dict(neither=.81,default_only=.09,transfer_only=.09,joint=.01))
    nested=combine_default_transfer(base,transfer,dependence='explicit',joint_probability=.1)
    assert nested.value['combined_pd']==.1
    assert nested.value['exclusive_probabilities']['default_only']==0
    with pytest.raises(CreditRiskError,match='TRANSFER_DOUBLE_COUNT'):
        combine_default_transfer(dict(nested.value,pd=.1),transfer,dependence='independent')


@pytest.mark.parametrize('joint',[.11,-.01,.8])
def test_invalid_joint_probability(joint):
    base,transfer=risks()
    with pytest.raises(CreditRiskError):combine_default_transfer(base,transfer,dependence='explicit',joint_probability=joint)


def test_exclusive_losses_and_zero_transfer_baseline():
    base,transfer=risks();combined=combine_default_transfer(base,transfer,dependence='independent')
    out=transfer_loss_components(combined,dict(neither=0,default_only=60,transfer_only=20,joint=70),currency='EUR',applied_loss_effects=[]).value
    assert out['expected_loss']==pytest.approx(7.9) # .09*60+.09*20+.01*70
    with pytest.raises(CreditRiskError,match='TRANSFER_DOUBLE_COUNT'):
        transfer_loss_components(combined,dict(neither=0,default_only=60,transfer_only=20,joint=70),currency='EUR',applied_loss_effects=['liquidity'])
    base,transfer=risks(event=0);combined=combine_default_transfer(base,transfer,dependence='explicit',joint_probability=0)
    assert transfer_loss_components(combined,dict(neither=0,default_only=60,transfer_only=20,joint=70),currency='EUR',applied_loss_effects=[]).value['expected_loss']==6


def policy_case():
    payment=dict(payment_id='p',due_at='2024-06-01',known_at='2024-01-01',currency='EUR',source_currency='EUR',
        payer_jurisdiction='A',payee_jurisdiction='B',payment_jurisdictions=['A','B'])
    event=dict(event_id='freeze',event_type='PAYMENT_FREEZE',jurisdictions=['A'],currencies=['EUR'],
        effective_from='2024-01-01',effective_until=None,known_at='2023-12-01',exemptions=[])
    return dict(payments=[payment],event_policy=dict(policy_id='explicit',version='1',events=[event]),as_of='2024-01-01',known_at='2024-01-01',horizon_end='2024-12-31')


def test_same_currency_freeze_applicable_and_known_exemption_reproducible():
    args=policy_case();assert assess_transfer_applicability(**args).value['payments'][0]['status']=='APPLICABLE'
    args['event_policy']['events'][0]['exemptions']=[dict(payment_id='p',reason='documented route exception',evidence_reference='synthetic-contract',known_at='2024-02-01')]
    assert assess_transfer_applicability(**args).value['payments'][0]['status']=='APPLICABLE'
    args['known_at']='2024-03-01'
    assert assess_transfer_applicability(**args).value['payments'][0]['status']=='EXEMPT'


def test_missing_route_is_not_assumed_no_risk():
    args=policy_case();args['payments'][0].pop('payment_jurisdictions')
    assert assess_transfer_applicability(**args).value['payments'][0]['status']=='INSUFFICIENT_INFORMATION'
    args=policy_case();args['event_policy']['events'][0]['event_type']='CONVERSION_RESTRICTION'
    assert assess_transfer_applicability(**args).value['payments'][0]['status']=='NO_MATCHING_EVENT'
