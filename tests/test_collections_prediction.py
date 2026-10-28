"""Ordered cure/re-default oracle differs from reversible state occupancy."""
import pytest
from creditratingsystem.collections import predict_cure_and_redefault
from creditratingsystem.survival import estimate_multistate_transitions
from creditratingsystem import CreditRiskError


def model_case():
    # Two initially defaulted accounts: A cures at1 and redefaults at2;
    # B remains in default. No historical default is erased by cure.
    target=dict(event='default',horizon=3,time_unit='years',probability_measure='PHYSICAL',conditioning='defaulted_episode')
    model=estimate_multistate_transitions([
        dict(entity_id='A',start=0,stop=1,from_state='D',to_state='C'),
        dict(entity_id='A',start=1,stop=2,from_state='C',to_state='D'),
        dict(entity_id='A',start=2,stop=3,from_state='D',to_state=None),
        dict(entity_id='B',start=0,stop=3,from_state='D',to_state=None)],
        states=['D','C'],allowed_transitions=[['D','C'],['C','D']],initial_state='D',target=target,
        training=dict(data_fingerprint='two-account-oracle',split_id='train',known_at='2023-01-01'),model_id='collections-aj',model_version='1').model
    model['collections_contract']=dict(cure_states=['C'],default_states=['D'],supported_actions=['baseline'],
        supported_initial_states=['D','C'],action_interpretation='observational_conditioning')
    return dict(current_episode=dict(episode_id='e',state='D',model_time=0,known_at='2024-01-01',ever_default=True),model=model,
        action_conditions=dict(action_id='baseline',assumption_reference='hand-oracle'),horizon_grid=[0,1,2],
        state_recoveries=[[0,0],[0,100],[0,0]],target=target,currency='EUR')


def test_cure_then_redefault_is_an_ordered_first_hit():
    v=predict_cure_and_redefault(**model_case()).value
    assert v['sustained_cure_probability']==[0,.5,.5]
    assert v['cure_then_redefault_probability']==[0,0,.5]
    assert [row[0] for row in v['state_occupancy']]==[1,.5,1]
    assert v['expected_recovery']==50
    assert v['ever_default'] and not v['causal_effect_identified']


def test_unsupported_action_and_future_model_not_hidden():
    args=model_case();args['action_conditions']['action_id']='unseen'
    with pytest.raises(CreditRiskError,match='COLLECTIONS_ACTION_SUPPORT'):predict_cure_and_redefault(**args)
    args['extrapolation']='flat'
    assert predict_cure_and_redefault(**args).value['treatment_support']=='REFERENCE_MODEL_EXTRAPOLATION'
    args=model_case();args['current_episode']['known_at']='2022-01-01'
    with pytest.raises(CreditRiskError,match='COLLECTIONS_LEAKAGE'):predict_cure_and_redefault(**args)
