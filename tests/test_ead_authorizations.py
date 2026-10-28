"""Hand-derived reservation/booked-exposure and exact monetary refund cases."""
from copy import deepcopy
import pytest
from creditratingsystem._ead_authorizations import reconcile_card_authorizations
from creditratingsystem.core import CreditRiskError


def inputs(amount=200):
    auth = dict(authorization_id='a', account_id='card', approved_at='2025-01-01',
                known_at='2025-01-01', sequence=0, currency='EUR', amount=amount,
                expires_at='2025-01-10')
    policy = dict(policy_id='p', version='1', known_at='2024-12-01',
                  expiry_boundary='EXCLUSIVE', late_settlement_allowed=False,
                  authorization_conversion_factor=1, minor_unit=.01)
    return [auth], policy


def event(identity, amount, kind='SETTLEMENT', day='2025-01-02', sequence=1, **extra):
    row = dict(event_id=identity, authorization_id='a', account_id='card',
               occurred_at=day, known_at=day, sequence=sequence, currency='EUR',
               amount=amount, kind=kind)
    if kind == 'SETTLEMENT': row['authorization_complete'] = False
    row.update(extra)
    return row


def calculate(events=(), amount=200, as_of='2025-01-05', **changes):
    auth, policy = inputs(amount)
    policy.update(changes)
    return reconcile_card_authorizations(auth, list(events), policy, as_of=as_of,
                                         known_at=as_of, currency='EUR').value


def test_reservations_are_replaced_by_settlement():
    assert calculate()['accounts'][0]['scenario_exposure'] == 200
    value = calculate([event('s', 200, authorization_complete=True)])
    assert value['accounts'][0] == dict(account_id='card', booked_exposure=200,
                                      pending_authorization=0, scenario_exposure=200)
    partial = calculate([event('s', 80)])['accounts'][0]
    assert partial['booked_exposure'] == 80
    assert partial['pending_authorization'] == 120
    assert partial['scenario_exposure'] == 200
    assert calculate([event('s',80,authorization_complete=True)])['accounts'][0]['scenario_exposure'] == 80


def test_expiry_cancellation_and_reversal_do_not_create_credit():
    assert calculate(as_of='2025-01-10')['accounts'][0]['scenario_exposure'] == 0
    cancellation = event('c',0,'CANCELLATION',sequence=2)
    assert calculate([event('s',80),cancellation])['accounts'][0]['scenario_exposure'] == 80
    reversal = event('r',40,'REVERSAL',sequence=3,reverses_settlement_id='s')
    row = calculate([event('s',80),reversal])['accounts'][0]
    assert row['booked_exposure'] == 40 and row['pending_authorization'] == 120
    # Reversed purchases do not restore reserved capacity.
    assert row['scenario_exposure'] == 160


def test_decimal_refunds_reconcile_exactly_and_factor_is_explicit():
    events = [event('s',.3,authorization_complete=True),
              event('r1',.1,'REVERSAL',sequence=2,reverses_settlement_id='s'),
              event('r2',.2,'REVERSAL',sequence=3,reverses_settlement_id='s')]
    assert calculate(events,amount=.3)['accounts'][0]['booked_exposure'] == 0
    assert calculate([event('s',80)],authorization_conversion_factor=.25)['accounts'][0]['scenario_exposure'] == 110


def test_knowledge_cutoff_and_late_settlement_policy():
    late = event('s',200,known_at='2025-01-07',authorization_complete=True)
    old = calculate([late])
    assert old['accounts'][0]['pending_authorization'] == 200
    assert old['excluded'] == [dict(record_id='s',reason='NOT_YET_KNOWN')]
    assert calculate([late],as_of='2025-01-07')['accounts'][0]['pending_authorization'] == 0
    expired = event('s',200,day='2025-01-10')
    with pytest.raises(CreditRiskError): calculate([expired],as_of='2025-01-10')
    assert calculate([expired],as_of='2025-01-10',late_settlement_allowed=True)['accounts'][0]['booked_exposure'] == 200


@pytest.mark.parametrize('events',[
    [event('s',201)],
    [event('s',200),event('s',0,sequence=2)],
    [event('s',100),event('x',100)],
    [event('r',1,'REVERSAL',reverses_settlement_id='absent')],
    [event('s',80),event('r',81,'REVERSAL',sequence=2,reverses_settlement_id='s')],
    [event('s',.001)],
])
def test_inconsistent_ledgers_rejected(events):
    with pytest.raises(CreditRiskError): calculate(deepcopy(events))
