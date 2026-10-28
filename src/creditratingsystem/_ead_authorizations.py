"""Dated reconciliation of card reservations and their actual booked charges."""
from copy import deepcopy
import math
from .core import _result, _fail, _context_value, _fingerprint
from .data import _rows, _required, _unique, _date


def reconcile_card_authorizations(authorizations, events, policy, *, as_of, known_at,
                                  currency, context=None):
    """Reconcile the supplied ledger without adding settled reservations twice.

    Expiry is exclusive at the stated date. Reversal does not reinstate a hold.
    Conversion of remaining reservations is an explicit scenario assumption.
    """
    anchor, cutoff = _date(as_of, 'as_of'), _date(known_at, 'known_at')
    ctx = _context_value(context)
    if not isinstance(currency, str) or not currency.strip():
        _fail('AUTHORIZATION_CURRENCY', 'Explicit currency required')
    for key, value in [('currency', currency), ('as_of', as_of), ('known_at', known_at)]:
        if ctx.get(key) is not None and ctx[key] != value:
            _fail('TARGET_MISMATCH', 'Authorization context differs from ' + key)
    if not isinstance(policy, dict):
        _fail('AUTHORIZATION_POLICY', 'Explicit policy required')
    _required(policy, ['policy_id', 'version', 'known_at', 'expiry_boundary',
                       'late_settlement_allowed', 'authorization_conversion_factor', 'minor_unit'])
    if not all(isinstance(policy[k], str) and policy[k].strip() for k in ('policy_id', 'version')):
        _fail('AUTHORIZATION_POLICY', 'Named versioned policy required')
    if _date(policy['known_at'], 'policy.known_at') > cutoff or policy['expiry_boundary'] != 'EXCLUSIVE':
        _fail('AUTHORIZATION_POLICY', 'Policy must be known; declare exclusive expiry boundary')
    if type(policy['late_settlement_allowed']) is not bool:
        _fail('AUTHORIZATION_POLICY', 'Explicit Boolean late settlement rule required')

    def number(value, name):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            _fail('AUTHORIZATION_AMOUNT', name + ' requires a finite nonnegative number')
        return float(value)

    minor = number(policy['minor_unit'], 'minor_unit')
    if minor <= 0:
        _fail('AUTHORIZATION_POLICY', 'Positive explicit monetary minor unit required')
    # Integer units keep partial captures/refunds exact in both Python and R.
    # The common bound also leaves headroom for intermediate binary64 sums.
    maximum_units = 2**40

    def units(value):
        scaled = number(value, 'amount') / minor
        if not math.isfinite(scaled) or scaled > maximum_units:
            _fail('AUTHORIZATION_AMOUNT', 'Amount exceeds supported minor-unit range')
        rounded = round(scaled)
        tolerance = max(1e-9, min(1e-3, 8 * 2**-52 * abs(scaled)))
        if abs(scaled - rounded) > tolerance:
            _fail('AUTHORIZATION_AMOUNT', 'Amount is not a whole monetary minor unit')
        return int(rounded)

    def money(value):
        result = value * minor
        if abs(value) > maximum_units or not math.isfinite(result):
            _fail('AUTHORIZATION_AMOUNT', 'Aggregated amount exceeds supported range')
        return result

    factor = number(policy['authorization_conversion_factor'], 'authorization_conversion_factor')
    if factor > 1:
        _fail('AUTHORIZATION_POLICY', 'Authorization conversion factor must lie in [0,1]')
    auths, entries = _rows(authorizations, 'authorizations'), _rows(events, 'events')
    _unique(auths, 'authorization_id', 'authorizations'); _unique(entries, 'event_id', 'events')
    keys = set(); all_auth = {}; visible = {}; excluded = []; ordered = []; accounts = set()

    def base(row, identity, date_key):
        _required(row, [identity, 'account_id', date_key, 'known_at', 'sequence', 'currency', 'amount'])
        for key in (identity, 'account_id'):
            if not isinstance(row[key], str) or not row[key].strip():
                _fail('AUTHORIZATION_ID', 'Nonempty identity required')
        date, known = _date(row[date_key], date_key), _date(row['known_at'], 'known_at')
        if known < date or row['currency'] != currency:
            _fail('AUTHORIZATION_CONTRACT', 'Knowledge/order or currency mismatch')
        seq = number(row['sequence'], 'sequence')
        if seq > 2**40 or seq != int(seq): _fail('AUTHORIZATION_ORDER', 'Integer event sequence required')
        key = (row['account_id'], row[date_key], int(seq))
        if key in keys: _fail('AUTHORIZATION_ORDER', 'Resolve same-account same-date sequence')
        keys.add(key); units(row['amount'])
        if date > anchor or known > cutoff:
            excluded.append(dict(record_id=row[identity], reason='FUTURE_OCCURRENCE' if date > anchor else 'NOT_YET_KNOWN'))
            return False
        accounts.add(row['account_id'])
        return True

    for row in auths:
        present = base(row, 'authorization_id', 'approved_at')
        _required(row, ['expires_at'])
        if _date(row['expires_at'], 'expires_at') <= _date(row['approved_at'], 'approved_at'):
            _fail('AUTHORIZATION_EXPIRY', 'Expiry must follow authorization')
        all_auth[row['authorization_id']] = row
        if present:
            visible[row['authorization_id']] = dict(source=deepcopy(row), captured=0, reversed=0, closed=False, cancellation=False)
            ordered.append((row['approved_at'], row['sequence'], 'AUTHORIZATION', row))
    for row in entries:
        present = base(row, 'event_id', 'occurred_at')
        _required(row, ['kind', 'authorization_id'])
        for key in ('kind', 'authorization_id'):
            if not isinstance(row[key], str) or not row[key].strip():
                _fail('AUTHORIZATION_REFERENCE', 'Nonempty event kind and reference required')
        if row['kind'] not in ('SETTLEMENT', 'REVERSAL', 'CANCELLATION'):
            _fail('AUTHORIZATION_EVENT', 'Unsupported event kind')
        if row['authorization_id'] not in all_auth:
            _fail('AUTHORIZATION_REFERENCE', 'Every event requires a supplied authorization')
        source = all_auth[row['authorization_id']]
        if row['account_id'] != source['account_id'] or (row['occurred_at'], row['sequence']) <= (source['approved_at'], source['sequence']):
            _fail('AUTHORIZATION_ORDER', 'Event must follow its authorization on the same account')
        if row['kind'] == 'SETTLEMENT' and type(row.get('authorization_complete')) is not bool:
            _fail('AUTHORIZATION_EVENT', 'Settlement requires explicit completion flag')
        if row['kind'] == 'REVERSAL':
            _required(row, ['reverses_settlement_id'])
            if not isinstance(row['reverses_settlement_id'], str) or not row['reverses_settlement_id'].strip():
                _fail('AUTHORIZATION_REFERENCE', 'Nonempty settlement reference required')
        if row['kind'] == 'CANCELLATION' and row['amount'] != 0:
            _fail('AUTHORIZATION_EVENT', 'Cancellation releases reservation, not booked money')
        if present:
            if row['authorization_id'] not in visible:
                _fail('AUTHORIZATION_REFERENCE', 'Known event refers to unavailable authorization')
            ordered.append((row['occurred_at'], row['sequence'], row['kind'], row))
    settlements = {}; ledger = []
    for date, seq, kind, row in sorted(ordered, key=lambda x: (x[0], x[1], x[3]['account_id'])):
        aid = row['authorization_id']; state = visible[aid]; source = state['source']; amount = units(row['amount'])
        if kind == 'SETTLEMENT':
            if state['closed']:
                _fail('AUTHORIZATION_CLOSED', 'Cannot settle cancelled or completed authorization')
            if date >= source['expires_at'] and not policy['late_settlement_allowed']:
                _fail('AUTHORIZATION_EXPIRED', 'Late settlement is not allowed by supplied policy')
            if state['captured'] + amount > units(source['amount']):
                _fail('AUTHORIZATION_OVERCAPTURE', 'Capture exceeds immutable authorization amount')
            state['captured'] += amount
            state['closed'] = row['authorization_complete']
            settlements[row['event_id']] = dict(authorization_id=aid, remaining=amount)
        elif kind == 'REVERSAL':
            original = settlements.get(row['reverses_settlement_id'])
            if original is None or original['authorization_id'] != aid or amount > original['remaining']:
                _fail('AUTHORIZATION_REVERSAL', 'Reversal requires earlier matching settlement and available booked amount')
            original['remaining'] -= amount; state['reversed'] += amount
        elif kind == 'CANCELLATION':
            state['closed'] = True; state['cancellation'] = True
        ledger.append(dict(record_id=row['authorization_id'] if kind == 'AUTHORIZATION' else row['event_id'],
                           authorization_id=aid, account_id=source['account_id'], date=date, sequence=seq,
                           kind=kind, cumulative_captured=money(state['captured']), cumulative_reversed=money(state['reversed']),
                           booked_exposure=money(state['captured']-state['reversed']), closed=state['closed']))
    output = []
    for aid, state in sorted(visible.items()):
        source = state['source']; expired = as_of >= source['expires_at']
        pending = 0 if expired or state['closed'] else units(source['amount'])-state['captured']
        booked = state['captured']-state['reversed']
        output.append(dict(authorization_id=aid, account_id=source['account_id'], authorization_amount=source['amount'],
                           captured_amount=money(state['captured']), reversed_amount=money(state['reversed']), booked_exposure=money(booked),
                           pending_authorization=money(pending), expired=expired, completed_or_cancelled=state['closed'],
                           scenario_exposure=money(booked+factor*pending)))
    totals = []
    for account in sorted(accounts):
        rows = [v for v in output if v['account_id'] == account]
        sums = {key: money(sum(units(v[key]) for v in rows))
                for key in ('booked_exposure', 'pending_authorization')}
        sums['scenario_exposure'] = money(units(sums['booked_exposure']) + factor * units(sums['pending_authorization']))
        totals.append(dict(account_id=account, **sums))
    value = dict(as_of=as_of, known_at=known_at, currency=currency, scope='SUPPLIED_AUTHORIZATION_LEDGER',
                 authorization_conversion_factor=factor, authorizations=output, accounts=totals, event_ledger=ledger,
                 excluded=sorted(excluded, key=lambda x: (x['record_id'], x['reason'])), policy=deepcopy(policy))
    value['artifact_hash'] = _fingerprint(value)
    return _result(value, 'ead.reconcile_card_authorizations', currency, ctx,
                   diagnostics=dict(input_hash=_fingerprint(dict(authorizations=auths, events=entries, policy=policy))),
                   limitations=('The supplied ledger is not total account exposure. Conversion of live reservations is an explicit scenario factor, not an estimated CCF. Reversal does not reopen a reservation. Additional captures require a new explicit authorization; no network or national legal convention is inferred.',))
