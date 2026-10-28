"""Shared-event asset pool cash ledgers and financed reinvestment."""
from copy import deepcopy
from datetime import timedelta
from .core import _fail, _result, _fingerprint, _context_value
from .data import _rows, _required, _date
from ._structured_pool import _number, _text, _sum, _bound, _checked


def _scenario_contract(contract, origin, context):
    c = deepcopy(contract)
    _required(c, ['contract_id', 'version', 'scenario_ids', 'weights', 'dates', 'currency', 'probability_measure', 'sampling_design', 'conditioning', 'default_timing', 'prepayment_timing', 'dilution_treatment', 'reinvestment_funding', 'initial_reinvestment_cash'])
    for key in ('contract_id', 'version', 'currency', 'conditioning'):
        _text(c[key], key)
    for key in ('scenario_ids', 'dates'):
        if not isinstance(c[key], list) or not c[key] or len(set(c[key])) != len(c[key]):
            _fail('STRUCTURED_SCENARIOS', 'Nonempty unique scenario/date axes required')
    for sid in c['scenario_ids']:
        _text(sid, 'scenario_id')
    previous = origin
    for date in c['dates']:
        if _date(date, 'date') <= _date(previous, 'previous'):
            _fail('STRUCTURED_SCENARIOS', 'Risk dates strictly increase after pool snapshot')
        previous = date
    if not isinstance(c['weights'], list) or len(c['weights']) != len(c['scenario_ids']):
        _fail('STRUCTURED_SCENARIOS', 'Aligned scenario weights required')
    weights = [_number(w, 'weight', 0) for w in c['weights']]
    if abs(_sum(weights) - 1) > 1e-12 or c['probability_measure'] != 'PHYSICAL' or c['sampling_design'] not in ('IID_UNWEIGHTED', 'FIXED_WEIGHTED_SCENARIOS', 'DEPENDENT_OR_QMC'):
        _fail('STRUCTURED_SCENARIOS', 'Normalized physical scenarios and explicit sampling design required')
    if c['sampling_design'] == 'IID_UNWEIGHTED' and any(abs(w - 1 / len(weights)) > 1e-14 for w in weights):
        _fail('STRUCTURED_SCENARIOS', 'IID declaration requires uniform weights')
    expected = dict(default_timing='BEFORE_DUE', prepayment_timing='AFTER_COUPON', dilution_treatment='PROPORTIONAL_REMAINING_SCHEDULE', reinvestment_funding='PRINCIPAL_CASH_ONLY')
    if any(c[k] != v for k, v in expected.items()):
        _fail('STRUCTURED_EVENT_CONTRACT', 'Explicit supported event and cash reinvestment ordering required')
    _number(c['initial_reinvestment_cash'], 'initial_reinvestment_cash', 0)
    ctx = _context_value(context)
    if ctx.get('currency') not in (None, c['currency']) or ctx.get('probability_measure') not in (None, 'PHYSICAL'):
        _fail('TARGET_MISMATCH', 'Pool scenario currency/measure conflicts with context')
    horizon_days = (_date(c['dates'][-1], 'date') - _date(origin, 'origin')).days
    expected_context = dict(conditioning=c['conditioning'], time_unit='days', horizon=horizon_days)
    if any(ctx.get(k) not in (None, v) for k, v in expected_context.items()):
        _fail('TARGET_MISMATCH', 'Pool path context must match conditioning and actual calendar-day horizon')
    return c, ctx


def simulate_asset_pool(asset_pool, scenario_contract, obligor_paths, asset_paths, shared_innovations, *, context=None):
    """Simulate owned contractual principal with shared absorbing obligor defaults."""
    pool = _checked(asset_pool, 'CreditAssetPool')
    contract, ctx = _scenario_contract(scenario_contract, pool['as_of'], context)
    if contract['currency'] != pool['profile']['currency']:
        _fail('TARGET_MISMATCH', 'Pool and scenario currency differ')
    sids, risk_dates = contract['scenario_ids'], contract['dates']
    horizon = risk_dates[-1]
    assignments = [x['assignment'] for x in pool['active_assignments'] + pool['reinvestment_candidates']]
    asset_ids = sorted({a['asset_id'] for a in assignments})
    tape = {aid: pool['asset_contracts'][aid] for aid in asset_ids}
    groups = sorted({a['default_group_id'] for a in tape.values()})
    for a in assignments:
        if a['effective_until'] is not None and a['effective_until'] <= horizon:
            _fail('STRUCTURED_TRANSFER', 'Intermediate legal transfer-out needs an explicit sale-price engine; assignment must cover simulation horizon')
        if a['effective_from'] > pool['as_of'] and a['effective_from'] not in risk_dates:
            _fail('STRUCTURED_CALENDAR', 'Reinvestment purchases must map to declared risk/payment dates')
    for a in tape.values():
        if any(p['date'] <= horizon and p['date'] not in risk_dates for p in a['schedule']):
            _fail('STRUCTURED_CALENDAR', 'Every contractual payment within horizon must be on the event grid')
    innovations, imap = _rows(shared_innovations, 'shared_innovations'), {}
    for row in innovations:
        _required(row, ['scenario_id', 'date', 'channel_id', 'uniform'])
        _text(row['channel_id'], 'channel_id')
        key = (row['scenario_id'], row['date'], row['channel_id'])
        u = _number(row['uniform'], 'uniform', 0, 1)
        if key in imap or row['scenario_id'] not in sids or row['date'] not in risk_dates or u == 1:
            _fail('STRUCTURED_INNOVATION', 'Unique scenario/risk-date uniforms in [0,1) required')
        imap[key] = u
    used, group_map, asset_map = set(), {}, {}
    for row in _rows(obligor_paths, 'obligor_paths'):
        _required(row, ['scenario_id', 'default_group_id', 'date', 'conditional_pd', 'default_channel'])
        key = (row['scenario_id'], row['default_group_id'], row['date'])
        if key in group_map or key[0] not in sids or key[1] not in groups or key[2] not in risk_dates:
            _fail('STRUCTURED_DEFAULT_GROUP', 'Unique aligned shared default-group paths required')
        _number(row['conditional_pd'], 'conditional_pd', 0, 1)
        _text(row['default_channel'], 'default_channel')
        innovation = (key[0], key[2], row['default_channel'])
        if innovation not in imap:
            _fail('STRUCTURED_INNOVATION', 'Missing shared obligor default innovation')
        used.add(innovation)
        group_map[key] = row
    if set(group_map) != {(s, g, d) for s in sids for g in groups for d in risk_dates}:
        _fail('STRUCTURED_DEFAULT_GROUP', 'Full group/scenario/date hazard grid required')
    event_dates = set(risk_dates)
    for row in _rows(asset_paths, 'asset_paths'):
        _required(row, ['scenario_id', 'asset_id', 'date', 'conditional_prepayment_probability', 'prepayment_channel', 'dilution_fraction', 'recovery_fraction', 'recovery_delay_days', 'recovery_cost', 'servicing_cost', 'interest_multiplier', 'overdue_days', 'disputed'])
        key = (row['scenario_id'], row['asset_id'], row['date'])
        if key in asset_map or key[0] not in sids or key[1] not in asset_ids or key[2] not in risk_dates:
            _fail('STRUCTURED_ASSET_PATH', 'Unique aligned asset paths required')
        for k in ('conditional_prepayment_probability', 'dilution_fraction', 'recovery_fraction'):
            _number(row[k], k, 0, 1)
        for k in ('recovery_cost', 'servicing_cost', 'interest_multiplier'):
            _number(row[k], k, 0)
        overdue = _number(row['overdue_days'], 'overdue_days', 0)
        if overdue != int(overdue) or type(row['disputed']) is not bool:
            _fail('STRUCTURED_ASSET_PATH', 'Dated acquisition eligibility needs integer aging and Boolean dispute state')
        delay = _number(row['recovery_delay_days'], 'recovery_delay_days', 0)
        if delay != int(delay):
            _fail('STRUCTURED_ASSET_PATH', 'Integer recovery delay required')
        try:
            recovery_date = (_date(row['date'], 'date') + timedelta(days=int(delay))).isoformat()
        except (OverflowError, ValueError):
            _fail('STRUCTURED_CALENDAR', 'Recovery date outside supported range')
        if recovery_date <= horizon:
            event_dates.add(recovery_date)
        _text(row['prepayment_channel'], 'prepayment_channel')
        innovation = (key[0], key[2], row['prepayment_channel'])
        if innovation not in imap:
            _fail('STRUCTURED_INNOVATION', 'Missing prepayment innovation')
        used.add(innovation)
        asset_map[key] = row
    if set(asset_map) != {(s, a, d) for s in sids for a in asset_ids for d in risk_dates} or used != set(imap):
        _fail('STRUCTURED_ASSET_PATH', 'Complete asset grids and no unused innovation channels required')
    reinvesting = pool['profile']['pool_mode'] == 'REVOLVING'
    until = pool['profile']['revolving_until']
    if not reinvesting and contract['initial_reinvestment_cash'] != 0:
        _fail('STRUCTURED_REINVESTMENT', 'Static pool cannot hold undeclared reinvestment cash')
    if reinvesting and until <= horizon:
        event_dates.add(until)
    calendar = sorted(event_dates)
    asset_results, group_events, pool_cashflows, acquisitions, endings = [], [], [], [], []
    for sid in sids:
        balances = {a: float(tape[a]['principal_balance']) for a in asset_ids}
        contractual = balances.copy()
        status = {a: 'PERFORMING' for a in asset_ids}
        defaulted = {g: False for g in groups}
        shares = {a: _sum(x['share'] for x in assignments if x['asset_id'] == a and x['effective_from'] <= pool['as_of']) for a in asset_ids}
        queue, acquired = [], {a['assignment_id'] for a in assignments if a['effective_from'] <= pool['as_of']}
        reinvestment_cash = float(contract['initial_reinvestment_cash'])
        cumulative_credit_loss = cumulative_dilution = 0.
        for date in calendar:
            opening_reinvestment = reinvestment_cash
            principal_terms, interest_terms, servicing_terms, recovery_terms, workout_terms, dilution_terms, credit_terms = [], [], [], [], [], [], []
            if date in risk_dates:
                for group in groups:
                    gp = group_map[(sid, group, date)]
                    u = imap[(sid, date, gp['default_channel'])]
                    event = not defaulted[group] and u < gp['conditional_pd']
                    defaulted[group] = defaulted[group] or event
                    group_events.append(dict(scenario_id=sid, default_group_id=group, date=date, new_default=event, in_default=defaulted[group]))
                for aid in asset_ids:
                    asset, path = tape[aid], asset_map[(sid, aid, date)]
                    opening, owned = balances[aid], shares[aid]
                    schedule = next((p for p in asset['schedule'] if p['date'] == date), None)
                    contractual_due = 0. if schedule is None else schedule['principal_due']
                    dilution = scheduled = prepayment = default_ead = interest = recovery_gross = recovery_cost = 0.
                    event = 'INACTIVE'
                    if status[aid] == 'PERFORMING' and opening > 0:
                        servicing_terms.append(owned * path['servicing_cost'])
                        dilution = opening * path['dilution_fraction']
                        balance = max(0., opening - dilution)
                        dilution_terms.append(owned * dilution)
                        factor = 0. if contractual[aid] == 0 else balance / contractual[aid]
                        if balance == 0:
                            event, status[aid] = 'DILUTED', 'DILUTED'
                        elif defaulted[asset['default_group_id']]:
                            event, status[aid], default_ead = 'DEFAULT', 'DEFAULTED', balance
                            recovery_gross = default_ead * path['recovery_fraction']
                            recovery_cost = path['recovery_cost']
                            recovery_date = (_date(date, 'date') + timedelta(days=int(path['recovery_delay_days']))).isoformat()
                            if owned > 0 and (recovery_gross > 0 or recovery_cost > 0):
                                queue.append(dict(asset_id=aid, date=recovery_date, gross=owned * recovery_gross, cost=owned * recovery_cost))
                            credit_terms.append(owned * (default_ead - recovery_gross + recovery_cost))
                            balance = 0.
                        else:
                            interest = 0. if schedule is None else schedule['interest_due'] * factor * path['interest_multiplier']
                            if imap[(sid, date, path['prepayment_channel'])] < path['conditional_prepayment_probability']:
                                event, status[aid], prepayment = 'PREPAYMENT', 'PREPAID', balance
                                balance = 0.
                            else:
                                scheduled = min(balance, contractual_due * factor)
                                balance = max(0., balance - scheduled)
                                event = 'MATURED' if balance == 0 else 'PERFORMING'
                                if balance == 0:
                                    status[aid] = 'MATURED'
                        balances[aid] = balance
                        principal_terms.append(owned * (scheduled + prepayment))
                        interest_terms.append(owned * interest)
                    contractual[aid] = max(0., contractual[aid] - contractual_due)
                    residual = _sum([opening, -dilution, -scheduled, -prepayment, -default_ead, -balances[aid]])
                    if abs(residual) > 1e-10 * max(1, opening):
                        _fail('STRUCTURED_CONSERVATION', 'Asset principal conservation failed')
                    asset_results.append(dict(scenario_id=sid, asset_id=aid, date=date, ownership_share_before_acquisition=owned, event=event, opening_principal=opening, dilution=dilution, scheduled_principal=scheduled, prepayment_principal=prepayment, default_ead=default_ead, closing_principal=balances[aid], interest=interest, modeled_recovery_gross=recovery_gross, modeled_recovery_cost=recovery_cost, principal_conservation_residual=residual))
            remaining_queue = []
            for flow in queue:
                if flow['date'] == date:
                    recovery_terms.append(flow['gross'])
                    workout_terms.append(flow['cost'])
                else:
                    remaining_queue.append(flow)
            queue = remaining_queue
            principal_cash = _sum(principal_terms)
            purchase_terms = []
            if reinvesting and date < until:
                reinvestment_cash = _sum([reinvestment_cash, principal_cash])
                principal_distributed = 0.
                purchases = sorted((a for a in assignments if a['effective_from'] == date and a['assignment_id'] not in acquired), key=lambda a: (a['purchase_priority'], a['assignment_id']))
                for a in purchases:
                    aid = a['asset_id']
                    reason = None
                    if status[aid] != 'PERFORMING' or balances[aid] <= 0:
                        reason = 'ASSET_NO_LONGER_PERFORMING'
                    elif asset_map[(sid, aid, date)]['overdue_days'] > pool['profile']['maximum_dpd'] or (asset_map[(sid, aid, date)]['disputed'] and pool['profile']['exclude_disputed']):
                        reason = 'ACQUISITION_ELIGIBILITY_FAILED'
                    elif a['purchase_price'] > reinvestment_cash:
                        reason = 'INSUFFICIENT_PRINCIPAL_CASH'
                    if reason is None:
                        reinvestment_cash = max(0., reinvestment_cash - a['purchase_price'])
                        shares[aid] += a['share']
                        if shares[aid] > 1 + 1e-12:
                            _fail('STRUCTURED_ASSIGNMENT', 'Acquired ownership exceeds whole asset')
                        acquired.add(a['assignment_id'])
                        purchase_terms.append(a['purchase_price'])
                    acquisitions.append(dict(scenario_id=sid, assignment_id=a['assignment_id'], asset_id=aid, date=date, acquired=reason is None, reason=reason, purchase_price=a['purchase_price'] if reason is None else 0., acquired_principal=a['share'] * balances[aid] if reason is None else 0.))
            else:
                principal_distributed = _sum([principal_cash, reinvestment_cash])
                reinvestment_cash = 0.
            purchase_cash = _sum(purchase_terms)
            cash_residual = _sum([opening_reinvestment, principal_cash, -purchase_cash, -principal_distributed, -reinvestment_cash])
            if abs(cash_residual) > 1e-10 * max(1, opening_reinvestment, principal_cash):
                _fail('STRUCTURED_CONSERVATION', 'Reinvestment cash conservation failed')
            dilution_amount, credit_loss = _sum(dilution_terms), _sum(credit_terms)
            cumulative_dilution = _sum([cumulative_dilution, dilution_amount])
            cumulative_credit_loss = _sum([cumulative_credit_loss, credit_loss])
            pool_cashflows.append(dict(scenario_id=sid, date=date, principal_collections=principal_cash, principal_available=principal_distributed, interest_available=_sum(interest_terms), recovery_gross=_sum(recovery_terms), workout_cost=_sum(workout_terms), servicing_cost=_sum(servicing_terms), opening_reinvestment_cash=opening_reinvestment, purchase_cash=purchase_cash, closing_reinvestment_cash=reinvestment_cash, closing_performing_principal=_sum(shares[a] * balances[a] for a in asset_ids), pending_recovery_gross=_sum(q['gross'] for q in queue), pending_workout_cost=_sum(q['cost'] for q in queue), dilution_loss=dilution_amount, modeled_credit_loss=credit_loss, cumulative_dilution_loss=cumulative_dilution, cumulative_modeled_credit_loss=cumulative_credit_loss, reinvestment_conservation_residual=cash_residual))
        endings.append(dict(scenario_id=sid, performing_principal=_sum(shares[a] * balances[a] for a in asset_ids), reinvestment_cash=reinvestment_cash, pending_recoveries=deepcopy(queue), terminal_complete=all(shares[a] * balances[a] == 0 for a in asset_ids) and not queue and reinvestment_cash == 0))
    value = _bound(dict(schema_version='1.0', kind='AssetPoolPaths', pool_id=pool['pool_id'], pool_hash=pool['sha256'], origin=pool['as_of'], scenario_contract=contract, scenario_ids=deepcopy(sids), scenario_weights=deepcopy(contract['weights']), dates=calendar, reference_principal=pool['reference_principal'], asset_events=asset_results, group_events=group_events, cashflows=pool_cashflows, acquisitions=acquisitions, ending_states=endings, innovation_hash=_fingerprint(innovations), path_input_hash=_fingerprint(dict(obligor_paths=obligor_paths, asset_paths=asset_paths))))
    return _result(value, 'structured_credit.simulate_asset_pool', 'pool_cashflows', ctx, limitations=('Initially performing assets with caller-supplied physical conditional hazards; all assets sharing a default group share its absorbing event. Prepayment is conditional on no obligor default and occurs after coupon. Dilution proportionally reduces remaining principal/coupon schedule and remains separate from credit default. Gross recoveries and workout costs are dated separately once; dates between risk endpoints add cash-only grid rows without new hazard draws. Reinvestment acquires only predeclared existing warehouse assets after date payments, using retained principal cash; failed purchase attempts are not silently retried. Servicing is charged on opening performing owned claims. Intermediate legal transfers need explicit sale-price mechanics and are rejected here. End-of-horizon unpaid assets/recoveries remain explicit, not assumed terminal losses or par recoveries.',))
