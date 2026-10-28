"""Dated structured-credit pools, invoice borrowing bases and static tranches."""
from copy import deepcopy
from datetime import date
import math
from .core import CalcResult, _fail, _result, _fingerprint, _context_value, _numeric
from .data import _rows, _required, _date

_CLASSES = {
    'ABS': ('CONSUMER_LOAN', 'AUTO_LOAN', 'LEASE'),
    'RMBS': ('RESIDENTIAL_MORTGAGE',),
    'CMBS': ('COMMERCIAL_MORTGAGE',),
    'CLO': ('CORPORATE_LOAN',),
    'MORTGAGE_COVER': ('RESIDENTIAL_MORTGAGE', 'COMMERCIAL_MORTGAGE'),
    'PUBLIC_SECTOR_COVER': ('PUBLIC_SECTOR_CLAIM',),
    'TRADE_RECEIVABLES': ('INVOICE',),
}


def _number(x, name, low=None, high=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or low is not None and x < low or high is not None and x > high:
        _fail('STRUCTURED_INPUT', f'{name} outside finite numeric domain')
    return float(x)


def _text(x, name):
    if not isinstance(x, str) or not x.strip():
        _fail('STRUCTURED_INPUT', f'{name} must be nonempty text')
    return x


def _sum(values):
    try:
        value = math.fsum(values)
    except (ValueError, OverflowError):
        _fail('NUMERICAL_FAILURE', 'Structured cash totals exceed finite representation')
    if not math.isfinite(value):
        _fail('NUMERICAL_FAILURE', 'Structured cash totals exceed finite representation')
    return value


def _bound(value):
    value = deepcopy(value)
    value['sha256'] = _fingerprint(value)
    return value


def _checked(value, kind):
    value = deepcopy(value.value if isinstance(value, CalcResult) else value)
    if not isinstance(value, dict):
        _fail('STRUCTURED_ARTIFACT', 'Bound structured-credit artifact required')
    digest = value.pop('sha256', None)
    if value.get('schema_version') != '1.0' or value.get('kind') != kind or digest != _fingerprint(value):
        _fail('STRUCTURED_ARTIFACT', 'Artifact kind, schema or content binding mismatch')
    value['sha256'] = digest
    return value


def _nullable(row, keys):
    if any(k not in row for k in keys):
        _fail('DATA_SCHEMA', 'Explicit nullable fields required: ' + ','.join(keys))


def _profile(profile, as_of, information_cutoff, context):
    _required(profile, ['profile_id', 'version', 'known_at', 'currency'])
    for k in ('profile_id', 'version', 'currency'):
        _text(profile[k], k)
    if not _date(profile['known_at'], 'known_at') <= _date(information_cutoff, 'information_cutoff') <= _date(as_of, 'as_of'):
        _fail('STRUCTURED_DATE', 'Known profile and knowledge cutoff no later than pool snapshot required')
    ctx = _context_value(context)
    if ctx.get('currency') not in (None, profile['currency']) or ctx.get('probability_measure') not in (None, 'PHYSICAL'):
        _fail('TARGET_MISMATCH', 'Structured currency and physical measure must match context')
    return ctx


def _eligibility(asset, profile, as_of):
    reasons = []
    if asset['asset_class'] not in profile['allowed_asset_classes']:
        reasons.append('ASSET_CLASS')
    if asset['currency'] != profile['currency']:
        reasons.append('CURRENCY')
    if asset['balance_date'] != as_of:
        reasons.append('STALE_BALANCE')
    if asset['overdue_days'] > profile['maximum_dpd']:
        reasons.append('AGING')
    if asset['disputed'] and profile['exclude_disputed']:
        reasons.append('DISPUTED')
    if asset['recourse'] not in profile['eligible_recourse']:
        reasons.append('RECOURSE')
    if asset['principal_balance'] <= 0:
        reasons.append('NO_OUTSTANDING_PRINCIPAL')
    return reasons


def build_credit_asset_pool(asset_contracts, pool_assignments, eligibility_profile, *, pool_id, as_of, information_cutoff, context=None):
    """Freeze a dated pool and eligible pre-existing reinvestment inventory."""
    profile = deepcopy(eligibility_profile)
    ctx = _profile(profile, as_of, information_cutoff, context)
    _text(pool_id, 'pool_id')
    _required(profile, ['product_kind', 'pool_mode', 'allowed_asset_classes', 'maximum_dpd', 'exclude_disputed', 'eligible_recourse'])
    _nullable(profile, ['revolving_until'])
    if profile['product_kind'] not in _CLASSES or profile['pool_mode'] not in ('STATIC', 'REVOLVING'):
        _fail('STRUCTURED_ROUTE', 'Explicit supported product and static/revolving pool route required')
    classes = profile['allowed_asset_classes']
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes) or any(c not in _CLASSES[profile['product_kind']] for c in classes):
        _fail('STRUCTURED_ROUTE', 'Product-specific compatible asset classes required')
    if not isinstance(profile['eligible_recourse'], list) or any(x not in ('NONE', 'SELLER') for x in profile['eligible_recourse']) or type(profile['exclude_disputed']) is not bool:
        _fail('STRUCTURED_ELIGIBILITY', 'Explicit recourse list and dispute policy required')
    dpd = _number(profile['maximum_dpd'], 'maximum_dpd', 0)
    if dpd != int(dpd):
        _fail('STRUCTURED_ELIGIBILITY', 'Integer aging threshold required')
    if profile['pool_mode'] == 'STATIC' and profile['revolving_until'] is not None:
        _fail('STRUCTURED_ROUTE', 'Static pools cannot declare reinvestment')
    if profile['pool_mode'] == 'REVOLVING' and (profile['revolving_until'] is None or _date(profile['revolving_until'], 'revolving_until') <= _date(as_of, 'as_of')):
        _fail('STRUCTURED_ROUTE', 'Future exclusive reinvestment end required')
    assets, tape, asset_ids, invoices, default_groups, collateral = _rows(asset_contracts, 'asset_contracts'), {}, set(), set(), {}, {}
    excluded = []
    for a in sorted(assets, key=lambda a: a.get('asset_id', '')):
        _required(a, ['asset_id', 'asset_class', 'obligor_id', 'default_group_id', 'seller_id', 'servicer_id', 'currency', 'observed_at', 'known_at', 'origination_date', 'balance_date', 'principal_balance', 'schedule', 'overdue_days', 'disputed', 'dilution_reserve', 'recourse', 'collateral_allocations'])
        _nullable(a, ['invoice_id'])
        for k in ('asset_id', 'asset_class', 'obligor_id', 'default_group_id', 'seller_id', 'servicer_id', 'currency'):
            _text(a[k], k)
        if a['asset_id'] in asset_ids:
            _fail('STRUCTURED_DUPLICATE', 'Each asset/invoice appears once in the local snapshot tape')
        asset_ids.add(a['asset_id'])
        if not _date(a['origination_date'], 'origination_date') <= _date(a['observed_at'], 'observed_at') <= _date(a['known_at'], 'known_at'):
            _fail('STRUCTURED_DATE', 'Origination, observation and knowledge dates are ordered')
        if _date(a['balance_date'], 'balance_date') > _date(a['observed_at'], 'observed_at'):
            _fail('STRUCTURED_DATE', 'Observed balance cannot be from a future date')
        if a['known_at'] > information_cutoff or a['observed_at'] > as_of:
            excluded.append(dict(asset_id=a['asset_id'], reasons=['NOT_YET_KNOWN_OR_OBSERVED']))
            continue
        if a['asset_class'] not in {c for cs in _CLASSES.values() for c in cs} or a['recourse'] not in ('NONE', 'SELLER'):
            _fail('STRUCTURED_ROUTE', 'Unsupported asset/recourse class')
        if a['asset_class'] == 'INVOICE':
            _text(a['invoice_id'], 'invoice_id')
            if a['invoice_id'] in invoices:
                _fail('STRUCTURED_DUPLICATE', 'Duplicate invoice cannot be financed twice')
            invoices.add(a['invoice_id'])
        elif a['invoice_id'] is not None:
            _fail('STRUCTURED_ROUTE', 'Noninvoice asset cannot carry invoice identity')
        if a['obligor_id'] in default_groups and default_groups[a['obligor_id']] != a['default_group_id']:
            _fail('STRUCTURED_DEFAULT_GROUP', 'One obligor cannot silently use independent default groups')
        default_groups[a['obligor_id']] = a['default_group_id']
        balance = _number(a['principal_balance'], 'principal_balance', 0)
        _number(a['dilution_reserve'], 'dilution_reserve', 0, balance)
        overdue = _number(a['overdue_days'], 'overdue_days', 0)
        if overdue != int(overdue) or type(a['disputed']) is not bool:
            _fail('STRUCTURED_INPUT', 'Integer overdue days and Boolean dispute required')
        schedule, principal, last = _rows(a['schedule'], 'schedule'), [], a['balance_date']
        for s in schedule:
            _required(s, ['date', 'principal_due', 'interest_due'])
            if _date(s['date'], 'date') <= _date(last, 'previous_date'):
                _fail('STRUCTURED_SCHEDULE', 'Chronological unique payments after balance date required')
            principal.append(_number(s['principal_due'], 'principal_due', 0))
            _number(s['interest_due'], 'interest_due', 0)
            last = s['date']
        if abs(_sum(principal) - balance) > 1e-12 * max(1, balance):
            _fail('STRUCTURED_SCHEDULE', 'Complete principal schedule must reconcile to observed balance')
        aids = set()
        for c in _rows(a['collateral_allocations'], 'collateral_allocations'):
            _required(c, ['collateral_id', 'allocation_fraction'])
            _text(c['collateral_id'], 'collateral_id')
            fraction = _number(c['allocation_fraction'], 'allocation_fraction', 0, 1)
            if c['collateral_id'] in aids:
                _fail('STRUCTURED_DUPLICATE', 'Unique collateral allocation per asset required')
            aids.add(c['collateral_id'])
            collateral.setdefault(c['collateral_id'], []).append(fraction)
        tape[a['asset_id']] = a
    if any(_sum(v) > 1 + 1e-12 for v in collateral.values()):
        _fail('STRUCTURED_COLLATERAL', 'Collateral allocation shares across assets exceed one')
    assignments, ids, intervals = _rows(pool_assignments, 'pool_assignments'), set(), {}
    visible, ignored = [], []
    for a in sorted(assignments, key=lambda a: a.get('assignment_id', '')):
        _required(a, ['assignment_id', 'pool_id', 'asset_id', 'known_at', 'effective_from', 'share', 'purchase_price', 'purchase_priority'])
        _nullable(a, ['effective_until'])
        for k in ('assignment_id', 'pool_id', 'asset_id'):
            _text(a[k], k)
        if a['assignment_id'] in ids:
            _fail('STRUCTURED_DUPLICATE', 'Unique pool assignment IDs required')
        ids.add(a['assignment_id'])
        start = _date(a['effective_from'], 'effective_from')
        end = date.max if a['effective_until'] is None else _date(a['effective_until'], 'effective_until')
        _date(a['known_at'], 'known_at')
        if end <= start:
            _fail('STRUCTURED_DATE', 'Assignment validity is nonempty and end-exclusive')
        share = _number(a['share'], 'share', 0, 1)
        if share == 0:
            _fail('STRUCTURED_ASSIGNMENT', 'Positive transfer share required')
        _number(a['purchase_price'], 'purchase_price', 0)
        priority = _number(a['purchase_priority'], 'purchase_priority', 0)
        if priority != int(priority):
            _fail('STRUCTURED_ASSIGNMENT', 'Integer acquisition priority required')
        if a['known_at'] > information_cutoff:
            ignored.append(dict(assignment_id=a['assignment_id'], reason='NOT_YET_KNOWN'))
            continue
        intervals.setdefault(a['asset_id'], []).append((start, end, share))
        visible.append(a)
    for aid, spans in intervals.items():
        for boundary in sorted({s for s, _, _ in spans}):
            if _sum(w for s, e, w in spans if s <= boundary < e) > 1 + 1e-12:
                _fail('STRUCTURED_ASSIGNMENT', 'Overlapping transfers across pools exceed asset ownership', {'asset_id': aid})
    active, pipeline, assignment_exclusions = [], [], []
    for a in visible:
        if a['pool_id'] != pool_id:
            continue
        reasons = []
        if a['effective_until'] is not None and a['effective_until'] <= as_of:
            reasons.append('EXPIRED_ASSIGNMENT')
        asset = tape.get(a['asset_id'])
        if asset is None:
            reasons.append('UNAVAILABLE_ASSET')
        else:
            reasons.extend(_eligibility(asset, profile, as_of))
        future = a['effective_from'] > as_of
        if future and profile['pool_mode'] != 'REVOLVING':
            reasons.append('STATIC_POOL_ADDITION')
        elif future and a['effective_from'] >= profile['revolving_until']:
            reasons.append('OUTSIDE_REINVESTMENT_WINDOW')
        if reasons:
            assignment_exclusions.append(dict(assignment_id=a['assignment_id'], asset_id=a['asset_id'], reasons=reasons))
        else:
            record = dict(assignment=deepcopy(a), allocated_reference_principal=a['share'] * asset['principal_balance'])
            (pipeline if future else active).append(record)
    value = _bound(dict(schema_version='1.0', kind='CreditAssetPool', pool_id=pool_id, as_of=as_of, information_cutoff=information_cutoff, profile=profile, asset_contracts=tape, active_assignments=active, reinvestment_candidates=pipeline, reference_principal=_sum(x['allocated_reference_principal'] for x in active), excluded_assets=excluded, excluded_assignments=assignment_exclusions, ignored_assignments=ignored, input_hash=_fingerprint(dict(asset_contracts=assets, pool_assignments=assignments))))
    return _result(value, 'structured_credit.build_credit_asset_pool', 'asset_pool', ctx, limitations=('Latest local snapshot tape supplied by caller; duplicate assets/invoices rejected rather than silently selecting a revision. Only exactly dated observed balances enter the current pool. Existing dated warehouse inventory may be reserved for future revolving purchases; it is not current collateral. Fractional transfers and collateral allocation shares are supplied legal assertions, not an enforceability opinion. Product-specific asset routing does not certify securitization or covered-bond eligibility under law.',))


def tranche_loss_from_pool(pool_loss, attachment, detachment, *, pool_notional, loss_unit='fraction', context=None):
    """Static bounded tranche loss, with distinct pool and tranche denominators."""
    a, d = _number(attachment, 'attachment', 0, 1), _number(detachment, 'detachment', 0, 1)
    notional = _number(pool_notional, 'pool_notional', 0)
    if a >= d or notional == 0 or loss_unit not in ('fraction', 'amount'):
        _fail('STRUCTURED_TRANCHE', 'Positive notional and 0<=attachment<detachment<=1 with explicit loss unit required')
    if isinstance(pool_loss, (int, float)) and not isinstance(pool_loss, bool):
        pool_loss = [pool_loss]
    x = _numeric(pool_loss, 'pool_loss', vector_only=True)
    if (x < 0).any():
        _fail('STRUCTURED_TRANCHE', 'Nonnegative pool loss required')
    losses = x.tolist()
    fractions = [_sum([v / notional]) if loss_unit == 'amount' else v for v in losses]
    width = d - a
    tranche_notional = _sum([notional * width])
    if tranche_notional == 0:
        _fail('NUMERICAL_FAILURE', 'Tranche notional underflows finite representation')
    tranche_fractions = [min(max(v - a, 0.), width) / width for v in fractions]
    amounts = [_sum([tranche_notional * v]) for v in tranche_fractions]
    value = dict(attachment=a, detachment=d, pool_notional=notional, tranche_notional=tranche_notional, pool_loss_fraction=fractions, tranche_loss_fraction=tranche_fractions, tranche_loss_amount=amounts, boundary_status=['BELOW_OR_AT_ATTACHMENT' if v <= a else 'AT_OR_ABOVE_DETACHMENT' if v >= d else 'PARTIAL' for v in fractions])
    return _result(value, 'structured_credit.tranche_loss_from_pool', 'tranche_loss', context, limitations=('Static sequential principal-loss allocation only, not a dated cashflow waterfall or tranche default probability. Attachment/detachment are fractions of reference pool notional; tranche loss fraction uses tranche notional. Excess pool losses above notional can occur from costs but tranche loss is capped at its contractual notional.',))


def receivables_borrowing_base(invoices, borrowing_profile, *, as_of, information_cutoff, context=None):
    """Apply explicit invoice eligibility, dilution, group caps and advance rates."""
    profile = deepcopy(borrowing_profile)
    ctx = _profile(profile, as_of, information_cutoff, context)
    _required(profile, ['maximum_overdue_days', 'exclude_disputed', 'allowed_recourse', 'advance_rates', 'default_group_cap', 'group_caps', 'additional_reserve', 'reserve_scope', 'coverage_status'])
    maximum = _number(profile['maximum_overdue_days'], 'maximum_overdue_days', 0)
    if maximum != int(maximum) or type(profile['exclude_disputed']) is not bool or profile['reserve_scope'] != 'ADDITIONAL_NOT_IN_INVOICE_DILUTION' or profile['coverage_status'] not in ('COMPLETE', 'PARTIAL', 'UNKNOWN'):
        _fail('STRUCTURED_BORROWING_BASE', 'Explicit eligibility, nonoverlapping reserve and coverage conventions required')
    if not isinstance(profile['allowed_recourse'], list) or any(x not in ('NONE', 'SELLER') for x in profile['allowed_recourse']):
        _fail('STRUCTURED_BORROWING_BASE', 'Explicit supported recourse eligibility required')
    for k in ('NONE', 'SELLER'):
        _number(profile['advance_rates'].get(k), 'advance_rate', 0, 1)
    default_cap = _number(profile['default_group_cap'], 'default_group_cap', 0, 1)
    if not isinstance(profile['group_caps'], dict):
        _fail('STRUCTURED_BORROWING_BASE', 'Explicit group-cap mapping required')
    for k, v in profile['group_caps'].items():
        _text(k, 'debtor_group_id')
        _number(v, 'group_cap', 0, 1)
    reserve = _number(profile['additional_reserve'], 'additional_reserve', 0)
    ids, rows, groups, debtor_groups = set(), [], {}, {}
    for row in sorted(_rows(invoices, 'invoices'), key=lambda r: r.get('invoice_id', '')):
        _required(row, ['invoice_id', 'debtor_id', 'debtor_group_id', 'seller_id', 'servicer_id', 'currency', 'issue_date', 'due_date', 'balance_date', 'known_at', 'outstanding', 'disputed', 'dilution_reserve', 'recourse'])
        for k in ('invoice_id', 'debtor_id', 'debtor_group_id', 'seller_id', 'servicer_id'):
            _text(row[k], k)
        if row['invoice_id'] in ids:
            _fail('STRUCTURED_DUPLICATE', 'Invoice ID may enter borrowing base once')
        ids.add(row['invoice_id'])
        if row['debtor_id'] in debtor_groups and debtor_groups[row['debtor_id']] != row['debtor_group_id']:
            _fail('STRUCTURED_DEFAULT_GROUP', 'One debtor cannot split across concentration groups')
        debtor_groups[row['debtor_id']] = row['debtor_group_id']
        issue, due, balance_date, known = [_date(row[k], k) for k in ('issue_date', 'due_date', 'balance_date', 'known_at')]
        if issue > due or issue > balance_date or balance_date > known:
            _fail('STRUCTURED_DATE', 'Invoice issue, due and observed-balance chronology invalid')
        amount = _number(row['outstanding'], 'outstanding', 0)
        dilution = _number(row['dilution_reserve'], 'dilution_reserve', 0, amount)
        if type(row['disputed']) is not bool or row['recourse'] not in ('NONE', 'SELLER'):
            _fail('STRUCTURED_BORROWING_BASE', 'Explicit dispute and recourse states required')
        aging = max(0, (_date(as_of, 'as_of') - due).days)
        reasons = []
        if row['known_at'] > information_cutoff:
            reasons.append('NOT_YET_KNOWN')
        if row['balance_date'] != as_of:
            reasons.append('STALE_OR_FUTURE_BALANCE')
        if row['currency'] != profile['currency']:
            reasons.append('CURRENCY')
        if aging > maximum:
            reasons.append('AGING')
        if row['disputed'] and profile['exclude_disputed']:
            reasons.append('DISPUTED')
        if row['recourse'] not in profile['allowed_recourse']:
            reasons.append('RECOURSE')
        eligible = 0. if reasons else amount - dilution
        groups.setdefault(row['debtor_group_id'], []).append(eligible)
        rows.append(dict(invoice_id=row['invoice_id'], debtor_id=row['debtor_id'], debtor_group_id=row['debtor_group_id'], overdue_days=aging, outstanding=amount, dilution_reserve=dilution, eligible_before_concentration=eligible, advance_rate=profile['advance_rates'][row['recourse']], exclusion_reasons=reasons))
    totals = {k: _sum(v) for k, v in groups.items()}
    denominator = _sum(totals.values())
    caps = {k: min(v, denominator * profile['group_caps'].get(k, default_cap)) for k, v in totals.items()}
    for row in rows:
        key = row['debtor_group_id']
        factor = 0. if totals[key] == 0 else caps[key] / totals[key]
        retained = row['eligible_before_concentration'] * factor
        row.update(eligible_after_concentration=retained, concentration_deduction=row['eligible_before_concentration'] - retained, concentration_binding=caps[key] < totals[key], advance_value=retained * row['advance_rate'])
    gross = _sum(r['advance_value'] for r in rows)
    known_base = max(0., gross - reserve)
    value = _bound(dict(schema_version='1.0', kind='ReceivablesBorrowingBase', as_of=as_of, information_cutoff=information_cutoff, profile=profile, invoice_results=rows, concentration_denominator=denominator, group_totals=totals, group_retained=caps, gross_advance_value=gross, additional_reserve=reserve, reserve_applied=min(gross, reserve), known_borrowing_base=known_base, borrowing_base=known_base if profile['coverage_status'] == 'COMPLETE' else None, status='COMPLETE' if profile['coverage_status'] == 'COMPLETE' else 'INCOMPLETE_COVERAGE'))
    return _result(value, 'structured_credit.receivables_borrowing_base', 'borrowing_base', ctx, limitations=('Caps use fixed total eligible face less invoice dilution reserves before concentration caps, not an implicit endogenous retained-pool denominator. Same-group invoices share one cap and proportional retention. Advance rates follow explicit recourse class; seller support is not assumed risk-free. Additional reserve is declared nonoverlapping with invoice dilution. Incomplete coverage returns a known subtotal and null total borrowing base; no actual credit limit is changed.',))
