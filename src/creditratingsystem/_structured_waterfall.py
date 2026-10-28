"""Contractual tranche cash accounts, triggers and terminal principal allocation."""
from copy import deepcopy
from .core import _fail, _result, _fingerprint, _context_value
from .data import _rows, _required, _date
from ._structured_pool import _number, _text, _sum, _bound, _checked


def _structure(profile, pool, context):
    p = deepcopy(profile)
    _required(p, ['structure_id', 'version', 'currency', 'payment_dates', 'tranches', 'fees', 'initial_reserve', 'reserve_target', 'liquidity_limit', 'principal_mode', 'principal_for_interest', 'excess_interest_to_principal', 'trigger_seniority', 'oc_threshold', 'ic_threshold', 'terminal_policy'])
    for k in ('structure_id', 'version', 'currency'):
        _text(p[k], k)
    dates = p['payment_dates']
    if not isinstance(dates, list) or not dates or dates != sorted(set(dates)):
        _fail('STRUCTURED_WATERFALL', 'Unique ordered payment dates required')
    for d in dates:
        if _date(d, 'payment_date') <= _date(pool['origin'], 'origin') or d > pool['dates'][-1]:
            _fail('STRUCTURED_CALENDAR', 'Payment dates must be within pool horizon after origin')
    if dates[-1] != pool['dates'][-1]:
        _fail('STRUCTURED_CALENDAR', 'Final pool date must be a waterfall payment date')
    for k in ('initial_reserve', 'reserve_target', 'liquidity_limit', 'oc_threshold', 'ic_threshold'):
        _number(p[k], k, 0)
    if p['principal_mode'] not in ('SEQUENTIAL', 'PRO_RATA') or p['principal_for_interest'] not in ('ALWAYS', 'TRIGGER_ONLY', 'NEVER') or not isinstance(p['excess_interest_to_principal'], bool) or p['terminal_policy'] not in ('REQUIRE_COMPLETE', 'KEEP_OPEN'):
        _fail('STRUCTURED_WATERFALL', 'Explicit supported priority and terminal conventions required')
    tranches = _rows(p['tranches'], 'tranches')
    if not tranches or len({t.get('tranche_id') for t in tranches}) != len(tranches):
        _fail('STRUCTURED_TRANCHE', 'Unique nonempty tranche identities required')
    ranks = []
    for t in tranches:
        _required(t, ['tranche_id', 'seniority', 'notional', 'legal_maturity', 'coupon_fractions'])
        _text(t['tranche_id'], 'tranche_id')
        rank = _number(t['seniority'], 'seniority', 1)
        if rank != int(rank):
            _fail('STRUCTURED_TRANCHE', 'Integer seniority required')
        ranks.append(rank)
        if _number(t['notional'], 'notional', 0) == 0:
            _fail('STRUCTURED_TRANCHE', 'Positive tranche notional required')
        if t['legal_maturity'] not in dates:
            _fail('STRUCTURED_CALENDAR', 'Each legal maturity must be an explicit payment date')
        if not isinstance(t['coupon_fractions'], dict) or set(t['coupon_fractions']) != set(dates):
            _fail('STRUCTURED_TRANCHE', 'Explicit coupon fractions for every payment date required')
        for d, v in t['coupon_fractions'].items():
            _number(v, 'coupon_fraction', 0)
            if d > t['legal_maturity'] and v != 0:
                _fail('STRUCTURED_TRANCHE', 'Post-maturity interest requires separate explicit claim contract')
    if len(set(ranks)) != len(ranks) or p['trigger_seniority'] not in ranks:
        _fail('STRUCTURED_TRANCHE', 'Unique seniority and existing trigger seniority required')
    if not isinstance(p['fees'], dict) or set(p['fees']) != set(dates):
        _fail('STRUCTURED_WATERFALL', 'Explicit additional fee schedule required')
    for fee in p['fees'].values():
        _number(fee, 'fee', 0)
    p['tranches'] = sorted(tranches, key=lambda t: t['seniority'])
    ctx = _context_value(context)
    if p['currency'] != pool['scenario_contract']['currency'] or ctx.get('currency') not in (None, p['currency']) or ctx.get('probability_measure') not in (None, 'PHYSICAL') or ctx.get('conditioning') not in (None, pool['scenario_contract']['conditioning']):
        _fail('TARGET_MISMATCH', 'Structure currency and physical scenario context must agree')
    horizon_days = (_date(dates[-1], 'date') - _date(pool['origin'], 'origin')).days
    if ctx.get('time_unit') not in (None, 'days') or ctx.get('horizon') not in (None, horizon_days):
        _fail('TARGET_MISMATCH', 'Waterfall context must use actual calendar-day horizon')
    return p, ctx


def run_tranche_waterfall(asset_pool_paths, structure_profile, *, context=None):
    """Allocate actual dated cash under explicit seniority and support accounts."""
    pool = _checked(asset_pool_paths, 'AssetPoolPaths')
    p, ctx = _structure(structure_profile, pool, context)
    endings = {e['scenario_id']: e for e in pool['ending_states']}
    if p['terminal_policy'] == 'REQUIRE_COMPLETE' and any(not endings[s]['terminal_complete'] for s in pool['scenario_ids']):
        _fail('STRUCTURED_UNRESOLVED', 'Ultimate write-down requires complete pool run-off in every scenario')
    by = {(c['scenario_id'], c['date']): c for c in pool['cashflows']}
    calendar = sorted(set(pool['dates']) | set(p['payment_dates']))
    tranche_rows, accounts, final = [], [], []
    for sid in pool['scenario_ids']:
        balances = {t['tranche_id']: float(t['notional']) for t in p['tranches']}
        arrears = {t['tranche_id']: 0. for t in p['tranches']}
        cumulative_principal = dict(arrears)
        timely_failure, maturity_failure = dict.fromkeys(balances, False), dict.fromkeys(balances, False)
        reserve, liquidity, interest_cash, principal_cash, fee_arrears = p['initial_reserve'], 0., 0., 0., 0.
        pool_value = pool['reference_principal']
        for date in calendar:
            opening_cash = _sum([interest_cash, principal_cash, reserve])
            c = by.get((sid, date))
            incoming_interest = c['interest_available'] if c else 0.
            incoming_principal = _sum([c['principal_available'], c['recovery_gross']]) if c else 0.
            cash_cost = _sum([c['workout_cost'], c['servicing_cost']]) if c else 0.
            interest_cash = _sum([interest_cash, incoming_interest])
            principal_cash = _sum([principal_cash, incoming_principal])
            fee_arrears = _sum([fee_arrears, cash_cost])
            if c:
                pool_value = max(0., _sum([c['closing_performing_principal'], c['pending_recovery_gross'], -c['pending_workout_cost'], c['closing_reinvestment_cash']]))
            draws, repayment, fees_paid, reserve_topup, reserve_release = 0., 0., 0., 0., 0.
            interest_paid_total, principal_paid_total, equity = 0., 0., 0.
            oc_ratio, ic_ratio, trigger = None, None, False
            terminal = date == calendar[-1] and p['terminal_policy'] == 'REQUIRE_COMPLETE'
            if date in p['payment_dates']:
                fee_arrears = _sum([fee_arrears, p['fees'][date]])
                coupons = {t['tranche_id']: _sum([balances[t['tranche_id']] * t['coupon_fractions'][date]]) for t in p['tranches']}
                protected = [t['tranche_id'] for t in p['tranches'] if t['seniority'] <= p['trigger_seniority']]
                protected_balance = _sum(balances[t] for t in protected)
                protected_due = _sum(_sum([coupons[t], arrears[t]]) for t in protected)
                oc_numerator = _sum([pool_value, interest_cash, principal_cash, reserve])
                oc_ratio = oc_numerator / protected_balance if protected_balance else None
                ic_ratio = max(0., interest_cash - fee_arrears) / protected_due if protected_due else None
                for ratio in (oc_ratio, ic_ratio):
                    if ratio is not None:
                        _sum([ratio])
                trigger = (oc_ratio is not None and oc_ratio < p['oc_threshold']) or (ic_ratio is not None and ic_ratio < p['ic_threshold'])
                allow_principal = p['principal_for_interest'] == 'ALWAYS' or (p['principal_for_interest'] == 'TRIGGER_ONLY' and trigger) or terminal
                if terminal:
                    reserve_release = reserve
                    principal_cash = _sum([principal_cash, reserve]); reserve = 0.

                def pay_claim(due, principal_allowed):
                    nonlocal interest_cash, principal_cash, reserve, liquidity, draws
                    paid = min(due, interest_cash); interest_cash -= paid; left = due - paid
                    # Pool workout/servicing fees and configured senior deal fees can
                    # consume principal. Tranche interest follows its separate rule.
                    if principal_allowed:
                        take = min(left, principal_cash); principal_cash -= take; paid += take; left -= take
                    take = min(left, reserve); reserve -= take; paid += take; left -= take
                    take = min(left, max(0., p['liquidity_limit'] - liquidity)) if not terminal else 0.
                    liquidity = _sum([liquidity, take]); draws = _sum([draws, take]); paid = _sum([paid, take])
                    return paid

                fees_paid = pay_claim(fee_arrears, True)
                fee_arrears = max(0., fee_arrears - fees_paid)
                paid_interest = {}
                for t in p['tranches']:
                    tid = t['tranche_id']; due = _sum([arrears[tid], coupons[tid]])
                    paid = pay_claim(due, allow_principal) if fee_arrears == 0 else 0.
                    paid_interest[tid] = paid; arrears[tid] = max(0., due - paid)
                    timely_failure[tid] |= arrears[tid] > 1e-12 * max(1., t['notional'])
                interest_paid_total = _sum(paid_interest.values())
                # Repay the support provider before reserve refill and debt principal.
                repayment = min(liquidity, _sum([interest_cash, principal_cash]))
                take = min(repayment, interest_cash); interest_cash -= take
                principal_cash -= repayment - take; liquidity -= repayment
                if not terminal:
                    reserve_topup = min(max(0., p['reserve_target'] - reserve), interest_cash)
                    reserve = _sum([reserve, reserve_topup]); interest_cash -= reserve_topup
                if p['excess_interest_to_principal'] or terminal:
                    principal_cash = _sum([principal_cash, interest_cash]); interest_cash = 0.
                else:
                    equity = interest_cash; interest_cash = 0.
                distributable = min(principal_cash, _sum(balances.values())) if fee_arrears == 0 and not any(arrears.values()) else 0.
                allocations = dict.fromkeys(balances, 0.)
                if p['principal_mode'] == 'SEQUENTIAL' or trigger:
                    remaining = distributable
                    for t in p['tranches']:
                        tid = t['tranche_id']; allocations[tid] = min(balances[tid], remaining); remaining -= allocations[tid]
                elif distributable:
                    denominator = _sum(balances.values())
                    for t in p['tranches']:
                        tid = t['tranche_id']; allocations[tid] = distributable * (balances[tid] / denominator)
                principal_paid_total = _sum(allocations.values()); principal_cash = max(0., principal_cash - principal_paid_total)
                for t in p['tranches']:
                    tid = t['tranche_id']; opening_balance = balances[tid]
                    balances[tid] = max(0., opening_balance - allocations[tid])
                    cumulative_principal[tid] = _sum([cumulative_principal[tid], allocations[tid]])
                    maturity_failure[tid] |= date == t['legal_maturity'] and balances[tid] > 1e-12 * max(1., t['notional'])
                    tranche_rows.append(dict(scenario_id=sid, date=date, tranche_id=tid, opening_principal=opening_balance, coupon_due=coupons[tid], interest_paid=paid_interest[tid], closing_interest_arrears=arrears[tid], principal_paid=allocations[tid], closing_principal=balances[tid], principal_write_down=balances[tid] if terminal else 0., timely_interest_failure=timely_failure[tid], legal_maturity_failure=maturity_failure[tid]))
                if all(b == 0 for b in balances.values()) and fee_arrears == 0 and not any(arrears.values()) and liquidity == 0:
                    equity = _sum([equity, principal_cash]); principal_cash = 0.
            closing_cash = _sum([interest_cash, principal_cash, reserve])
            uses = _sum([fees_paid, interest_paid_total, principal_paid_total, repayment, equity, closing_cash])
            residual = _sum([opening_cash, incoming_interest, incoming_principal, draws, -uses])
            if abs(residual) > 1e-10 * max(1., opening_cash, incoming_interest, incoming_principal):
                _fail('STRUCTURED_CONSERVATION', 'Waterfall cash conservation failed')
            accounts.append(dict(scenario_id=sid,date=date,opening_cash=opening_cash,incoming_interest=incoming_interest,incoming_principal=incoming_principal,fees_paid=fees_paid,fee_arrears=fee_arrears,interest_paid=interest_paid_total,principal_paid=principal_paid_total,liquidity_draw=draws,liquidity_repayment=repayment,liquidity_outstanding=liquidity,reserve_release=reserve_release,reserve_topup=reserve_topup,closing_reserve=reserve,closing_interest_cash=interest_cash,closing_principal_cash=principal_cash,equity_distribution=equity,oc_ratio=oc_ratio,ic_ratio=ic_ratio,trigger_breached=trigger,cash_conservation_residual=residual))
        for t in p['tranches']:
            tid = t['tranche_id']
            final.append(dict(scenario_id=sid,tranche_id=tid,notional=t['notional'],principal_paid=cumulative_principal[tid],remaining_principal=balances[tid],ultimate_principal_loss=balances[tid] if p['terminal_policy']=='REQUIRE_COMPLETE' else None,interest_arrears=arrears[tid],timely_interest_failure=timely_failure[tid],legal_maturity_failure=maturity_failure[tid],ultimate_identified=p['terminal_policy']=='REQUIRE_COMPLETE'))
    value = _bound(dict(schema_version='1.0',kind='TrancheWaterfallPaths',origin=pool['origin'],currency=p['currency'],pool_paths_hash=pool['sha256'],innovation_hash=pool['innovation_hash'],scenario_ids=deepcopy(pool['scenario_ids']),scenario_weights=deepcopy(pool['scenario_weights']),scenario_contract=deepcopy(pool['scenario_contract']),structure_profile=p,structure_hash=_fingerprint(p),dates=calendar,tranche_cashflows=tranche_rows,cash_accounts=accounts,ending_states=final))
    return _result(value,'structured_credit.run_tranche_waterfall','tranche_cashflows',ctx,limitations=('Configured senior additional deal fees must exclude costs already supplied by the pool. Opening reserve is external funded support; liquidity is a repayable senior claim, not credit recovery. OC uses performing principal plus pending net modeled recovery and actual cash; it is a declared contractual valuation proxy. Coupon fractions apply to opening outstanding principal; no interest on arrears or post-maturity penalty claim is invented. Triggers switch principal allocation to sequential and may permit principal-to-interest diversion. Ultimate principal write-down is identified only after explicitly complete pool run-off; legal-maturity and timely-interest failures remain separate even if subsequently cured.',))
