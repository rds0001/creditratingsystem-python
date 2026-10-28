"""Capped economic dual recourse under caller-supplied legal conventions."""
from copy import deepcopy
from .core import _fail, _result, _fingerprint, _context_value
from .data import _required, _date, _rows
from ._structured_pool import _number, _text, _sum, _bound, _checked, _nullable


def covered_bond_dual_recourse(cover_pool, cover_pool_paths, bond_contract, issuer_paths, *, context=None):
    """Allocate issuer, segregated cover and residual-estate cash once per claim."""
    pool=_checked(cover_pool,'CreditAssetPool');paths=_checked(cover_pool_paths,'AssetPoolPaths')
    if pool['profile']['product_kind'] not in ('MORTGAGE_COVER','PUBLIC_SECTOR_COVER') or paths['pool_hash']!=pool['sha256']:
        _fail('STRUCTURED_ROUTE','Matched mortgage/public-sector cover pool required')
    b=deepcopy(bond_contract)
    _required(b,['bond_id','version','currency','legal_profile_id','principal','original_maturity','interest_schedule','extension_interest_schedule','extension_rule','cover_cash_convention','estate_rule','innovation_hash'])
    _nullable(b,['extended_maturity'])
    for k in ('bond_id','version','currency','legal_profile_id'):_text(b[k],k)
    principal=_number(b['principal'],'principal',0)
    if principal==0:_fail('STRUCTURED_COVERED','Positive principal required')
    origin=paths['origin'];maturity=b['original_maturity'];extended=b['extended_maturity']
    if _date(maturity,'original_maturity')<=_date(origin,'origin') or (extended is not None and _date(extended,'extended_maturity')<=_date(maturity,'original_maturity')):
        _fail('STRUCTURED_CALENDAR','Original maturity after origin and extension strictly later required')
    if b['extension_rule'] not in ('NONE','ISSUER_DEFAULT_AND_DECLARED_TRIGGER') or (b['extension_rule']=='NONE'and extended is not None) or (b['extension_rule']!='NONE'and extended is None) or b['cover_cash_convention']!='SEGREGATED_UNTIL_ISSUER_DEFAULT' or b['estate_rule']!='RESIDUAL_AFTER_COMPLETE_COVER':
        _fail('STRUCTURED_COVERED','Explicit extension, segregation and residual-estate conventions required')
    if b['innovation_hash']!=paths['innovation_hash'] or b['currency']!=pool['profile']['currency']:
        _fail('STRUCTURED_SHARED_PATHS','Issuer and cover scenarios must share innovation binding and currency')
    for key in ('interest_schedule','extension_interest_schedule'):
        if not isinstance(b[key],dict):_fail('STRUCTURED_COVERED','Interest schedules must be dated mappings')
        for date,amount in b[key].items():
            _number(amount,'interest_due',0);_date(date,'interest_date')
            if (key=='interest_schedule'and not origin<date<=maturity) or (key=='extension_interest_schedule'and (extended is None or not maturity<date<=extended)):
                _fail('STRUCTURED_CALENDAR','Coupon dates must fit original or extension term')
    rows=_rows(issuer_paths,'issuer_paths');imap={}
    last_claim_date=extended or maturity
    for row in rows:
        _required(row,['scenario_id','issuer_capacity','extension_triggered','estate_payment_date','estate_recovery_fraction','estate_capacity'])
        _nullable(row,['issuer_default_date'])
        sid=row['scenario_id']
        if sid not in paths['scenario_ids'] or sid in imap:_fail('STRUCTURED_COVERED','Unique aligned issuer scenario rows required')
        default=row['issuer_default_date'];settlement=row['estate_payment_date']
        if default is not None and _date(default,'issuer_default_date')<_date(origin,'origin'):_fail('STRUCTURED_CALENDAR','Issuer default may start at origin, not before')
        if _date(settlement,'estate_payment_date')<max(_date(paths['dates'][-1],'pool_end'),_date(last_claim_date,'last_claim_date')):
            _fail('STRUCTURED_CALENDAR','Residual estate settlement must follow full cover calendar and possible maturity')
        if default is not None and default>settlement:_fail('STRUCTURED_CALENDAR','Default after settlement is outside scenario horizon')
        if not isinstance(row['extension_triggered'],bool) or (row['extension_triggered'] and (b['extension_rule']=='NONE' or default is None or default>maturity)):
            _fail('STRUCTURED_EXTENSION','Extension requires declared rule and issuer default by original maturity')
        _number(row['estate_recovery_fraction'],'estate_recovery_fraction',0,1);_number(row['estate_capacity'],'estate_capacity',0)
        if not isinstance(row['issuer_capacity'],dict):_fail('STRUCTURED_COVERED','Dated issuer cash capacity mapping required')
        for date,amount in row['issuer_capacity'].items():
            _number(amount,'issuer_capacity',0)
            if not _date(origin,'origin')<=_date(date,'capacity_date')<=_date(settlement,'estate_payment_date'):_fail('STRUCTURED_CALENDAR','Issuer capacity outside scenario horizon')
        imap[sid]=row
    if set(imap)!=set(paths['scenario_ids']):_fail('STRUCTURED_COVERED','Full aligned issuer scenarios required')
    if any(not e['terminal_complete']for e in paths['ending_states']):_fail('STRUCTURED_UNRESOLVED','Residual estate calculation requires complete cover runoff')
    ctx=_context_value(context)
    if ctx.get('currency') not in (None,b['currency']) or ctx.get('probability_measure') not in (None,'PHYSICAL') or ctx.get('conditioning') not in (None,paths['scenario_contract']['conditioning']):_fail('TARGET_MISMATCH','Covered bond result context conflicts with physical paths')
    horizon_end=max(row['estate_payment_date'] for row in rows)
    if ctx.get('time_unit') not in (None,'days') or ctx.get('horizon') not in (None,(_date(horizon_end,'horizon_end')-_date(origin,'origin')).days):_fail('TARGET_MISMATCH','Covered bond context must match actual final settlement-day horizon')
    ledgers=[];ends=[];cash={(r['scenario_id'],r['date']):r for r in paths['cashflows']}
    for sid in paths['scenario_ids']:
        row=imap[sid];default=row['issuer_default_date'];extension=row['extension_triggered'];due_date=extended if extension else maturity
        interest=deepcopy(b['interest_schedule'])
        if extension:
            for d,a in b['extension_interest_schedule'].items():interest[d]=_sum([interest.get(d,0),a])
        calendar=sorted(set(paths['dates'])|set(interest)|set(row['issuer_capacity'])|{maturity,due_date,row['estate_payment_date']}|({default}if default else set()))
        outstanding_principal=0.;outstanding_interest=0.;cover_balance=0.;issuer_total=cover_total=estate_total=0.;late_interest=False;original_shortfall=0.;effective_maturity_shortfall=0.
        for date in calendar:
            principal_due=principal if date==due_date else 0.;coupon_due=interest.get(date,0.)
            outstanding_principal=_sum([outstanding_principal,principal_due]);outstanding_interest=_sum([outstanding_interest,coupon_due])
            pc=cash.get((sid,date));net_cover=_sum([pc['principal_available'],pc['interest_available'],pc['recovery_gross'],-pc['workout_cost'],-pc['servicing_cost']])if pc else 0.
            opening_cover=cover_balance;cover_balance=_sum([cover_balance,net_cover]);claim=_sum([outstanding_principal,outstanding_interest]);in_default=default is not None and date>=default
            issuer_payment=min(claim,row['issuer_capacity'].get(date,0.))if not in_default else 0.;claim-=issuer_payment
            cover_payment=min(claim,max(0.,cover_balance))if in_default else 0.;cover_balance-=cover_payment;claim-=cover_payment
            estate_payment=min(claim*row['estate_recovery_fraction'],row['estate_capacity'])if in_default and date==row['estate_payment_date'] else 0.
            paid=_sum([issuer_payment,cover_payment,estate_payment]);interest_paid=min(outstanding_interest,paid);outstanding_interest=max(0.,outstanding_interest-interest_paid);principal_paid=min(outstanding_principal,max(0.,paid-interest_paid));outstanding_principal=max(0.,outstanding_principal-principal_paid)
            issuer_total=_sum([issuer_total,issuer_payment]);cover_total=_sum([cover_total,cover_payment]);estate_total=_sum([estate_total,estate_payment]);late_interest |= outstanding_interest>1e-12*max(1.,principal)
            if date==maturity:original_shortfall=principal if extension else outstanding_principal
            if date==due_date:effective_maturity_shortfall=outstanding_principal
            residual=_sum([opening_cover,net_cover,-cover_payment,-cover_balance])
            ledgers.append(dict(scenario_id=sid,date=date,issuer_in_default=in_default,principal_due=principal_due,interest_due=coupon_due,issuer_payment=issuer_payment,cover_payment=cover_payment,estate_payment=estate_payment,interest_paid=interest_paid,principal_paid=principal_paid,outstanding_interest=outstanding_interest,outstanding_principal=outstanding_principal,cover_net_collection=net_cover,closing_cover_balance=cover_balance,cover_conservation_residual=residual))
        total_claim=_sum([principal,*interest.values()]);total_payment=_sum([issuer_total,cover_total,estate_total]);shortfall=_sum([outstanding_principal,outstanding_interest])
        if abs(_sum([total_claim,-total_payment,-shortfall]))>1e-10*max(1.,total_claim):_fail('STRUCTURED_CONSERVATION','Dual recourse claim reconciliation failed')
        ends.append(dict(scenario_id=sid,extension_activated=extension,effective_maturity=due_date,contractual_claim=total_claim,issuer_payment=issuer_total,cover_payment=cover_total,estate_payment=estate_total,ultimate_shortfall=shortfall,ultimate_principal_shortfall=outstanding_principal,ultimate_interest_shortfall=outstanding_interest,original_date_principal_deferral_or_shortfall=original_shortfall,effective_maturity_principal_shortfall=effective_maturity_shortfall,timely_interest_failure=late_interest,unused_cover_cash=max(0.,cover_balance),unfunded_cover_cost=max(0.,-cover_balance)))
    value=_bound(dict(schema_version='1.0',kind='CoveredBondRecoursePaths',origin=origin,horizon_end=horizon_end,pool_hash=pool['sha256'],pool_paths_hash=paths['sha256'],bond_contract=b,issuer_paths_hash=_fingerprint(rows),innovation_hash=paths['innovation_hash'],scenario_ids=deepcopy(paths['scenario_ids']),scenario_weights=deepcopy(paths['scenario_weights']),cashflows=ledgers,ending_states=ends))
    return _result(value,'structured_credit.covered_bond_dual_recourse','covered_bond_recourse',ctx,limitations=('National insolvency priority, enforceability and maturity extension are externally supplied legal assumptions. Cover collections are segregated until issuer default; pre-default cash is not distributed to issuer. Normal issuer capacity pays due claims before default; thereafter cover pays and residual estate recovery is capped by remaining contractual claim and estate capacity. Estate settlement follows complete cover runoff and all possible maturity dates, avoiding competing future cover/estate recovery. Interest is paid before principal. Extension moves principal only and activates explicitly supplied extra coupons; original-date deferral is reported separately from effective-maturity failure. No penalty interest, issuer recovery uplift or double recovery is inferred.',))
