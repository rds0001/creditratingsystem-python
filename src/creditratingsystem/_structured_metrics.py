"""Definition-specific tranche event and discounted contractual cash shortfall."""
from copy import deepcopy
from math import sqrt, hypot
from scipy.stats import beta
from .core import _fail, _result, _context_value
from .data import _required, _date, _rows
from ._structured_pool import _number, _text, _sum, _bound, _checked


def estimate_tranche_credit_metrics(tranche_paths, target, discount_factors, *, confidence_level=.95, context=None):
    """Keep payment-event probability separate from actual discounted cash loss."""
    paths = _checked(tranche_paths, 'TrancheWaterfallPaths')
    _required(target, ['event','horizon','time_unit','probability_measure','conditioning'])
    events = {'TIMELY_INTEREST':'timely_interest_failure','LEGAL_MATURITY_PRINCIPAL':'legal_maturity_failure','ULTIMATE_PRINCIPAL':'ultimate_principal_loss'}
    if target['event'] not in events or target['time_unit'] != 'days' or target['probability_measure'] != 'PHYSICAL' or target['conditioning'] != paths['scenario_contract']['conditioning'] or target['horizon'] != (_date(paths['dates'][-1],'date')-_date(paths['origin'],'origin')).days:
        _fail('TARGET_MISMATCH','Explicit event and actual full calendar-day physical horizon required')
    ctx = _context_value(context)
    if any(ctx.get(k) not in (None,v) for k,v in target.items() if k != 'event') or ctx.get('currency') not in (None,paths['currency']):
        _fail('TARGET_MISMATCH','Tranche metric target conflicts with context')
    dates = paths['structure_profile']['payment_dates']
    if not isinstance(discount_factors,dict) or set(discount_factors)!=set(dates):
        _fail('STRUCTURED_DISCOUNT','One origin-based discount factor for each payment date required')
    for v in discount_factors.values():
        if _number(v,'discount_factor',0)==0:
            _fail('STRUCTURED_DISCOUNT','Strictly positive origin-based discount factors required')
    level = _number(confidence_level,'confidence_level',0,1)
    if level in (0,1):
        _fail('STRUCTURED_CONFIDENCE','Interior confidence level required')
    sids = paths['scenario_ids']
    weight_total = _sum(paths['scenario_weights'])
    weights = [w / weight_total for w in paths['scenario_weights']]
    final = {(e['scenario_id'],e['tranche_id']):e for e in paths['ending_states']}
    rows, results = [], []
    iid = paths['scenario_contract']['sampling_design']=='IID_UNWEIGHTED'
    for tranche in paths['structure_profile']['tranches']:
        tid=tranche['tranche_id']; notional=tranche['notional']; selected=[]; ultimate=[]; cashloss=[]; severities=[]
        for sid in sids:
            e=final[sid,tid]; identified=e['ultimate_identified']
            if target['event']=='ULTIMATE_PRINCIPAL' and not identified:
                _fail('STRUCTURED_UNRESOLVED','Ultimate event requires complete pool run-off')
            event = e[events[target['event']]] > 1e-12*max(1.,notional) if target['event']=='ULTIMATE_PRINCIPAL' else e[events[target['event']]]
            cash = [r for r in paths['tranche_cashflows'] if r['scenario_id']==sid and r['tranche_id']==tid]
            expected_principal_pv = _sum([notional*discount_factors[tranche['legal_maturity']]])
            principal_pv = _sum(r['principal_paid']*discount_factors[r['date']] for r in cash)
            coupon_pv = _sum(r['coupon_due']*discount_factors[r['date']] for r in cash)
            interest_pv = _sum(r['interest_paid']*discount_factors[r['date']] for r in cash)
            signed = _sum([expected_principal_pv,coupon_pv,-principal_pv,-interest_pv]) if identified else None
            shortfall = max(0.,signed) if identified else None
            rows.append(dict(scenario_id=sid,tranche_id=tid,event=bool(event),ultimate_principal_loss=e['ultimate_principal_loss'],contractual_principal_pv=expected_principal_pv,principal_received_pv=principal_pv,coupon_due_pv=coupon_pv,interest_received_pv=interest_pv,signed_contractual_cash_shortfall=signed,discounted_contractual_cash_shortfall=shortfall))
            selected.append(float(event));ultimate.append(e['ultimate_principal_loss']);cashloss.append(shortfall)
        pd=_sum(w*x for w,x in zip(weights,selected));known=all(x is not None for w,x in zip(weights,ultimate) if w>0)
        expected_loss=_sum(w*x for w,x in zip(weights,ultimate) if w>0) if known else None
        discounted_loss=_sum(w*x for w,x in zip(weights,cashloss) if w>0) if known else None
        event_loss=_sum(w*x*y for w,x,y in zip(weights,selected,ultimate) if w>0) if known else None
        interval=mcse=None
        if iid:
            n=len(sids);k=int(sum(selected));alpha=1-level
            interval=[0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),1. if k==n else float(beta.isf(alpha/2,k+1,n-k))]
            if known and n>1:
                mean=discounted_loss
                mcse=_sum([hypot(*(_sum([x,-mean])/sqrt(n*(n-1)) for x in cashloss))])
        results.append(dict(tranche_id=tid,notional=notional,event_probability=pd,event_count=int(sum(selected)),positive_weight_scenarios=sum(w>0 for w in weights),event_probability_interval=interval,expected_ultimate_principal_loss=expected_loss,expected_ultimate_principal_loss_rate=expected_loss/notional if known else None,expected_discounted_contractual_cash_shortfall=discounted_loss,conditional_principal_loss_given_selected_event=event_loss/pd if known and pd>0 else None,expected_principal_loss_outside_selected_event=_sum([expected_loss,-event_loss]) if known else None,discounted_cash_shortfall_mcse=mcse,ultimate_identified=known))
    value=_bound(dict(schema_version='1.0',kind='TrancheCreditMetrics',paths_hash=paths['sha256'],innovation_hash=paths['innovation_hash'],target=deepcopy(target),discount_factors=deepcopy(discount_factors),confidence_level=level,sampling_design=paths['scenario_contract']['sampling_design'],metrics=results,scenario_metrics=rows))
    return _result(value,'structured_credit.estimate_tranche_credit_metrics','tranche_risk',ctx,limitations=('Targets distinguish timely interest, legal-maturity principal and ultimate principal; issuer default probability is never substituted. Discounted cash shortfall compares principal promised at legal maturity plus coupons actually due on opening outstanding balances against actual dated receipts; each scenario floors the combined signed shortfall at zero. This explicitly includes timing loss and is not a spread-implied fair value. Undiscounted ultimate principal loss is reported separately and is not forced to zero outside a selected payment event. Open-runoff cash shortfall and ultimate loss remain unidentified. Exact binomial intervals and sample-mean loss standard errors require caller-declared IID equally weighted scenarios; fixed weights and dependent/QMC paths receive no IID uncertainty. Conditional severity is explicitly conditional on the selected event, not a generic LGD.',))


def stress_tranche_structure(baseline_paths, alternatives, target, discount_factors, *, confidence_level=.95, context=None):
    """Compare complete paired waterfall artifacts on identical shared innovations."""
    base=_checked(baseline_paths,'TrancheWaterfallPaths'); alternatives=_rows(alternatives,'alternatives')
    if not alternatives:
        _fail('STRUCTURED_STRESS','At least one explicit alternative required')
    baseline=estimate_tranche_credit_metrics(base,target,discount_factors,confidence_level=confidence_level,context=context).value
    base_metrics={m['tranche_id']:m for m in baseline['metrics']};comparisons=[];seen=set()
    for alternative in alternatives:
        _required(alternative,['alternative_id','tranche_paths','assumption']);aid=_text(alternative['alternative_id'],'alternative_id');_text(alternative['assumption'],'assumption')
        if aid in seen:
            _fail('STRUCTURED_STRESS','Unique alternative IDs required')
        seen.add(aid);paths=_checked(alternative['tranche_paths'],'TrancheWaterfallPaths')
        for key in ('scenario_ids','scenario_weights','innovation_hash','origin','currency'):
            if paths[key]!=base[key]:
                _fail('STRUCTURED_SHARED_PATHS','Stress requires identical scenario axes, weights, origin, currency and shared innovations')
        metrics=estimate_tranche_credit_metrics(paths,target,discount_factors,confidence_level=confidence_level,context=context).value
        if {m['tranche_id']for m in metrics['metrics']}!=set(base_metrics):
            _fail('STRUCTURED_STRESS','Comparable tranche identities required; changed notionals remain explicit')
        deltas=[]
        for m in metrics['metrics']:
            old=base_metrics[m['tranche_id']];delta={}
            for key in ('event_probability','expected_ultimate_principal_loss','expected_ultimate_principal_loss_rate','expected_discounted_contractual_cash_shortfall'):
                delta[key]=_sum([m[key],-old[key]]) if m[key]is not None and old[key]is not None else None
            deltas.append(dict(tranche_id=m['tranche_id'],baseline_notional=old['notional'],alternative_notional=m['notional'],changes=delta))
        comparisons.append(dict(alternative_id=aid,assumption=alternative['assumption'],paths_hash=paths['sha256'],structure_hash=paths['structure_hash'],metrics=metrics,deltas=deltas))
    value=_bound(dict(schema_version='1.0',kind='TrancheStructureStress',baseline=baseline,baseline_structure_hash=base['structure_hash'],innovation_hash=base['innovation_hash'],comparisons=comparisons))
    return _result(value,'structured_credit.stress_tranche_structure','paired_tranche_stress',context,limitations=('Alternatives are explicit evaluated economic structures or joint asset states sharing the exact innovation binding and scenario axes. Changes are paired scenario comparisons, not causal effects or an optimization guarantee. Changed notionals are reported alongside amount and rate deltas. Individual IID intervals are not intervals for paired differences; no unrequested independent-samples interval is fabricated.',))
