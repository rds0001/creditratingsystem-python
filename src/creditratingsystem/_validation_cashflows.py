"""Factual dated collections cashflow validation with explicit complete coverage."""
from copy import deepcopy
from collections import Counter
import math
import numpy as np
from .core import _fail,_fingerprint,_result,_context_value
from .data import _date
from ._expert_common import _ex_object,_ex_ids,_ex_read
from ._validation_common import _vb_design,_vb_records,_vb_number,_vb_string,_vb_draws,_vb_bands,_vb_errors,_vb_result
from ._validation_retail import _vr_rows


def collections_backtest(forecasts,episodes,cashflows,actions,cashflow_coverage,*,discount_contract,design,uncertainty,cluster_resamples=None,confidence_level=.95,context=None):
    """Compare frozen fixed-horizon recovery/payment/cost forecasts to known cash."""
    d=_vb_design(design,context);rows=_vb_records(forecasts,d);history=_ex_read(episodes.value if hasattr(episodes,'value')else episodes,'history_hash');cash=_vr_rows(cashflows,'cashflows');action_rows=_vr_rows(actions,'actions');coverage=_vr_rows(cashflow_coverage,'cashflow_coverage');discount=deepcopy(_ex_object(discount_contract,'discount_contract'))
    if history.get('evaluation_cutoff')!=d['evaluation_cutoff']:_fail('COLLECTIONS_VALIDATION_CUTOFF','Episode history must use the declared evaluation cutoff')
    currency=_vb_string(d.get('currency'),'validation currency');_vb_string(discount.get('reference'),'discount reference');rate=_vb_number(discount.get('annual_rate'),'annual_rate');known=_date(discount.get('known_at'),'discount known_at')
    if _context_value(context).get('currency',currency)!=currency:_fail('COLLECTIONS_VALIDATION_CURRENCY','Context and validation currencies differ')
    if discount.get('method')!='flat_effective_annual'or discount.get('day_count')!='ACT_365F'or rate<=-1 or discount.get('currency')!=currency:_fail('COLLECTIONS_VALIDATION_DISCOUNT','Declare currency-matched flat effective annual ACT_365F discount with rate > -1')
    if any(known>_date(r['forecast_known_at'],'forecast_known_at')for r in rows):_fail('COLLECTIONS_VALIDATION_LOOKAHEAD','Discount curve must be known before forecasts')
    eps={e['episode_id']:e for e in history['episodes']};_ex_ids([r.get('cashflow_id')for r in cash],'cashflow IDs',empty=True);_ex_ids([r.get('action_id')for r in action_rows],'action IDs',empty=True);_ex_ids([r.get('episode_id')for r in coverage],'cashflow coverage episode IDs',empty=True);action_index={r['action_id']:r for r in action_rows};coverage_index={r['episode_id']:r for r in coverage};cash_index={};cutoff=_date(d['evaluation_cutoff'],'evaluation_cutoff')
    for c in cash:
        if c.get('episode_id')not in eps or c.get('kind')not in ('REPAYMENT','RECOVERY','COST')or c.get('currency')!=currency:_fail('COLLECTIONS_VALIDATION_CASH','Known episode, explicit cash kind and matching currency required')
        _vb_number(c.get('amount'),'cash amount',0);at=_date(c.get('occurred_at'),'cash occurred_at');available=_date(c.get('known_at'),'cash known_at')
        if available<at or at<_date(eps[c['episode_id']]['started_at'],'episode start'):_fail('COLLECTIONS_VALIDATION_DATE','Cash knowledge follows occurrence inside the assigned episode lineage')
        cash_index.setdefault(c['episode_id'],[]).append(c)
    for c in coverage:
        start=_date(c.get('observed_from'),'cash coverage from');end=_date(c.get('observed_through'),'cash coverage through');available=_date(c.get('known_at'),'cash coverage known')
        if c['episode_id']not in eps or start>end or available<end:_fail('COLLECTIONS_VALIDATION_COVERAGE','Cashflow capture coverage must be a valid known interval on an episode')
    for a in action_rows:
        for key in ('account_id','treatment_id','action_type'):_vb_string(a.get(key),key)
        if a.get('stage')not in ('ASSIGNED','OFFERED','ACCEPTED','PERFORMED','CANCELLED'):_fail('COLLECTIONS_VALIDATION_ACTION','Unknown action stage')
        if _date(a.get('known_at'),'action known_at')<_date(a.get('occurred_at'),'action occurred_at'):_fail('COLLECTIONS_VALIDATION_ACTION','Action knowledge precedes occurrence')
    kinds=('REPAYMENT','RECOVERY','COST');keys=('discounted_repayment','discounted_recovery','discounted_cost','discounted_net','recovery_weighted_days');p={k:np.zeros(len(rows))for k in keys};y={k:np.zeros(len(rows))for k in keys};eligible=np.zeros(len(rows),bool);timing=np.zeros(len(rows),bool);weights=np.ones(len(rows));groups=[];traces=[];exclusions=[]
    def value(flows,asof):
        sums={k:0. for k in kinds};recovery_mass=0.;weighted_days=0.
        for c in flows:
            days=(_date(c['occurred_at'],'cash date')-asof).days
            try:factor=math.exp(-math.log1p(rate)*days/365.)
            except OverflowError:_fail('COLLECTIONS_VALIDATION_NUMERICAL','Discount factor exceeds finite range')
            amount=float(c['amount']);sums[c['kind']]+=amount*factor
            if c['kind']=='RECOVERY':recovery_mass+=amount;weighted_days+=amount*days
        result=dict(discounted_repayment=sums['REPAYMENT'],discounted_recovery=sums['RECOVERY'],discounted_cost=sums['COST'],discounted_net=sums['REPAYMENT']+sums['RECOVERY']-sums['COST'],recovery_weighted_days=None if recovery_mass==0 else weighted_days/recovery_mass)
        if any(v is not None and not math.isfinite(v)for v in result.values()):_fail('COLLECTIONS_VALIDATION_NUMERICAL','Discounted cashflow exceeds finite range')
        return result
    for i,row in enumerate(rows):
        episode=eps.get(row.get('episode_id'));asof=_date(row['as_of'],'as_of');end=_date(row['horizon_end'],'horizon_end');weights[i]=_vb_number(row.get('weight',1),'weight',0)
        if episode is None or row.get('account_id')!=episode['account_id']or asof<_date(episode['started_at'],'episode start'):_fail('COLLECTIONS_VALIDATION_EPISODE','Forecast must belong to its already started account episode')
        if episode['timeline']and _date(episode['timeline'][0]['known_at'],'episode first known')>_date(row['forecast_known_at'],'forecast_known_at'):_fail('COLLECTIONS_VALIDATION_LOOKAHEAD','Episode onset unavailable at forecast')
        action=action_index.get(row.get('index_action_id'))
        if action is None or action['account_id']!=row['account_id']or action['stage']not in ('ASSIGNED','OFFERED')or action['known_at']>row['forecast_known_at']or action['occurred_at']>row['as_of']:_fail('COLLECTIONS_VALIDATION_ACTION','Frozen forecast must reference a known assignment/offer, not eventual uptake')
        embedded=next((a for a in episode['actions']if a['action_id']==action['action_id']),None)
        if embedded is None or any(embedded.get(k)!=action.get(k)for k in ('account_id','treatment_id','action_type','stage','occurred_at','known_at')):_fail('COLLECTIONS_VALIDATION_ACTION','Action assignment must match the frozen episode-linked record')
        groups.append(action['action_type']);projected=_vr_rows(row.get('predicted_cashflows'),'predicted_cashflows');_ex_ids([c.get('cashflow_id')for c in projected],'predicted cashflow IDs',empty=True)
        for c in projected:
            if c.get('kind')not in kinds or c.get('currency')!=currency or not asof<=_date(c.get('occurred_at'),'predicted cash date')<=end:_fail('COLLECTIONS_VALIDATION_FORECAST','Forecast cash kinds/currency/dates must match fixed horizon')
            _vb_number(c.get('amount'),'predicted amount',0)
        actual=[c for c in cash_index.get(row['episode_id'],[])if asof<=_date(c['occurred_at'],'cash date')<=min(end,cutoff)and _date(c['known_at'],'cash known')<=cutoff];pv=value(projected,asof);av=value(actual,asof);cover=coverage_index.get(row['episode_id']);reason=None
        if end>cutoff:reason='IMMATURE_HORIZON'
        elif cover is None or _date(cover['known_at'],'coverage known')>cutoff or _date(cover['observed_from'],'coverage from')>asof or _date(cover['observed_through'],'coverage through')<end:reason='INCOMPLETE_CASHFLOW_CAPTURE'
        elif weights[i]==0:reason='ZERO_EVALUATION_WEIGHT'
        else:eligible[i]=True
        timing[i]=eligible[i]and pv['recovery_weighted_days']is not None and av['recovery_weighted_days']is not None
        for key in keys:p[key][i]=pv[key]if pv[key]is not None else 0.;y[key][i]=av[key]if av[key]is not None else 0.
        if reason:exclusions.append(dict(prediction_id=row['prediction_id'],reason=reason))
        traces.append(dict(prediction_id=row['prediction_id'],episode_id=row['episode_id'],episode_status_at_cutoff=episode['status'],index_action_id=action['action_id'],treatment_id=action['treatment_id'],action_type=action['action_type'],predicted=pv,observed_known_to_cutoff=av,observed_cashflow_ids=[c['cashflow_id']for c in actual],complete_fixed_horizon=bool(eligible[i]),timing_applicability='APPLICABLE'if timing[i]else'NOT_APPLICABLE',timing_reason=None if timing[i]else'NO_PAIRED_POSITIVE_RECOVERY_MASS_OR_INCOMPLETE_WINDOW'))
    def calculate(draw):
        def group(mask):return {k:_vb_errors(p[k],y[k],weights*draw*mask*(timing if k=='recovery_weighted_days'else eligible))for k in keys}
        return dict(overall=group(np.ones(len(rows))),by_action={g:group(np.array([v==g for v in groups]))for g in sorted(set(groups))})
    values=calculate(np.ones(len(rows)));draws=_vb_draws(uncertainty,cluster_resamples,rows,confidence_level);samples=None
    if draws is not None:
        samples=[]
        for draw in draws:
            m=calculate(draw);flat={k+'/'+metric:m['overall'][k][metric]for k in keys for metric in ('bias','rmse')}
            for g,v in m['by_action'].items():flat.update({g+'/'+k+'/'+metric:v[k][metric]for k in keys for metric in ('bias','rmse')})
            samples.append(flat)
    diagnostics=dict(discount_contract=discount,episode_history_hash=history['history_hash'],cashflow_fingerprint=_fingerprint(cash),action_fingerprint=_fingerprint(action_rows),cashflow_coverage_fingerprint=_fingerprint(coverage),forecast_traces=traces,episode_status_counts=dict(sorted(Counter(eps[r['episode_id']]['status']for r in rows).items())),estimand='FACTUAL_FIXED_HORIZON_CASHFLOWS',causal_action_effect_identified=False,open_episodes_treated_as_closed=False,cashflow_window='[forecast_as_of, horizon_end]',timing_estimand='UNDISCOUNTED_RECOVERY_AMOUNT_WEIGHTED_MEAN_DAYS_CONDITIONAL_ON_POSITIVE_PREDICTED_AND_OBSERVED_RECOVERY')
    return _vb_result('collections_backtest',values,d,rows,exclusions,_vb_bands(samples,confidence_level),context,diagnostics,currency)
