"""Joint path and exclusive-state loss expectations; dependence is preserved."""
import math
import numpy as np
from .core import _numeric, _probability, _fail, _result, _context_value, _fingerprint


def _contract(currency, probability_measure, context):
    if not isinstance(currency,str) or not currency.strip():
        _fail("INVALID_CURRENCY","Explicit reporting currency required")
    if probability_measure not in ("PHYSICAL","RISK_NEUTRAL"):
        _fail("INCOMPATIBLE_MEASURE","Declare PHYSICAL or RISK_NEUTRAL")
    ctx=_context_value(context)
    for key,value in (("currency",currency),("probability_measure",probability_measure)):
        if ctx.get(key) is not None and ctx[key]!=value:
            _fail("TARGET_MISMATCH","Context conflicts with declared loss convention")
    return ctx


def _ids(values,name,n=None):
    if not isinstance(values,(list,tuple)) or not values or any(not isinstance(x,str) or not x for x in values) or len(set(values))!=len(values) or (n is not None and len(values)!=n):
        _fail("INVALID_IDENTIFIERS",name+" requires unique nonempty strings of matching length")
    return list(values)


def _weights(weights,n):
    w=_numeric(weights,"scenario_weights",vector_only=True)
    if len(w)!=n or np.any(w<0):
        _fail("INVALID_WEIGHTS","Scenario probabilities must match paths and be nonnegative")
    try: total=math.fsum(w)
    except OverflowError: _fail("INVALID_WEIGHTS","Scenario probabilities must sum to one")
    if not math.isfinite(total) or abs(total-1)>1e-12:
        _fail("INVALID_WEIGHTS","Scenario probabilities must sum to one within 1e-12")
    return w/total,total-1


def _multiply(*values):
    arrays=np.broadcast_arrays(*values)
    mantissa=np.ones_like(arrays[0],dtype=float); exponent=np.zeros(arrays[0].shape,dtype=np.int64)
    for value in arrays:
        m,e=np.frexp(value);mantissa*=m;exponent+=e
    with np.errstate(over="ignore",under="ignore",invalid="ignore"):
        result=np.ldexp(mantissa,exponent)
    if not np.all(np.isfinite(result)):
        _fail("NUMERICAL_FAILURE","Loss product outside finite range")
    return result


def _sum(values):
    try: result=math.fsum(values)
    except OverflowError: _fail("NUMERICAL_FAILURE","Loss sum outside finite range")
    if not math.isfinite(result):_fail("NUMERICAL_FAILURE","Loss sum outside finite range")
    return result


def _time(time_grid,n):
    t=_numeric(time_grid,"time_grid",vector_only=True)
    if len(t)!=n or np.any(t<=0) or np.any(np.diff(t)<=0):
        _fail("INVALID_TIME_GRID","Positive increasing period-end times in years required")
    return t


def _paths(default_indicator,lgd_paths,ead_paths,discount_factors,scenario_weights,scenario_ids,time_grid):
    d=_numeric(default_indicator,"default_indicator");l=_numeric(lgd_paths,"lgd_paths");e=_numeric(ead_paths,"ead_paths")
    if d.ndim!=2 or l.shape!=d.shape or e.shape!=d.shape:
        _fail("INCOMPATIBLE_SHAPE","Default, LGD and EAD require matching paths by periods matrices")
    if np.any((d!=0)&(d!=1)) or np.any(d.sum(axis=1)>1):
        _fail("INVALID_DEFAULT_PATH","Only one first-default indicator per path is allowed")
    if np.any(e<0):_fail("INVALID_EAD","EAD must be nonnegative")
    ids=_ids(scenario_ids,"scenario_ids",d.shape[0]);times=_time(time_grid,d.shape[1]);w,residual=_weights(scenario_weights,d.shape[0])
    df=_numeric(discount_factors,"discount_factors")
    if df.ndim==0:df=np.full(d.shape,float(df))
    elif df.ndim==1 and len(df)==d.shape[1]:df=np.broadcast_to(df,d.shape)
    elif df.shape!=d.shape:_fail("INCOMPATIBLE_SHAPE","Discounts must be scalar, period vector or matching matrix")
    if np.any(df<0):_fail("INVALID_DISCOUNT","Discount factors must be nonnegative")
    return d,l,e,df,w,ids,times,residual


def expected_loss_from_paths(default_indicator,lgd_paths,ead_paths,scenario_weights,discount_factors,
                             *,scenario_ids,time_grid,currency,probability_measure,uncertainty="none",context=None):
    """Average joint discounted first-default losses with explicit scenario alignment."""
    ctx=_contract(currency,probability_measure,context)
    d,l,e,df,w,ids,times,residual=_paths(default_indicator,lgd_paths,ead_paths,discount_factors,scenario_weights,scenario_ids,time_grid)
    cells=_multiply(d,l,e,df)
    losses=np.array([_sum(row) for row in cells])
    period=np.array([_sum(w*cells[:,j]) for j in range(d.shape[1])])
    pd=np.array([_sum(w*d[:,j]) for j in range(d.shape[1])])
    lgd_cond=[];ead_df_cond=[];separate=[]
    for j,p in enumerate(pd):
        if p==0:lgd_cond.append(None);ead_df_cond.append(None);separate.append(0.)
        else:
            q=w*d[:,j]/p
            a=_sum(q*l[:,j]);b=_sum(_multiply(q,e[:,j],df[:,j]))
            lgd_cond.append(a);ead_df_cond.append(b);separate.append(float(_multiply(p,a,b)))
    mean=_sum(period);se=None
    if uncertainty not in ("none","iid"):_fail("INVALID_UNCERTAINTY","Use none or iid")
    if uncertainty=="iid":
        if len(w)<2 or not np.all(w==w[0]):
            _fail("INCOMPATIBLE_UNCERTAINTY","IID Monte Carlo SE requires at least two equally weighted independent paths")
        scale=float(np.max(np.abs(losses)))
        se=0. if scale==0 else float(np.std(losses/scale,ddof=1)/math.sqrt(len(w))*scale)
    return _result(dict(expected_loss=mean,path_losses=losses.tolist(),period_contributions=period.tolist(),
        marginal_default_probability=pd.tolist(),default_weighted_lgd=lgd_cond,
        default_weighted_discounted_ead=ead_df_cond,conditional_independence_loss=_sum(separate),
        conditional_lgd_exposure_covariance_contribution=_sum((mean,-_sum(separate))),standard_error=se),
        "losses.expected_loss_from_paths",currency,ctx,
        diagnostics=dict(probability_measure=probability_measure,time_unit="years",time_grid=times.tolist(),
            scenario_ids=ids,scenario_weight_residual=residual,uncertainty=uncertainty,
            path_fingerprint=_fingerprint(dict(default=d.tolist(),lgd=l.tolist(),ead=e.tolist(),discount=df.tolist(),scenario_ids=ids))),
        limitations=("All matrices share the declared scenario order and first-default periods. Conditional means do not justify unconditional factorization.",
                     "IID standard error excludes parameter, discretization and model uncertainty; independence of supplied paths is not verified.",))


def lifetime_expected_loss(marginal_pd,conditional_loss,discount_factors,time_grid,*,currency,probability_measure,context=None):
    """Sum marginal first-default mass times conditional joint loss and discount."""
    ctx=_contract(currency,probability_measure,context)
    p=_probability(marginal_pd);l=_numeric(conditional_loss,"conditional_loss",vector_only=True);df=_numeric(discount_factors,"discount_factors",vector_only=True)
    if p.ndim!=1 or p.shape!=l.shape or p.shape!=df.shape:
        _fail("INCOMPATIBLE_SHAPE","Equal-length period vectors required")
    times=_time(time_grid,len(p))
    if _sum(p)>1+1e-12:_fail("INVALID_PD_MASS","Marginal first-default masses cannot sum above one")
    if np.any(df<0):_fail("INVALID_DISCOUNT","Nonnegative discount factors required")
    contributions=_multiply(p,l,df)
    return _result(dict(lifetime_expected_loss=_sum(contributions),period_contributions=contributions.tolist(),
                        total_default_probability=_sum(p)),"losses.lifetime_expected_loss",currency,ctx,
        diagnostics=dict(time_grid=times.tolist(),time_unit="years",probability_measure=probability_measure,
                         conditioning="conditional_loss is E[LGD*EAD | first default in period]"),
        limitations=("Pass marginal, not cumulative PD. Discount factors are fixed per period; stochastic loss-discount dependence needs joint path integration.",))


def event_conditioned_loss(states,*,event_names,currency,probability_measure,context=None):
    """Integrate a fully enumerated exclusive Boolean event partition once."""
    ctx=_contract(currency,probability_measure,context)
    names=_ids(event_names,"event_names")
    if len(names)>12:_fail("INVALID_STATE_SPACE","Explicit partition supports at most twelve binary events")
    if not isinstance(states,list) or len(states)!=2**len(names):
        _fail("INCOMPLETE_PARTITION","Provide all exclusive Boolean event combinations")
    seen=set();ids=set();masses=[];contributions=[];details=[]
    for row in states:
        if not isinstance(row,dict) or not isinstance(row.get("state_id"),str) or not row["state_id"] or row["state_id"] in ids:
            _fail("INVALID_STATE_SPACE","Unique state IDs required")
        ids.add(row["state_id"]);events=row.get("events")
        if not isinstance(events,dict) or set(events)!=set(names) or any(type(x) is not bool for x in events.values()):
            _fail("INVALID_STATE_SPACE","Each state needs one Boolean per event")
        key=tuple(events[n] for n in names)
        if key in seen:_fail("OVERLAPPING_STATES","Repeated event combination")
        seen.add(key)
        p=_probability(row.get("probability"));loss=_numeric(row.get("conditional_loss"),"conditional_loss")
        if p.ndim or loss.ndim:_fail("INVALID_STATE_SPACE","State probability and joint conditional loss must be scalar")
        contribution=float(_multiply(p,loss));masses.append(float(p));contributions.append(contribution)
        details.append(dict(state_id=row["state_id"],events=dict(events),probability=float(p),conditional_loss=float(loss),contribution=contribution))
    total=_sum(masses)
    if abs(total-1)>1e-12:_fail("INVALID_PD_MASS","Exclusive state probabilities must sum to one")
    return _result(dict(expected_loss=_sum(contributions),states=details,probability_sum=total),
        "losses.event_conditioned_loss",currency,ctx,
        diagnostics=dict(probability_measure=probability_measure,event_names=names,complete_boolean_partition=True),
        limitations=("State losses are joint conditional monetary expectations, not products of arbitrary conditional means. Event meaning is supplied by the caller.",))


def counterparty_expected_loss(default_indicator,net_exposure_paths,recovery_paths,scenario_weights,discount_factors,
                              *,counterparty_id,netting_set_id,scenario_ids,time_grid,currency,probability_measure,uncertainty="none",context=None):
    """Joint loss on one declared netting set, with an explicit dependence comparator."""
    _ids([counterparty_id],"counterparty_id");_ids([netting_set_id],"netting_set_id")
    exposure=_numeric(net_exposure_paths,"net_exposure_paths");recovery=_probability(recovery_paths,name="recovery_paths")
    if exposure.shape!=recovery.shape:_fail("INCOMPATIBLE_SHAPE","Exposure and recovery paths must match")
    result=expected_loss_from_paths(default_indicator,1-recovery,np.maximum(exposure,0),scenario_weights,discount_factors,
        scenario_ids=scenario_ids,time_grid=time_grid,currency=currency,probability_measure=probability_measure,uncertainty=uncertainty,context=context)
    d,l,e,df,w,_,_,_=_paths(default_indicator,1-recovery,np.maximum(exposure,0),discount_factors,scenario_weights,scenario_ids,time_grid)
    independent=[]
    for j in range(d.shape[1]):
        independent.append(float(_multiply(_sum(w*d[:,j]),_sum(_multiply(w,l[:,j],e[:,j],df[:,j])))))
    benchmark=_sum(independent)
    return _result(dict(result.value,independent_default_exposure_loss=benchmark,
                        default_exposure_dependence_contribution=_sum((result.value["expected_loss"],-benchmark))),
        "losses.counterparty_expected_loss",currency,result.context,
        diagnostics=dict(result.diagnostics,counterparty_id=counterparty_id,netting_set_id=netting_set_id,
            negative_net_values_floored=int(np.sum(exposure<0)),interpretation="physical_loss_forecast" if probability_measure=="PHYSICAL" else "risk_neutral_discounted_credit_loss"),
        limitations=result.limitations+("One externally established netting set only; legal netting, margin closeout and own-default effects are not inferred. No capital or CVA framework compliance claim.",))


def aggregate_portfolio_loss(facility_loss_paths,portfolio_mapping,scenario_weights,fx_rates,*,scenario_ids,
                             reporting_currency,probability_measure,quantile_levels,context=None):
    """Aggregate aligned paths before computing weighted quantiles and tail means."""
    ctx=_contract(reporting_currency,probability_measure,context);ids=_ids(scenario_ids,"scenario_ids");w,residual=_weights(scenario_weights,len(ids))
    levels=_numeric(quantile_levels,"quantile_levels",vector_only=True)
    if np.any((levels<=0)|(levels>=1)) or len(np.unique(levels))!=len(levels):_fail("INVALID_QUANTILE","Distinct levels in (0,1) required")
    if not isinstance(facility_loss_paths,list) or not facility_loss_paths or not isinstance(portfolio_mapping,list) or not portfolio_mapping or not isinstance(fx_rates,dict):
        _fail("INVALID_INPUT","Facility rows, allocation rows and explicit FX object required")
    facilities={};allocations={};shares={}
    for row in facility_loss_paths:
        if not isinstance(row,dict):_fail("INVALID_INPUT","Facility loss row must be an object")
        identifier=_ids([row.get("facility_id")],"facility_id")[0]
        if identifier in facilities:_fail("INVALID_IDENTIFIERS","Repeated facility")
        local_ids=_ids(row.get("scenario_ids"),"facility scenario ids",len(ids))
        if set(local_ids)!=set(ids):_fail("SCENARIO_MISMATCH","Every facility must cover the same named scenarios")
        values=_numeric(row.get("losses"),"facility losses",vector_only=True)
        if len(values)!=len(ids):_fail("INCOMPATIBLE_SHAPE","Losses require one value per scenario")
        currency=row.get("currency")
        if not isinstance(currency,str) or not currency:_fail("INVALID_CURRENCY","Facility currency required")
        if currency not in fx_rates:_fail("MISSING_FX","Explicit rate required for every currency")
        fx=_numeric(fx_rates[currency],"fx_rate")
        if fx.ndim==0:fx=np.full(len(ids),float(fx))
        if fx.ndim!=1 or len(fx)!=len(ids) or np.any(fx<=0):_fail("INVALID_FX","Positive FX scalar or scenario vector required")
        if currency==reporting_currency and np.any(fx!=1):_fail("INVALID_FX","Reporting currency FX must equal one")
        positions={key:i for i,key in enumerate(local_ids)}
        facilities[identifier]=_multiply(np.array([values[positions[s]] for s in ids]),fx)
    for row in portfolio_mapping:
        if not isinstance(row,dict):_fail("INVALID_INPUT","Allocation row must be an object")
        fid=_ids([row.get("facility_id")],"facility_id")[0];pid=_ids([row.get("portfolio_id")],"portfolio_id")[0];share=_numeric(row.get("share"),"share")
        if fid not in facilities or share.ndim or not 0<=share<=1:_fail("INVALID_ALLOCATION","Known facility and fraction in [0,1] required")
        if fid in allocations.get(pid,{}):_fail("INVALID_ALLOCATION","Duplicate facility/portfolio allocation")
        allocations.setdefault(pid,{})[fid]=float(share);shares[fid]=shares.get(fid,0)+float(share)
    if any(v>1+1e-12 for v in shares.values()):_fail("DOUBLE_COUNTING","Allocated share exceeds one")
    results=[]
    for pid,mapping in sorted(allocations.items()):
        facility_ids=sorted(mapping);components=np.array([_multiply(facilities[f],mapping[f]) for f in facility_ids]).T
        paths=np.array([_sum(row) for row in components]);mean=_sum(w*paths)
        expected=[_sum(w*components[:,i]) for i in range(len(facility_ids))]
        active=w>0;ordered=np.argsort(paths[active],kind="stable");values=paths[active][ordered];cum=np.cumsum(w[active][ordered]);cum[-1]=1.
        tails=[]
        for level in levels:
            var=float(values[np.searchsorted(cum,level,side="left")]);above=paths>var;equal=paths==var
            tail_weights=np.where(above,w,0.);remaining=max(0.,1-float(level)-_sum(tail_weights));equal_mass=_sum(w[equal])
            if equal_mass:tail_weights[equal]=w[equal]*(remaining/equal_mass)
            tail_weights=tail_weights/(1-float(level))
            contributions=[_sum(tail_weights*components[:,i]) for i in range(len(facility_ids))]
            tails.append(dict(level=float(level),value_at_risk=var,expected_shortfall=_sum(tail_weights*paths),
                              facility_es_contributions=dict(zip(facility_ids,contributions))))
        results.append(dict(portfolio_id=pid,path_losses=paths.tolist(),expected_loss=mean,
                            facility_expected_loss=dict(zip(facility_ids,expected)),tail_metrics=tails))
    return _result(dict(portfolios=results,unallocated_shares={f:1-shares.get(f,0) for f in sorted(facilities)}),
        "losses.aggregate_portfolio_loss",reporting_currency,ctx,
        diagnostics=dict(scenario_ids=ids,scenario_weight_residual=residual,probability_measure=probability_measure,
            quantile_convention="left_inverse_weighted_cdf",expected_shortfall_convention="upper_(1-alpha)_mass_with_proportional_boundary_ties"),
        limitations=("Loss paths and FX must share the same scenario meaning. Tail contributions allocate portfolio tail losses, not sums of standalone facility VaR.",))
