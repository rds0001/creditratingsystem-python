"""Credit portfolio distributions with explicit scenario and atom conventions."""
import math
from collections import defaultdict
import numpy as np
from scipy.special import ndtr,ndtri
from .core import _fail,_result,_numeric,_context_value,_fingerprint
from .data import _rows,_required


def _sum(values):
    try:value=math.fsum(values)
    except OverflowError:_fail("NUMERICAL_FAILURE","Portfolio sum exceeds finite representation")
    if not math.isfinite(value):_fail("NUMERICAL_FAILURE","Portfolio sum exceeds finite representation")
    return value


def _ids(values,name):
    if not isinstance(values,(list,tuple))or not values or any(not isinstance(x,str)or not x for x in values)or len(set(values))!=len(values):_fail('PORTFOLIO_IDS',name+' must be nonempty distinct strings')
    return list(values)


def _weights(weights,n):
    w=_numeric(weights,'weights',vector_only=True)
    if w.shape!=(n,)or np.any(w<0)or not np.any(w>0):_fail('PORTFOLIO_WEIGHTS','Aligned nonnegative weights with positive mass required')
    original=w.copy();w=w/float(w.max());w=w/_sum(w)
    if np.any((original>0)&(w==0)):_fail('NUMERICAL_FAILURE','Positive scenario weight underflowed')
    return w


def _mean(x,w):
    scale=float(np.max(np.abs(x)))
    value=0. if scale==0 else scale*_sum((x/scale)*w)
    if not math.isfinite(value):_fail('NUMERICAL_FAILURE','Portfolio mean exceeds finite representation')
    return value


def _atoms(loss,w):
    order=np.flatnonzero(w>0);order=order[np.argsort(loss[order],kind="stable")];x=loss[order];starts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1]
    return x[starts],np.add.reduceat(w[order],starts)


def _tail(loss,w,level):
    level=float(level)
    if not math.isfinite(level)or not 0<=level<=1:_fail('PORTFOLIO_LEVEL','Tail level must be in [0,1]')
    support,masses=_atoms(loss,w);cdf=np.cumsum(masses);cdf[-1]=1
    index=min(int(np.searchsorted(cdf,level,side='left')),len(support)-1);var=float(support[index]);tail=np.zeros(len(loss));equal=loss==var
    if level==1:tail[equal]=w[equal]/_sum(w[equal])
    else:
        above=loss>var;tail[above]=w[above];remaining=max(0.,1-level-_sum(tail));tail[equal]=w[equal]*(remaining/_sum(w[equal]));tail/=1-level
    return var,_mean(loss,tail),tail


def credit_loss_distribution(loss_paths,weights,*,scenario_ids,levels=(.95,.99),currency,context=None):
    """Weighted empirical/exact loss distribution, left quantiles and upper-mass ES."""
    ids=_ids(scenario_ids,'scenario_ids');loss=_numeric(loss_paths,'loss_paths',vector_only=True)
    if loss.shape!=(len(ids),):_fail('PORTFOLIO_SHAPE','One loss per scenario required')
    w=_weights(weights,len(ids));ctx=_context_value(context)
    if not isinstance(currency,str)or not currency or ctx.get('currency')not in (None,currency):_fail('TARGET_MISMATCH','Explicit matching loss currency required')
    levels=_numeric(levels,'levels',vector_only=True)
    if len(set(levels.tolist()))!=len(levels):_fail('PORTFOLIO_LEVEL','Distinct levels required')
    tails=[]
    for level in levels:
        var,es,tw=_tail(loss,w,level);tails.append(dict(level=float(level),value_at_risk=var,expected_shortfall=es,normalized_tail_weights=tw.tolist()))
    mean=_mean(loss,w);scale=float(np.max(np.abs(loss)));sd=0. if scale==0 else scale*math.sqrt(_sum(w*(loss/scale-mean/scale)**2))
    if not math.isfinite(sd):_fail('NUMERICAL_FAILURE','Portfolio standard deviation exceeds finite representation')
    support,masses=_atoms(loss,w);atoms=[dict(loss=float(x),probability=float(m))for x,m in zip(support,masses)]
    return _result(dict(scenario_ids=ids,loss_paths=loss.tolist(),normalized_weights=w.tolist(),currency=currency,expected_loss=mean,standard_deviation=sd,atoms=atoms,tails=tails,quantile_convention='LEFT_INVERSE_CDF',tail_convention='UPPER_MASS_PROPORTIONAL_BOUNDARY_TIES_LEVEL_ONE_LIMIT'),'portfolio.credit_loss_distribution',currency,ctx,diagnostics=dict(paths_hash=_fingerprint(dict(scenario_ids=ids,loss_paths=loss.tolist(),weights=w.tolist()))),limitations=('A supplied weighted distribution is not proof of independent sampling. No confidence interval is inferred. Signed losses are retained. ES takes exactly the upper 1-alpha probability mass, allocating the boundary atom proportionally; alpha=1 is the maximum-loss limit.',))


def allocate_expected_shortfall(position_loss_paths,weights,*,scenario_ids,position_ids,level,currency,groups=None,context=None):
    """Allocate exact discrete-tail expected loss using the same portfolio tail mass."""
    ids=_ids(position_ids,'position_ids');scenarios=_ids(scenario_ids,'scenario_ids');x=_numeric(position_loss_paths,'position_loss_paths')
    if x.shape!=(len(scenarios),len(ids)):_fail('PORTFOLIO_SHAPE','Scenario-by-position monetary loss matrix required')
    total=np.array([_sum(row)for row in x]);distribution=credit_loss_distribution(total,weights,scenario_ids=scenarios,levels=[level],currency=currency,context=context).value;tail=distribution['tails'][0];tw=np.asarray(tail['normalized_tail_weights']);contributions=[_mean(x[:,j],tw)for j in range(len(ids))]
    group_results=[]
    if groups is not None:
        if not isinstance(groups,dict)or set(groups)!=set(ids)or any(not isinstance(v,str)or not v for v in groups.values()):_fail('PORTFOLIO_GROUP','Exactly one explicit group per position required')
        group_results=[dict(group_id=g,contribution=_sum(contributions[j]for j,p in enumerate(ids)if groups[p]==g))for g in sorted(set(groups.values()))]
    residual=_sum(contributions)-tail['expected_shortfall']
    if not math.isclose(residual,0,rel_tol=0,abs_tol=1e-12*max(1,max(abs(v)for v in contributions))):_fail('NUMERICAL_FAILURE','ES allocation does not reconcile')
    return _result(dict(level=float(level),value_at_risk=tail['value_at_risk'],expected_shortfall=tail['expected_shortfall'],scenario_ids=scenarios,normalized_tail_weights=tail['normalized_tail_weights'],contributions=[dict(position_id=p,contribution=contributions[j])for j,p in enumerate(ids)],groups=group_results,reconciliation_residual=residual),'portfolio.allocate_expected_shortfall',currency,context,diagnostics=dict(position_paths_hash=_fingerprint(dict(position_ids=ids,scenario_ids=scenarios,values=x.tolist()))),limitations=('Conditional loss allocation under the declared proportional boundary-tie convention, not a unique derivative at a discrete VaR atom. Signed hedge contributions are retained. All positions use the same scenario weights and portfolio tail event.',))


def incremental_credit_risk(base_loss_paths,change_loss_paths,weights,*,scenario_ids,levels=(.95,.99),currency,change_id,context=None):
    """Compare paired base and changed monetary loss paths without resampling."""
    base=_numeric(base_loss_paths,'base_loss_paths',vector_only=True);change=_numeric(change_loss_paths,'change_loss_paths',vector_only=True)
    if base.shape!=change.shape or not isinstance(change_id,str)or not change_id:_fail('PORTFOLIO_CHANGE','Aligned paired change paths and explicit change ID required')
    before=credit_loss_distribution(base,weights,scenario_ids=scenario_ids,levels=levels,currency=currency,context=context).value;after=credit_loss_distribution(base+change,weights,scenario_ids=scenario_ids,levels=levels,currency=currency,context=context).value
    return _result(dict(change_id=change_id,base=before,changed=after,expected_loss_change=after['expected_loss']-before['expected_loss'],tail_changes=[dict(level=b['level'],value_at_risk_change=a['value_at_risk']-b['value_at_risk'],expected_shortfall_change=a['expected_shortfall']-b['expected_shortfall'])for b,a in zip(before['tails'],after['tails'])]),'portfolio.incremental_credit_risk',currency,context,diagnostics=dict(change_hash=_fingerprint(change.tolist())),limitations=('Changes are paired finite differences on fixed scenarios, not marginal derivatives or causal effects. Addition, removal and scaling are represented by explicit signed monetary changes; no new random worlds are generated.',))


def credit_concentration_measures(positions,*,weight_field,group_fields,weight_unit,context=None):
    """HHI, entropy and effective counts on an explicit nonnegative risk denominator."""
    rows=_rows(positions,'positions');fields=_ids(group_fields,'group_fields');ids=[];amounts=[]
    if not isinstance(weight_unit,str)or not weight_unit:_fail('PORTFOLIO_UNIT','Explicit exposure/risk weight unit required')
    for row in rows:
        _required(row,['position_id',weight_field]+fields)
        if row['position_id']in ids:_fail('PORTFOLIO_IDS','Duplicate position ID')
        ids.append(row['position_id']);v=_numeric(row[weight_field],weight_field)
        if v.size!=1 or float(v)<0:_fail('PORTFOLIO_WEIGHT','Finite nonnegative scalar concentration weight required')
        if any(not isinstance(row[f],str)or not row[f]for f in fields):_fail('PORTFOLIO_GROUP','Explicit nonmissing group memberships required')
        amounts.append(float(v))
    total=_sum(amounts);reports=[]
    for field in ['position_id']+fields:
        collected=defaultdict(list)
        for row,v in zip(rows,amounts):collected[row[field]].append(v)
        names=sorted(collected);mass={g:_sum(collected[g])for g in names};shares={g:v/total if total>0 else 0. for g,v in mass.items()};hhi=_sum(v*v for v in shares.values())if total>0 else None;entropy=-_sum(v*math.log(v)for v in shares.values()if v>0)if total>0 else None
        reports.append(dict(dimension=field,groups=[dict(group_id=g,weight=mass[g],share=shares[g])for g in names],hhi=hhi,effective_number=None if hhi is None else 1/hhi,entropy=entropy,entropy_effective_number=None if entropy is None else math.exp(entropy),largest_share=max(shares.values(),default=0)))
    return _result(dict(status='ESTIMABLE'if total>0 else'ZERO_WEIGHT_DENOMINATOR',weight_field=weight_field,weight_unit=weight_unit,total_weight=total,dimensions=reports),'portfolio.credit_concentration_measures','concentration',context,diagnostics=dict(position_hash=_fingerprint(rows)),limitations=('Descriptive concentration on the explicitly supplied nonnegative denominator. Group definitions are caller-supplied, not a legal connected-client determination. HHI and effective counts are not tail loss or regulatory large-exposure limit calculations.',))


def asymptotic_factor_loss(pd,lgd,asset_correlation,systematic_factor,*,context=None):
    """Homogeneous infinitesimal-exposure Gaussian one-factor conditional loss rate."""
    values=[]
    for v,name in [(pd,'pd'),(lgd,'lgd'),(asset_correlation,'asset_correlation')]:
        a=_numeric(v,name)
        if a.size!=1:_fail('PORTFOLIO_ASRF','Homogeneous scalar parameters required')
        values.append(float(a))
    ctx=_context_value(context)
    if ctx.get('probability_measure')not in (None,'PHYSICAL'):_fail('TARGET_MISMATCH','ASRF benchmark requires physical probabilities')
    p,l,rho=values;y=_numeric(systematic_factor,'systematic_factor',vector_only=True)
    if not 0<=p<=1 or not 0<=rho<1:_fail('PORTFOLIO_ASRF','PD in [0,1] and asset correlation in [0,1) required')
    conditional=np.full(len(y),p)if p in (0,1)or rho==0 else ndtr((ndtri(p)-math.sqrt(rho)*y)/math.sqrt(1-rho));loss=conditional*l
    return _result(dict(pd=p,lgd=l,asset_correlation=rho,systematic_factor=y.tolist(),conditional_pd=conditional.tolist(),conditional_loss_rate=loss.tolist(),unconditional_expected_loss_rate=p*l),'portfolio.asymptotic_factor_loss','loss_rate',context,diagnostics=dict(factor_hash=_fingerprint(y.tolist())),limitations=('Physical Gaussian one-factor benchmark with identical PD, fixed LGD and infinitesimal exposures; adverse factor is negative. Asset correlation is latent asset correlation, not default correlation. Not a finite concentrated portfolio, multifactor model or regulatory capital calculation.',))

from ._portfolio_simulation import prepare_credit_portfolio, simulate_credit_portfolio, simulate_portfolio_migrations, stress_portfolio_factor_structure
