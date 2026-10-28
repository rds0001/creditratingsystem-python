"""Structural credit risk under declared asset dynamics and probability measures."""
from collections.abc import Mapping
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr, log_ndtr
from .core import _numeric, _fail, _result, _context_value, _fingerprint


def _scalar(x, name, lower=None, strict=False):
    a = _numeric(x, name)
    if a.ndim or (lower is not None and (a <= lower if strict else a < lower)):
        _fail('INVALID_INPUT', f'{name} must be a scalar in its declared domain')
    return float(a)


def _measure(measure, horizon, context):
    if measure not in ('PHYSICAL', 'RISK_NEUTRAL'):
        _fail('INCOMPATIBLE_MEASURE', 'Declare PHYSICAL or RISK_NEUTRAL')
    ctx = _context_value(context)
    for key, value in [('probability_measure',measure),('time_unit','years'),('horizon',horizon)]:
        if ctx.get(key) is not None and ctx[key] != value:
            _fail('TARGET_MISMATCH', f'Context {key} conflicts with structural contract')
    return ctx


def _args(asset_value,barrier,volatility,drift,horizon,payout):
    values = (_scalar(asset_value,'asset_value',0,True),_scalar(barrier,'barrier',0),
            _scalar(volatility,'volatility',0),_scalar(drift,'drift'),
            _scalar(horizon,'horizon',0),_scalar(payout,'payout'))
    a,b,s,mu,t,q = values
    if t > 0 and (not all(math.isfinite(v) for v in (s*s*t,mu*t,q*t)) or (s > 0 and s*s == 0)):
        _fail('NUMERICAL_RANGE','Derived variance or drift outside supported floating-point range')
    return values


def _equity(a,d,s,r,t,q):
    ad = a*math.exp(-q*t); dd = d*math.exp(-r*t)
    if not math.isfinite(ad) or not math.isfinite(dd) or ad == 0 or (d > 0 and dd == 0):
        _fail('NUMERICAL_OVERFLOW', 'Discounted values exceed finite range')
    if d == 0:
        return dict(equity_value=ad,delta=math.exp(-q*t),d1=None,d2=None)
    if t == 0 or s == 0:
        return dict(equity_value=max(ad-dd,0.),delta=math.exp(-q*t)*(1. if ad>dd else 0. if ad<dd else .5),d1=None,d2=None)
    v=s*math.sqrt(t); d1=(math.log(a)-math.log(d)+(r-q+.5*s*s)*t)/v; d2=d1-v
    # Compute the call from log terms, avoiding subtraction of tiny close values.
    l1=math.log(ad)+float(log_ndtr(d1)); l2=math.log(dd)+float(log_ndtr(d2))
    value=math.exp(l1)*(-math.expm1(min(0.,l2-l1)))
    return dict(equity_value=value,delta=math.exp(-q*t)*float(ndtr(d1)),d1=d1,d2=d2)


def merton_equity_value(asset_value,debt_face,asset_volatility,risk_free_rate,maturity,payout=0,*,currency,context=None):
    """Value terminal equity as a European call under Q; annual rates and years."""
    a,d,s,r,t,q=_args(asset_value,debt_face,asset_volatility,risk_free_rate,maturity,payout)
    if not isinstance(currency,str) or not currency.strip(): _fail('INVALID_UNIT','Explicit common currency required')
    ctx=_measure('RISK_NEUTRAL',t,context)
    try: value=_equity(a,d,s,r,t,q)
    except (OverflowError,ValueError): _fail('NUMERICAL_OVERFLOW','Valuation exceeds numerical domain')
    return _result(value,'structural.merton_equity_value',currency,ctx,diagnostics={'probability_measure':'RISK_NEUTRAL','time_unit':'years','maturity':t,'degenerate_delta_convention':'one_half_at_kink'},limitations=('Single terminal debt face and constant geometric Brownian asset dynamics; payout is a continuous yield.',))


def infer_merton_assets(equity_value,equity_volatility,debt_face,risk_free_rate,maturity,*,asset_bounds,volatility_bounds,currency,payout=0,tolerance=1e-9,max_iterations=1000,context=None):
    """Invert equity price and equity volatility jointly within explicit bounds."""
    e=_scalar(equity_value,'equity_value',0,True); se=_scalar(equity_volatility,'equity_volatility',0,True)
    d=_scalar(debt_face,'debt_face',0); r=_scalar(risk_free_rate,'risk_free_rate'); t=_scalar(maturity,'maturity',0,True); q=_scalar(payout,'payout')
    tol=_scalar(tolerance,'tolerance',0,True); limit=_scalar(max_iterations,'max_iterations',1)
    if tol>=.01 or limit!=int(limit): _fail('INVALID_INPUT','Invalid solver controls')
    bounds=[]
    for values,name in [(asset_bounds,'asset_bounds'),(volatility_bounds,'volatility_bounds')]:
        b=_numeric(values,name,vector_only=True)
        if b.size!=2 or b[0]<=0 or b[1]<=b[0]: _fail('INVALID_BOUNDS','Strict positive lower and larger upper bound required')
        bounds.append(b)
    if not isinstance(currency,str) or not currency.strip(): _fail('INVALID_UNIT','Explicit currency required')
    ctx=_measure('RISK_NEUTRAL',t,context)
    lo=np.log([b[0] for b in bounds]); hi=np.log([b[1] for b in bounds])
    def residual(x):
        a,s=np.exp(x); model=_equity(a,d,s,r,t,q)
        return np.array([(model['equity_value']-e)/e,(a*model['delta']*s-e*se)/(e*se)])
    initial=np.array([e+d*math.exp(-r*t),se*e/(e+d*math.exp(-r*t))])
    start=np.minimum(np.maximum(np.log(initial),lo+1e-8*(hi-lo)),hi-1e-8*(hi-lo))
    fitted=least_squares(residual,start,bounds=(lo,hi),xtol=max(tol*.01,1e-14),ftol=max(tol*.01,1e-14),gtol=max(tol*.01,1e-14),max_nfev=int(limit))
    error=residual(fitted.x)
    if not fitted.success or max(abs(error))>tol:
        _fail('NONCONVERGENCE','No in-bounds solution satisfies both pricing and volatility equations',{'scaled_residuals':error.tolist(),'solver_message':fitted.message})
    a,s=np.exp(fitted.x)
    return _result({'asset_value':float(a),'asset_volatility':float(s)},'structural.infer_merton_assets','asset_parameters',ctx,
        diagnostics={'scaled_residuals':error.tolist(),'converged':True,'evaluations':fitted.nfev,'asset_bounds':list(map(float,asset_bounds)),'volatility_bounds':list(map(float,volatility_bounds)),'currency':currency,'probability_measure':'RISK_NEUTRAL','time_unit':'years','maturity':t},
        limitations=('Numerical inversion of a specified equity option model does not identify the physical asset drift or validate its debt proxy.',))


def distance_to_default(asset_value,default_barrier,asset_volatility,physical_drift,horizon,payout=0,*,context=None):
    """Physical standardized terminal log-distance, with explicit degenerate cases."""
    a,b,s,mu,t,q=_args(asset_value,default_barrier,asset_volatility,physical_drift,horizon,payout)
    ctx=_measure('PHYSICAL',t,context)
    logdistance=None if b==0 else math.log(a)-math.log(b)+(mu-q-.5*s*s)*t
    sd=s*math.sqrt(t)
    dd=logdistance/sd if b>0 and sd>0 else None
    status='FINITE' if dd is not None else 'NO_POSITIVE_BARRIER' if b==0 else 'DETERMINISTIC_AT_BARRIER' if logdistance==0 else 'DETERMINISTIC_ABOVE' if logdistance>0 else 'DETERMINISTIC_BELOW'
    return _result({'distance_to_default':dd,'log_distance':logdistance,'log_standard_deviation':sd,'boundary_status':status},'structural.distance_to_default','standard_deviations',ctx,diagnostics={'probability_measure':'PHYSICAL','time_unit':'years','horizon':t,'event':'terminal_asset_at_or_below_barrier'},limitations=('No empirical default-frequency mapping or automatic TTC interpretation.',))


def merton_default_probability(asset_value,debt_face,asset_volatility,drift_or_rate,horizon,*,measure,payout=0,context=None):
    """Probability of assets at or below terminal debt, under explicit P or Q drift."""
    a,d,s,mu,t,q=_args(asset_value,debt_face,asset_volatility,drift_or_rate,horizon,payout)
    ctx=_measure(measure,t,context); dd=None
    if d==0: pd=0.
    else:
        x=math.log(a)-math.log(d)+(mu-q-.5*s*s)*t
        if t==0 or s==0: pd=float(x<=0)
        else: dd=x/(s*math.sqrt(t)); pd=float(ndtr(-dd))
    return _result(pd,'structural.merton_default_probability','probability',ctx,diagnostics={'distance_statistic':dd,'probability_measure':measure,'time_unit':'years','horizon':t,'event':'terminal_asset_at_or_below_debt','drift_interpretation':'physical_expected_asset_return' if measure=='PHYSICAL' else 'risk_free_rate'},limitations=('Risk-neutral PD is a pricing-model quantity, not a historical default forecast.',))


def first_passage_pd(asset_value,barrier_spec,volatility,drift,horizon,*,measure,payout=0,context=None):
    """Continuous GBM first passage to a constant or exponentially growing barrier."""
    if not isinstance(barrier_spec,Mapping) or barrier_spec.get('kind') not in ('constant','exponential'):
        _fail('INVALID_BARRIER','Declare constant or exponential barrier')
    growth=0. if barrier_spec['kind']=='constant' else _scalar(barrier_spec.get('growth_rate'),'growth_rate')
    if barrier_spec['kind']=='constant' and barrier_spec.get('growth_rate',0)!=0: _fail('INVALID_BARRIER','Constant barrier cannot have growth')
    a,b,s,mu,t,q=_args(asset_value,barrier_spec.get('initial_level'),volatility,drift,horizon,payout)
    ctx=_measure(measure,t,context)
    if b==0: pd=0.
    elif a<=b: pd=1.
    elif t==0: pd=0.
    else:
        x=math.log(a)-math.log(b); m=mu-q-.5*s*s-growth
        if s==0: pd=float(x+min(0.,m*t)<=0)
        else:
            v=s*math.sqrt(t)
            loga=float(log_ndtr((-x-m*t)/v)); logb=-2*m*x/(s*s)+float(log_ndtr((-x+m*t)/v))
            logp=float(np.logaddexp(loga,logb))
            if math.isnan(logp): _fail('NUMERICAL_RANGE','First-passage tail outside numerical domain')
            pd=float(math.exp(min(0.,logp)))
    return _result(pd,'structural.first_passage_pd','probability',ctx,diagnostics={'barrier_spec':dict(barrier_spec),'probability_measure':measure,'time_unit':'years','horizon':t,'monitoring':'continuous','event':'asset_at_or_below_barrier_any_time_including_start'},limitations=('Constant GBM dynamics and a constant or exponential barrier are required; this is not terminal-only default.',))


def simulate_structural_paths(asset_dynamics,barrier_schedule,time_grid,innovations,*,measure,jump_counts=None,jump_innovations=None,bridge_uniforms=None,context=None):
    """Exact grid GBM/lognormal-jump steps using supplied independent innovations.

    Rows are paths, columns intervals. Jump innovations are standard normals for
    the aggregate log jump conditional on the supplied Poisson count. Brownian
    bridges monitor continuous diffusion against log-linearly interpolated barriers.
    """
    if not isinstance(asset_dynamics,Mapping): _fail('INVALID_INPUT','asset_dynamics object required')
    times=_numeric(time_grid,'time_grid',vector_only=True)
    if times.size<2 or times[0]!=0 or np.any(np.diff(times)<=0): _fail('INVALID_TIME_GRID','Increasing years starting at zero required')
    a=_scalar(asset_dynamics.get('asset_value'),'asset_value',0,True)
    s=_scalar(asset_dynamics.get('volatility'),'volatility',0)
    mu=_scalar(asset_dynamics.get('drift'),'drift'); q=_scalar(asset_dynamics.get('payout',0),'payout')
    intensity=_scalar(asset_dynamics.get('jump_intensity',0),'jump_intensity',0)
    jm=_scalar(asset_dynamics.get('jump_log_mean',0),'jump_log_mean'); js=_scalar(asset_dynamics.get('jump_log_sd',0),'jump_log_sd',0)
    ctx=_measure(measure,float(times[-1]),context)
    z=_numeric(innovations,'innovations')
    if z.ndim!=2 or z.shape[0]<1 or z.shape[1]!=times.size-1: _fail('INCOMPATIBLE_SHAPE','Innovations require paths by intervals matrix')
    b=_numeric(barrier_schedule,'barrier_schedule',vector_only=True)
    if b.size!=times.size or np.any(b<0): _fail('INVALID_BARRIER','Nonnegative barrier level required at every grid point')
    if (jump_counts is None)!=(jump_innovations is None): _fail('MISSING_INPUT','Provide both jump counts and jump innovations')
    if intensity>0 and jump_counts is None: _fail('MISSING_INPUT','Positive intensity needs explicit jump draws')
    counts=np.zeros_like(z) if jump_counts is None else _numeric(jump_counts,'jump_counts')
    jz=np.zeros_like(z) if jump_innovations is None else _numeric(jump_innovations,'jump_innovations')
    if counts.shape!=z.shape or jz.shape!=z.shape: _fail('INCOMPATIBLE_SHAPE','Jump draws must match diffusion matrix')
    if np.any(counts<0) or np.any(counts!=np.floor(counts)) or (intensity==0 and np.any(counts!=0)): _fail('INVALID_JUMPS','Counts must be nonnegative integers; zero intensity requires zero counts')
    bridge=None
    if bridge_uniforms is not None:
        bridge=_numeric(bridge_uniforms,'bridge_uniforms')
        if bridge.shape!=z.shape or np.any((bridge<=0)|(bridge>=1)): _fail('INVALID_BRIDGE','Bridge draws must match shape and lie in (0,1)')
        if intensity>0 or np.any(b<=0): _fail('INVALID_BRIDGE','Continuous bridge supports positive barriers and pure diffusion only')
    dt=np.diff(times)
    try: compensation=0. if intensity == 0 else intensity*math.expm1(jm+.5*js*js)
    except OverflowError: _fail('NUMERICAL_OVERFLOW','Jump compensation overflow')
    increment=(mu-q-compensation-.5*s*s)*dt+s*np.sqrt(dt)*z+counts*jm+np.sqrt(counts)*js*jz
    logs=np.column_stack([np.full(z.shape[0],math.log(a)),math.log(a)+np.cumsum(increment,axis=1)])
    with np.errstate(over='ignore',under='ignore'): paths=np.exp(logs)
    paths[:,0] = a
    if np.any(~np.isfinite(paths)) or np.any(paths==0): _fail('NUMERICAL_OVERFLOW','Simulated asset path outside representable positive range')
    hits=paths<=b
    if bridge is not None and s>0:
        x=logs[:,:-1]-np.log(b[:-1]); y=logs[:,1:]-np.log(b[1:])
        probability=np.exp(-2*np.maximum(x,0)*np.maximum(y,0)/(s*s*dt))
        hits[:,1:] |= bridge<probability
    defaulted=np.maximum.accumulate(hits,axis=1)
    first=np.argmax(defaulted,axis=1)
    defaults=[float(times[index]) if defaulted[i,-1] else None for i,index in enumerate(first)]
    survival=np.mean(~defaulted,axis=0); n=z.shape[0]
    se=np.sqrt(survival*(1-survival)/n)
    return _result({'asset_paths':paths,'default_time_upper_bounds':defaults,'survival_curve':survival,'survival_standard_error':se},'structural.simulate_structural_paths','paths',ctx,
        diagnostics={'probability_measure':measure,'time_unit':'years','time_grid':times.tolist(),'n_paths':n,'monitoring':'continuous_diffusion_bridge' if bridge is not None else 'grid_only','jump_law':'compound_poisson_lognormal','jump_drift_compensation':compensation,'innovations_fingerprint':_fingerprint({'normal':z.tolist(),'counts':counts.tolist(),'jump_normal':jz.tolist(),'bridge_uniform':None if bridge is None else bridge.tolist()}),'barrier_miss_between_grid_points':bridge is None,'default_times_are_interval_upper_bounds':True,'iid_mc_assumption_verified':False},
        limitations=('Supplied diffusion, jump-count, jump-normal and bridge draws must have the declared independent laws. Standard errors assume independent paths.',
                     'Jump path crossings between observation times are not observed; bridge correction is only supported without jumps. Exact within-interval hitting times are not returned.'))


def calibrate_distance_pd_mapping(distances,outcomes,*,metadata,weights=None,evaluation_distances=None,bootstrap_indices=None,confidence_level=.95,out_of_bounds='error',context=None):
    """Fit a decreasing empirical DD map, optionally bootstrap on supplied resamples.

    Bootstrap row indices are zero-based in both languages. Frequency weights are
    supported for fitting; bootstrap requires unit weights and IID row resampling.
    Quantile bands are pointwise empirical bootstrap summaries, not guaranteed
    confidence coverage for this nonregular shape-constrained estimator.
    """
    from .calibration import calibrate_isotonic, apply_calibrator
    if not isinstance(metadata,Mapping) or metadata.get('target',{}).get('probability_measure')!='PHYSICAL':
        _fail('INCOMPATIBLE_MEASURE','Observed default mapping requires a PHYSICAL target')
    c=_scalar(confidence_level,'confidence_level',0,True)
    if c>=1: _fail('INVALID_INPUT','confidence_level must lie in (0,1)')
    fitted=calibrate_isotonic(distances,outcomes,weights,metadata=metadata,increasing=False,out_of_bounds=out_of_bounds,context=context)
    x=_numeric(distances,'distances',vector_only=True); y=_numeric(outcomes,'outcomes',vector_only=True)
    evaluation=x if evaluation_distances is None else _numeric(evaluation_distances,'evaluation_distances',vector_only=True)
    predicted=apply_calibrator(fitted,evaluation,target=metadata['target'],base_model_version=metadata['base_model_version']).value
    lower=upper=None; count=0
    if bootstrap_indices is not None:
        if weights is not None and np.any(_numeric(weights,'weights')!=1): _fail('INVALID_BOOTSTRAP','Bootstrap requires unit observation weights')
        indices=_numeric(bootstrap_indices,'bootstrap_indices')
        if indices.ndim!=2 or indices.shape[0]<2 or indices.shape[1]!=len(x) or np.any(indices!=np.floor(indices)) or np.any((indices<0)|(indices>=len(x))): _fail('INVALID_BOOTSTRAP','At least two complete zero-based row resamples required')
        samples=[]
        for index in indices.astype(int):
            boot=calibrate_isotonic(x[index],y[index],metadata=metadata,increasing=False,out_of_bounds='clip',context=context)
            samples.append(apply_calibrator(boot,evaluation,target=metadata['target'],base_model_version=metadata['base_model_version']).value)
        samples=np.asarray(samples); lower=np.quantile(samples,(1-c)/2,axis=0); upper=np.quantile(samples,(1+c)/2,axis=0); count=len(samples)
    return _result({'calibration_artifact':fitted.value,'evaluation_distances':evaluation,'pd':predicted,'lower':lower,'upper':upper},'structural.calibrate_distance_pd_mapping','empirical_pd_mapping',fitted.context,
        diagnostics={'monotonicity':'decreasing','bootstrap_samples':count,'confidence_level':c,'bootstrap_interpolation':'linear','bootstrap_boundary_policy':'clip','out_of_sample_validation_performed':False,'uncertainty':'none' if count==0 else 'pointwise_iid_percentile_bootstrap'},provenance=fitted.provenance,
        limitations=fitted.limitations+('Bootstrap quantile bands do not guarantee coverage for data-adaptive isotonic blocks. An independent time-separated test set remains necessary.',))
