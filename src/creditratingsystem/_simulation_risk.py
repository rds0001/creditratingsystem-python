"""Sampling-aware default frequency and joint monetary-loss estimates."""
import math
from copy import deepcopy
import numpy as np
from scipy.stats import beta
from .core import _fail,_result,_context_value,_fingerprint,_numeric,CalcResult,CreditRiskError
from .data import _rows,_required
from .simulation import _number,_text


def estimate_simulation_risk(path_records,sampling_contract,*,confidence_level=.95,context=None):
    """Estimate risk for one initially-performing subject and declared sampling law."""
    rows=_rows(path_records,'path_records');contract=deepcopy(sampling_contract)
    if not isinstance(contract,dict):_fail('SIMULATION_SAMPLING','Explicit sampling contract required')
    _required(contract,['contract_id','version','subject_id','currency','probability_measure','horizon','time_unit','conditioning','sampling_design','event'])
    for key in ('contract_id','version','subject_id','currency','probability_measure','time_unit','conditioning','sampling_design','event'):_text(contract[key],key)
    horizon=_number(contract['horizon'],'horizon',0);level=_number(confidence_level,'confidence_level',0)
    if horizon<=0 or not 0<level<1 or contract['time_unit']not in ('days','months','years')or contract['probability_measure']not in ('PHYSICAL','RISK_NEUTRAL')or contract['conditioning']!='initially_performing'or contract['event']!='first_default':_fail('SIMULATION_SAMPLING','Declare positive horizon, common probability measure and initially_performing target')
    design=contract['sampling_design']
    if design not in ('IID_UNWEIGHTED','FIXED_WEIGHTED_SCENARIOS','DEPENDENT_OR_QMC'):_fail('SIMULATION_SAMPLING','Unsupported sampling design; importance sampling requires its own density-ratio estimator')
    ctx=_context_value(context)
    for key in ('currency','probability_measure','horizon','time_unit','conditioning'):
        if ctx.get(key)is not None and ctx[key]!=contract[key]:_fail('TARGET_MISMATCH','Sampling target conflicts with context '+key)
    seen=set();active=[];excluded=[]
    for row in rows:
        _required(row,['scenario_id','subject_id','initial_default','defaulted','horizon_complete','weight','ead_at_default','economic_loss'])
        if row['scenario_id']in seen:_fail('SIMULATION_DUPLICATE','One independent sampling unit per unique scenario required')
        seen.add(row['scenario_id'])
        if row['subject_id']!=contract['subject_id']:_fail('SIMULATION_SAMPLING','One subject per risk estimate; correlated obligors are not extra Monte Carlo trials')
        for key in ('initial_default','defaulted','horizon_complete'):
            if not isinstance(row[key],bool):_fail('SIMULATION_SAMPLING',key+' must be Boolean')
        weight=_number(row['weight'],'weight',0);ead=_number(row['ead_at_default'],'ead_at_default',0);loss=_number(row['economic_loss'],'economic_loss')
        if design=='IID_UNWEIGHTED'and weight!=1:_fail('SIMULATION_SAMPLING','IID exact-binomial calculations require unit trial weights')
        if row['initial_default']and not row['defaulted']:_fail('SIMULATION_SAMPLING','Initial default cannot become a nondefault first-event path')
        if not row['defaulted']and(ead!=0 or loss!=0):_fail('SIMULATION_SAMPLING','Nondefault paths have zero default-conditioned EAD and loss')
        if row['initial_default']or weight==0:
            excluded.append(dict(scenario_id=row['scenario_id'],reason='INITIAL_DEFAULT'if row['initial_default']else'ZERO_WEIGHT'));continue
        if not row['horizon_complete']:_fail('SIMULATION_CENSORING','Positive-weight performing trials require complete target horizon; do not discard informative censoring')
        active.append((weight,int(row['defaulted']),ead,loss))
    value=dict(subject_id=contract['subject_id'],currency=contract['currency'],target=contract,sampling_design=design,input_trials=len(rows),risk_set_trials=len(active),excluded=excluded,confidence_level=level)
    if not active:
        value.update(status='EMPTY_RISK_SET',pd=None,pd_interval=None,expected_loss=None,conditional_mean_loss=None,conditional_mean_ead=None,conditional_mean_lgd=None,ead_weighted_lgd=None,effective_weight_count=None,pd_mc_standard_error=None,loss_mc_standard_error=None)
    else:
        # Binary64 products followed by compensated accumulation avoid BLAS/FMA
        # reduction differences in content-sealed simulation assessments.
        a=np.asarray(active,dtype=float);w=a[:,0];w=w/float(np.max(w));w=w/math.fsum(w);
        if np.any(w==0):_fail('NUMERICAL_FAILURE','Positive scenario weight underflowed during normalization')
        default=a[:,1];ead=a[:,2];loss=a[:,3];pd=math.fsum(w[default==1])/math.fsum(w);el=math.fsum(w*loss);k=int(default.sum());n=len(active);mass=w*default;dm=math.fsum(mass)
        if not math.isfinite(el):_fail('NUMERICAL_FAILURE','Weighted monetary loss exceeds finite representation')
        conditional_loss=None if dm==0 else math.fsum((mass/dm)*loss);conditional_ead=None if dm==0 else math.fsum((mass/dm)*ead);individual_lgd=None
        if dm>0 and np.all(ead[default==1]>0):
            ratios=loss[default==1]/ead[default==1]
            if np.any(~np.isfinite(ratios)):_fail('NUMERICAL_FAILURE','Individual LGD exceeds finite representation')
            individual_lgd=math.fsum((mass[default==1]/dm)*ratios)
        weighted_lgd=None if conditional_ead is None or conditional_ead==0 else conditional_loss/conditional_ead
        if weighted_lgd is not None and not math.isfinite(weighted_lgd):_fail('NUMERICAL_FAILURE','EAD-weighted LGD exceeds finite representation')
        interval=None;pdse=None;lossse=None
        if design=='IID_UNWEIGHTED':
            alpha=1-level;interval=[0.0 if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),1.0 if k==n else float(beta.isf(alpha/2,k+1,n-k))]
            pdse=math.sqrt(pd*(1-pd)/n)
            if n>1:
                scale=float(np.max(np.abs(loss)))
                lossse=0.0 if scale==0 else scale*(float(np.std(loss/scale,ddof=1))/math.sqrt(n))
        value.update(status='ESTIMABLE',pd=pd,pd_interval=interval,pd_interval_method='CLOPPER_PEARSON_TWO_SIDED'if interval is not None else'NOT_AVAILABLE_FOR_SAMPLING_DESIGN',observed_default_trials=k,expected_loss=el,conditional_mean_loss=conditional_loss,conditional_mean_ead=conditional_ead,conditional_mean_lgd=individual_lgd,ead_weighted_lgd=weighted_lgd,effective_weight_count=1/math.fsum(w*w),pd_mc_standard_error=pdse,loss_mc_standard_error=lossse,uncertainty_status='IID_EXACT_PD_AND_SAMPLE_LOSS_MOMENTS'if design=='IID_UNWEIGHTED'else'NO_INDEPENDENT_MONTE_CARLO_INTERVAL',pd_boundary_sample=k in (0,n),zero_ead_default_trials=int(np.sum((default==1)&(ead==0))))
    return _result(value,'simulation.estimate_simulation_risk','simulation_risk',ctx,diagnostics=dict(input_fingerprint=_fingerprint(rows),contract_hash=_fingerprint(contract)),limitations=('Independence and representative sampling are explicit caller assumptions, not proven by path IDs. Exact binomial intervals apply only to IID unit-weight trials; Kish effective weight count is descriptive and never substituted into binomial calculations. A zero-default plug-in standard error of zero is not certainty: the exact interval has a positive upper bound. Weighted scenarios and dependent/QMC points have no IID interval here. Expected loss is the joint mean of default loss; separate mean EAD and mean LGD generally do not factor. Raw losses/LGD are not capped. Initially defaulted paths are excluded; incomplete positive-weight initially-performing paths are rejected.',))


def importance_sample_defaults(standard_normal_innovations, proposal_shift, evaluator, importance_contract, *, context=None):
    """Evaluate a complete Gaussian mean-shift proposal with its density ratio."""
    z=_numeric(standard_normal_innovations,'standard_normal_innovations');shift=_numeric(proposal_shift,'proposal_shift',vector_only=True);contract=deepcopy(importance_contract)
    if z.ndim!=2 or not z.shape[0]or not z.shape[1]or shift.shape!=(z.shape[1],):_fail('SIMULATION_IMPORTANCE','Nonempty trial/coordinate matrix and complete shift vector required')
    if not callable(evaluator)or not isinstance(contract,dict):_fail('SIMULATION_IMPORTANCE','Callable evaluator and explicit importance contract required')
    _required(contract,['contract_id','version','evaluator_id','evaluator_version','base_law','proposal','all_randomness_in_innovations','deterministic_given_innovations','sampling_design','subject_id','currency','probability_measure','event','conditioning','horizon','time_unit'])
    for key in ('contract_id','version','evaluator_id','evaluator_version','base_law','proposal','sampling_design','subject_id','currency','probability_measure','event','conditioning','time_unit'):_text(contract[key],key)
    if contract['base_law']!='INDEPENDENT_STANDARD_NORMAL' or contract['proposal']!='MEAN_SHIFT' or contract['all_randomness_in_innovations']is not True or contract['deterministic_given_innovations']is not True:_fail('SIMULATION_IMPORTANCE','All random coordinates must have the declared base law and be supplied to a deterministic evaluator')
    if contract['sampling_design']not in ('IID','DEPENDENT_OR_QMC')or contract['event']!='first_default'or contract['conditioning']!='initially_performing'or contract['probability_measure']not in ('PHYSICAL','RISK_NEUTRAL')or contract['time_unit']not in ('days','months','years')or _number(contract['horizon'],'horizon',0)<=0:_fail('SIMULATION_IMPORTANCE','Explicit initially-performing first-default target and sampling design required')
    ctx=_context_value(context)
    for key in ('currency','probability_measure','horizon','time_unit','conditioning'):
        if ctx.get(key)is not None and ctx[key]!=contract[key]:_fail('TARGET_MISMATCH','Importance target conflicts with context '+key)
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        x=z+shift;norm2=float(shift@shift);logw=-(z@shift)-.5*norm2;weights=np.exp(logw)
    if np.any(~np.isfinite(x))or not math.isfinite(norm2)or np.any(~np.isfinite(weights))or np.any(weights<=0):_fail('SIMULATION_IMPORTANCE_RANGE','Proposal density ratios must remain finite and positive; no clipping or silent weight truncation is performed')
    try:raw=evaluator(x.copy())
    except CreditRiskError:raise
    except Exception as error:_fail('SIMULATION_EVALUATOR','Target evaluator failed: '+type(error).__name__)
    result=raw.value if isinstance(raw,CalcResult)else raw
    if not isinstance(result,dict):_fail('SIMULATION_EVALUATOR','Evaluator must return defaulted and economic_loss vectors')
    _required(result,['defaulted','economic_loss'])
    default=np.asarray(result['defaulted'])
    if default.shape!=(len(z),)or default.dtype.kind not in 'bifu' or np.any(~np.isfinite(default))or np.any((default!=0)&(default!=1)):_fail('SIMULATION_EVALUATOR','One binary default outcome per proposal trial required')
    default=default.astype(float);loss=_numeric(result['economic_loss'],'economic_loss',vector_only=True)
    if loss.shape!=default.shape or np.any(loss[default==0]!=0):_fail('SIMULATION_EVALUATOR','One finite monetary default loss per trial; nondefaults have zero default loss')
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        pd_contribution=weights*default;loss_contribution=weights*loss
    if np.any(~np.isfinite(loss_contribution))or np.any((loss!=0)&(loss_contribution==0)):_fail('SIMULATION_IMPORTANCE_RANGE','Weighted losses exceed finite representation; weights are never silently truncated')
    n=len(z)
    def moments(values):
        scale=float(np.max(np.abs(values)))
        if scale==0:return 0.0,0.0 if n>1 and contract['sampling_design']=='IID'else None
        standardized=values/scale;mean=scale*(math.fsum(standardized)/n)
        se=scale*(float(np.std(standardized,ddof=1))/math.sqrt(n))if n>1 and contract['sampling_design']=='IID'else None
        return mean,se
    pd,pdse=moments(pd_contribution);el,lossse=moments(loss_contribution);meanw,_=moments(weights)
    relative=weights/float(weights.max());sum_relative=math.fsum(relative);normalized=relative/sum_relative;ess=sum_relative*sum_relative/float(relative@relative)
    value=dict(contract=contract,proposal_shift=shift.tolist(),proposal_innovations=x.tolist(),log_likelihood_ratios=logw.tolist(),likelihood_ratios=weights.tolist(),defaulted=default.astype(bool).tolist(),economic_loss=loss.tolist(),weighted_default_contributions=pd_contribution.tolist(),weighted_loss_contributions=loss_contribution.tolist(),pd_estimate=pd,expected_loss_estimate=el,pd_mc_standard_error=pdse,loss_mc_standard_error=lossse,mean_likelihood_ratio=meanw,self_normalized_pd=float(normalized@default),effective_weight_count=ess,max_normalized_weight=float(normalized.max()),proposal_default_count=int(default.sum()),trials=n,pd_estimate_in_unit_interval=0<=pd<=1,zero_proposal_defaults=not bool(default.any()),target_zero_default_bound=None,uncertainty_status='IID_SAMPLE_CONTRIBUTION_MOMENTS'if contract['sampling_design']=='IID'else'NO_IID_MONTE_CARLO_INTERVAL')
    return _result(value,'simulation.importance_sample_defaults','importance_sampling',ctx,diagnostics=dict(innovation_fingerprint=_fingerprint(z.tolist()),contract_hash=_fingerprint(contract)),limitations=('The density ratio includes every shifted standard-normal coordinate: log(p/q)=-mu dot x+||mu||^2/2. The ordinary importance mean is unbiased only for IID draws from the declared proposal and complete support; dependent/QMC inputs provide a numerical estimate without IID uncertainty. A finite-sample probability estimate can exceed one and is not clipped. Self-normalized PD is separately labelled and generally biased. Zero proposal defaults do not give an exact target-PD upper bound; a zero sample standard error is not certainty. Effective weight count measures concentration, not achieved rare-event precision. The evaluator is caller-supplied local code, identified by declared ID/version and required to be deterministic given all innovations.',))
