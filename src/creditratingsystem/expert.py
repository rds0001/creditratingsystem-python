"""Explicit elicitation, probability pooling and documented expert evidence."""
from copy import deepcopy
from itertools import combinations
import math
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import expit, logit, logsumexp, betaln, gammaln
from .core import CalcResult, _numeric, _probability, _fail, _result, _fingerprint, _context_value
from .data import _date


from ._expert_common import _ex_object, _ex_ids, _ex_number, _ex_probability, _ex_integer, _ex_seal, _ex_read, _ex_target, _ex_elicitation


def fit_elicited_beta_prior(elicitation,*,elicitation_contract,max_cdf_error=.01,context=None):
    """Fit a proper Beta prior from moments or weighted elicited CDF points."""
    c=_ex_elicitation(elicitation_contract,context);e=_ex_object(elicitation,'elicitation');mode=e.get('method');tol=_ex_number(max_cdf_error,'max_cdf_error',True)
    if tol>=1: _fail('EXPERT_CONTRACT','max_cdf_error must be below one')
    diagnostics={}
    if mode=='moments':
        m=_ex_probability(e.get('mean'),'mean',True)
        if ('variance' in e)==('concentration' in e): _fail('EXPERT_ELICITATION','Supply exactly variance or concentration')
        if 'variance' in e:
            var=_ex_number(e['variance'],'variance',True)
            if var>=m*(1-m): _fail('EXPERT_ELICITATION','Variance incompatible with a proper Beta prior')
            concentration=m*(1-m)/var-1
        else: concentration=_ex_number(e['concentration'],'concentration',True)
        alpha=m*concentration;beta=(1-m)*concentration;residual=[];diagnostics['method']='ANALYTIC_MOMENTS'
    elif mode=='quantiles':
        x=_numeric(e.get('quantiles'),'quantiles',vector_only=True);p=_numeric(e.get('probabilities'),'probabilities',vector_only=True);w=_numeric(e.get('weights'),'weights',vector_only=True)
        if len(x)<2 or x.shape!=p.shape or x.shape!=w.shape or np.any((x<=0)|(x>=1)) or np.any((p<=0)|(p>=1)) or np.any(w<=0) or np.any(np.diff(x)<=0) or np.any(np.diff(p)<=0):
            _fail('EXPERT_ELICITATION','At least two strictly increasing interior quantile/probability pairs and positive weights required')
        w=w/w.max();w=w/w.sum();fun=lambda z:np.sqrt(w)*(stats.beta.cdf(x,*np.exp(z))-p)
        fits=[least_squares(fun,np.log(start),bounds=(-18,18),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=3000) for start in ((1,1),(2,18),(18,2))]
        fit=min(fits,key=lambda a:float(a.fun@a.fun));alpha,beta=np.exp(fit.x);residual=(stats.beta.cdf(x,alpha,beta)-p).tolist();singular=np.linalg.svd(fit.jac,compute_uv=False)
        if not fit.success or max(abs(v)for v in residual)>tol or singular[-1]<=1e-8*singular[0] or np.any(abs(fit.x)>17.99):
            _fail('EXPERT_ELICITATION_FIT','No identified interior Beta fit within declared CDF residual tolerance')
        diagnostics.update(method='WEIGHTED_CDF_LEAST_SQUARES',weighted_squared_error=float(fit.fun@fit.fun),max_cdf_error=max(abs(v)for v in residual),jacobian_condition=float(singular[0]/singular[-1]),log_shape_bounds=[-18,18],optimizer_status=int(fit.status))
    else: _fail('EXPERT_ELICITATION','method must be moments or quantiles')
    if not np.isfinite(alpha+beta) or min(alpha,beta)<=0: _fail('EXPERT_ELICITATION','Beta shapes must be positive and finite')
    value=_ex_seal(dict(distribution='beta',alpha=float(alpha),beta=float(beta),mean=float(alpha/(alpha+beta)),variance=float((alpha/(alpha+beta))*(beta/(alpha+beta))/(alpha+beta+1)),prior_concentration=float(alpha+beta),elicitation=deepcopy(e),elicitation_contract=c,known_at=c['known_at'],evidence_ids=list(c['evidence_ids']),cdf_residuals=residual),'prior_hash')
    return _result(value,'expert.fit_elicited_beta_prior','probability_distribution',context,diagnostics=diagnostics,limitations=('Prior concentration is elicited strength, not an observed sample size. A fitted distribution does not establish that an elicitation workshop occurred.',))


def update_expert_beta_binomial(prior,defaults,exposures,*,observation_contract,predictive_exposures=0,credible_level=.95,context=None):
    """Conjugate update using explicitly compatible complete conditionally iid counts."""
    pr=_ex_read(prior,'prior_hash');contract=_ex_object(observation_contract,'observation_contract');base=pr['elicitation_contract'];_ex_target(base['target'],context)
    if contract.get('target')!=base['target'] or contract.get('population_id')!=base['population_id']: _fail('TARGET_MISMATCH','Observations and elicitation population/target differ')
    if contract.get('conditional_iid') is not True or contract.get('outcome_complete') is not True: _fail('EXPERT_OBSERVATIONS','Complete conditionally iid observations must be declared; clustered or censored counts are unsupported')
    ids=_ex_ids(contract.get('evidence_ids'),'evidence_ids')
    if set(ids)&set(pr['evidence_ids']): _fail('EXPERT_DOUBLE_COUNT','Evidence already used in the prior or previous updates')
    if _date(contract.get('known_at'),'known_at')<_date(pr['known_at'],'known_at'): _fail('EXPERT_LOOKAHEAD','Observation cutoff precedes elicitation availability')
    d=_ex_integer(defaults,'defaults');n=_ex_integer(exposures,'exposures');future=_ex_integer(predictive_exposures,'predictive_exposures',100000);level=_ex_probability(credible_level,'credible_level',True)
    if d>n: _fail('EXPERT_OBSERVATIONS','defaults exceed exposures')
    a=pr['alpha']+d;b=pr['beta']+n-d;k=np.arange(future+1,dtype=float)
    pmf=np.exp(gammaln(future+1)-gammaln(k+1)-gammaln(future-k+1)+betaln(k+a,future-k+b)-betaln(a,b))
    if abs(float(pmf.sum())-1)>1e-7: _fail('NUMERICAL_FAILURE','Predictive count probabilities do not normalize')
    posterior=_ex_seal(dict(distribution='beta',alpha=a,beta=b,mean=a/(a+b),variance=(a/(a+b))*(b/(a+b))/(a+b+1),prior_concentration=a+b,elicitation=pr['elicitation'],elicitation_contract=base,known_at=contract['known_at'],evidence_ids=pr['evidence_ids']+ids,cdf_residuals=pr['cdf_residuals'],previous_prior_hash=pr['prior_hash']),'prior_hash')
    return _result(dict(posterior=posterior,posterior_mean=a/(a+b),posterior_median=float(stats.beta.ppf(.5,a,b)),credible_level=level,credible_interval=stats.beta.ppf([(1-level)/2,(1+level)/2],a,b).tolist(),prior_mean=pr['alpha']/(pr['alpha']+pr['beta']),prior_weight=(pr['alpha']+pr['beta'])/(a+b),observed_exposures=n,observed_defaults=d,predictive_counts=list(range(future+1)),predictive_probabilities=pmf.tolist(),predictive_mean=future*a/(a+b),predictive_variance=future*(a/(a+b))*(b/(a+b))*(a+b+future)/(a+b+1),observation_contract=deepcopy(contract)),
        'expert.update_expert_beta_binomial','probability_distribution',context,limitations=('Credible intervals describe the common latent PD under a conditionally iid binomial model; future counts follow a Beta-binomial predictive distribution. Opinion correlation is not additional independent evidence.',))


def pool_expert_probability_distributions(components,weights,*,support,rule,exclusion_reasons=None,context=None):
    """Exact linear/log pooling of finite masses or piecewise-constant densities."""
    a=_numeric(components,'components');w=_numeric(weights,'weights',vector_only=True);s=_ex_object(support,'support');kind=s.get('kind')
    if a.ndim!=2 or not a.shape[0] or not a.shape[1] or w.shape!=(a.shape[0],) or np.any(a<0) or np.any(w<0) or not w.sum()>0: _fail('EXPERT_POOL','Nonnegative aligned components and nonzero nonnegative weights required')
    if kind=='discrete':
        values=_ex_ids(s.get('values'),'support values');width=np.ones(len(values))
    elif kind=='histogram':
        edges=_numeric(s.get('breaks'),'breaks',vector_only=True)
        if len(edges)<2 or np.any(np.diff(edges)<=0): _fail('EXPERT_POOL','Increasing finite histogram breaks required')
        width=np.diff(edges)
    else: _fail('EXPERT_POOL','support kind must be discrete or histogram')
    if a.shape[1]!=len(width) or np.any(abs(a@width-1)>1e-10): _fail('EXPERT_POOL','Every component must be normalized on common declared support')
    reasons={}if exclusion_reasons is None else _ex_object(exclusion_reasons,'exclusion_reasons')
    for i in np.flatnonzero(w==0):
        if not isinstance(reasons.get(str(i)),str) or not reasons[str(i)]: _fail('EXPERT_POOL','Every zero-weight component requires a zero-based exclusion reason')
    w=w/w.max();w=w/w.sum();positive=w>0;lognormalizer=0.
    if rule=='linear': pooled=w@a
    elif rule=='logarithmic':
        with np.errstate(divide='ignore'): logs=np.sum(w[positive,None]*np.log(a[positive]),axis=0)
        lognormalizer=float(logsumexp(logs+np.log(width)))
        if not math.isfinite(lognormalizer): _fail('EXPERT_DISJOINT_SUPPORT','Positive-weight distributions have no shared positive support')
        pooled=np.exp(logs-lognormalizer)
    else: _fail('EXPERT_POOL','rule must be linear or logarithmic')
    mass=pooled*width;mixture=w@a;tv=.5*np.sum(abs(a-mixture)*width,axis=1)
    return _result(dict(support=deepcopy(s),rule=rule,normalized_weights=w.tolist(),density_or_mass=pooled.tolist(),bin_probabilities=mass.tolist(),cumulative_probabilities=np.cumsum(mass).tolist(),log_normalization_constant=lognormalizer,normalization_constant=float(np.exp(lognormalizer)),component_total_variation_to_linear_pool=tv.tolist(),weighted_total_variation=float(w@tv),exclusion_reasons=deepcopy(reasons)),
        'expert.pool_expert_probability_distributions','probability_distribution',context,limitations=('Pooling is opinion aggregation, not multiplication of independent observations. Histogram densities are constant inside each declared bin; reported CDF values are at bin right endpoints.',))


def update_odds_from_evidence(prior_pd,evidence_groups,*,dependence_contract,previous_evidence_ids=None,context=None):
    """Apply one joint LR per distinct evidence group under explicit independence."""
    p=_ex_probability(prior_pd,'prior_pd');groups=evidence_groups
    if not isinstance(groups,(list,tuple)): _fail('EXPERT_EVIDENCE','evidence_groups must be a list')
    c=_ex_object(dependence_contract,'dependence_contract');ids=_ex_ids([g.get('group_id')for g in groups],'group_ids',True);prior_ids=_ex_ids([]if previous_evidence_ids is None else previous_evidence_ids,'previous_evidence_ids',True);seen=set(prior_ids);rows=[];log_lr=0.
    if len(groups)>1 and (c.get('conditional_independence_given_default_and_survival') is not True or not isinstance(c.get('reference'),str) or not c['reference']): _fail('EXPERT_DEPENDENCE','Multiple group LRs require evidenced conditional independence under BOTH outcomes')
    for group in groups:
        group=_ex_object(group,'evidence group');eids=_ex_ids(group.get('evidence_ids'),'evidence_ids');lr=_ex_number(group.get('joint_likelihood_ratio'),'joint_likelihood_ratio',True)
        if set(eids)&seen: _fail('EXPERT_DOUBLE_COUNT','An evidence item appears more than once')
        if not isinstance(group.get('reference'),str) or not group['reference']: _fail('EXPERT_EVIDENCE','Joint LR evidence reference required')
        seen.update(eids);contribution=math.log(lr);log_lr+=contribution;rows.append(dict(group_id=group['group_id'],evidence_ids=eids,log_odds_contribution=contribution,joint_likelihood_ratio=lr,reference=group['reference']))
    posterior=p if p in (0.,1.) else float(expit(logit(p)+log_lr))
    return _result(dict(prior_pd=p,posterior_pd=posterior,total_log_likelihood_ratio=log_lr,contributions=rows,evidence_ids=sorted(seen),prior_boundary_absorbing=p in (0.,1.),dependence_contract=deepcopy(c)),
        'expert.update_odds_from_evidence','probability',context,limitations=('Finite strictly positive likelihood ratios cannot change a dogmatic zero/one prior. Correlated observations within one group need a supplied joint LR; their marginal LRs are never multiplied implicitly.',))


def check_expert_preference_consistency(comparisons,*,items,incomplete_policy,tolerance=1e-8,context=None):
    """Diagnose reciprocal ratios, directed cycles and log-triads; log-LS priorities."""
    labels=_ex_ids(items,'items');n=len(labels);tol=_ex_number(tolerance,'tolerance',True)
    if incomplete_policy not in ('error','connected_components'): _fail('EXPERT_PREFERENCE','Declare error or connected_components incomplete policy')
    try: a=np.array(comparisons,dtype=float)
    except (ValueError,TypeError): _fail('EXPERT_PREFERENCE','Numeric comparison matrix with null/NaN for missing pairs required')
    if a.shape!=(n,n) or np.any(np.isinf(a)) or np.any(np.isfinite(a)&(a<=0)) or np.any(~np.isfinite(np.diag(a))) or np.any(abs(np.log(np.diag(a)))>tol): _fail('EXPERT_PREFERENCE','Positive square ratios and unit diagonal required')
    observed=np.isfinite(a)
    if np.any(observed!=observed.T) or np.any(abs(np.log(a[observed])+np.log(a.T[observed]))>tol): _fail('EXPERT_PREFERENCE','Observed pairs must be reciprocal within declared log tolerance')
    if incomplete_policy=='error' and not np.all(observed): _fail('EXPERT_PREFERENCE','Incomplete comparisons prohibited by policy')
    loga=np.zeros_like(a);loga[observed]=np.log(a[observed]);neighbors={i:[j for j in range(n)if j!=i and observed[i,j]]for i in range(n)};components=[];seen=set()
    for start in range(n):
        if start in seen: continue
        stack=[start];seen.add(start);component=[]
        while stack:
            i=stack.pop();component.append(i)
            for j in neighbors[i]:
                if j not in seen: seen.add(j);stack.append(j)
        components.append(sorted(component))
    solved=[];edge_residuals=[]
    for component in components:
        pairs=[(i,j)for i,j in combinations(component,2)if observed[i,j]];design=np.zeros((len(pairs)+1,len(component)));rhs=np.zeros(len(pairs)+1)
        for k,(i,j) in enumerate(pairs): design[k,component.index(i)]=1;design[k,component.index(j)]=-1;rhs[k]=loga[i,j]
        design[-1]=1;theta=np.linalg.lstsq(design,rhs,rcond=None)[0];priority=np.exp(theta-logsumexp(theta))
        solved.append(dict(items=[labels[i]for i in component],priorities=priority.tolist()))
        for (i,j),res in zip(pairs,(design@theta-rhs)[:-1]): edge_residuals.append(dict(first=labels[i],second=labels[j],log_residual=float(res)))
    triads=[dict(items=[labels[i],labels[j],labels[k]],log_residual=float(loga[i,j]+loga[j,k]-loga[i,k]))for i,j,k in combinations(range(n),3)if observed[i,j] and observed[j,k] and observed[i,k]]
    directed={i:[j for j in range(n)if observed[i,j] and loga[i,j]>tol]for i in range(n)};color=[0]*n;cycles=[]
    # Iterative DFS: one witness for each encountered back edge, not exhaustive cycle enumeration.
    for start in range(n):
        if color[start]:continue
        path=[start];color[start]=1;stack=[iter(directed[start])]
        while stack:
            j=next(stack[-1],None)
            if j is None: color[path.pop()]=2;stack.pop()
            elif color[j]==0: color[j]=1;path.append(j);stack.append(iter(directed[j]))
            elif color[j]==1: cycles.append([labels[k]for k in path[path.index(j):]+[j]])
    maximum=max([abs(r['log_residual'])for r in edge_residuals]+[0.])
    return _result(dict(components=solved,globally_identified=len(components)==1,cycle_witnesses=cycles,triad_residuals=triads,pair_fit_residuals=edge_residuals,max_pair_log_residual=maximum,consistent_with_tolerance=not cycles and maximum<=tol,tolerance=tol,uncertainty_status='JUDGMENT_DISAGREEMENT_ONLY_NO_SAMPLING_MODEL'),
        'expert.check_expert_preference_consistency','expert_priorities',context,limitations=('Ratios compare first-item importance to second-item importance. Log least squares diagnoses supplied judgments without repairing the matrix. Disconnected components have separate normalized priorities and no identified common scale; no universal consistency cutoff or sampling interval is asserted.',))


def assess_expert_rating_agreement(first_ratings,second_ratings,*,ordered_grades,rater_pair,weighting,missing_policy,bootstrap_indices=None,confidence_level=.95,context=None):
    """Paired-rater weighted Cohen kappa with explicit missingness and resamples."""
    grades=_ex_ids(ordered_grades,'ordered_grades');raters=_ex_ids(rater_pair,'rater_pair')
    if len(raters)!=2 or len(grades)<2 or not isinstance(first_ratings,(list,tuple)) or not isinstance(second_ratings,(list,tuple)) or len(first_ratings)!=len(second_ratings): _fail('EXPERT_AGREEMENT','Two raters, aligned case lists and at least two ordered grades required')
    if weighting not in ('unweighted','linear','quadratic') or missing_policy not in ('error','pairwise_complete'): _fail('EXPERT_AGREEMENT','Explicit weighting and missingness policy required')
    pairs=[];missing=[]
    for i,(a,b) in enumerate(zip(first_ratings,second_ratings)):
        if a is None or b is None:
            if missing_policy=='error': _fail('EXPERT_AGREEMENT','Missing rating forbidden')
            missing.append(i);continue
        if a not in grades or b not in grades: _fail('EXPERT_AGREEMENT','Unknown rating grade')
        pairs.append((grades.index(a),grades.index(b)))
    k=len(grades);distance=abs(np.arange(k)[:,None]-np.arange(k)[None,:])/(k-1);weights=(distance==0).astype(float)if weighting=='unweighted'else 1-distance**(1 if weighting=='linear'else 2)
    def calculate(rows):
        table=np.zeros((k,k),dtype=int)
        for a,b in rows: table[a,b]+=1
        if not len(rows): return table,None,None,None
        p=table/len(rows);observed=float(np.sum(p*weights));expected=float(np.sum(np.outer(p.sum(axis=1),p.sum(axis=0))*weights))
        return table,observed,expected,None if expected>=1-1e-15 else (observed-expected)/(1-expected)
    table,observed,expected,kappa=calculate(pairs);interval=None;valid=0;invalid=0
    if bootstrap_indices is not None:
        idx=_numeric(bootstrap_indices,'bootstrap_indices')
        if idx.ndim!=2 or idx.shape[1]!=len(pairs) or idx.shape[0]<2 or not len(pairs) or np.any(idx!=np.floor(idx)) or np.any((idx<0)|(idx>=len(pairs))): _fail('EXPERT_AGREEMENT','Supplied resamples must index paired-complete cases, zero-based, with unchanged sample size')
        values=[]
        for row in idx.astype(int):
            v=calculate([pairs[i]for i in row])[3]
            if v is None:invalid+=1
            else:values.append(v)
        valid=len(values);level=_ex_probability(confidence_level,'confidence_level',True)
        if invalid==0: interval=np.quantile(values,[(1-level)/2,(1+level)/2]).tolist()
    return _result(dict(rater_pair=raters,ordered_grades=grades,weighting=weighting,confusion_matrix=table.tolist(),agreement_weights=weights.tolist(),observed_agreement=observed,expected_agreement=expected,kappa=kappa,status='NO_PAIRED_CASES'if not pairs else 'DEGENERATE_MARGINALS'if kappa is None else 'ESTIMABLE',complete_cases=len(pairs),total_cases=len(first_ratings),missing_case_indices=missing,bootstrap_confidence_level=None if bootstrap_indices is None else level,bootstrap_percentile_interval=interval,bootstrap_valid=valid,bootstrap_degenerate=invalid,uncertainty_status='NOT_REQUESTED'if bootstrap_indices is None else 'UNAVAILABLE_DEGENERATE_RESAMPLES'if invalid else 'PAIRED_IID_PERCENTILE'),
        'expert.assess_expert_rating_agreement','agreement',context,limitations=('Agreement does not establish predictive validity. Chance agreement uses independent empirical rater marginals. Bootstrap uncertainty assumes independent sampled cases; clustered/repeated cases require caller-supplied cluster-valid resamples and separate interpretation. Any degenerate resample suppresses the interval rather than silently dropping it.',))

# Reexports retain one granular expert family while keeping finite factors separate.
from ._expert_network import compile_discrete_bayesian_network, infer_discrete_credit_network

from ._expert_qualitative import compile_qualitative_rubric, evaluate_qualitative_evidence
