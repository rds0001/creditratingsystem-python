"""Observed-outcome validation metrics with explicit applicability and boundaries."""
import math
import numpy as np
from scipy.stats import binomtest

from .core import _numeric, _probability, _fail, _result


def _sample(outcomes, predictions, weights, probability=False):
    y = _numeric(outcomes, "outcomes")
    if y.ndim > 1:
        _fail("INVALID_INPUT", "outcomes must be scalar or a one-dimensional vector")
    y = y.reshape(-1)
    p = (_probability(predictions) if probability else _numeric(predictions, "scores"))
    if p.ndim > 1:
        _fail("INCOMPATIBLE_SHAPE", "Validation requires one-dimensional observations")
    p = p.reshape(-1)
    w = np.ones(len(y)) if weights is None else _numeric(weights, "weights")
    if w.ndim > 1:
        _fail("INVALID_INPUT", "weights must be scalar or a one-dimensional vector")
    w = w.reshape(-1)
    if len(y) != len(p) or len(y) != len(w):
        _fail("INCOMPATIBLE_SHAPE", "Outcomes, predictions and weights must have equal lengths")
    if np.any((y != 0) & (y != 1)):
        _fail("INVALID_OUTCOMES", "Observed binary outcomes must be zero or one; missing outcomes are not zero")
    if np.any(w < 0) or not np.any(w > 0):
        _fail("INVALID_WEIGHTS", "Nonnegative weights with positive total required")
    with np.errstate(over="ignore"):
        total = float(np.sum(w))
    if not math.isfinite(total):
        _fail("NUMERICAL_RANGE", "Weight sum exceeds finite range")
    return y, p, w, total


def _rank_groups(y, score, w, direction):
    if direction not in ("higher_is_risk", "lower_is_risk"):
        _fail("INVALID_DIRECTION", "Declare higher_is_risk or lower_is_risk")
    transformed = score if direction == "higher_is_risk" else -score
    active = w > 0
    values = np.unique(transformed[active])
    group = np.searchsorted(values, transformed[active])
    positives = np.bincount(group, weights=w[active]*y[active], minlength=len(values))
    negatives = np.bincount(group, weights=w[active]*(1-y[active]), minlength=len(values))
    positive_count = np.bincount(group, weights=y[active], minlength=len(values)).astype(int)
    negative_count = np.bincount(group, weights=1-y[active], minlength=len(values)).astype(int)
    return values, positives, negatives, positive_count, negative_count


def auc_gini(outcomes, scores, weights=None, *, score_direction, context=None):
    """Weighted concordance with half credit for ties; higher risk predicts y=1."""
    y, scores, w, total = _sample(outcomes, scores, weights)
    values, pos, neg, np_, nn_ = _rank_groups(y, scores, w, score_direction)
    ptotal, ntotal = float(pos.sum()), float(neg.sum())
    pair_count = int(np_.sum()) * int(nn_.sum())
    tie_count = sum(int(a)*int(b) for a,b in zip(np_,nn_))
    if ptotal == 0 or ntotal == 0:
        value = dict(auc=None, gini=None, applicability="NOT_APPLICABLE", reason="ONE_CLASS",
                     pair_count=pair_count, tied_pair_count=tie_count,
                     concordant_pair_mass=None, tied_pair_mass=None)
    else:
        p, n = pos/ptotal, neg/ntotal
        lower_negative = np.concatenate(([0.], np.cumsum(n)[:-1]))
        concordance, ties = float(np.dot(p, lower_negative)), float(np.dot(p, n))
        auc = concordance + .5*ties
        value = dict(auc=auc, gini=2*auc-1, applicability="APPLICABLE", reason=None,
                     pair_count=pair_count, tied_pair_count=tie_count,
                     concordant_pair_mass=concordance, tied_pair_mass=ties)
    return _result(value, "validation.auc_gini", "discrimination", context,
        diagnostics=dict(score_direction=score_direction, event_weight=ptotal, nonevent_weight=ntotal,
                         total_weight=total, excluded_zero_weight=int(np.sum(w==0)), pair_mass_normalization="within_each_class"),
        limitations=("Descriptive paired concordance; repeated entities and sample selection affect inferential interpretation. No confidence interval is implied.",))


def ks_discrimination(outcomes, scores, weights=None, *, score_direction, context=None):
    """Maximum absolute difference of weighted class CDFs, evaluating ties together."""
    y, scores, w, total = _sample(outcomes, scores, weights)
    values, pos, neg, _, _ = _rank_groups(y, scores, w, score_direction)
    ptotal, ntotal = float(pos.sum()), float(neg.sum())
    if ptotal == 0 or ntotal == 0:
        value = dict(ks=None, threshold=None, signed_difference=None, cdf=[], applicability="NOT_APPLICABLE", reason="ONE_CLASS")
    else:
        pcdf, ncdf = np.cumsum(pos/ptotal), np.cumsum(neg/ntotal)
        difference = ncdf-pcdf
        index = int(np.argmax(np.abs(difference)))
        thresholds = values if score_direction == "higher_is_risk" else -values
        rows = [dict(threshold=float(t), event_cdf=float(p), nonevent_cdf=float(n)) for t,p,n in zip(thresholds,pcdf,ncdf)]
        value = dict(ks=float(abs(difference[index])), threshold=float(thresholds[index]),
                     signed_difference=float(difference[index]), cdf=rows, applicability="APPLICABLE", reason=None)
    return _result(value, "validation.ks_discrimination", "discrimination", context,
        diagnostics=dict(score_direction=score_direction, cdf_order="ascending_oriented_risk_score", total_weight=total,
                         threshold_tie_break="first_in_ascending_oriented_score"),
        limitations=("Descriptive KS distance only; weighted, tied or dependent data do not justify a classical continuous IID KS p-value.",))


def brier_score(outcomes, predicted_pd, weights=None, *, context=None):
    """Binary Brier mean squared error on the [0,1] convention."""
    y, p, w, total = _sample(outcomes, predicted_pd, weights, True)
    loss = (y-p)**2
    contributions = (w/total)*loss
    rate = float(np.dot(w/total, y))
    return _result(dict(brier=float(contributions.sum()), contributions=contributions.tolist(),
                        squared_errors=loss.tolist(), observed_rate=rate, outcome_variance_baseline=rate*(1-rate)),
        "validation.brier_score", "mean_squared_probability_error", context,
        diagnostics=dict(total_weight=total, excluded_zero_weight=int(np.sum(w==0))),
        limitations=("Outcomes must be observed on the same target and horizon as predictions; this metric does not repair selection or censoring.",))


def log_loss(outcomes, predicted_pd, weights=None, *, boundary_policy="exact", epsilon=None, context=None):
    """Binary natural-log loss, preserving impossible predictions as infinite status.

    JSON-safe null plus INFINITE explicitly represents positive infinite loss.
    Clipping is available only with a user-specified epsilon and is disclosed.
    """
    y, p, w, total = _sample(outcomes, predicted_pd, weights, True)
    if boundary_policy not in ("exact", "clip"):
        _fail("INVALID_BOUNDARY_POLICY", "Use exact or clip")
    if boundary_policy == "exact" and epsilon is not None:
        _fail("INVALID_BOUNDARY_POLICY", "Exact policy does not take epsilon")
    if boundary_policy == "clip":
        e = _numeric(epsilon, "epsilon")
        if e.ndim or not 0 < e < .5:
            _fail("INVALID_BOUNDARY_POLICY", "Explicit epsilon must lie in (0,.5)")
        used = np.clip(p, float(e), 1-float(e))
        if np.any((used<=0)|(used>=1)):
            _fail("INVALID_BOUNDARY_POLICY", "Epsilon must produce representable interior bounds")
    else:
        used = p
    losses = np.zeros(len(y))
    with np.errstate(divide="ignore"):
        losses[y==1] = -np.log(used[y==1])
        losses[y==0] = -np.log1p(-used[y==0])
    impossible = np.isinf(losses) & (w>0)
    contributions = [0. if weight==0 else None if math.isinf(loss) else float(weight/total*loss)
                     for weight,loss in zip(w,losses)]
    infinite = bool(impossible.any())
    return _result(dict(log_loss=None if infinite else float(sum(contributions)),
                        loss_status="INFINITE" if infinite else "FINITE", contributions=contributions,
                        infinite_loss_flags=impossible.tolist()), "validation.log_loss", "nats", context,
        diagnostics=dict(total_weight=total,boundary_policy=boundary_policy,epsilon=epsilon,
                         clipped_count=int(np.sum(used!=p)),positive_infinity_representation="null plus INFINITE"),
        limitations=("Predictions fixed before observing outcomes are required for an honest validation interpretation. Any clipping changes the scoring rule at the selected boundaries.",))


def binomial_pd_test(n, observed_defaults, common_pd, *, alternative, independence_assumption,
                     confidence_level=.95, context=None):
    """Exact independent common-PD count test and Clopper-Pearson interval."""
    n_, k_, p, confidence = (_numeric(x, name) for x,name in
        [(n,"n"),(observed_defaults,"observed_defaults"),(common_pd,"common_pd"),(confidence_level,"confidence_level")])
    if any(x.ndim for x in (n_,k_,p,confidence)) or n_<0 or k_<0 or k_>n_ or n_!=math.floor(n_) or k_!=math.floor(k_) or n_>2**53-1:
        _fail("INVALID_COUNTS", "Require exact integer counts 0 <= defaults <= n")
    if not 0<=p<=1 or not 0<confidence<1:
        _fail("INVALID_PROBABILITY", "PD in [0,1] and confidence in (0,1) required")
    if alternative not in ("two-sided", "less", "greater"):
        _fail("INVALID_ALTERNATIVE", "Declare two-sided, less or greater")
    if independence_assumption != "independent_common_probability":
        _fail("INCOMPATIBLE_ASSUMPTIONS", "Binomial test requires declared independent common-probability trials")
    if n_ == 0:
        value = dict(p_value=None, lower=None, upper=None, observed_rate=None, applicability="NOT_APPLICABLE", reason="EMPTY_COHORT")
    else:
        test = binomtest(int(k_),int(n_),float(p),alternative=alternative)
        interval = test.proportion_ci(confidence_level=float(confidence),method="exact")
        value = dict(p_value=float(test.pvalue),lower=float(interval.low),upper=float(interval.high),
                     observed_rate=float(k_/n_),applicability="APPLICABLE",reason=None)
    return _result(value,"validation.binomial_pd_test","test",context,
        diagnostics=dict(n=int(n_),observed_defaults=int(k_),common_pd=float(p),alternative=alternative,
                         confidence_level=float(confidence),independence_assumption=independence_assumption,
                         two_sided_ordering="probability_mass",interval_method="Clopper-Pearson"),
        limitations=("Independent equal-probability trials are an explicit assumption; correlated obligors or heterogeneous PDs require a different null model.",))


# Extended diagnostics use explicit frozen forecast and sampling contracts.
def _validation_design(design, pd, allow_fitted=False):
    from .core import _fingerprint, _json_data
    from .data import _date, _required
    from .calibration import _target
    if not isinstance(design, dict):
        _fail('INVALID_VALIDATION_DESIGN', 'An explicit frozen-forecast design is required')
    d = _json_data(design)
    _required(d, ['prediction_origin', 'forecast_known_at', 'evaluation_at', 'split_id', 'target'])
    if not isinstance(d['split_id'], str) or not d['split_id']:
        _fail('INVALID_VALIDATION_DESIGN', 'Nonempty split_id required')
    allowed = ('HOLDOUT', 'EXTERNAL', 'FITTED_SAMPLE') if allow_fitted else ('HOLDOUT', 'EXTERNAL')
    if d['prediction_origin'] not in allowed:
        _fail('INVALID_VALIDATION_DESIGN', 'Prediction origin incompatible with this diagnostic')
    if _date(d['forecast_known_at'], 'forecast_known_at') > _date(d['evaluation_at'], 'evaluation_at'):
        _fail('INVALID_VALIDATION_DESIGN', 'Forecast knowledge must precede evaluation')
    if d.get('forecast_fingerprint') is not None and d['forecast_fingerprint'] != _fingerprint(pd.tolist()):
        _fail('FORECAST_INTEGRITY', 'Frozen forecast fingerprint differs from supplied PDs')
    d['forecast_fingerprint'] = _fingerprint(pd.tolist())
    d['target'] = _target(d['target'])
    if d['target']['probability_measure'] != 'PHYSICAL':
        _fail('INCOMPATIBLE_TARGET', 'Observed default validation requires physical probabilities')
    return d


def _alternative(alternative):
    if alternative not in ('two-sided', 'less', 'greater'):
        _fail('INVALID_ALTERNATIVE', 'Declare two-sided, less or greater')


def _confidence(confidence_level):
    value = _numeric(confidence_level, 'confidence_level')
    if value.ndim or not 0 < value < 1:
        _fail('INVALID_PROBABILITY', 'Confidence must lie in (0,1)')
    return float(value)


def _bins(predicted_pd, bin_edges):
    edges = _probability(bin_edges, vector_only=True)
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0):
        _fail('INVALID_BINS', 'Prespecified increasing edges must cover [0,1]')
    assignments = np.minimum(np.searchsorted(edges, predicted_pd, side='right') - 1, len(edges)-2)
    return edges, assignments


def calibration_intercept_slope(predicted_pd, outcomes, *, design, covariance_policy,
                                cluster_ids=None, confidence_level=.95,
                                boundary_policy='reject', epsilon=None,
                                max_iter=100, tolerance=1e-10, context=None):
    """Diagnostic logistic calibration fit, with explicit IID or cluster covariance."""
    from scipy.special import expit, logit
    from scipy.stats import norm, t
    from .classification import fit_binary_glm
    from .core import CreditRiskError, _fingerprint
    y, p, _, _ = _sample(outcomes, predicted_pd, None, True)
    d = _validation_design(design, p); confidence = _confidence(confidence_level)
    if covariance_policy not in ('iid', 'cluster_cr1', 'none'):
        _fail('INVALID_COVARIANCE_POLICY', 'Use iid, cluster_cr1 or none')
    groups = None
    if covariance_policy == 'cluster_cr1':
        if not isinstance(cluster_ids, (list, tuple)) or len(cluster_ids) != len(y) or any(not isinstance(g, str) or not g for g in cluster_ids):
            _fail('INVALID_CLUSTERS', 'One nonempty cluster ID per observation required')
        groups = sorted(set(cluster_ids))
        if len(groups) < 2 or len(y) <= 2:
            _fail('INVALID_CLUSTERS', 'CR1 covariance needs at least two independent clusters and n>2')
    elif cluster_ids is not None:
        _fail('INVALID_COVARIANCE_POLICY', 'Cluster keys require cluster_cr1 policy')
    if boundary_policy not in ('reject', 'clip') or (boundary_policy == 'reject' and epsilon is not None):
        _fail('INVALID_BOUNDARY_POLICY', 'Use reject, or clip with explicit epsilon')
    used = p.copy()
    if boundary_policy == 'clip':
        e = _numeric(epsilon, 'epsilon')
        if e.ndim or not 0 < e < .5 or 1-float(e) == 1:
            _fail('INVALID_BOUNDARY_POLICY', 'Epsilon must give interior representable bounds')
        used = np.clip(p, float(e), 1-float(e))
    diagnostics = dict(design=d, covariance_policy=covariance_policy, boundary_policy=boundary_policy,
                       epsilon=epsilon, clipped_count=int(np.sum(used != p)), confidence_level=confidence,
                       cluster_count=None if groups is None else len(groups), fit_updates_model=False)
    empty = dict(intercept=None, slope=None, covariance=None, standard_errors=None, confidence_intervals=None,
                 applicability='NOT_APPLICABLE', reason=None, converged=False)
    reason = 'BOUNDARY_PD' if np.any((used == 0) | (used == 1)) else 'CONSTANT_PD' if np.ptp(used) == 0 else 'ONE_CLASS' if np.unique(y).size < 2 else None
    if reason:
        empty['reason'] = reason
        return _result(empty, 'validation.calibration_intercept_slope', 'calibration_diagnostic', context, diagnostics=diagnostics)
    x = logit(used)
    try:
        fit = fit_binary_glm(x[:, None], y, link='logit', feature_order=['frozen_log_odds'], fit_intercept=True,
            target=d['target'], training={'data_fingerprint':_fingerprint({'pd':p.tolist(),'outcomes':y.tolist()}),
            'split_id':d['split_id'], 'known_at':d['evaluation_at']}, model_id='calibration-diagnostic',
            model_version='1.0', max_iter=max_iter, tolerance=tolerance)
    except CreditRiskError as error:
        if error.code not in ('SEPARATION','NON_CONVERGENCE','RANK_DEFICIENT','NON_IDENTIFIABLE','NUMERICAL_FAILURE'):
            raise
        empty['reason'] = error.code; diagnostics['fit_error'] = error.details
        return _result(empty, 'validation.calibration_intercept_slope', 'calibration_diagnostic', context, diagnostics=diagnostics)
    beta = np.array(fit.model['parameters']['coefficients'])
    covariance = np.array(fit.model['parameters']['covariance'])
    critical = float(norm.ppf((1+confidence)/2))
    if groups is not None:
        X = np.column_stack([np.ones(len(y)), x]); residual = y-expit(X@beta)
        scores = np.vstack([(X[np.array(cluster_ids) == g]*residual[np.array(cluster_ids) == g,None]).sum(axis=0) for g in groups])
        adjustment = len(groups)/(len(groups)-1)*(len(y)-1)/(len(y)-2)
        covariance = covariance @ (scores.T @ scores) @ covariance * adjustment
        covariance = (covariance+covariance.T)/2
        critical = float(t.ppf((1+confidence)/2, len(groups)-1))
        diagnostics.update(cluster_correction=adjustment, reference_distribution='Student-t', reference_df=len(groups)-1)
    else:
        diagnostics['reference_distribution'] = 'normal' if covariance_policy == 'iid' else None
    se = np.sqrt(np.maximum(0, np.diag(covariance)))
    value = dict(intercept=float(beta[0]), slope=float(beta[1]), covariance=None if covariance_policy == 'none' else covariance.tolist(),
                 standard_errors=None if covariance_policy == 'none' else se.tolist(),
                 confidence_intervals=None if covariance_policy == 'none' else np.column_stack([beta-critical*se,beta+critical*se]).tolist(),
                 applicability='APPLICABLE', reason=None, converged=True)
    return _result(value,'validation.calibration_intercept_slope','calibration_diagnostic',context,diagnostics=diagnostics,
        limitations=('Diagnostic fit on frozen forecasts; coefficients are not automatically applied. IID and independent-cluster asymptotic intervals do not include original model fitting uncertainty.',))


def calibration_curve(predicted_pd, outcomes, bin_edges, *, design, interval_method,
                      dependence_assumption, confidence_level=.95, minimum_count=5, context=None):
    """Prespecified reliability bins with explicit empty cells and interval assumptions."""
    from .data import _days
    y, p, _, _ = _sample(outcomes, predicted_pd, None, True)
    d = _validation_design(design,p); confidence = _confidence(confidence_level)
    edges, assignment = _bins(p,bin_edges); _days(minimum_count,'minimum_count',1)
    if interval_method not in ('none','clopper_pearson'):
        _fail('INVALID_INTERVAL_METHOD','Use none or clopper_pearson')
    if dependence_assumption not in ('descriptive_only','independent_common_probability_within_bins') or (interval_method == 'clopper_pearson' and dependence_assumption != 'independent_common_probability_within_bins'):
        _fail('INCOMPATIBLE_ASSUMPTIONS','Exact binomial intervals require a common independent event probability within bins')
    rows=[]
    for j in range(len(edges)-1):
        mask=assignment==j;n=int(mask.sum());k=int(y[mask].sum());mean=None if n==0 else float(p[mask].mean());rate=None if n==0 else k/n
        lower=upper=None
        if n and interval_method=='clopper_pearson':
            interval=binomtest(k,n).proportion_ci(confidence_level=confidence,method='exact');lower=float(interval.low);upper=float(interval.high)
        rows.append(dict(bin=j+1,lower_edge=float(edges[j]),upper_edge=float(edges[j+1]),upper_closed=j==len(edges)-2,
                         count=n,defaults=k,mean_pd=mean,observed_rate=rate,residual=None if n==0 else rate-mean,
                         lower=lower,upper=upper,small_bin=n<minimum_count,status='EMPTY' if n==0 else 'OBSERVED'))
    return _result(dict(bins=rows,assignments=(assignment+1).tolist(),total_count=len(y),total_defaults=int(y.sum())),
        'validation.calibration_curve','probability',context,diagnostics=dict(design=d,interval_method=interval_method,dependence_assumption=dependence_assumption,confidence_level=confidence,minimum_count=minimum_count,boundary_convention='left_closed_last_bin_includes_one'),
        limitations=('Bins must be chosen before inspecting outcomes. Exact count intervals assume common independent event probability within each bin; none is the appropriate descriptive policy when this is unsupported.',))


def spiegelhalter_test(predicted_pd, outcomes, *, design, alternative, independence_assumption, context=None):
    """Fixed-forecast Spiegelhalter Z using heterogeneous independent Bernoulli variance."""
    from scipy.stats import norm
    y,p,_,_=_sample(outcomes,predicted_pd,None,True);d=_validation_design(design,p);_alternative(alternative)
    if independence_assumption!='independent_fixed_bernoulli':
        _fail('INCOMPATIBLE_ASSUMPTIONS','Require independent_fixed_bernoulli')
    numerator=float(np.sum((y-p)*(1-2*p)));terms=p*(1-p)*(1-2*p)**2;variance=float(terms.sum())
    z=pvalue=None;status='NOT_APPLICABLE';reason='ZERO_VARIANCE'
    if variance>0:
        z=numerator/math.sqrt(variance);pvalue=float(2*norm.sf(abs(z)) if alternative=='two-sided' else norm.cdf(z) if alternative=='less' else norm.sf(z));status='APPLICABLE';reason=None
    return _result(dict(z=z,numerator=numerator,variance=variance,p_value=pvalue,applicability=status,reason=reason),
        'validation.spiegelhalter_test','test',context,diagnostics=dict(design=d,alternative=alternative,independence_assumption=independence_assumption,
        maximum_variance_share=None if variance==0 else float(terms.max()/variance),reference_distribution='asymptotic_standard_normal'),
        limitations=('Normal approximation needs an adequate non-dominant sum of Bernoulli terms. This statistic tests one calibration moment and is not an omnibus test; zero variance has no defined Z.',))


def poisson_binomial_pd_test(individual_pd, observed_default_count, *, design, alternative,
                             independence_assumption, method='dynamic_programming', context=None):
    """Exact Bernoulli convolution, exposing finite-precision mass loss and underflow."""
    p=_probability(individual_pd,vector_only=True);d=_validation_design(design,p);_alternative(alternative)
    k=_numeric(observed_default_count,'observed_default_count')
    if k.ndim or k<0 or k>len(p) or k!=np.floor(k):
        _fail('INVALID_COUNTS','Observed default count must be an integer in [0,n]')
    if independence_assumption!='independent_fixed_bernoulli':
        _fail('INCOMPATIBLE_ASSUMPTIONS','Require independent_fixed_bernoulli')
    if method!='dynamic_programming':
        _fail('INVALID_METHOD','Supported exact method is dynamic_programming')
    pmf=np.array([1.])
    for prob in p:
        pmf=np.convolve(pmf,np.array([1-prob,prob]))
    count=int(k);mass_error=float(abs(pmf.sum()-1));minimum=int(np.sum(p==1));maximum=int(np.sum(p>0))
    underflow=int(np.sum(pmf[minimum:maximum+1]==0))
    if mass_error>1e-10:
        _fail('NUMERICAL_FAILURE','Bernoulli convolution lost probability mass')
    tail=float(pmf[:count+1].sum() if alternative=='less' else pmf[count:].sum() if alternative=='greater' else pmf[pmf<=pmf[count]*(1+1e-7)].sum())
    reason='EMPTY_COHORT' if len(p)==0 else 'NUMERICAL_UNDERFLOW' if (alternative=='two-sided' and pmf[count]==0 and minimum<=count<=maximum) or (tail==0 and minimum<=count<=maximum) else None
    return _result(dict(counts=list(range(len(p)+1)),probability_mass=pmf.tolist(),p_value=None if reason else min(1.,tail),expected_count=float(p.sum()),variance=float(np.sum(p*(1-p))),applicability='NOT_APPLICABLE' if reason else 'APPLICABLE',reason=reason),
        'validation.poisson_binomial_pd_test','test',context,diagnostics=dict(design=d,alternative=alternative,independence_assumption=independence_assumption,method=method,mass_error=mass_error,underflowed_support_cells=underflow,two_sided_ordering='probability_mass',mass_order_relative_tolerance=1e-7),
        limitations=('Exact finite-precision dynamic programming takes quadratic work in trial count. Model estimation uncertainty and dependence are not included in this conditional null.',))


def hosmer_lemeshow_test(predicted_pd,outcomes,bin_edges,*,design,degrees_of_freedom_policy,
                        independence_assumption,minimum_expected=5.,context=None):
    """Grouped Pearson calibration statistic with explicit external/fitted design policy."""
    from scipy.stats import chi2
    y,p,_,_=_sample(outcomes,predicted_pd,None,True);d=_validation_design(design,p,allow_fitted=True);edges,assignment=_bins(p,bin_edges)
    threshold=_numeric(minimum_expected,'minimum_expected')
    if threshold.ndim or threshold<=0:
        _fail('INVALID_INPUT','minimum_expected must be a positive explicit cell-size rule')
    if independence_assumption!='independent_fixed_bernoulli':
        _fail('INCOMPATIBLE_ASSUMPTIONS','Require independent_fixed_bernoulli')
    if degrees_of_freedom_policy not in ('external_common_bin_pd','fitted_logistic_approximation'):
        _fail('INVALID_DF_POLICY','Declare external_common_bin_pd or fitted_logistic_approximation')
    fitted=degrees_of_freedom_policy=='fitted_logistic_approximation'
    if fitted!=(d['prediction_origin']=='FITTED_SAMPLE'):
        _fail('INVALID_VALIDATION_DESIGN','DF policy must match external versus fitted prediction origin')
    rows=[];reason=None;statistic=0.;active=0
    for j in range(len(edges)-1):
        mask=assignment==j;n=int(mask.sum());observed=int(y[mask].sum());expected=float(p[mask].sum());mean=expected/n if n else None
        contribution=None;variance=None;flags=[]
        if n:
            variance=expected*(1-mean)
            if variance==0:
                if observed!=expected:flags.append('IMPOSSIBLE_BIN_OUTCOME')
                else:contribution=0.;flags.append('DETERMINISTIC_BIN')
            else:
                active+=1;contribution=(observed-expected)**2/variance;statistic+=contribution
                if min(expected,n-expected)<float(threshold):flags.append('SMALL_EXPECTED_CELL')
            if not fitted and np.ptp(p[mask])!=0:flags.append('HETEROGENEOUS_BIN_PD')
        else:flags.append('EMPTY_BIN')
        if any(f in flags for f in ('IMPOSSIBLE_BIN_OUTCOME','SMALL_EXPECTED_CELL','HETEROGENEOUS_BIN_PD')) and reason is None:
            reason=next(f for f in flags if f in ('IMPOSSIBLE_BIN_OUTCOME','SMALL_EXPECTED_CELL','HETEROGENEOUS_BIN_PD'))
        rows.append(dict(bin=j+1,count=n,observed_defaults=observed,expected_defaults=expected,mean_pd=mean,variance=variance,contribution=contribution,flags=flags))
    df=active-2 if fitted else active
    if df<=0 and reason is None:reason='NONPOSITIVE_DF'
    if any('IMPOSSIBLE_BIN_OUTCOME' in r['flags'] for r in rows):statistic=None
    return _result(dict(statistic=statistic,degrees_of_freedom=df,p_value=None if reason else float(chi2.sf(statistic,df)),bins=rows,applicability='NOT_APPLICABLE' if reason else 'APPLICABLE',reason=reason),
        'validation.hosmer_lemeshow_test','test',context,diagnostics=dict(design=d,degrees_of_freedom_policy=degrees_of_freedom_policy,independence_assumption=independence_assumption,minimum_expected=float(threshold),reference_distribution='asymptotic_chi_square'),
        limitations=('The G-2 fitted-logistic policy is an explicitly chosen approximation, not a universal degrees-of-freedom correction. External fixed-bin inference requires common probabilities within each bin. Small expected cells withhold a p-value.',))


def dependent_pd_backtest(predicted_pd,outcomes,cluster_ids,time_ids,*,design,null_model,
                          innovations,alternative,confidence_level=.95,context=None):
    """Shared-innovation Gaussian-factor null for clustered and time-dependent counts."""
    from scipy.special import ndtri
    from .core import _fingerprint, _json_data
    y,p,_,_=_sample(outcomes,predicted_pd,None,True);d=_validation_design(design,p);_alternative(alternative);confidence=_confidence(confidence_level)
    n=len(y)
    for keys in (cluster_ids,time_ids):
        if not isinstance(keys,(list,tuple)) or len(keys)!=n or any(not isinstance(k,str) or not k for k in keys):
            _fail('INVALID_CLUSTERS','Require one nonempty cluster and time key per row')
    ids=d.get('observation_ids')
    if not isinstance(ids,list) or len(ids)!=n or len(set(ids))!=n or any(not isinstance(i,str) or not i for i in ids):
        _fail('INVALID_VALIDATION_DESIGN','Unique ordered observation_ids required for shared innovations')
    if not isinstance(null_model,dict) or not isinstance(innovations,dict):
        _fail('INVALID_NULL_MODEL','Explicit null model and innovations objects required')
    null_model=_json_data(null_model);innovations=_json_data(innovations)
    if null_model.get('family')!='gaussian_factor' or innovations.get('draw_assumption')!='independent_standard_normal_rows':
        _fail('INVALID_NULL_MODEL','Use declared Gaussian factor null and independent standard normal innovation rows')
    factors=sorted(set(zip(cluster_ids,time_ids)))
    factor_keys=[dict(cluster_id=c,time_id=t) for c,t in factors]
    if innovations.get('observation_ids')!=ids or innovations.get('factor_keys')!=factor_keys:
        _fail('INNOVATION_ALIGNMENT','Innovation columns must exactly match observation IDs and sorted cluster/time factors')
    Z=_numeric(innovations.get('factor_normals'),'factor_normals');E=_numeric(innovations.get('idiosyncratic_normals'),'idiosyncratic_normals')
    if Z.ndim!=2 or E.ndim!=2 or Z.shape[1]!=len(factors) or E.shape!=(Z.shape[0],n) or Z.shape[0]<2:
        _fail('INCOMPATIBLE_SHAPE','Normal innovation matrices need B>=2 rows and declared factor/observation columns')
    rho=_probability(null_model.get('asset_correlation'))
    if rho.ndim>1:_fail('INCOMPATIBLE_SHAPE','Correlation must be scalar or vector')
    rho=rho.reshape(-1)
    if len(rho)==1:rho=np.repeat(rho,n)
    if len(rho)!=n:_fail('INCOMPATIBLE_SHAPE','Correlation must be scalar or one per observation')
    C=_numeric(null_model.get('factor_correlation'),'factor_correlation')
    if C.shape!=(len(factors),len(factors)) or not np.allclose(C,C.T,rtol=0,atol=1e-12) or not np.allclose(np.diag(C),1,rtol=0,atol=1e-12):
        _fail('INVALID_CORRELATION','Factor correlation must be symmetric with unit diagonal')
    try:L=np.linalg.cholesky(C)
    except np.linalg.LinAlgError:_fail('INVALID_CORRELATION','Factor correlation must be positive definite; use shared factor keys for perfect common shocks')
    paths=Z@L.T;index=[factors.index(key) for key in zip(cluster_ids,time_ids)]
    latent=paths[:,index]*np.sqrt(rho)+E*np.sqrt(1-rho)
    uncertainty=null_model.get('parameter_uncertainty')
    if uncertainty=='conditional_fixed':
        if 'parameter_pd_draws' in innovations:_fail('INVALID_NULL_MODEL','Fixed-parameter policy cannot include parameter draws')
        thresholds=ndtri(p)[None,:]
    elif uncertainty=='supplied_predictive_pd_draws':
        draws=_probability(innovations.get('parameter_pd_draws'))
        if draws.shape!=E.shape:_fail('INCOMPATIBLE_SHAPE','PD parameter paths must align with simulation rows and observation columns')
        if not isinstance(null_model.get('parameter_draw_source'),str) or not null_model['parameter_draw_source']:
            _fail('INVALID_NULL_MODEL','Predictive parameter draws require an explicit source/identification description')
        thresholds=ndtri(draws)
    else:_fail('INVALID_NULL_MODEL','Declare conditional_fixed or supplied_predictive_pd_draws uncertainty')
    counts=np.sum(latent<=thresholds,axis=1);expected=float(p.sum());observed=int(y.sum());statistic=observed-expected;null_stat=counts-expected
    tail=(null_stat<=statistic) if alternative=='less' else (null_stat>=statistic) if alternative=='greater' else (np.abs(null_stat)>=abs(statistic))
    B=len(counts);extreme=int(tail.sum());pvalue=(extreme+1)/(B+1)
    interval=binomtest(extreme,B).proportion_ci(confidence_level=confidence,method='exact')
    sorted_counts=np.sort(counts)
    def empirical_quantile(prob):return int(sorted_counts[max(0,min(B-1,math.ceil(prob*B)-1))])
    low=empirical_quantile((1-confidence)/2);high=empirical_quantile((1+confidence)/2)
    return _result(dict(observed_count=observed,expected_count=expected,statistic=statistic,p_value=pvalue,null_counts=counts.astype(int).tolist(),null_statistics=null_stat.tolist(),null_count_interval=[low,high],tail_probability_interval=[float(interval.low),float(interval.high)],applicability='APPLICABLE',reason=None),
        'validation.dependent_pd_backtest','test',context,diagnostics=dict(design=d,null_model=null_model,alternative=alternative,confidence_level=confidence,replicates=B,extreme_replicates=extreme,p_value_resolution=1/(B+1),innovation_fingerprint=_fingerprint(innovations),factor_keys=factor_keys,monte_carlo_rule='(extreme+1)/(B+1)',two_sided_ordering='absolute_count_residual',parameter_uncertainty=uncertainty),
        limitations=('Inference is conditional on the declared dependence null and exchangeable independent simulation rows. Supplied parameter draws define a predictive mixture, not an automatically estimated frequentist sampling distribution. Tail-probability intervals measure Monte Carlo error, and count intervals are null prediction intervals.',))


def default_count_backtest(observation_panel,frozen_forecasts,group_by,*,unit,horizon_days,
                           default_definition,redefault_policy,evaluation_cutoff,context=None):
    """Compare mature complete cohorts, deduplicating the declared default unit."""
    from .data import _date,_days,_rows,_unique,_required
    from .core import _fingerprint
    rows=_rows(observation_panel,'observation_panel');forecasts=_rows(frozen_forecasts,'frozen_forecasts')
    _unique(rows,['observation_id'],'observation_panel');_unique(forecasts,['observation_id'],'frozen_forecasts')
    horizon=_days(horizon_days,'horizon_days',1);cutoff=_date(evaluation_cutoff,'evaluation_cutoff')
    if unit not in ('obligor','facility') or redefault_policy!='exclude_previous_default':
        _fail('INVALID_COHORT_POLICY','Declare obligor/facility unit and exclude_previous_default')
    if not isinstance(default_definition,str) or not default_definition:
        _fail('INVALID_COHORT_POLICY','Nonempty default definition required')
    if not isinstance(group_by,(list,tuple)) or len(set(group_by))!=len(group_by) or any(not isinstance(g,str) or not g for g in group_by):
        _fail('INVALID_COHORT_POLICY','group_by must contain distinct field names')
    known_ids={r['observation_id'] for r in rows};forecast_map={f['observation_id']:f for f in forecasts}
    if any(f['observation_id'] not in known_ids for f in forecasts):_fail('FORECAST_FOREIGN_KEY','Forecast references an unknown observation')
    key='entity_id' if unit=='obligor' else 'facility_id';units={};excluded=[];duplicate_count=0
    for row in sorted(rows,key=lambda r:r['observation_id']):
        _required(row,['observation_id',key,'as_of','performance_start','outcome_end','eligible','prior_default','default_definition',*group_by])
        if type(row['eligible']) is not bool or type(row['prior_default']) is not bool or 'target' not in row:
            _fail('INVALID_COHORT_POLICY','Rows require explicit eligibility/prior-default booleans and target, possibly null')
        start,end,asof=(_date(row[f],f) for f in ('performance_start','outcome_end','as_of'))
        if start<asof or (end-start).days!=horizon or row['default_definition']!=default_definition:
            _fail('INCOMPATIBLE_TARGET','Observation timing/default definition differs from cohort contract')
        if row['target'] is not None:
            if isinstance(row['target'],bool) or row['target'] not in (0,1):_fail('INVALID_OUTCOMES','Observed target must be zero/one or missing')
            _required(row,['outcome_known_at'])
            observed_at=_date(row['outcome_known_at'],'outcome_known_at')
            if observed_at<start or (row['target']==0 and observed_at<end):_fail('INVALID_COHORT_POLICY','Outcome cannot be known before occurrence/completion')
        f=forecast_map.get(row['observation_id'])
        if f is None:
            excluded.append(dict(observation_id=row['observation_id'],reason='MISSING_FORECAST'));continue
        _required(f,['pd','forecast_at','known_at','default_definition','horizon_days','unit','probability_measure','model_id','model_version'])
        prob=_probability(f['pd'])
        if prob.ndim:_fail('INCOMPATIBLE_SHAPE','Each forecast must be scalar')
        if not _date(f['forecast_at'],'forecast_at')<=_date(f['known_at'],'known_at')<=asof:
            _fail('FORECAST_LEAKAGE','Forecast must have been created and known by observation date')
        if f['default_definition']!=default_definition or f['horizon_days']!=horizon or f['unit']!=unit or f['probability_measure']!='PHYSICAL':
            _fail('INCOMPATIBLE_TARGET','Forecast target/unit differs from cohort contract')
        grouping={g:row[g] for g in group_by}
        normalized=dict(unit_id=row[key],as_of=row['as_of'],performance_start=row['performance_start'],outcome_end=row['outcome_end'],eligible=row['eligible'],prior_default=row['prior_default'],target=row['target'],outcome_known_at=row.get('outcome_known_at'),pd=float(prob),group=grouping)
        dedup=(row[key],row['as_of'],row['performance_start'],row['outcome_end'])
        if dedup in units:
            if _fingerprint(units[dedup]['normalized'])!=_fingerprint(normalized):_fail('DUPLICATE_UNIT_CONFLICT','Repeated default-unit/window has inconsistent forecasts, group or outcome')
            duplicate_count+=1;continue
        reason='FUTURE_OBSERVATION' if asof>cutoff else 'NOT_ELIGIBLE' if not row['eligible'] else 'PRIOR_DEFAULT' if row['prior_default'] else 'IMMATURE' if end>cutoff else 'MISSING_OUTCOME' if row['target'] is None else 'OUTCOME_NOT_YET_KNOWN' if observed_at>cutoff else None
        units[dedup]=dict(normalized=normalized,reason=reason,observation_id=row['observation_id'])
    groups={}
    for record in units.values():
        info=record['normalized'];digest=_fingerprint(info['group'])
        groups.setdefault(digest,dict(group=info['group'],unit_windows=0,observed_count=0,observed_defaults=0,expected_defaults=0.,exclusions=[]))
        g=groups[digest];g['unit_windows']+=1
        if record['reason']:
            item=dict(observation_id=record['observation_id'],reason=record['reason']);g['exclusions'].append(item);excluded.append(item)
        else:g['observed_count']+=1;g['observed_defaults']+=int(info['target']);g['expected_defaults']+=info['pd']
    def finish(g):
        expected=g['expected_defaults'];observed=g['observed_defaults']
        g['observed_expected_ratio']=None if expected==0 else observed/expected
        g['ratio_status']='INFINITE' if expected==0 and observed>0 else 'UNDEFINED' if expected==0 else 'FINITE'
        g['observed_rate']=None if g['observed_count']==0 else observed/g['observed_count']
        g['mean_pd']=None if g['observed_count']==0 else expected/g['observed_count']
        return g
    output=[finish(groups[k]) for k in sorted(groups)]
    overall=finish(dict(unit_windows=sum(g['unit_windows'] for g in output),observed_count=sum(g['observed_count'] for g in output),observed_defaults=sum(g['observed_defaults'] for g in output),expected_defaults=sum(g['expected_defaults'] for g in output)))
    return _result(dict(groups=output,overall=overall,excluded=sorted(excluded,key=lambda e:e['observation_id']),deduplicated_rows=duplicate_count),
        'validation.default_count_backtest','count',context,diagnostics=dict(unit=unit,horizon_days=horizon,default_definition=default_definition,redefault_policy=redefault_policy,evaluation_cutoff=evaluation_cutoff,group_by=list(group_by),forecast_fingerprint=_fingerprint(forecasts),observation_fingerprint=_fingerprint(rows),comparison_population='mature_complete_eligible_unit_windows'),
        limitations=('Only mature complete windows contribute to both observed and expected counts, including when a default is observed early. Repeated time windows may remain dependent; this descriptive report supplies no binomial p-value. Eligibility and default identity are caller-provided evidence.',))

from ._validation_losses import lgd_backtest, ead_backtest
from ._validation_survival import survival_backtest
from ._validation_stability import population_stability
from ._validation_migration import rating_migration_backtest

from ._validation_benchmarks import benchmark_predictions, benchmark_ratings

from ._validation_interventions import override_monitoring, support_transfer_backtest

from ._validation_numerics import simulation_convergence, sensitivity_analysis

from ._validation_retail import validate_retail_cohort, roll_rate_backtest

from ._validation_vintage import vintage_backtest

from ._validation_cashflows import collections_backtest

from ._validation_system import validate_system_coverage

from ._validation_release import release_candidate_assessment

from ._validation_report import validation_report

from ._validation_lineage import validate_source_lineage

from ._validation_replay import replay_consistency

from ._validation_counterparty import counterparty_exposure_backtest

from ._validation_affordability import affordability_stress_backtest

from ._validation_selection import selection_bias_sensitivity

from ._validation_policy import decision_policy_backtest

from ._validation_retail_pools import retail_pool_diagnostics

from ._validation_asset_pools import asset_pool_backtest
