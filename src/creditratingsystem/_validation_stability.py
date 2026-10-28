"""Frozen-bin descriptive population stability with explicit empty-cell limits."""
import math
import numpy as np
from scipy.stats import wasserstein_distance
from .core import _numeric,_fail,_result,_fingerprint
from .data import _date
from ._expert_common import _ex_object,_ex_ids
from ._validation_common import _vb_design,_vb_number,_vb_string


def population_stability(reference,current,*,binning,metrics,smoothing,design,reference_weights=None,current_weights=None,context=None):
    """PSI, Jensen-Shannon divergence and raw numeric first Wasserstein distance."""
    d=_vb_design(design,context,False);spec=_ex_object(binning,'binning');names=_ex_ids(metrics,'metrics');alpha=_vb_number(smoothing,'smoothing',0)
    if set(names)-{'psi','jsd','wasserstein'}:_fail('VALIDATION_DRIFT','Unknown population-stability metric')
    for key in ('binning_id','version'):_vb_string(spec.get(key),key)
    known=_date(spec.get('known_at'),'binning known_at');reference_asof=_date(d.get('reference_as_of'),'reference_as_of');current_asof=_date(d.get('current_as_of'),'current_as_of')
    if known>min(reference_asof,current_asof)or max(reference_asof,current_asof)>_date(d['evaluation_cutoff'],'evaluation_cutoff'):_fail('VALIDATION_LOOKAHEAD','Bins must precede both samples and samples precede evaluation')
    if not isinstance(reference,(list,tuple))or not isinstance(current,(list,tuple)):_fail('VALIDATION_DRIFT','One-dimensional value arrays required')
    missing=lambda v:v is None or isinstance(v,(float,np.floating))and math.isnan(v)
    kind=spec.get('kind')
    if kind=='numeric':
        cuts=_numeric(spec.get('cutpoints'),'cutpoints',allow_empty=True,vector_only=True);support=_numeric(spec.get('reference_support'),'reference_support',vector_only=True)
        if np.any(np.diff(cuts)<=0)or support.shape!=(2,)or support[0]>=support[1]:_fail('VALIDATION_BINS','Strictly increasing cutpoints and ordered finite support required')
        size=len(cuts)+2
        def encode(v):return size-1 if missing(v)else int(np.searchsorted(cuts,_vb_number(v,'value'),side='left'))
        def novel(v):return not missing(v)and not support[0]<=_vb_number(v,'value')<=support[1]
    elif kind=='categorical':
        levels=_ex_ids(spec.get('levels'),'levels');size=len(levels)+2
        def encode(v):
            if missing(v):return size-2
            _vb_string(v,'categorical value');return levels.index(v)if v in levels else size-1
        def novel(v):return not missing(v)and v not in levels
        if 'wasserstein'in names:_fail('VALIDATION_DRIFT','Wasserstein requires numeric observations with a meaningful distance')
    else:_fail('VALIDATION_BINS','Use numeric or categorical bins')
    def sample(values,weights):
        w=np.ones(len(values))if weights is None else _numeric(weights,'weights',allow_empty=True,vector_only=True)
        if w.shape!=(len(values),)or np.any(w<0)or not np.isfinite(w.sum()):_fail('VALIDATION_WEIGHT','Aligned nonnegative finite-sum weights required')
        counts=np.bincount([encode(v)for v in values],weights=w,minlength=size).astype(float)
        return w,counts
    wr,a=sample(reference,reference_weights);wc,b=sample(current,current_weights);supported=a.sum()>0 and b.sum()>0;findings={};pa=a/a.sum()if a.sum()>0 else None;pb=b/b.sum()if b.sum()>0 else None;psi_a=psi_b=None
    if supported:
        def smooth(counts):
            scale=max(float(np.max(counts)),alpha,1.);v=counts/scale+alpha/scale;return v/v.sum()
        psi_a=smooth(a);psi_b=smooth(b)
        if 'psi'in names:
            contributions=[];infinite=False
            for p,q in zip(psi_a,psi_b):
                if p==q:contributions.append(0.)
                elif p==0 or q==0:contributions.append(None);infinite=True
                else:contributions.append(float((q-p)*(math.log(q)-math.log(p))))
            findings['psi']=dict(value=None if infinite else float(sum(contributions)),status='INFINITE'if infinite else'FINITE',contributions=contributions,smoothing_pseudocount=alpha)
        if 'jsd'in names:
            middle=(pa+pb)/2;parts=np.zeros(size);left=pa>0;right=pb>0;parts[left]+=.5*pa[left]*np.log(pa[left]/middle[left]);parts[right]+=.5*pb[right]*np.log(pb[right]/middle[right]);findings['jsd']=dict(value=float(parts.sum()),status='FINITE',contributions=parts.tolist(),log_base='natural',smoothing_pseudocount=0.)
        if 'wasserstein'in names:
            ia=np.array([not missing(v)for v in reference]);ib=np.array([not missing(v)for v in current]);wa=wr[ia];wb=wc[ib];good=wa.sum()>0 and wb.sum()>0
            distance=float(wasserstein_distance(np.array(reference,dtype=float)[ia],np.array(current,dtype=float)[ib],wa,wb))if good else None
            if distance is not None and not np.isfinite(distance):_fail('VALIDATION_NUMERICAL','Wasserstein distance exceeds finite range')
            findings['wasserstein']=dict(value=distance,status='FINITE'if good else'NOT_APPLICABLE',reason=None if good else'NO_NONMISSING_SUPPORT',missing_policy='EXCLUDED_AND_REPORTED_SEPARATELY')
    else:
        findings={name:dict(value=None,status='NOT_APPLICABLE',reason='EMPTY_WEIGHTED_POPULATION')for name in names}
    coverage=dict(reference_count=len(reference),current_count=len(current),reference_weight=float(wr.sum()),current_weight=float(wc.sum()),reference_missing_count=sum(missing(v)for v in reference),current_missing_count=sum(missing(v)for v in current),reference_missing_share=float(sum(w for v,w in zip(reference,wr)if missing(v))/wr.sum())if wr.sum()>0 else None,current_missing_share=float(sum(w for v,w in zip(current,wc)if missing(v))/wc.sum())if wc.sum()>0 else None,reference_novel_count=sum(novel(v)for v in reference),current_novel_count=sum(novel(v)for v in current),novel_definition='OUTSIDE_DECLARED_REFERENCE_SUPPORT'if kind=='numeric'else'UNSEEN_FIXED_CATEGORY')
    coverage['missing_share_change']=None if coverage['reference_missing_share']is None or coverage['current_missing_share']is None else coverage['current_missing_share']-coverage['reference_missing_share']
    value=dict(artifact_type='ValidationResult',metrics=findings,uncertainty=dict(status='NOT_REQUESTED',method=None),p_value=None,decision=dict(status='NOT_ASSESSED',rule=None),applicability='APPLICABLE'if supported else'NOT_APPLICABLE',reason=None if supported else'EMPTY_WEIGHTED_POPULATION',design=d,binning=spec,coverage=coverage,reference_counts=a.tolist(),current_counts=b.tolist(),reference_shares=None if pa is None else pa.tolist(),current_shares=None if pb is None else pb.tolist(),psi_reference_shares=None if psi_a is None else psi_a.tolist(),psi_current_shares=None if psi_b is None else psi_b.tolist(),input_fingerprint=_fingerprint(dict(reference=[None if missing(v)else v for v in reference],current=[None if missing(v)else v for v in current],reference_weights=wr.tolist(),current_weights=wc.tolist())))
    return _result(value,'validation.population_stability','distribution_distance',context,limitations=('Distances are descriptive; no universal PSI threshold, statistical significance or concept-drift conclusion follows. PSI smoothing applies to fixed histogram counts only; JSD uses empirical masses and Wasserstein uses observed raw numeric values.',))
