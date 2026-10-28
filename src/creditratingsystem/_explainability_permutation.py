"""Frozen held-out prediction loss under explicit shared feature permutations."""
from copy import deepcopy
import math
import numpy as np
from .core import _fail,_numeric,_result,_fingerprint,CreditRiskError
from .data import _required,_rows,_date
from .portfolio import _ids,_weights,_sum
from .explainability import _ex_contract,_ex_predict,_ex_groups


def credit_permutation_importance(predictor,X,y,model_contract,feature_groups,permutations,design,loss_contract,*,row_metadata,weights=None,strata=None,custom_loss=None,support_evaluator=None,support_contract=None,context=None):
    """Compare loss without refitting; fixed plans carry no IID uncertainty claim."""
    features,ctx=_ex_contract(model_contract,context);x=_numeric(X,'X');labels=_numeric(y,'y',vector_only=True)
    if x.ndim!=2 or x.shape!=(len(labels),len(features)):_fail('PERMUTATION_SHAPE','Aligned held-out row matrix and labels required')
    n=len(labels);w=_weights([1]*n if weights is None else weights,n);groups,indices=_ex_groups(feature_groups,features)
    if not isinstance(design,dict):_fail('PERMUTATION_DESIGN','Explicit frozen evaluation design required')
    _required(design,['split','training_end','data_cutoff','train_row_ids','train_subject_ids','model_and_calibration_frozen','selection_use','permutation_design'])
    if design['split']not in ('OUT_OF_TIME','HOLDOUT')or design['model_and_calibration_frozen']is not True or design['selection_use']!='FORBIDDEN'or design['permutation_design']not in ('FIXED_PERMUTATIONS','IID_RANDOM_PERMUTATIONS'):_fail('PERMUTATION_DESIGN','Held-out frozen evaluation without feature selection and explicit permutation law required')
    train_end=_date(design['training_end'],'training_end');cutoff=_date(design['data_cutoff'],'data_cutoff')
    if train_end>cutoff:_fail('PERMUTATION_DESIGN','Training cutoff cannot follow evaluation cutoff')
    for k in ('train_row_ids','train_subject_ids'):
        if not isinstance(design[k],list)or any(not isinstance(v,str)or not v for v in design[k])or len(set(design[k]))!=len(design[k]):_fail('PERMUTATION_DESIGN','Explicit unique excluded training identities required')
    rows=_rows(row_metadata,'row_metadata')
    if len(rows)!=n:_fail('PERMUTATION_SHAPE','One evaluation metadata record per row required')
    ids=[]
    for row in rows:
        _required(row,['row_id','subject_id','observation_time','target_end','label_known_time']);ids.append(row['row_id']);_ids([row['subject_id']],'subject_id')
        observed=_date(row['observation_time'],'observation_time');end=_date(row['target_end'],'target_end');known=_date(row['label_known_time'],'label_known_time')
        if row['row_id']in design['train_row_ids']or row['subject_id']in design['train_subject_ids']:_fail('PERMUTATION_LEAKAGE','Evaluation row or subject occurs in excluded fitting identities')
        if not observed<end<=cutoff or not observed<=known<=cutoff or (design['split']=='OUT_OF_TIME'and observed<=train_end):_fail('PERMUTATION_TIMING','Mature, known outcomes and declared out-of-time observations required')
    _ids(ids,'row_ids');plan=_numeric(permutations,'permutations')
    if plan.ndim!=2 or plan.shape[1]!=n or np.any(plan!=np.floor(plan))or np.any(plan<0)or np.any(plan>=n):_fail('PERMUTATION_PLAN','Zero-based complete integer permutation rows required')
    plan=plan.astype(int)
    if any(sorted(p.tolist())!=list(range(n))for p in plan):_fail('PERMUTATION_PLAN','Each plan row must be a bijection')
    if strata is not None:
        if not isinstance(strata,list)or len(strata)!=n or any(not isinstance(s,str)or not s for s in strata):_fail('PERMUTATION_STRATA','One explicit stratum label per row required')
        if any(strata[i]!=strata[int(p[i])]for p in plan for i in range(n)):_fail('PERMUTATION_STRATA','Permutation must preserve every recipient stratum')
    if not isinstance(loss_contract,dict):_fail('PERMUTATION_LOSS','Versioned loss contract required')
    _required(loss_contract,['loss_id','version','kind']);kind=loss_contract['kind']
    if kind not in ('BRIER','LOG_LOSS','MSE','MAE','CUSTOM')or (kind=='CUSTOM')!=(custom_loss is not None):_fail('PERMUTATION_LOSS','Supported declared loss and matching custom callback required')
    if kind in ('BRIER','LOG_LOSS')and (model_contract['output_scale']!='probability'or np.any((labels!=0)&(labels!=1))):_fail('PERMUTATION_LOSS','Binary probability loss requires binary mature outcomes')
    def loss(pred):
        if kind in ('BRIER','MSE'):v=(pred-labels)**2
        elif kind=='MAE':v=np.abs(pred-labels)
        elif kind=='LOG_LOSS':
            with np.errstate(divide='ignore'):v=np.where(labels==1,-np.log(pred),-np.log1p(-pred))
            # Infinite losses have no finite paired-difference estimate. Do not clip.
            if np.any(~np.isfinite(v)&(w>0)):_fail('PERMUTATION_INFINITE_LOSS','Positive-weight log loss is infinite; paired importance is not estimable without a separately declared model probability floor')
            v=np.where(w==0,0,v)
        else:
            try:v=custom_loss(labels.copy(),pred.copy())
            except CreditRiskError:raise
            except Exception:_fail('PERMUTATION_LOSS','Custom loss callback failed')
        v=_numeric(v,'loss values',vector_only=True)
        if v.shape!=(n,):_fail('PERMUTATION_LOSS','One finite loss per evaluation row required')
        return _sum(w*v)
    if (support_evaluator is None)!=(support_contract is None)or (support_evaluator is not None and not callable(support_evaluator)):_fail('PERMUTATION_SUPPORT','Support evaluator and frozen support contract required together')
    if support_contract is not None:
        if not isinstance(support_contract,dict):_fail('PERMUTATION_SUPPORT','Versioned support contract required')
        _required(support_contract,['support_id','version'])
    def support(z):
        if support_evaluator is None:return None
        try:mask=np.asarray(support_evaluator(z.copy()))
        except Exception:_fail('PERMUTATION_SUPPORT','Support callback failed')
        if mask.dtype!=bool or mask.shape!=(n,):_fail('PERMUTATION_SUPPORT','One boolean support flag per row required')
        return dict(unsupported_row_ids=[ids[i]for i in range(n)if not mask[i]],unsupported_weight=float(w@(~mask)))
    base_pred=_ex_predict(predictor,x,model_contract);base_loss=loss(base_pred);base_support=support(x);reports=[]
    for gid,idx in zip(groups,indices):
        losses=[];support_reports=[]
        for p in plan:
            z=x.copy();z[:,idx]=x[p][:,idx];losses.append(loss(_ex_predict(predictor,z,model_contract)));support_reports.append(support(z))
        changes=np.asarray(losses)-base_loss;se=None
        if design['permutation_design']=='IID_RANDOM_PERMUTATIONS'and len(plan)>1:se=float(np.std(changes,ddof=1)/math.sqrt(len(plan)))
        reports.append(dict(group_id=gid,feature_ids=[features[i]for i in idx],permuted_losses=losses,loss_increases=changes.tolist(),mean_loss_increase=float(np.mean(changes)),permutation_standard_error=se,support_findings=support_reports))
    means=w@x;center=x-means;cov=(center*w[:,None]).T@center;sd=np.sqrt(np.diag(cov));corr=[[None if sd[i]==0 or sd[j]==0 else float(cov[i,j]/(sd[i]*sd[j]))for j in range(len(features))]for i in range(len(features))]
    return _result(dict(baseline_loss=base_loss,baseline_predictions=base_pred.tolist(),baseline_support=base_support,row_ids=ids,normalized_weights=w.tolist(),groups=reports,feature_order=features,weighted_reference_correlation=corr,permutation_count=len(plan),design=deepcopy(design),loss_contract=deepcopy(loss_contract),support_status='NOT_ASSESSED'if support_evaluator is None else'DECLARED_SUPPORT_CHECKED_WITHOUT_EXCLUDING_ROWS'),'explainability.credit_permutation_importance','loss_increase',ctx,diagnostics=dict(model_contract_hash=_fingerprint(model_contract),data_hash=_fingerprint(dict(X=x.tolist(),y=labels.tolist(),row_metadata=rows)),permutation_hash=_fingerprint(plan.tolist()),support_contract_hash=None if support_contract is None else _fingerprint(support_contract)),limitations=('No fitting or feature selection is performed. Training exclusion and frozen status are declared provenance, not authenticated. Blocks are permuted jointly but dependence across blocks can be broken; correlation is descriptive. Unsupported rows remain in loss denominators. IID permutation standard errors concern permutation randomness only, not cohort, model or selection uncertainty. Fixed plans have no IID standard error.',))
