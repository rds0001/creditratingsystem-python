"""Estimated logistic scorecards, declared odds scales and contribution explanations."""
from collections.abc import Mapping
import math
import numpy as np
from scipy.special import expit
from .core import _fail,_numeric,_model_artifact,_fit_result,_result,_fingerprint
from .classification import fit_binary_glm,predict_pd


def fit_scorecard(design_matrix,outcomes,*,feature_order,fit_spec,target,training,model_id,model_version,weights=None,max_iter=100,tolerance=1e-10,context=None):
    """Fit a logit scorecard on already frozen WOE or expert feature columns."""
    if not isinstance(fit_spec,Mapping):_fail('INVALID_SCORECARD_SPEC','Explicit feature kinds and neutral profile required')
    neutral=_numeric(fit_spec.get('neutral_profile'),'neutral_profile',vector_only=True)
    kinds=fit_spec.get('feature_kinds')
    if len(neutral)!=len(feature_order) or not isinstance(kinds,(list,tuple)) or len(kinds)!=len(feature_order) or any(k not in ('woe','expert') for k in kinds):_fail('INVALID_SCORECARD_SPEC','One neutral value and declared kind per feature required')
    fitted=fit_binary_glm(design_matrix,outcomes,link='logit',feature_order=feature_order,fit_intercept=True,target=target,training=training,model_id=model_id,model_version=model_version,weights=weights,max_iter=max_iter,tolerance=tolerance,context=context)
    model=fitted.model;model['method_id']='scorecard.fit_scorecard';model['scorecard_spec']={'neutral_profile':neutral.tolist(),'feature_kinds':list(kinds),'design_status':'CALLER_FROZEN','design_fingerprint':_fingerprint(_numeric(design_matrix,'design_matrix').tolist())}
    model=_model_artifact(model)
    return _fit_result(model,'scorecard.fit_scorecard',diagnostics=model['diagnostics'],context=context)


def score_scorecard(fitted_model,transformed_features,*,feature_order,expected_target,context=None):
    """Replay PD, log-odds and feature contributions, including neutral contrasts."""
    model=_model_artifact(fitted_model)
    if model['method_id']!='scorecard.fit_scorecard' or model['parameters'].get('link')!='logit':_fail('UNSUPPORTED_MODEL','Logistic scorecard artifact required')
    copy=dict(model);copy['method_id']='classification.fit_binary_glm'
    prediction=predict_pd(copy,transformed_features,feature_order=feature_order,expected_target=expected_target,context=context)
    x=_numeric(transformed_features,'transformed_features');beta=_numeric(model['parameters']['coefficients'],'coefficients',vector_only=True)
    neutral=_numeric(model.get('scorecard_spec',{}).get('neutral_profile'),'neutral_profile',vector_only=True)
    if len(beta)!=x.shape[1]+1 or len(neutral)!=x.shape[1]:_fail('INVALID_MODEL','Scorecard coefficient or neutral dimensions differ')
    contributions=x*beta[1:];eta=beta[0]+contributions.sum(axis=1);reference=neutral*beta[1:];differences=contributions-reference
    if np.any(~np.isfinite(eta)) or np.any(~np.isfinite(contributions)):_fail('NUMERICAL_RANGE','Scorecard contribution overflow')
    return _result({'pd':prediction.value,'log_odds':eta,'intercept':float(beta[0]),'feature_contributions':contributions,'reference_contributions':reference,'contribution_differences':differences,'feature_order':list(feature_order)},'scorecard.score_scorecard','log_odds_default_over_nondefault',context,
        diagnostics={'model_id':model['model_id'],'model_version':model['model_version'],'target':model['target'],'neutral_profile':neutral.tolist(),'refitted':False})


def scale_score_points(log_odds,*,base_score,base_odds,points_to_double_odds,odds_convention,score_direction,context=None):
    """Map default/nondefault log-odds to points with explicit inverse parameters."""
    eta=_numeric(log_odds,'log_odds')
    if eta.ndim>1:_fail('INVALID_INPUT','Scalar or vector log odds required')
    values=[]
    for v,name in [(base_score,'base_score'),(base_odds,'base_odds'),(points_to_double_odds,'points_to_double_odds')]:
        x=_numeric(v,name)
        if x.ndim:_fail('INVALID_INPUT',f'{name} must be scalar')
        values.append(float(x))
    base,odds,pdo=values
    if odds<=0 or pdo<=0 or odds_convention not in ('default_over_nondefault','nondefault_over_default') or score_direction not in ('higher_is_safer','higher_is_riskier'):_fail('INVALID_SCORE_SCALE','Declare positive odds/PDO, odds convention and direction')
    base_eta=math.log(odds)*(1 if odds_convention=='default_over_nondefault' else -1)
    slope=pdo/math.log(2)*(-1 if score_direction=='higher_is_safer' else 1);intercept=base-slope*base_eta
    points=intercept+slope*eta
    if not math.isfinite(intercept) or np.any(~np.isfinite(points)):_fail('NUMERICAL_RANGE','Score scaling overflow')
    return _result({'points':float(points) if points.ndim==0 else points,'inverse':{'score_intercept':intercept,'score_slope_per_default_log_odds':slope,'formula':'pd=expit((points-score_intercept)/score_slope_per_default_log_odds)'}},'scorecard.scale_score_points','score_points',context,
        diagnostics={'base_score':base,'base_odds':odds,'points_to_double_odds':pdo,'odds_convention':odds_convention,'score_direction':score_direction,'input_log_odds':'default_over_nondefault','rounding':'none'})


def apply_qualitative_adjustments(qualitative_inputs,coefficient_spec,neutral_levels,*,base_value,adjustment_scale,context=None):
    """Translate levels relative to declared neutral levels on one explicit scale."""
    if not all(isinstance(x,Mapping) for x in (qualitative_inputs,coefficient_spec,neutral_levels)) or not coefficient_spec:_fail('INVALID_ADJUSTMENT','Named inputs, coefficients and neutral levels required')
    base=_numeric(base_value,'base_value')
    if base.ndim or adjustment_scale not in ('log_odds','rating_notches'):_fail('INVALID_ADJUSTMENT','Scalar base and explicit log_odds/rating_notches scale required')
    contributions={}
    for feature,levels in coefficient_spec.items():
        value=qualitative_inputs.get(feature);neutral=neutral_levels.get(feature)
        if value is None:_fail('MISSING_INPUT',f'{feature}: missing is not neutral')
        if not isinstance(levels,Mapping) or not isinstance(value,str) or not isinstance(neutral,str) or value not in levels or neutral not in levels:_fail('UNKNOWN_CATEGORY','Observed and neutral levels require coefficients')
        selected=_numeric(levels[value],'coefficient');reference=_numeric(levels[neutral],'neutral_coefficient')
        if selected.ndim or reference.ndim:_fail('INVALID_ADJUSTMENT','Scalar level coefficients required')
        contribution=float(selected-reference)
        if adjustment_scale=='rating_notches' and contribution!=round(contribution):_fail('INCOMPATIBLE_UNIT','Rating-notch adjustments must be integer differences')
        contributions[feature]=contribution
    adjusted=float(base)+sum(contributions.values())
    if not math.isfinite(adjusted):_fail('NUMERICAL_RANGE','Adjustment overflow')
    return _result({'contributions':contributions,'total_adjustment':sum(contributions.values()),'adjusted_value':adjusted,'adjusted_pd':float(expit(adjusted)) if adjustment_scale=='log_odds' else None},'scorecard.apply_qualitative_adjustments',adjustment_scale,context,
        diagnostics={'neutral_levels':dict(neutral_levels),'missing_is_neutral':False},limitations=('Rating notches are not log-odds; notch results require a separately declared rating scale and boundary policy.',))


def scorecard_reason_codes(score_contributions,reference_profile,reason_policy,*,context=None):
    """Rank contribution contrasts deterministically; model explanations are not causes."""
    if not all(isinstance(x,Mapping) for x in (score_contributions,reference_profile,reason_policy)) or set(score_contributions)!=set(reference_profile) or not score_contributions:_fail('INVALID_REASON_POLICY','Matching contribution and reference feature sets required')
    direction=reason_policy.get('adverse_direction');limit=_numeric(reason_policy.get('max_reasons'),'max_reasons');codes=reason_policy.get('reason_codes')
    if direction not in ('positive','negative') or limit.ndim or limit<1 or limit!=int(limit) or not isinstance(codes,Mapping) or set(codes)!=set(score_contributions):_fail('INVALID_REASON_POLICY','Explicit direction, positive count and code for every feature required')
    rows=[];total=0.
    for name in sorted(score_contributions):
        if not isinstance(codes[name],str) or not codes[name]:_fail('INVALID_REASON_POLICY','Nonempty reason codes required')
        contribution=_numeric(score_contributions[name],'contribution');reference=_numeric(reference_profile[name],'reference')
        if contribution.ndim or reference.ndim:_fail('INVALID_INPUT','One scalar contribution per feature required')
        delta=float(contribution-reference);total+=delta;adverse=delta*(1 if direction=='positive' else -1)
        rows.append({'feature':name,'reason_code':codes[name],'difference':delta,'adverse_magnitude':adverse})
    if not math.isfinite(total):_fail('NUMERICAL_RANGE','Contribution difference overflow')
    selected=sorted((r for r in rows if r['adverse_magnitude']>0),key=lambda r:(-r['adverse_magnitude'],r['feature']))[:int(limit)]
    return _result({'reasons':selected,'all_differences':rows,'total_difference':total,'unreported_difference':total-sum(r['difference'] for r in selected)},'scorecard.scorecard_reason_codes','contribution_differences',context,
        diagnostics={'adverse_direction':direction,'tie_break':'feature_name_lexicographic','causal_interpretation':False})
