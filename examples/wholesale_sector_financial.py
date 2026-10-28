"""Sector-specific statements, genuine native fits and dated support/override evidence."""
from importlib.resources import files
from copy import deepcopy
import math
import json
import creditratingsystem as cr


def fit_sector_case_models(case,target):
    training=case['score_training']
    prep=cr.fit_preprocessor(case['preprocessor_training'],[dict(name='risk_ratio',type='numeric',imputation='error',scaling='standard')],target=target,training=training['training'],model_id=case['case_id']+'-preprocessor',model_version='1')
    score=cr.fit_scorecard(training['X'],training['y'],feature_order=training['feature_order'],fit_spec=training['fit_spec'],target=target,training=training['training'],model_id=case['case_id']+'-scorecard',model_version='1',weights=training['weights'])
    cal=case['calibration'];calibrator=cr.calibrate_logistic(cal['values'],cal['outcomes'],cal['weights'],metadata=cal['metadata'],**cal['fit_controls'])
    return dict(preprocessor=prep,scorecard=score,calibrator=calibrator)


def run(data=None,verify_reference=True,return_fitted_models=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sector_financial.json').read_text())
    out=[];raw=[];target=data['target'];as_of=data['as_of'];known=data['known_at']
    for case in data['cases']:
        sid=case['case_id'];left=[dict(entity_id=sid,as_of=as_of,known_at=known)]
        selected=cr.join_as_of(left,case['statement_versions']).value[0]
        later=cr.join_as_of([dict(left[0],known_at='2024-04-02')],case['statement_versions']).value[0]
        ratio=cr.build_financial_ratios([selected['statement']],case['ratio_spec']).value[0]['ratios']['risk_ratio']['value']
        late_ratio=cr.build_financial_ratios([later['statement']],case['ratio_spec']).value[0]['ratios']['risk_ratio']['value']
        fitted=fit_sector_case_models(case,target)
        transformed=cr.transform_features([dict(risk_ratio=ratio)],fitted['preprocessor'])
        fundamental=cr.score_scorecard(fitted['scorecard'],transformed.value['design_matrix'],feature_order=['risk_ratio'],expected_target=target)
        calibrated=cr.apply_calibrator(fitted['calibrator'],fundamental.value['pd'],target=target,base_model_version='1')
        unrounded=calibrated.value[0]
        if verify_reference:
            assert abs(fundamental.value['pd'][0]-case['reference']['pd'])<data['precision_contract']['raw_fit_oracle_absolute_tolerance']
            assert abs(unrounded-case['reference']['pd'])<data['precision_contract']['raw_fit_oracle_absolute_tolerance']
            assert abs(ratio-case['reference']['ratio'])<1e-12 and abs(late_ratio-case['reference']['late_ratio'])<1e-12
            assert selected['release_id']=='initial' and later['release_id']=='revision'
        pd=round(unrounded,data['precision_contract']['pd_decimal_places'])
        base=dict(entity_id=sid,pd=pd,stage='STANDALONE',target=target,applied_effects=[])
        parent=dict(entity_id=sid+'-sponsor',pd=.02,stage='STANDALONE',target=target,applied_effects=[])
        joint=min(pd*.25,.01)
        supported=cr.apply_parent_support(base,parent,case['support'],joint_default_probability=joint).value
        grade=cr.map_pd_to_grade(supported['pd'],case['rating_scale']).value
        constrained=cr.apply_rating_constraints(dict(pd=supported['pd'],grade=grade),case['rating_scale'],case['constraints'],representative_pd_policy='representative_on_binding').value
        request=dict(override_id=sid+'-review',target='rating_grade',value='C',reason='Explicit synthetic qualitative deterioration review; no silent continuous-PD adjustment',author='local_analyst',approved_by='local_reviewer',recorded_at='2024-03-01',valid_from='2024-03-01',valid_until='2024-04-01')
        policy=dict(policy_id=sid+'-override-policy',version='1',targets={'rating_grade':dict(path=['grade'],kind='grade',allowed_values=['A','B','C'])},require_dual_control=True)
        override=cr.apply_override(constrained,request,policy,as_of=as_of,known_at=known).value
        expired=cr.apply_override(constrained,request,policy,as_of='2024-04-01',known_at='2024-04-01').value
        snapshot=dict(subject_id=sid,subject_type='obligor',as_of=as_of,known_at=known,attributes={'sector_variant':dict(value=case['variant'],observed_at=as_of,known_at=known),'financial_evidence_verified':dict(value=True,observed_at=as_of,known_at=known)})
        rule=dict(rule_id=sid,profile_id=sid,profile_version='1',priority=1,effective_from='2023-01-01',effective_to=None,known_at='2023-01-01',conditions=[dict(field='sector_variant',operator='eq',value=case['variant']),dict(field='financial_evidence_verified',operator='eq',value=True)])
        route=cr.route_rating_profile(snapshot,[rule],as_of,known).value
        snapshot['attributes']['financial_evidence_verified']['value']=False
        negative=cr.route_rating_profile(snapshot,[rule],as_of,known).value
        if verify_reference:
            assert math.isclose(supported['pd'],pd-(pd-joint)*case['reference']['rescue_success'],abs_tol=1e-12)
            assert override['status']=='APPLIED'and expired['status']=='EXPIRED'and negative['status']=='NO_ROUTE'
            assert override['current']['pd']==constrained['pd']
        out.append(dict(product=case['product'],variant=case['variant'],case_id=sid,economic_definition=case['economic_definition'],selected_release=selected['release_id'],later_release=later['release_id'],ratio=ratio,later_ratio=late_ratio,transformed=transformed.to_dict()['value'],reported_standalone_pd=pd,supported=supported,constrained=constrained,override=override,expired_override=expired,profile=route['profile_id'],missing_evidence_status=negative['status']))
        if return_fitted_models:raw.append(dict(case_id=sid,preprocessor=fitted['preprocessor'].to_dict(),scorecard=fitted['scorecard'].to_dict(),calibrator=fitted['calibrator'].to_dict(),fundamental=fundamental.to_dict(),calibrated=calibrated.to_dict()))
    report=dict(data_origin='SYNTHETIC',precision_contract=data['precision_contract'],cases=out)
    return dict(report=report,raw_native_fit_details=raw)if return_fitted_models else report


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
