"""Paired out-of-time model review with dated rollback and exact pinned replay."""
from importlib.resources import files
from copy import deepcopy
import json,math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('champion_challenger.json').read_text())
    registry=None
    for model in data['models']:
        registry=cr.register_rating_model_version(registry,dict(model=model,dependencies={'source_reference':'explicit analytic model'},configuration={'purpose':'paired out-of-time review'}),dict(model_id=model['model_id'],model_version=model['model_version'],registered_at='2024-01-01')).value
    def event(version,identity,role,slot,effective):
        entry=next(e for e in registry['entries']if e['model_version']==version)
        return dict(event_id=identity,profile_id='paired-example',purpose='monitoring',role=role,slot_id=slot,model_id=entry['model_id'],model_version=version,model_hash=entry['model_hash'],effective_from=effective,known_at='2024-12-01',evaluation_reference='Explicit prior local assignment; current test outcomes unavailable',reason='Synthetic governance history',required_feature_order=['x','z'])
    history=cr.assign_model_role(None,registry,event('1','original','champion','champion','2024-12-01')).value
    history=cr.assign_model_role(history,registry,event('2','shadow','challenger','candidate','2024-12-01')).value
    history=cr.assign_model_role(history,registry,event('2','prior-promotion','champion','champion','2025-01-01')).value
    original_request=dict(row_id='historical',known_at='2024-12-15',model_id=data['models'][0]['model_id'],model_version='1',model_hash=registry['entries'][0]['model_hash'],recipe=dict(method='pd_model',version='1.0',parameters=dict(expected_target=data['target'])),input=dict(X=[[1,1]],feature_order=['x','z']))
    original=cr.rate_subject_batch([original_request],registry,'historic','2024-12-15').value['rows']
    cohorts={};panels={};predictions={};forecast_inputs={}
    for model in data['models']:
        version=model['model_version'];p=cr.predict_pd(model,[c['X']for c in data['cases']],feature_order=['x','z'],expected_target=data['target']).value
        predictions[version]=p.tolist()
        forecasts=[dict(forecast_id=version+'-'+c['case_id'],case_id=c['case_id'],entity_id=c['entity_id'],cluster_id=c['cluster_id'],cohort=c['cohort'],segment='paired-example',model_id=model['model_id'],model_version=version,forecast_at=c['forecast_at'],known_at=c['known_at'],pd=q,target=data['target'],origin_grade='A',decision=None,override=False,prediction_origin='OUT_OF_TIME')for c,q in zip(data['cases'],p)]
        forecast_inputs[version]=forecasts
        cohorts[version]=cr.build_monitoring_cohorts(forecasts,data['events'],data['coverage'],target_contract=data['target'],evaluation_cutoff=data['evaluation_cutoff']).value
        panels[version]=cr.monitor_rating_cohort(cohorts[version],group_by=['cohort'],reference_window={'start':'2024-01-01','end':'2024-12-31'},minimum_count=4,uncertainty='independent_cluster_bootstrap',cluster_resamples=data['cluster_resamples']).value
    comparison=cr.compare_challenger_cohorts(cohorts['1'],cohorts['2'],cluster_resamples=data['cluster_resamples'],selection_protocol=data['selection_protocol']).value
    fairness=cr.diagnose_fairness_slices(cohorts['2'],group_labels=[dict(case_id=c['case_id'],group=c['group'])for c in data['cases']],reference_group='A',use_basis='Supplied synthetic groups for demonstration, no inferred protected traits',minimum_count=8,risk_threshold=.85,cluster_resamples=data['cluster_resamples']).value
    current=[dict(zip(['x','z'],c['X']))for c in data['cases'][:4]]
    drift=cr.detect_credit_feature_drift(data['reference_features'],current,feature_order=['x','z'],method='ks',permutations=data['drift_permutations'],multiple_testing='holm',dependence_assumption='exchangeable_rows').value
    ordered=sorted(panels['2'],key=lambda r:r['group']['cohort'])
    residuals=[dict(time=r['group']['cohort']+'-28',value=r['metrics']['calibration_error'],mature=True)for r in ordered]
    # Monitoring time is when the complete cohort outcome is available, not origin.
    for row,time in zip(residuals,['2025-02-15','2025-03-15','2025-04-15']):row['time']=time
    sequential=cr.detect_pd_calibration_drift(residuals,method='ewma',parameters={'smoothing':.5},null_paths=data['null_paths'],null_spec=dict(model_id=data['models'][1]['model_id'],model_version='2',calibration_id='prior-Bernoulli-reference',residual_scale='observed_minus_expected_rate',dependence='independent',probability_measure='PHYSICAL',calibrated_before='2024-12-01',validation_paths_independent=True),alpha=.05,validation_null_paths=data['validation_null_paths']).value
    metric_panel=[dict(series_id='candidate',metric='absolute_calibration_gap',period=r['group']['cohort'],status='OK',value=abs(r['metrics']['calibration_error']))for r in ordered]
    findings=cr.classify_monitoring_findings(metric_panel,data['monitor_policy']).value
    model=data['models'][1];coeff=model['parameters']['coefficients']
    contract=dict(model_id=model['model_id'],model_version='2',feature_order=['x','z'],output_scale='probability',deterministic_frozen=True,target=data['target'])
    linear=cr.explain_linear_credit_prediction(dict(contract,coefficients=coeff[1:],intercept=coeff[0],link='logit',transformed_feature_lineage={'x':['x'],'z':['z']}),[[1,1]],[0,0],feature_groups={'x':['x'],'z':['z']}).value
    shapley=cr.estimate_credit_shapley_values(lambda x:cr.predict_pd(model,x,feature_order=['x','z'],expected_target=data['target']).value,[1,1],dict(values=[[0,0]],weights=[1]),contract,feature_groups={'x':['x'],'z':['z']},masking_contract=dict(masking='INTERVENTIONAL_EMPIRICAL',reference_id='fixed-zero',reference_version='1')).value
    rolled=cr.rollback_model_assignment(history,registry,'original',dict(event_id='explicit-rollback',effective_from='2025-04-20',known_at='2025-04-16',reason='Explicit reviewer action after worse Brier and calibration; AUC alone insufficient',evaluation_reference='held-out-paired-comparison',required_feature_order=['x','z'])).value
    selections={label:cr.select_effective_model(registry,rolled,'paired-example','monitoring',day,day).value for label,day in [('historical','2024-12-15'),('promoted','2025-02-01'),('rolled_back','2025-04-21')]}
    replay=cr.replay_assessment_history([original_request],registry,original,'2025-04-21').value
    if verify_reference:
        assert math.isclose(comparison['champion_metrics']['auc'],.75) and comparison['challenger_metrics']['auc']==1
        assert math.isclose(comparison['champion_metrics']['brier'],.1875) and math.isclose(comparison['challenger_metrics']['brier'],.325)
        assert math.isclose(comparison['differences']['calibration_error'],-.225)
        assert all(math.isclose(a,b)for a,b in zip(sequential['statistic_path'],[.175,.2625,.30625]))
        assert findings['status']=='WARNING' and replay['status']=='REPRODUCED'
        assert [selections[k]['selected']['model_version']for k in ['historical','promoted','rolled_back']]==['1','2','1']
        assert math.isclose(shapley['values'][0],0,abs_tol=1e-15) and math.isclose(shapley['values'][1],.9-6/7)
    return dict(forecast_inputs=forecast_inputs,predictions=predictions,cohorts=cohorts,panels=panels,comparison=comparison,fairness=fairness,feature_drift=drift,sequential=sequential,findings=findings,linear_explanation=linear,shapley=shapley,role_history=rolled,selections=selections,historical_request=original_request,historical_rows=original,replay=replay,interpretation='Synthetic prior-specified frozen models and untouched paired outcomes. Descriptive monitoring with explicit human review and local role events; no automatic deployment or empirical performance claim.')


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
