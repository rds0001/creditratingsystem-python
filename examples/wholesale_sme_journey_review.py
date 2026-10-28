"""Actual corporate/commitment source bindings, reference comparisons and frozen replay."""
from importlib.resources import files
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import json,math,runpy
import creditratingsystem as cr


def run(data=None,verify_reference=True,accounts_path=None,releases_path=None,corporate_data=None,commitment_data=None,observed_data=None,artifact_paths=None,return_artifacts=False):
    resources=files('creditratingsystem.datasets');here=Path(__file__).parent
    def read(name):return json.loads(resources.joinpath(name+'.json').read_text())
    data=deepcopy(read('wholesale_sme_journey_review')if data is None else data);corporate_data=read('wholesale_corporate_lifecycle')if corporate_data is None else corporate_data;commitment_data=deepcopy(read('wholesale_commitment_governance')if commitment_data is None else commitment_data);observed_data=read('wholesale_commitment_observed')if observed_data is None else observed_data
    lineage=runpy.run_path(str(here/'input_lineage.py'))['prepare_snapshot'](accounts_path or resources.joinpath('wholesale_sme_positions.csv'),releases_path or resources.joinpath('wholesale_sme_releases.csv'),data['lineage_config'],data['lineage_config']['cutoffs'][0]);rows={x['entity_id']:x for x in lineage['rows']};tools=runpy.run_path(str(here/'_wholesale_journey_tools.py'));out=[];audits=[]
    for case in data['cases']:
        sid=case['case_id'];source=rows[sid]
        if source['balance']!=case['source_balance']or source['product_id']!=case['product']or source['x']!=case['source_feature']:raise ValueError('Source must bind actual exposure, product and frozen financial input')
        model=cr.load_fitted_model(artifact_paths[case['model_resource']]if artifact_paths else resources.joinpath(case['model_resource']))
        if case['product']=='CORPORATE_LOAN_BOND':
            if sid!=corporate_data['subject_id']or source['balance']!=corporate_data['positions'][0]['ead']or source['x']!=2:raise ValueError('Corporate source differs from original historical subject/claim')
            audit=runpy.run_path(str(here/'wholesale_corporate_lifecycle.py'))['_governance'](corporate_data,SimpleNamespace(model=model.value),return_audit_artifacts=True);policy=cr.compile_credit_decision_policy(corporate_data['decision_policy']).value;timing=dict(as_of='2024-01-01',known_at='2024-01-01');p=audit['report']['origination_pd'];gate=corporate_data['credit_gate_inputs'];decision=cr.apply_credit_decision_policy(policy,risk=dict(timing,pd=p,target=case['target'],freshness='CURRENT'),affordability=dict(timing,status='AFFORDABLE'if gate['available_cashflow']>=gate['debt_service_due']else'NOT_AFFORDABLE'),eligibility=dict(timing,status='ELIGIBLE'if gate['legal_eligibility']else'INELIGIBLE'),operations=dict(timing,status=gate['operations_status']),**timing).value
            reference=corporate_data['monitoring']['predictions'][0]['pd'];assert all(x['pd']==reference for x in corporate_data['monitoring']['predictions']);prediction_case=dict(case,features=[[source['x']]],feature_order=corporate_data['development']['feature_order'],reference_pd=reference,calibrator=dict(applied=False,reference='Previously recorded fixed reference forecasts; no separate calibration applied'));observed=dict(monitoring=corporate_data['monitoring']);reference_id='recorded_reference';defaults=2;mature=20
        else:
            c=deepcopy(next(x for x in commitment_data['cases']if x['case_id']==sid));j=next(x for cohort in observed_data['cohorts']for x in cohort['journeys']if x['case_id']==sid)
            if source['x']!=c['features'][0][0]or j['forecast_contract']['target']!=c['target']:raise ValueError('Commitment source/observed target differs from pinned historical forecast')
            c['features']=[[source['x']]];result=runpy.run_path(str(here/'wholesale_commitment_governance.py'))['run'](dict(commitment_data,cases=[c]),artifact_paths=artifact_paths,return_audit_artifacts=True);audit=dict(result['audits'][0],report=result['report']['cases'][0]);decision=audit['results'][0]['value']['decision'];policy=audit['registry']['entries'][0]['bundle']['dependencies']['artifacts']['credit_policy']['content'];prediction_case=c;observed=j;reference_id='calibrated';defaults=j['cohort_defaults'];mature=20
            forecast=j['forecast_contract'];raw=cr.predict_pd(model,forecast['features'],feature_order=forecast['feature_order'],expected_target=forecast['target']).value;cal=cr.apply_calibrator(forecast['calibrator'],raw,target=forecast['target'],base_model_version='1',**forecast['calibrator_arguments']).value[0]
            if abs(cal-c['reference_pd'])>1e-12 or any(abs(x['pd']-cal)>1e-12 for x in j['monitoring']['predictions']):raise ValueError('Observed forecast must equal actual calibrated pinned inference')
        reason={r['reason_code']:'Declared policy rule '+r['rule_id']for r in decision['rule_trace']};explanation=cr.decompose_policy_decision(decision,reason).value;replay=tools['_replay'](audit,source,policy);comparison=tools['_prediction_review'](prediction_case,model,observed,reference_id);p=comparison['raw_pd'];oracle=defaults/mature*(1-p)**2+(1-defaults/mature)*p*p
        if verify_reference:assert lineage['audit']['decision']['status']=='PASS'and math.isclose(comparison['benchmark']['metrics']['by_model']['raw']['brier'],oracle,abs_tol=1e-12)
        out.append(dict(case_id=sid,product=case['product'],claim_ids=case['claim_ids'],source_balance=source['balance'],source_snapshot_hash=lineage['selected_snapshot_hash'],target=case['target'],governance=audit['report'],decision=decision,policy_explanation=explanation,replay=replay['result'],negative_replay=replay['negative'],prediction_review=comparison,observed_brier_oracle=oracle,observed_mature_count=mature,observed_default_count=defaults))
        audits.append(dict(case_id=sid,governance=audit,replay_inputs=replay['inputs']))
    report=dict(data_origin='SYNTHETIC',scope=data['scope'],lineage=lineage['audit'],archive_sha256=lineage['archive_sha256'],cases=out)
    return dict(report=report,artifacts=audits)if return_artifacts else report

if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
