"""Signed legal netting, collateral MPOR and joint counterparty credit loss."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('counterparty_credit_products.json').read_text())
    data=deepcopy(data);out={};errors={}
    for name,case in data['variants'].items():
        route=cr.route_rating_profile(**case['route_input']).value
        pd=cr.predict_pd(case['model'],[[0]],feature_order=['x'],expected_target=case['model']['target']).value.tolist()
        net=cr.aggregate_netting_set_exposure(**case['netting_input']).value
        collateral=cr.simulate_collateral_margin(net,**case['margin_input']).value
        closeargs=dict(freeze_timing='BEFORE_DATE_EVENTS',cashflow_convention='UNSETTLED_EXCLUDED_FROM_CLOSEOUT_VALUE')
        closed=cr.closeout_exposure_at_default(net,collateral,case['default_events'],case['intermediate_cashflows'],**closeargs).value
        exposure=[[r['closeout_exposure']] for r in closed['closeouts']]
        loss=cr.counterparty_expected_loss([[1],[0]],exposure,[[.4],[.4]],[.5,.5],[[1],[1]],counterparty_id='cp',netting_set_id='n',scenario_ids=['h','l'],time_grid=[2/365],currency='EUR',probability_measure='PHYSICAL').value
        # Forecast scenarios and one synthetic realized holdout path are explicit.
        actual_inputs=deepcopy(case['netting_input'])
        for key in ['trade_value_paths','fx_paths']:actual_inputs[key]=[r for r in actual_inputs[key] if r['path_id']=='h']
        actual=cr.aggregate_netting_set_exposure(**actual_inputs).value
        am=deepcopy(case['margin_input'])
        for key in ['call_calendar','collateral_market_paths']:am[key]=[r for r in am[key] if r['path_id']=='h']
        actual_coll=cr.simulate_collateral_margin(actual,**am).value
        forecast=deepcopy(case['forecast']);observation=deepcopy(case['observation']);forecast['netting_paths']=net;observation['netting_paths']=actual
        if name=='collateralized':
            forecast.update(collateral_paths=collateral,csa_version='1',default_events=[case['default_events'][0]],intermediate_cashflows=[case['intermediate_cashflows'][0]],**closeargs,default_sampling_contract=dict(mode='JOINT_SIMULATED_PATHS',known_at='2024-01-01',reference='synthetic-joint-default-paths'))
            observation.update(collateral_paths=actual_coll,csa_version='1',closeout_event=case['default_events'][0],intermediate_cashflows=[case['intermediate_cashflows'][0]],reported_closeout_exposure=80,closeout_known_at='2024-01-04')
        validation=cr.counterparty_exposure_backtest([forecast],[observation],quantile_levels=[.5,.95],design=case['validation_design'],uncertainty='none').value
        subject=dict(subject_id=name+'-netting',subject_type='transaction');target=dict(case['model']['target'])
        components=[dict(component_id=name+'-el',quantity='EL',value=loss['expected_loss'],currency='EUR',**subject,variant='joint_default_exposure',definition=target)]
        binding=dict(profile_id=route['profile_id'],profile_version='1',model_id='ccr-reference',model_version='v1',model_hash=collateral['artifact_hash'],input_hash=net['artifact_hash'])
        assessment=cr.compose_risk_assessment(name+'-assessment',subject,components,binding,'2024-01-01','2024-01-01','2024-01-01').value
        separate=cr.aggregate_netting_set_exposure(**case['separate_netting_input']).value
        try:
            if name=='netting_set':
                wrong=deepcopy(case['netting_input']);wrong['netting_agreements'][0]['legally_enforceable']=False;cr.aggregate_netting_set_exposure(**wrong)
            elif name=='collateralized':
                wrong=deepcopy(case['default_events']);wrong[0]['closeout_date']='2024-01-04';cr.closeout_exposure_at_default(net,collateral,wrong,case['intermediate_cashflows'],**closeargs)
            elif name=='uncollateralized':
                wrong=deepcopy(case['netting_input']);wrong['netting_agreements'].append(dict(wrong['netting_agreements'][0],netting_set_id='duplicate'));cr.aggregate_netting_set_exposure(**wrong)
            else:cr.counterparty_expected_loss([[1],[0]],exposure,[[.4],[.4]],[.5,.5],[[1],[1]],counterparty_id='cp',netting_set_id='n',scenario_ids=['h','h'],time_grid=[2/365],currency='EUR',probability_measure='PHYSICAL')
        except cr.CreditRiskError as exc:errors[name]=exc.code
        else:raise AssertionError('Invalid counterparty contract accepted')
        out[name]=dict(route=route,pd=pd,netting=net,collateral=collateral,closeout=closed,loss=loss,validation=validation,assessment=assessment,separate_sets=separate)
    if verify_reference:
        def same(a,b):
            if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10):raise AssertionError((a,b))
        for name,r in out.items():
            same(r['pd'][0],.5);same(r['loss']['expected_loss'],24 if name=='collateralized'else 27)
            same(r['netting']['paths'][0]['positive_exposure'],40)
            same(sum(x['positive_exposure'] for x in r['separate_sets']['paths'] if x['date']=='2024-01-01'),100)
            assert r['assessment']['composition_status']=='CONSISTENT'and not r['validation']['diagnostics']['reconciliation_violations']
        same(out['wrong_way']['loss']['independent_default_exposure_loss'],15);same(out['wrong_way']['loss']['default_exposure_dependence_contribution'],12)
        c=out['collateralized'];same(c['closeout']['closeouts'][0]['closeout_exposure'],80);same(c['validation']['metrics']['market_exposure']['mean_outcome'],0);same(c['validation']['metrics']['conditional_default_closeout']['mean_outcome'],80)
        assert errors==dict(netting_set='NETTING_ELIGIBILITY',collateralized='INSUFFICIENT_PATH_EXTENSION',uncollateralized='NETTING_OVERLAP',wrong_way='INVALID_IDENTIFIERS')
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out,contract_error_codes=errors)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
