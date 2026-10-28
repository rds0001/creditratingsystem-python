"""Sovereign vintage, credit default, transfer union and restructuring economics."""
from importlib.resources import files
import json,math
import creditratingsystem as cr


def run(data=None,verify_reference=True,return_fitted_models=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sovereign.json').read_text())
    rows=[];raw=[]
    for c in data['cases']:
        sid=c['case_id'];variant=c['variant'];local=variant=='local_currency'
        if (c['currency']=='LOC')!=local:raise ValueError('Variant and payment currency are incompatible')
        if variant not in ('local_currency','foreign_currency','restructuring') or (c['legal_claim']=='RESTRUCTURED_BOND')!=(variant=='restructuring') or (c['cashflow_type']=='EXCHANGED_SCHEDULE')!=(variant=='restructuring'):raise ValueError('Variant and restructuring rights are incompatible')
        query=[dict(entity_id=sid,as_of='2024-01-01',known_at='2024-03-01')]
        vintages=[dict(x,release_id='revision'if x['known_at']=='2024-04-01'else x['observed_at']) for x in c['series']]
        selected=cr.join_as_of(query,vintages).value[0]
        lag_spec=[dict(name='fiscal_change',feature='fiscal_balance',lag=1,operation='difference')]
        lag=cr.build_distributed_lags(c['series'],query,lag_spec,frequency='years',gap_policy='error').value[0]
        late=cr.build_distributed_lags(c['series'],[dict(query[0],known_at='2024-04-02')],lag_spec,frequency='years',gap_policy='error').value[0]
        risk=-lag['features']['fiscal_change']+12/c['reserves_months']+2*(1-c['institution_index'])
        p=c['pretransfer_pd'];X=[[risk],[risk],[risk+1],[risk+1]];weights=[100*(1-p),100*p,100*(1-2*p),200*p]
        model=cr.fit_binary_glm(X,[0,1,0,1],link='logit',feature_order=['sovereign_capacity_stress'],fit_intercept=True,target=data['target'],training=c['training'],model_id=sid,model_version='1',weights=weights)
        pred=cr.predict_pd(model,[[risk]],feature_order=['sovereign_capacity_stress'],expected_target=data['target'])
        a=c['calibration'];cal=cr.calibrate_logistic(a['values'],a['outcomes'],a['weights'],metadata=a['metadata'],**a['fit_controls'])
        calibrated=cr.apply_calibrator(cal,pred.value,target=data['target'],base_model_version='1')
        pd=round(calibrated.value[0],data['precision_contract']['pd_decimal_places'])
        payment=dict(payment_id=sid+'-coupon',due_at='2024-06-01',known_at='2024-01-01',currency=c['currency'],source_currency='LOC',payer_jurisdiction='HOME',payee_jurisdiction='HOME'if local else'FOREIGN',payment_jurisdictions=['HOME']if local else['HOME','FOREIGN'])
        event=dict(event_id=sid+'-freeze',event_type='PAYMENT_FREEZE'if local else'BORDER_TRANSFER_BLOCK',jurisdictions=['HOME'],currencies=[c['currency']],effective_from='2024-01-01',known_at='2024-01-01',exemptions=[])
        applicability=cr.assess_transfer_applicability([payment],dict(policy_id=sid,version='1',events=[event]),as_of='2024-03-01',known_at='2024-03-01',horizon_end='2025-03-01').value
        target=dict(exposure_id=sid,horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='PIT')
        transfer=cr.transfer_event_default_probability(c['event_probability'],c['failure_given_event'],dict(effect_id=sid+'-liquidity-mitigation',success_probability_given_event_and_unmitigated_failure=c['mitigation_success'],conditioning='event_and_unmitigated_failure',assumption_reference='Declared synthetic payment-route stress contract'),event_id=event['event_id'],target=target).value
        base=dict(stage='PRE_TRANSFER',pd=pd,target=target,applied_effects=[])
        combined=cr.combine_default_transfer(base,transfer,dependence='explicit',joint_probability=c['joint']).value
        loss=cr.transfer_loss_components(combined,c['losses'],currency=c['currency'],applied_loss_effects=[]).value
        rejected=False
        try:cr.combine_default_transfer(dict(base,stage='TRANSFER_ADJUSTED'),transfer,dependence='explicit',joint_probability=c['joint'])
        except ValueError:rejected=True
        pv=math.fsum(x['amount']/(1+c['discount_rate'])**x['time_years']for x in c['restructured_cashflows']) if c['restructured_cashflows']else None
        restructuring=dict(assessment_date='2024-03-01',exchange_date='2024-06-01'if variant=='restructuring'else None,observed_exchange=variant=='restructuring',payment_delay_days=0,contractual_default_policy='Distressed exchange counts as default; mere technical delay is a separate event',original_claim=c['original_claim'],restructured_pv=pv,economic_loss=None if pv is None else c['original_claim']-pv)
        facts=dict(product='SOVEREIGN_DEBT',variant=variant,borrower_class='SOVEREIGN',legal_claim=c['legal_claim'],cashflow_type=c['cashflow_type'],currency=c['currency'])
        snapshot=dict(subject_id=sid,subject_type='issue',as_of='2024-03-01',known_at='2024-03-01',attributes={k:dict(value=v,observed_at='2024-01-01',known_at='2024-01-01')for k,v in facts.items()})
        rule=dict(rule_id=sid,profile_id=sid,profile_version='1',priority=1,effective_from='2023-01-01',effective_to=None,known_at='2023-01-01',conditions=[dict(field=k,operator='eq',value=v)for k,v in facts.items()])
        route=cr.route_rating_profile(snapshot,[rule],'2024-03-01','2024-03-01').value
        del snapshot['attributes']['legal_claim'];missing=cr.route_rating_profile(snapshot,[rule],'2024-03-01','2024-03-01').value
        if verify_reference:
            assert abs(calibrated.value[0]-p)<1e-7 and abs(pred.value[0]-p)<1e-7
            assert selected['value']==c['fiscal_balance'] and late['features']['fiscal_change']!=lag['features']['fiscal_change']
            q=c['event_probability']*c['failure_given_event']*(1-c['mitigation_success'])
            assert math.isclose(combined['combined_pd'],p+q-c['joint'],abs_tol=1e-12)
            expected=(p-c['joint'])*c['losses']['default_only']+(q-c['joint'])*c['losses']['transfer_only']+c['joint']*c['losses']['joint']
            assert math.isclose(loss['expected_loss'],expected,abs_tol=1e-12)
            assert applicability['payments'][0]['status']=='APPLICABLE'and rejected and missing['status']=='NO_ROUTE'
            if pv is not None:assert math.isclose(pv,8400/121,abs_tol=1e-12)
        rows.append(dict(variant=variant,case_id=sid,selected_release=selected['release_id'],lags=lag,later_lags=late,capacity_stress=risk,reported_pretransfer_pd=pd,applicability=applicability,transfer=transfer,combined=combined,loss=loss,grade=cr.map_pd_to_grade(combined['combined_pd'],data['rating_scale']).value,restructuring=restructuring,double_count_rejected=rejected,profile=route['profile_id'],missing_claim_status=missing['status']))
        if return_fitted_models:raw.append(dict(case_id=sid,model=model.to_dict(),calibrator=cal.to_dict(),prediction=pred.to_dict(),calibrated=calibrated.to_dict()))
    report=dict(data_origin='SYNTHETIC',product='SOVEREIGN_DEBT',precision_contract=data['precision_contract'],cases=rows)
    return dict(report=report,raw_native_fit_details=raw)if return_fitted_models else report


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
