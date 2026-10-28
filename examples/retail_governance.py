"""Product-specific retail capacity, frozen risk and dated assessment governance."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import itertools
import creditratingsystem as cr


def contractual_service(case):
    """Small explicit payment contracts, never a generic debt-to-income proxy."""
    t=case['terms'];kind=case['kind'];aux={}
    if kind=='MORTGAGE':service=[t['principal']*t['annual_rate']+t['annual_principal']]
    elif kind=='HELOC':service=[t['drawn']*t['annual_rate']/12+p for p in t['monthly_principal']]
    elif kind=='CONSOLIDATION':
        service=[t['new_payment']];aux['old_service_still_counted']=sum(t['old_payments'])
        if case['budget']['existing_debt_service']!=[sum(t['old_payments'])]:raise ValueError('Unsettled old facilities must remain in application capacity')
    elif kind=='EDUCATION':service=t['monthly_dues']
    elif kind=='AUTO_BALLOON':
        remaining=t['principal'];service=[]
        for principal in t['annual_principal']:service.append(principal+remaining*t['annual_rate']);remaining-=principal
        aux['remaining_principal']=remaining
    elif kind=='OPERATING_LEASE':
        if t['customer_bears_residual']:raise ValueError('Operating-lease reference requires a separate residual liability contract')
        service=[t['rent']];aux['lessor_residual_excluded']=t['lessor_residual_value']
    elif kind=='CREDIT_CARD':service=[max(t['minimum_floor'],t['minimum_fraction']*t['statement_balance'])];aux['full_repayment_scenario']=t['statement_balance']
    elif kind=='CHARGE_CARD':
        if t['deferred']:raise ValueError('Deferred charge requires its own dated due schedule')
        service=[t['statement_balance']]
    elif kind=='OVERDRAFT':
        if t['demand_known_at']>'2025-01-01' or t['demand_due_at']<'2025-01-01':raise ValueError('Demand must be known and prospective')
        service=[t['demand_amount']]
    elif kind=='BNPL':service=[sum(period)for period in t['weekly_plan_dues']]
    elif kind=='MICRO':service=t['weekly_dues']
    elif kind=='OWNER_BUSINESS':
        service=[t['new_payment']];aux['business_after_draw']=t['sales']-t['operating_cost']-t['tax']-t['capex']-t['business_debt']-t['owner_draw']
        if case['budget']['net_income']!=[t['owner_draw']+t['other_household_income']]:raise ValueError('Owner transfer must enter household income once')
    else:raise ValueError('Unsupported contractual payment type')
    return service,aux


def run(data=None,verify_reference=True,product_ids=None):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('retail_governance.json').read_text())
    data=deepcopy(data);D='2025-01-01';P='2025-01-31';R='2025-02-28';out={}
    models=[data['application_model'],data['behavioral_model']];registry=None
    for model in models:
        if model['training']['known_at']>=D:raise ValueError('Risk model must be frozen before application date')
        registry=cr.register_rating_model_version(registry,dict(model=model,dependencies={},configuration=dict(population='explicit pooled synthetic illustration')),dict(model_id=model['model_id'],model_version='1',registered_at='2024-12-31')).value
    hashes={e['model_id']:e['model_hash']for e in registry['entries']};appmodel,bmodel=models
    policy=cr.compile_credit_decision_policy(data['policy']).value
    for product,c in data['cases'].items():
        if product_ids is not None and product not in product_ids:continue
        if c['term_evidence']['known_at']>D or c['income_evidence']['known_at']>D:raise ValueError('Capacity evidence not known at application')
        service,aux=contractual_service(c)
        residual=cr.residual_income(**c['budget'],new_debt_service=service).value
        evidence=dict(income_verified=c['income_evidence']['income_verified'],terms_verified=True)
        capacity=cr.assess_affordability(dict(minimum_residual=residual['minimum']),data['affordability_profile'],evidence).value
        no_evidence=cr.assess_affordability(dict(minimum_residual=residual['minimum']),data['affordability_profile'],dict(income_verified=False,terms_verified=True)).value
        aid=product+'-application';sid=product+'-customer';account=product+'-existing-account'
        app=dict(application_id=aid,customer_id=sid,product_id=product,population_id='synthetic_pooled_new_customer',application_at=D,information_cutoff=D,requested_terms=c['requested_terms'])
        features=c['application_features']
        snapshot=cr.build_application_snapshot([app],features,required_features=['recent_arrears']).value[0]
        risk=cr.predict_application_risk(snapshot,appmodel,data['application_target']).value
        subject=dict(subject_id=sid,subject_type='customer');subjects=cr.build_rating_subjects(dict(customer=[dict(subject_id=sid,version='1',effective_from=D,effective_to=None,known_at=D,attributes=dict(product=product))]),[],D,D).value
        rs=dict(subject,as_of=D,known_at=D,attributes=dict(product=dict(value=product,observed_at=D,known_at=D),recent_arrears=dict(value=c['recent_arrears'],observed_at='2024-12-30',known_at='2024-12-31')))
        rules=[dict(rule_id=product,profile_id=product+'-application',profile_version='1',priority=1,effective_from=D,effective_to=None,known_at='2024-12-31',conditions=[dict(field='product',operator='eq',value=product)])]
        route=cr.route_rating_profile(rs,rules,D,D).value
        applicability=cr.assess_rating_applicability(rs,dict(contract_id=product+'-support',version='1',required_features=['product','recent_arrears'],allowed_values=dict(product=[product]),support_bounds=dict(recent_arrears=[0,1]),maximum_age_days=dict(recent_arrears=31)),D,D).value
        mh=hashes[appmodel['model_id']];assignment=[dict(assignment_id=product+'-champion',profile_id=route['profile_id'],purpose='application',model_id=appmodel['model_id'],model_version='1',model_hash=mh,effective_from=D,effective_to='2027-01-01',known_at='2024-12-31')]
        selected=cr.select_effective_model(registry,assignment,route['profile_id'],'application',R,R).value
        request=dict(row_id=aid,known_at=D,model_id=appmodel['model_id'],model_version='1',model_hash=mh,recipe=dict(method='application_risk',version='1.0',parameters=dict(target=data['application_target'])),input=dict(snapshot=snapshot))
        batch=cr.rate_subject_batch([request],registry,product+'-batch',D).value
        binding=dict(profile_id=route['profile_id'],profile_version='1',model_id=appmodel['model_id'],model_version='1',model_hash=mh,input_hash=batch['rows'][0]['input_hash'])
        components=[dict(component_id=product+'-pd',quantity='PD',value=risk['pd'],**subject,variant='application_given_funding',definition=data['application_target'])]
        assessment=cr.compose_risk_assessment(aid+'-assessment',subject,components,binding,D,D,D).value
        dates=dict(as_of=D,known_at=D)
        decision=cr.apply_credit_decision_policy(policy,dict(risk,**dates,freshness='CURRENT'),dict(status=capacity['assessment'],**dates),dict(status='ELIGIBLE',**dates),dict(status='CLEAR',**dates),**dates).value
        missing_decision=cr.apply_credit_decision_policy(policy,dict(risk,**dates,freshness='CURRENT'),dict(status=no_evidence['assessment'],**dates),dict(status='ELIGIBLE',**dates),dict(status='CLEAR',**dates),**dates).value
        # Separate already-funded account, not a hypothetical acceptance of this application.
        history=cr.build_payment_history_features(**c['payment_history']).value
        panel=cr.build_account_month_panel(**c['historical_account']).value['accounts']
        if panel[-1]['features']['dpd']!=history['features']['max_dpd'] or not math.isclose(panel[-1]['features']['past_due_amount'],history['daily_history'][-1]['unpaid_due_amount'],abs_tol=1e-10):raise ValueError('Statement arrears must reconcile to the dated payment ledger')
        behavior=[cr.predict_behavioral_risk(s,bmodel,data['behavioral_target']).value for s in panel]
        breqs=[dict(row_id=account+'-'+s['as_of'],known_at=s['known_at'],model_id=bmodel['model_id'],model_version='1',model_hash=hashes[bmodel['model_id']],recipe=dict(method='behavioral_risk',version='1.0',parameters=dict(target=data['behavioral_target'])),input=dict(snapshot=s))for s in panel]
        bbatch=cr.rate_subject_batch(breqs,registry,product+'-behavioral-batch',R).value
        bs=dict(subject_id=account,subject_type='facility');bcomponents=[dict(component_id=account+'-pd',quantity='PD',value=behavior[0]['pd'],**bs,variant='active_account',definition=data['behavioral_target'])]
        bbinding=dict(profile_id=product+'-behavioral',profile_version='1',model_id=bmodel['model_id'],model_version='1',model_hash=hashes[bmodel['model_id']],input_hash=bbatch['rows'][0]['input_hash'])
        performing=cr.compose_risk_assessment(account+'-assessment',bs,bcomponents,bbinding,P,P,P).value
        pool=[]
        for label,score,date in [('application',risk['pd'],D),('performing',behavior[0]['pd'],P)]:
            row=dict(observation_id=product+'-'+label,index_date=date,feature_observed_at=date,feature_known_at=date,score=score,product=product,borrower_type='person',facility_type=c['kind'],delinquency_state='not_defaulted')
            pool.append(cr.assign_retail_risk_pool([row],data['pool_model'],as_of=date,information_cutoff=date,expected_target=data['pool_target'],uncovered_policy='error').value)
        trigger=dict(event_id=account+'-delay',subject_id=account,type='payment_delay',occurred_at=R,known_at=R)
        fresh=cr.check_assessment_freshness(performing,dict(maximum_age_days=60,maximum_information_age_days=60,trigger_types=['payment_delay']),[],R,R).value
        stale=cr.check_assessment_freshness(performing,dict(maximum_age_days=60,maximum_information_age_days=60,trigger_types=['payment_delay']),[trigger],R,R).value
        deps=[dict(assessment_id=performing['assessment_id'],dependency_type='facility',dependency_id=account)]
        event=dict(event_id=trigger['event_id'],revision='1',occurred_at=R,known_at=R,dependencies=[dict(type='facility',id=account)],payload=dict(dpd=history['features']['max_dpd']))
        impact=cr.apply_assessment_event(None,event,deps,R).value
        proposal=deepcopy(event);proposal['event_id']=account+'-proposal';proposal['payload']=dict(action='PROPOSED_REPAYMENT_REVIEW',contract_changed=False,default_status_changed=False)
        proposal_impact=cr.apply_assessment_event(impact['state'],proposal,deps,R).value
        target=dict(event='default',probability_measure='PHYSICAL',default_definition='DECLARED_FIRST_DEFAULT',unit='account',horizon_days=365,default_grade='D')
        forecast=dict(forecast_id=account+'-forecast',case_id=account,entity_id=account,cluster_id=sid,cohort='2025-01',segment=product,model_id=bmodel['model_id'],model_version='1',forecast_at=P,known_at=P,pd=behavior[0]['pd'],target=target,origin_grade='P',override=False,prediction_origin='OUT_OF_TIME')
        cohort=cr.build_monitoring_cohorts([forecast],c['monitoring_events'],c['monitoring_coverage'],target_contract=target,evaluation_cutoff='2026-02-01').value
        monitor=cr.monitor_rating_cohort(cohort,group_by=['segment'],reference_window=dict(start=P,end=P),minimum_count=30,uncertainty='none').value
        drift=cr.detect_credit_feature_drift([dict(dpd=x)for x in [0,0]],[dict(dpd=x)for x in [0,history['features']['max_dpd']]],feature_order=['dpd'],method='ks',permutations=list(itertools.permutations(range(4))),multiple_testing='none',dependence_assumption='exchangeable_rows').value
        nullspec=dict(model_id=bmodel['model_id'],model_version='1',calibration_id='arithmetic-zero-reference',residual_scale='observed_minus_expected_rate',dependence='prespecified_dependent',probability_measure='PHYSICAL',calibrated_before='2025-01-30',validation_paths_independent=True)
        calibration=cr.detect_pd_calibration_drift([dict(time='2026-02-01',value=monitor[0]['metrics']['calibration_error'],mature=True)],method='ewma',parameters=dict(smoothing=.5),null_paths=[[0]]*19,null_spec=nullspec,alpha=.05,validation_null_paths=[[0]]*20).value
        transitions=[dict(from_state='DRAFT',to_state='FINAL',event_type='approve',required_gates=['review']),dict(from_state='FINAL',to_state='SUPERSEDED',event_type='close',required_gates=['closed'])]
        finalized=cr.transition_assessment_state(performing,dict(event_id=account+'-final',type='approve',occurred_at=P,known_at=P,reason='Separate historical account assessment reviewed',gates=dict(review=True)),transitions).value
        exited=cr.transition_assessment_state(finalized,dict(event_id=account+'-closed',type='close',occurred_at='2026-02-01',known_at='2026-02-01',reason='Historical account settlement completed',gates=dict(closed=True)),transitions).value
        newer=deepcopy(appmodel);newer['model_version']='2';newer['parameters']['coefficients'][0]+=1
        later=cr.register_rating_model_version(registry,dict(model=newer,dependencies={},configuration={}),dict(model_id=appmodel['model_id'],model_version='2',registered_at='2025-02-01')).value
        replay=cr.replay_assessment_history([request],later,batch['rows'],D).value
        breplay=cr.replay_assessment_history(breqs,later,bbatch['rows'],R).value
        out[product]=dict(contractual_service=service,contract_note=c['contract_note'],economic_details=aux,residual_income=residual,capacity=capacity,missing_evidence_capacity=no_evidence,subjects=subjects,route=route,applicability=applicability,snapshot=snapshot,application_risk=risk,selected_model=selected,assessment=assessment,decision=decision,missing_evidence_decision=missing_decision,payment_history=history,account_panel=panel,behavioral_risk=behavior,performing_assessment=performing,pool_assignments=pool,freshness=fresh,triggered_freshness=stale,event=impact,restructuring_proposal_event=proposal_impact,cohort=cohort,monitoring=monitor,feature_drift=drift,calibration_drift=calibration,exit_assessment=exited,replay=replay,behavioral_replay=breplay)
        if verify_reference:
            assert math.isclose(max(service),c['reference_due'],abs_tol=1e-10)
            assert math.isclose(risk['pd'],.02,abs_tol=1e-8) and math.isclose(behavior[0]['pd'],.01,abs_tol=1e-8) and math.isclose(behavior[1]['pd'],1/(1+99/(11**(28/30))),abs_tol=1e-8)
            assert history['features']['max_dpd']==28 and fresh['status']=='CURRENT'and stale['status']=='STALE'
            assert decision['action']==('APPROVE'if residual['minimum']>=0 else'DECLINE')and missing_decision['action']!='APPROVE'
            assert monitor[0]['status']=='INSUFFICIENT_INFORMATION'and monitor[0]['metrics']['n']==1
            assert replay['status']=='REPRODUCED'and exited['state']=='SUPERSEDED'
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
