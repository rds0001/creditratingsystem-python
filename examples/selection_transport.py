"""Observed funding stages and explicit, nonidentified reject transport scenarios."""
from copy import deepcopy
from datetime import date,timedelta
from importlib.resources import files
from pathlib import Path
import importlib.util,json,math
import creditratingsystem as cr


def helper(name):
    spec=importlib.util.spec_from_file_location('selection_'+name,Path(__file__).with_name(name+'.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def audit(product,records,target,model,calibrator,training_entities,cutoff,assumptions):
    """Records preserve application/first-draw anchors and observed own-contract labels."""
    target={k:v for k,v in target.items()if k!='performance_origin'};population=product+'-through-door';model_id=model['model_id'];pred=[];rows=[];apps=[];decisions=[];bookings=[];outcomes=[]
    for r in records:
        key=r['application_id'];funded=r['funded'];observed=r['observed'];anchor=r['first_draw_at']if funded else r['application_at'];end=(date.fromisoformat(anchor)+timedelta(days=target['horizon'])).isoformat()
        apps.append(dict(application_id=key,entity_id=r['customer_id'],submitted_at=r['application_at']))
        decisions.append(dict(application_id=key,as_of=r['decision_at'],known_at=r['decision_at'],decision='offered'if r['offered']else'rejected',policy_version=r['policy_version']))
        if r['offered']:bookings.append(dict(application_id=key,as_of=r['first_draw_at']if funded else r['decision_at'],known_at=r['first_draw_at']if funded else r['decision_at'],taken_up=funded,funded=funded))
        if funded:
            status='observed'if observed else'immature'if end>cutoff else'censored'
            outcomes.append(dict(application_id=key,as_of=end if end<=cutoff else cutoff,known_at=cutoff,outcome_scope='own_contract',status=status,outcome=r['outcome']if observed else None,target=target))
        rows.append(dict(prediction_id=key,application_id=key,entity_id=r['customer_id'],cluster_id=r['customer_id'],segment=product,as_of=anchor,forecast_known_at=r['application_at'],horizon_end=end,known_at=cutoff,complete_through=end if observed and r['outcome']==0 else None,model_id=model_id,model_version='1',parameter_version='1',funded=funded,label_origin='OBSERVED_OWN_CONTRACT'if observed else'UNOBSERVED',outcome=r['outcome']if observed else None,target=target,weight=1))
        pred.append(dict(application_id=key,pd=r['pd'],known_at=r['application_at']))
    panel_args=dict(applications=apps,decisions=decisions,bookings=bookings,outcomes=outcomes,as_of=cutoff,target_population=population,target=target)
    panel=cr.build_selection_dataset(**panel_args).value
    candidate=dict(model_id=model_id,model_hash=cr.content_fingerprint(dict(model=model,calibrator=calibrator)).value,training_entities=training_entities,training_known_at=model['training']['known_at'],predictions=pred)
    flags=[int(r['observed'])for r in records];ys=[r['outcome']if r['observed']else None for r in records];weights=[1]*len(records);base=dict(observed=flags,target_weights=weights);scenarios=[]
    configurations=[('MAR_FULL_SUPPORT','IPW',dict(base,propensities=[assumptions['observation_propensity']]*len(records)),cr.selection_ipw_weights),('MANSKI','BOUNDS',dict(base,outcomes=ys),cr.reject_inference_bounds),('MNAR_LOG_ODDS','MNAR_SHIFT',dict(base,outcomes=ys,baseline_pd=[r['pd']for r in records],log_odds_shifts=assumptions['log_odds_shifts']),cr.reject_inference_sensitivity),('NO_REJECT_SUPPORT','IPW',dict(base,propensities=[assumptions['observation_propensity']if r['observed']else 0 for r in records]),None)]
    for name,kind,inputs,fn in configurations:
        scenarios.append(dict(result_id=name,kind=kind,model_id=model_id,population_id=population,target=target,application_ids=[r['application_id']for r in records],inputs=inputs,result=None if fn is None else fn(**inputs).to_dict()['value']))
    design=dict(model_id=model_id,model_version='1',parameter_version='1',data_fingerprint=cr.content_fingerprint(records).value,population_id=population,split_id=product+'-independent-live-holdout',prediction_origin='HOLDOUT',evaluation_cutoff=cutoff,target=target)
    args=dict(candidates=[candidate],evaluation_rows=rows,selection_results=scenarios,design=design,reference_model_id=model_id)
    validated=cr.selection_bias_sensitivity(**args).to_dict()['value']
    # Neither fractional pseudo labels nor missing borrower outcomes are ground truth.
    pseudo_args=deepcopy(args);pseudo_args['selection_results']=[]
    for row in pseudo_args['evaluation_rows']:
        if not row['funded']:row.update(label_origin='PSEUDO',outcome=1)
    pseudo=cr.selection_bias_sensitivity(**pseudo_args).to_dict()['value']
    n=len(records);m=sum(flags);defaults=sum(x for x in ys if x is not None);bound=scenarios[1]['result']
    assert bound['lower']==defaults/n and bound['upper']==(defaults+n-m)/n
    assert validated['metrics']['mature_factual_count']==m==pseudo['metrics']['mature_factual_count']
    assert validated['diagnostics']['transport_scenarios'][-1]['status']=='NOT_IDENTIFIED_ZERO_SUPPORT'
    assert validated['decision']['status']=='NOT_ASSESSED'
    brier=sum((r['pd']-r['outcome'])**2 for r in records if r['observed'])/m
    assert math.isclose(validated['metrics']['evidence_scopes']['FUNDED_OBSERVED_TEST'][model_id]['brier'],brier,abs_tol=1e-12)
    shift0=next(x for x in scenarios[2]['result']['surface']if x['log_odds_shift']==0)
    oracle=(defaults+sum(r['pd']for r in records if not r['observed']))/n
    assert math.isclose(shift0['population_pd'],oracle,abs_tol=1e-12)
    summary=dict(application_ids=[r['application_id']for r in records],funded_facilities=[r['facility_id']for r in records if r['funded']],target=target,index_convention='First actual draw for funded observations; application-date hypothetical funding for unbooked scenario rows. No unbooked performance is observed.',stage_counts=panel['counts'],factual_count=m,funded_count=sum(r['funded']for r in records),factual_defaults=defaults,brier=round(brier,8),bounds=bound,mar_effective_sample_size=scenarios[0]['result']['effective_sample_size'],mnar_population_pd=[dict(log_odds_shift=x['log_odds_shift'],population_pd=round(x['population_pd'],8))for x in scenarios[2]['result']['surface']],zero_support_status=validated['diagnostics']['transport_scenarios'][-1]['status'],pseudo_labels_change_factual_count=False,selection_validation=validated['decision']['status'],assumptions=assumptions)
    return dict(summary=summary,records=records,panel_inputs=panel_args,panel=panel,validation_inputs=args,validation=validated,pseudo_control=pseudo,model=model,calibrator=calibrator)


def application_records(origin,case):
    bookings={b['application_id']:b for b in case['bookings']};records=[]
    for row,d in zip(origin['applications'],case['decisions']):
        a=row['application'];label=row['label'];b=bookings.get(a['application_id']);observed=bool(label.get('mature')and label.get('label_observed'))
        records.append(dict(application_id=a['application_id'],customer_id=a['customer_id'],facility_id=label.get('facility_id'),application_at=a['application_at'],decision_at=d['decided_at'],first_draw_at=b['first_draw_at']if b else None,funded=label['funded'],observed=observed,outcome=label.get('target')if observed else None,offered=d['status']=='approved',policy_version='1',pd=row['risk']['pd']))
    return records


def run(data=None,product_ids=None,return_artifacts=False):
    resources=files('creditratingsystem.datasets')
    if data is None:data=json.loads(resources.joinpath('selection_transport.json').read_text())
    chosen=list(data['products'])if product_ids is None else product_ids;out={};chains={};application=helper('retail_application_journey');base=json.loads(resources.joinpath('retail_application_journey.json').read_text());regular=[p for p in chosen if p in base['products']]
    if regular:
        chain=application.run(product_ids=regular,return_artifacts=True)
        for p in regular:chains[p]=(chain['products'][p],base['products'][p],base['target'],base['evaluation_cutoff'])
    if 'RETAIL_SMALL_BUSINESS_FINANCE'in chosen:
        owned=helper('retail_owner_journey').run(return_artifacts=True,include_lifecycle=False);ownerdata=json.loads(resources.joinpath('retail_owner_journey.json').read_text());p='RETAIL_SMALL_BUSINESS_FINANCE';chains[p]=(owned['application_chain']['products'][p],ownerdata['products'][p],ownerdata['target'],ownerdata['evaluation_cutoff'])
    for p,(origin,case,target,cutoff)in chains.items():
        fitted=origin['fitted'];out[p]=audit(p,application_records(origin,case),target,fitted['model'],fitted['calibrator'],[r['customer_id']for r in fitted['training_rows']+fitted['calibration_rows']],cutoff,data['products'][p]);out[p]['source_origin']=origin;out[p]['source_case']=case;out[p]['prediction_design']=[[r['snapshot']['features']['driver']]for r in origin['applications']];out[p]['summary'].update(cohort_status=origin['performing']['validation']['decision']['status'],policy_factual=origin['policy_validation']['result']['metrics']['factual'],policy_violations=origin['policy_validation']['result']['violation_counts'],replay_status=origin['monitoring']['replay_validation']['decision']['status'])
    if 'RETAIL_CREDIT_CARD'in chosen:out['RETAIL_CREDIT_CARD']=card_chain(data['products']['RETAIL_CREDIT_CARD'])
    result=dict(journey_id='J_SELECTION_TRANSPORT',data_origin='SYNTHETIC',products={p:x['summary']for p,x in out.items()},interpretation='Observed own-contract holdout and hypothetical transport are separate. MAR propensity and MNAR shifts are assumptions, not estimated selection effects. No full-population validation or causal policy approval is claimed.')
    if return_artifacts:result['raw_artifacts']=out
    return result





def card_chain(assumptions,case=None):
    product='RETAIL_CREDIT_CARD';module=helper('revolving_lifecycle')
    source_data=json.loads(files('creditratingsystem.datasets').joinpath('revolving_lifecycle.json').read_text())
    if case is not None:source_data['cases'][product]=case
    case=source_data['cases'][product];journey=module.run(source_data,product_ids=[product],return_artifacts=True)
    origin=journey['raw_artifacts']['products'][product]['origin'];history=module.account_history(case,product,origin);replay=journey['raw_artifacts']['products'][product]['replay'];records=[]
    for app,d,b,label,risk in zip(case['applications'],case['decisions'],case['bookings'],origin['targets'],origin['risks']):
        observed=bool(label.get('mature')and label.get('label_observed'))
        records.append(dict(application_id=app['application_id'],customer_id=app['customer_id'],facility_id=b['facility_id'],application_at=app['application_at'],decision_at=d['decided_at'],first_draw_at=b['first_draw_at'],funded=True,observed=observed,outcome=label.get('target')if observed else None,offered=True,policy_version='1',pd=risk['pd']))
    additional=[]
    for row in assumptions['unbooked_applications']:
        # Raw covariate replay is explicitly a conditional transport assumption;
        # no funded-only model population is silently widened to reject applicants.
        pd=cr.predict_pd(origin['model'],[[row['recent_arrears']]],feature_order=['recent_arrears'],expected_target=case['application_target']).value[0]
        pd=float(cr.apply_calibrator(origin['calibrator'],[pd],target=case['application_target'],base_model_version=origin['model']['model_version']).value[0])
        if not row['evidence_observed_at']<=row['evidence_known_at']<=row['application_at']:raise ValueError('Affordability evidence must precede application')
        residual=cr.residual_income(row['monthly_net_income'],row['monthly_expenses'],row['existing_debt_service'],row['proposed_minimum_payment'],period='monthly',currency='EUR').value
        capacity='AFFORDABLE'if residual['minimum']>=0 else'NOT_AFFORDABLE'
        if capacity!=row['affordability']:raise ValueError('Documentary cash budget does not support declared affordability')
        dated=dict(as_of=row['application_at'],known_at=row['application_at']);inputs=dict(risk=dict(pd=pd,target=case['application_target'],freshness='CURRENT',**dated),affordability=dict(status=row['affordability'],**dated),eligibility=dict(status='ELIGIBLE',**dated),operations=dict(status='CLEAR',**dated))
        decision=cr.apply_credit_decision_policy(origin['policy'],**inputs,**dated).to_dict()['value']
        if decision['action']!=row['recorded_action']:raise ValueError('Added card application decision does not reconcile')
        records.append(dict(application_id=row['application_id'],customer_id=row['customer_id'],facility_id=None,application_at=row['application_at'],decision_at=row['application_at'],first_draw_at=None,funded=False,observed=False,outcome=None,offered=decision['action']=='APPROVE',policy_version='1',pd=pd))
        additional.append(dict(source=row,decision_inputs=inputs,decision=decision,conditional_transport_pd=pd,residual_income=residual))
    policy_validation=card_policy_audit(case,origin,additional,records,'2025-01-02')
    result=audit(product,records,case['application_target'],origin['model'],origin['calibrator'],[], '2025-01-02',assumptions)
    result['summary'].update(cohort_status=history['cohort']['decision']['status'],replay_status=replay['validation']['decision']['status'],policy_stage_actions=[r['action']for r in origin['decisions']]+[r['decision']['action']for r in additional],economic_policy_outcomes='UNKNOWN_365_DAY_NET_PROFIT; actual approval, uptake and funded amounts validated',policy_factual=policy_validation['result']['metrics']['factual'],policy_violations=policy_validation['result']['violation_counts'],policy_validation_status=policy_validation['result']['decision']['status'],transport_population='Original funded-only model unchanged; two unbooked PDs are explicitly assumed conditional transfers',source_funded_population=origin['model']['application_contract']['population_ids'],source_lineage_status=journey['source_lineage_status'],governance_replayed=journey['products'][product]['governance_replayed'])
    result.update(policy_validation=policy_validation,origin=origin,history=history,replay=replay,source_case=case,funded_journey=journey,additional_applications=additional,prediction_design=[[s['features']['recent_arrears']]for s in origin['snapshots']]+[[r['source']['recent_arrears']]for r in additional])
    return result





def card_policy_audit(case,origin,additional,records,cutoff):
    target=case['application_target'];product='RETAIL_CREDIT_CARD';candidate=dict(candidate_id='historical',policy=origin['policy'],offer_rule_reference='Recorded initial card authorization terms; no identified uptake or profitability model');cases=[];inputs=[];outcomes=[];historical=[];bookings={b['application_id']:b for b in case['bookings']}
    for i,r in enumerate(records):
        key=r['application_id'];D=r['decision_at'];dated=dict(as_of=D,known_at=D);funded=r['funded']
        if i<len(case['applications']):
            decision_inputs=dict(risk=dict(origin['risks'][i],freshness='CURRENT',**dated),affordability=dict(status=origin['capacities'][i]['assessment']['assessment'],**dated),eligibility=dict(status='ELIGIBLE',**dated),operations=dict(status='CLEAR',**dated));decision=origin['decisions'][i];amount=case['applications'][i]['requested_terms']['amount']
        else:
            a=additional[i-len(case['applications'])];decision_inputs=a['decision_inputs'];decision=a['decision'];amount=a['source']['requested_limit']
        offer=dict(offer_id=key,version='1',known_at=D,currency='EUR',target=target,risk_conditioning='CONDITIONAL_ON_THIS_ACCEPTED_OFFER',assumption_reference='Original initial authorization terms. No take-up probability or complete365day profitability model identified.',support_status='UNSUPPORTED',takeup_probability=None,conditional_booked_expected_loss=None,conditional_booked_npv=None,ead=None,offer_cost=None,contract_terms=dict(limit=amount,tenor_days=365,currency='EUR',interest_rate_status='NOT_SUPPLIED'))
        cases.append(dict(case_id=key,decision_date=D,weight=1,split_id='card-through-door-heldout',prediction_reference='Original frozen funded-only model; unbooked predictions explicitly assumption-conditional',excluded_from_model_training=True,excluded_from_calibration=True))
        inputs.append(dict(candidate_id='historical',case_id=key,**decision_inputs,offer=offer,nonapproval_costs=dict(REFER=0,DECLINE=0,BLOCKED=0)))
        b=bookings.get(key);end=(date.fromisoformat(b['first_draw_at'])+timedelta(days=365)).isoformat()if funded else None
        outcomes.append(dict(case_id=key,funded=funded,complete=False,booked_offer_hash=cr.content_fingerprint(offer).value if funded else None,funded_at=b['first_draw_at']if funded else None,funding_known_at=b['first_draw_known_at']if funded else None,outcome_end=end,known_at=None,defaulted=None,economic_loss=None,net_profit=None))
        initial_draw=sum(t['amount']for t in case['transactions']if funded and t['account_id']==b['facility_id']and t['date']==b['first_draw_at']and t['kind']=='DRAW')if funded else None
        historical.append(dict(prediction_id=key,case_id=key,entity_id=r['customer_id'],cluster_id=r['customer_id'],segment=product,as_of=D,forecast_known_at=D,horizon_end=(date.fromisoformat(D)+timedelta(days=365)).isoformat(),model_id='card-policy-audit',model_version='1',parameter_version='1',weight=1,historical_candidate_id='historical',historical_decision=decision,offered=r['offered'],taken_up=funded,offered_limit=amount if r['offered']else None,booked_amount=initial_draw,stages_known_at=b['first_draw_known_at']if funded else D,application_cost=None,cost_known_at=None,cost_complete=False))
    profile=dict(profile_id='card-policy-evidence',version='1',target_contract_version='1.1',known_at='2023-12-31',registered_at='2023-12-31',outcome_access_date=cutoff,currency='EUR',evaluation_role='TEST',prediction_design='HELD_OUT',split_id='card-through-door-heldout',data_fingerprint=cr.content_fingerprint(records).value,target=target,candidate_hashes={'historical':cr.content_fingerprint(candidate).value},objective='expected_npv_per_application',constraints=[],tie_break='candidate_id',tolerance=1e-12,loss_bounds=[0,2000],profit_bounds=[-2000,2000])
    gridargs=dict(policy_candidates=[candidate],cases=cases,candidate_inputs=inputs,observed_outcomes=outcomes,evaluation_profile=profile,evaluation_cutoff=cutoff);grid=cr.evaluate_decision_grid(**gridargs).value
    design=dict(model_id='card-policy-audit',model_version='1',parameter_version='1',data_fingerprint=profile['data_fingerprint'],population_id=product+'-through-door',split_id=profile['split_id'],prediction_origin='HOLDOUT',evaluation_cutoff=cutoff,currency='EUR',target={k:v for k,v in target.items()if k!='performance_origin'},policy_target_contract_version='1.1',performance_origin='first_draw_at')
    args=dict(evaluated_grid=grid,grid_inputs=gridargs,historical_cases=historical,design=design,reference_candidate_id='historical');validated=cr.decision_policy_backtest(**args).to_dict()['value']
    assert validated['violation_counts']=={} and validated['metrics']['factual']['funded_rate']==.6 and validated['metrics']['factual']['offered_rate']==.8, validated
    assert validated['metrics']['factual']['observed_pd']is None
    return dict(inputs=args,result=validated)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
