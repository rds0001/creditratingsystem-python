"""Same sole-trader facility: proposal, observed default, settlement and original-book risk."""
from copy import deepcopy
from itertools import product
from datetime import date
import math
import numpy as np
import creditratingsystem as cr


def run(case,origin,cutoff,verify_reference=True):
    x=case['lifecycle'];aid=x['account_id'];key=x['application_id'];app=next(a for a in origin['applications']if a['application']['application_id']==key);booking=next(b for b in case['bookings']if b['application_id']==key)
    default=next(e for e in case['defaults']if e['facility_id']==aid);closed=next(e for e in case['exits']if e['facility_id']==aid);resolution=x['resolution']
    if resolution['known_at']>cutoff or resolution['resolution_date']!=closed['exit_at']or resolution['principal_received']+resolution['principal_written_off']+resolution['claim_remaining']!=x['opening_principal']:raise ValueError('Actual closure requires a reconciled dated settlement')
    economics=next(e for e in case['realized_economics']if e['application_id']==key)
    if economics['principal_cash']!=resolution['principal_received']:raise ValueError('Policy and workout principal ledgers differ')
    if x['proposal_date']>=default['default_at']or x['proposal_status']!='OFFERED_NOT_ACCEPTED':raise ValueError('This scenario requires an unaccepted prior proposal, not retrospective contract substitution')
    arrears=next(r for r in origin['performing']['payment_history']if r['account_id']==aid)
    if x['arrangement']['contract']['principal_arrears']!=arrears['dpd']['past_due_amount']:raise ValueError('Proposal arrears must equal actual unpaid principal')
    proposal=cr.simulate_repayment_arrangement(**x['arrangement']).value
    schedule=cr.exposure_schedule(contract=dict(facility_id=aid,currency='EUR',as_of=booking['first_draw_at'],drawn_principal=x['opening_principal'],credit_limit=x['opening_principal'],accrued_interest=0),events=[],annual_interest_rates=[0],period_end_dates=[default['default_at']],day_count='ACT/365F',default_timing='AFTER_PERIOD_END_EVENTS',allow_overlimit=False,currency='EUR')
    ead=cr.predict_ead(schedule,None,feature_order=[],expected_target=None,reference_dates=[booking['first_draw_at']],default_dates=[default['default_at']],currency='EUR').value
    discounted=cr.discount_recoveries(x['cashflows'],x['discount_curve'],x['fx_contract'],default_date=default['default_at'],evaluation_cutoff=cutoff,reporting_currency='EUR')
    loss=cr.workout_lgd(ead['conditional_ead'][0],discounted,episode_status='CLOSED',default_date=default['default_at'],evaluation_cutoff=cutoff,currency='EUR').value
    linked=cr.build_recovery_episodes([dict(episode_id=aid+'-default',entity_id=aid,default_at=default['default_at'],known_at=default['known_at'],ead=1000,currency='EUR')],[dict(cashflow_id=f['cashflow_id'],episode_id=f['episode_id'],cashflow_at=f['cashflow_date'],known_at=f['known_at'],amount=f['amount'],kind='recovery',currency=f['currency'])for f in x['cashflows']],cures=[],evaluation_cutoff=cutoff).value
    states=[dict(account_id=aid,state_event_id=aid+'-'+state,state=state,occurred_at=D,known_at=K)for state,D,K in [('CLEAN',booking['first_draw_at'],booking['first_draw_known_at']),('ARREARS','2025-01-15','2025-01-15'),('DEFAULT',default['default_at'],default['known_at']),('CLOSED',closed['exit_at'],closed['known_at'])]]
    episodes=cr.build_collections_episodes(states,[],[dict(account_id=aid,observed_from=booking['first_draw_at'],observed_through=cutoff,known_at=cutoff)],dict(rule_id='actual-owner-collection',version='1',problem_states=['ARREARS','DEFAULT'],clean_states=['CLEAN'],terminal_states=['CLOSED'],default_states=['DEFAULT'],probation_days=10,separation_days=15),evaluation_cutoff=cutoff)
    cure=cr.build_cure_targets(episodes,horizon_days=30,redefault_horizon_days=90,evaluation_cutoff=cutoff).value
    prior=x['prior_lgd']
    if max(prior['resolution_dates'])>=prior['training']['known_at']or prior['training']['known_at']>=booking['first_draw_at']:raise ValueError('LGD training must be closed and known before live funding')
    lgdfit=cr.fit_fractional_lgd(np.empty((len(prior['values']),0)),prior['values'],feature_order=[],link='logit',covariance='HC0',target=prior['target'],training=prior['training'],model_id='owner-principal-lgd-prior',model_version='1')
    lgd=cr.predict_lgd(lgdfit,np.empty((1,0)),feature_order=[],expected_target=prior['target'],default_condition='given_default').value['mean_lgd'][0]
    portfolio=original_portfolio(case,origin,lgd)
    audit=benchmark(case,origin,cutoff)
    horizon_loss=cr.lifetime_expected_loss([app['risk']['pd']],[1000*lgd],[1],[30/365],currency='EUR',probability_measure='PHYSICAL').value
    prior_rows=origin['fitted']['training_rows'];survival_target=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='newly_funded_owner')
    survival=cr.estimate_survival_curve([0]*len(prior_rows),[(date.fromisoformat(r['default_at']if r['outcome']else r['outcome_end'])-date.fromisoformat(r['first_draw_at'])).days for r in prior_rows],[r['outcome']for r in prior_rows],weight_policy='frequency',target=survival_target,training=origin['fitted']['model']['training'],model_id='owner-prior-first-default',model_version='1')
    curves=cr.predict_event_curves(survival,[],[0,17,30],feature_order=[],target=survival_target).value
    assert curves['curves'][0]['survival']==[1,.5,.5]
    # Proposal/default/settlement/exit deliveries genuinely rerun the public reducer.
    dependency=[dict(assessment_id=app['assessment']['assessment_id'],dependency_type='account',dependency_id=aid)];events=[]
    for kind,D,K,payload in [('PROPOSAL',x['proposal_date'],x['proposal_known_at'],dict(accepted=False,proposal_hash=cr.content_fingerprint(proposal).value)),('DEFAULT',default['default_at'],default['known_at'],dict(ever_default=True,ead=ead['conditional_ead'][0])),('SETTLEMENT',resolution['resolution_date'],resolution['known_at'],dict(remaining_claim=0,realized_principal_loss=loss['economic_loss'])),('EXIT',closed['exit_at'],closed['known_at'],dict(state='CLOSED',ever_default=True))]:events.append(dict(event_id=aid+'-'+kind,idempotency_key=aid+'-'+kind,occurred_at=D,known_at=K,sequence=len(events),payload=payload,revision_of=None))
    for event in events:event['payload_hash']=cr.content_fingerprint(event['payload']).value
    events.append(deepcopy(events[0]))
    def execute():
        state=None;seen=set();ids=[]
        for e in sorted(events,key=lambda z:(z['known_at'],z['sequence'],z['event_id'])):
            if e['idempotency_key']in seen:continue
            seen.add(e['idempotency_key']);ids.append(e['event_id']);state=cr.apply_assessment_event(state,dict(event_id=e['event_id'],revision='1',occurred_at=e['occurred_at'],known_at=e['known_at'],dependencies=[dict(type='account',id=aid)],payload=e['payload']),dependency,cutoff).value['state']
        return ids,dict(dependencies=state,remaining_claim=resolution['claim_remaining'],ever_default=True,state='CLOSED')
    ids,state=execute();ids2,state2=execute();snap=[dict(snapshot_id=kind,kind=kind,content_hash=cr.content_fingerprint(value).value,known_at=known,effective_from=known,effective_until=None)for kind,value,known in [('MODEL',origin['fitted']['model'],origin['fitted']['model']['training']['known_at']),('CALIBRATION',origin['fitted']['calibrator'],origin['fitted']['calibrator']['metadata']['calibration']['known_at']),('POLICY',origin['policy'],'2024-01-01'),('SOURCE',x,resolution['known_at'])]];base=dict(checkpoint_id=aid+'-closed',as_of=cutoff,known_at=cutoff,event_log_hash=cr.content_fingerprint(events).value,snapshot_hashes={r['snapshot_id']:r['content_hash']for r in snap})
    replayargs=dict(event_log=events,original_checkpoints=[dict(base,applied_event_ids=ids,state=state)],replayed_checkpoints=[dict(base,applied_event_ids=ids2,state=state2,mode='REPLAY')],snapshot_manifest=snap,replay_contract=dict(contract_id=aid+'-lifecycle-replay',version='1',reference='Actual separate execution of public event reducer on unaccepted proposal and dated principal settlement',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=[base['checkpoint_id']],absolute_tolerance=1e-12,relative_tolerance=0))
    replay=cr.replay_consistency(**replayargs).value
    terminal=cr.compose_risk_assessment(aid+'-closed-assessment',dict(subject_id=aid,subject_type='facility'),[dict(component_id='remaining-principal',quantity='EAD',value=0,subject_id=aid,subject_type='facility',variant='remaining_contractual_claim',definition=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='after_documented_full_and_final_settlement'),currency='EUR')],dict(profile_id='closed-owner-facility',profile_version='1',model_id='dated-settlement-ledger',model_version='1',model_hash=cr.content_fingerprint(resolution).value,input_hash=cr.content_fingerprint(x['cashflows']).value),cutoff,cutoff,cutoff).value
    if verify_reference:
        assert arrears['dpd']['dpd']==5 and ead['conditional_ead']==[1000] and loss['economic_loss']==400
        assert not proposal['contract_changed'] and not proposal['default_status_changed'] and proposal['candidate']['remaining_claim']==700
        assert abs(lgd-.4)<1e-12 and replay['decision']['status']=='PASS' and state['ever_default']
        assert abs(portfolio['distribution']['expected_loss']-400)<1e-10
    return dict(account_id=aid,application_id=key,cash_scope=x['cash_scope'],arrears=arrears,proposal=proposal,ead=ead,linked_recoveries=linked,workout=loss,episodes=episodes.value,cure_targets=cure,lgd_model=lgdfit.model,predicted_lgd=lgd,portfolio=portfolio,benchmark=audit,forecast_horizon_loss=horizon_loss,loss_horizon_scope='First30days only, not full contractual lifetime',prior_event_model=survival.model,prior_event_curve=curves,replay_inputs=replayargs,replay=replay,terminal_assessment=terminal,terminal_state=state)


def original_portfolio(case,origin,lgd):
    bookings=[b for b in case['bookings']if b['first_draw_at']=='2024-12-31'];positions=[];pds={};entities=[]
    for b in bookings:
        row=next(r for r in origin['applications']if r['application']['application_id']==b['application_id']);cid=row['application']['customer_id'];entities.append(dict(entity_id=cid));positions.append(dict(position_id=b['facility_id'],claim_id=b['facility_id']+'-principal-claim',facility_id=b['facility_id'],entity_id=cid,ead=1000,facility_ead=1000,currency='EUR',sector='sole_trader',region='synthetic-region',exposure_basis='GROSS',collateral_treatment='NONE',household_id=cid+'-household'));pds[b['facility_id']]=row['risk']['pd']
    rule=dict(policy_id='separate-sole-traders',version='1',edge_types=['LEGAL_IDENTITY'],direction_policy='mutual_only',transitive=True);graph=cr.build_entity_graph(entities,[],rule,as_of='2024-12-31',known_at='2024-12-31');contract=dict(portfolio_id='owner-origin-book',version='1',currency='EUR',as_of='2024-12-31',known_at='2024-12-31',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='initially_performing',event='first_default')
    prepared=cr.prepare_credit_portfolio(positions,graph,rule,contract).value;groups=prepared['default_group_ids'];gp={r['default_group_id']:pds[r['position_id']]for r in prepared['positions']};uniforms=[];weights=[]
    for state in product([0,1],repeat=len(groups)):
        uniforms.append([gp[g]/2 if bit else(1+gp[g])/2 for g,bit in zip(groups,state)]);weights.append(math.prod(gp[g]if bit else 1-gp[g]for g,bit in zip(groups,state)))
    ids=['state-'+str(i)for i in range(len(weights))];dep=dict(kind='SUPPLIED_DEPENDENT_UNIFORMS',group_ids=groups);draws=dict(scenario_ids=ids,weights=weights,group_ids=groups,sampling_design='EXACT_STATES',group_uniforms=uniforms);paths=dict(scenario_ids=ids,position_ids=[r['position_id']for r in prepared['positions']],driver_hash=cr.content_fingerprint(dict(dependence_contract=dep,shared_innovations=draws)).value,ead=[[1000]*2 for _ in ids],lgd=[[lgd]*2 for _ in ids],guarantee_treatment='NOT_YET_APPLIED')
    simulated=cr.simulate_credit_portfolio(prepared,gp,dep,draws,paths,levels=[.95]).value;distribution=cr.credit_loss_distribution(simulated['portfolio_loss_paths'],weights,scenario_ids=ids,levels=[.95],currency='EUR').value
    return dict(prepared=prepared,group_pds=gp,simulation=simulated,distribution=distribution,dependence_assumption='Independent distinct owners conditional on supplied covariates; no ownership edge between different businesses',exclusions='Immature account first funded2025-01-31 excluded from2024-12-31 book')


def benchmark(case,origin,cutoff):
    old=origin['fitted'];training=old['training_rows'];target=dict(event='default',horizon=30,time_unit='days',probability_measure='PHYSICAL',conditioning='performing');fit=cr.fit_binary_glm(np.empty((len(training),0)),[r['outcome']for r in training],feature_order=[],link='logit',fit_intercept=True,target=target,training=old['model']['training'],model_id='owner-intercept-reference',model_version='1');p=cr.predict_pd(fit,np.empty((1,0)),feature_order=[],expected_target=target).value[0]
    mt=origin['monitoring']['cohort']['target_contract'];cohorts={};events=[dict(event_id=e['event_id'],entity_id=e['facility_id'],event='default',event_at=e['default_at'],known_at=e['known_at'],default_definition=mt['default_definition'])for e in case['defaults']];events.extend(dict(event_id=e['event_id'],entity_id=e['facility_id'],event='exit',event_at=e['exit_at'],known_at=e['known_at'],default_definition=mt['default_definition'])for e in case['exits']);coverage=[dict(entity_id=b['facility_id'],complete_through=b['followup_end'],known_at=b['outcome_available_at'])for b in case['bookings']]
    for mid,model,isref in [('reference',fit.model,True),('candidate',old['model'],False),('post_review',old['model'],False)]:
        forecasts=[]
        for b in case['bookings']:
            row=next(r for r in origin['applications']if r['application']['application_id']==b['application_id']);forecasts.append(dict(forecast_id=mid+'-'+b['facility_id'],case_id=b['application_id'],entity_id=b['facility_id'],cluster_id=row['application']['customer_id'],cohort='owner-book',segment='owner',model_id=mid,model_version='1',forecast_at=b['first_draw_at'],known_at=b['first_draw_at'],pd=p if isref else row['risk']['pd'],target=mt,prediction_origin='HOLDOUT'))
        bundle=cr.build_monitoring_cohorts(forecasts,events,coverage,target_contract=mt,evaluation_cutoff=cutoff).value;cohorts[mid]=dict(cohort=bundle,model_version='1',parameter_version='1',model_fingerprint=cr.content_fingerprint(model).value,training_known_at=model['training']['known_at'])
    design=dict(model_id='reference',model_version='1',parameter_version='1',data_fingerprint=cr.content_fingerprint(case).value,population_id='owner-first-funded',split_id='live-heldout',prediction_origin='HOLDOUT',evaluation_cutoff=cutoff,target=target)
    result=cr.benchmark_predictions({k:v for k,v in cohorts.items()if k!='post_review'},reference_model_id='reference',metrics=['brier'],matching_policy='strict',multiple_comparisons='none',design=design,uncertainty='none').value
    override=cr.override_monitoring(cohorts['candidate'],cohorts['post_review'],[],policy=dict(approved_authorities=['committee'],reference='No actual PD override in this owner cohort',concentration_threshold=.6),design=design,metrics=['brier'],uncertainty='none').value
    assert abs(result['metrics']['paired_differences']['candidate']['brier']['difference']+.109375)<1e-10 and override['metrics']['override_rate']==0
    return dict(reference_model=fit.model,cohorts=cohorts,design=design,result=result,overrides=override)
