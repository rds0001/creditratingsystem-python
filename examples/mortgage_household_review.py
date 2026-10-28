"""Joint mortgage budget and observed funded-account validation.

The worked cohort is synthetic and too small for an empirical performance claim.
"""
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import runpy
import json
import math
import creditratingsystem as cr


def fingerprint(value):
    return cr.content_fingerprint(value).value


def run(data=None, verify_reference=True, return_artifacts=False):
    load=lambda name: json.loads(files('creditratingsystem.datasets').joinpath(name+'.json').read_text())
    if data is None: data=load('mortgage_household_review')
    life=load('mortgage_lifecycle')
    helper=runpy.run_path(str(Path(__file__).with_name('input_lineage.py')))['prepare_snapshot']
    resource=files('creditratingsystem.datasets')
    lineage=helper(resource.joinpath('mortgage_applications.csv'),resource.joinpath('mortgage_releases.csv'),data['lineage_config'],data['lineage_config']['cutoffs'][0])
    assert lineage['audit']['decision']['status']=='PASS'
    for row in lineage['rows']:
        feature=next(f for f in life['features']if f['application_id']==row['entity_id'])
        assert row['x']==feature['value']
        feature['value']=row['x']
    retail=runpy.run_path(str(Path(__file__).with_name('retail_slice.py')))['run'](life,return_artifacts=True)
    lifecycle=runpy.run_path(str(Path(__file__).with_name('mortgage_lifecycle.py')))['run'](life)
    assert lifecycle['application_and_monitoring']==retail['summary']
    D='2024-02-01';subject=dict(subject_id='customer_A',subject_type='customer')
    subjects=cr.build_rating_subjects(dict(customer=[dict(subject_id='customer_A',version='1',effective_from=D,effective_to=None,known_at=D,attributes=dict(product_id='residential_mortgage'))]),[],D,D).value
    snapshot=deepcopy(life['route_snapshot']);snapshot['attributes']['recent_arrears']=dict(value=retail['requests'][0]['input']['snapshot']['features']['recent_arrears'],observed_at=D,known_at=D)
    contract=dict(contract_id='mortgage-support',version='1',required_features=['product_id','recent_arrears'],allowed_values=dict(product_id=['residential_mortgage']),support_bounds=dict(recent_arrears=[0,1]),maximum_age_days=dict(recent_arrears=31))
    applicability=cr.assess_rating_applicability(snapshot,contract,D,D).value
    bad=deepcopy(snapshot);bad['attributes']['product_id']['value']='unsupported_product'
    negative=cr.assess_rating_applicability(bad,contract,D,D).value
    row=retail['batch']['rows'][0];binding=dict(profile_id='residential-mortgage',profile_version='1',model_id=retail['model']['model_id'],model_version='1',model_hash=retail['requests'][0]['model_hash'],input_hash=row['input_hash'])
    assessment=cr.compose_risk_assessment('mortgage-A',subject,[dict(component_id='application-PD',quantity='PD',value=row['value']['pd'],**subject,variant='application_given_funding',definition=life['target'])],binding,D,D,D).value
    explanation=cr.decompose_policy_decision(retail['decision_artifacts'][0],dict(ELIGIBILITY='Declared eligibility failed',AFFORDABILITY='Verified household capacity failed',RISK_HIGH='PD exceeds risk threshold',RISK_LOW='PD within approval threshold')).value
    governance=dict(subjects=subjects,applicability=applicability,negative_applicability=negative,assessment=assessment,policy_explanation=explanation)
    graph=cr.build_customer_household_graph(**data['household']['graph'])
    household=cr.aggregate_household_obligations(**data['household']['household'],household_graph=graph).value
    joint=cr.assemble_joint_application(**data['joint_application']).value
    collateral=cr.combined_loan_to_value(**data['collateral']).value
    a=data['affordability'];stress=cr.stress_affordability(a['baseline'],a['scenario']).value
    case=dict(prediction_id='joint-mortgage-budget',entity_id='customer_A',cluster_id='household_A',segment='residential_mortgage',as_of='2024-02-01',forecast_known_at='2024-02-01',horizon_end='2024-03-01',model_id='contract-budget',model_version='1',parameter_version='1',household_id='household_A',baseline=a['baseline'],scenario=a['scenario'],assessment=stress,evidence_bindings={r['economic_id']:r['evidence_id']for r in a['economic_evidence']},expected_economic_ids={k:[x['economic_id']for x in a['baseline'][k]]for k in ['incomes','expenses','contracts','collateral']},scenario_known_at='2024-02-01',profile_known_at='2024-02-01',profile_version='1')
    design=dict(model_id='contract-budget',model_version='1',parameter_version='1',data_fingerprint=fingerprint(a),population_id='joint-household',split_id='mortgage-holdout',prediction_origin='HOLDOUT',evaluation_cutoff='2025-03-01',currency='EUR',target=dict(event='default',horizon=1,time_unit='months',probability_measure='PHYSICAL',conditioning='household_budget'))
    # No observed February repayment ledger is supplied: do not invent its outcome.
    affordability_inputs=dict(cases=[case],economic_evidence=a['economic_evidence'],allocations=a['allocations'],design=design,payment_outcomes=[])
    affordability=cr.affordability_stress_backtest(**affordability_inputs).value
    profile=dict(profile_id='mortgage-first-default',version='1',reference='Original actual mortgage target and account histories',known_at='2023-12-31',default_definition='DECLARED_FIRST_DEFAULT',default_group_rule='facility-only',default_unit='facility',first_default=True,exit_treatment='censor',default_event='DEFAULT',chargeoff_event='CHARGEOFF',cure_event='CURED',competing_events=['CLOSED'],risk_set_by_product={'residential_mortgage':dict(active_states=['ACTIVE'],include_zero_balance=True)},split_isolation_keys=['customer_id','household_id'],application_horizon_days=365,behavioral_horizon_months=12)
    splits=[dict(customer_id=r['customer_id'],household_id='household_'+r['application_id'],role='test')for r in life['applications']]
    sources=[dict(source_id='application-'+r['row_id'],prediction_key='application:'+r['row_id'],source_hash=fingerprint(r['input']['snapshot']),kind='DOCUMENT',observed_at='2024-02-01',available_at='2024-02-01')for r in retail['requests']]
    for snapshot in retail['account_panel']:
        sources.append(dict(source_id='account-'+snapshot['as_of'],prediction_key='account:'+snapshot['account_id']+'@'+snapshot['as_of'],source_hash=fingerprint(snapshot),kind='STATEMENT',observed_at=snapshot['as_of'],available_at=snapshot['known_at']))
    cohort_design=dict(design,model_id=retail['behavioral_model']['model_id'],target=dict(life['behavioral']['target'],event='default'))
    cohort_inputs=dict(application_targets=retail['application_targets'],account_panel=retail['account_panel'],behavioral_targets=retail['behavioral_targets'],target_profile=profile,split_memberships=splits,source_lineage=sources,design=cohort_design)
    cohort=cr.validate_retail_cohort(**cohort_inputs).value
    pool=data['pool'];fit=cr.fit_retail_risk_pools(**pool['fit']);assigned=cr.assign_retail_risk_pool(fitted_model=fit,**pool['assignment_inputs'])
    lgdfit=cr.fit_fractional_lgd(**pool['lgd_fit']);severity=cr.predict_lgd(lgdfit,[[]],feature_order=[],expected_target=pool['lgd_fit']['target'],default_condition='given_default').value['mean_lgd'][0]
    selected=assigned.value['assignments'][0];pid=selected['pool_id']
    statistics=next(p['statistics']for s in fit.model['parameters']['segments']for p in s['pools']if p['pool_id']==pid)
    pd=statistics['means']['pd'];d=pool['design'];outcome=pool['cohort'][0]
    forecast=dict(prediction_id='mortgage-at-funding',observation_id='facility_A',entity_id='facility_A',cluster_id='household_A',segment='residential_mortgage',as_of=outcome['index_date'],forecast_known_at=outcome['index_date'],horizon_end=outcome['label_window_end'],model_id=d['model_id'],model_version='1',parameter_version='1',pd=pd,lgd=severity,ead=pool['forecast_ead'],reference_exposure=100000,excluded_from_training=True,excluded_from_calibration=True)
    pool_inputs=dict(fitted_model=fit.model,pool_assignments=assigned.value,assignment_inputs=pool['assignment_inputs'],predictions=[forecast],cohort_outcomes=pool['cohort'],workouts=pool['workouts'],links=pool['links'],estimation_profile=pool['estimation_profile'],design=d)
    diagnostics=cr.retail_pool_diagnostics(**pool_inputs).value
    discounted=cr.discount_recoveries(**life['realized_recovery'])
    closed=cr.workout_lgd(life['default_ead'],discounted,episode_status='CLOSED',default_date=life['realized_recovery']['default_date'],evaluation_cutoff=life['realized_recovery']['evaluation_cutoff'],currency='EUR').value
    loss_design=deepcopy(d);loss_design['target']['conditioning']='default'
    record=dict(forecast,predicted_lgd=severity,predicted_ead=pool['forecast_ead'],realized_lgd=closed['economic_lgd'],realized_ead=95000,realized_ead_date='2024-09-01',default_date='2024-09-01',default_known_at='2024-09-02',defaulted=True,workout_status='CLOSED',workout_closed_at='2025-09-15',outcome_known_at='2025-09-30',ead_at_default=95000,currency='EUR',drawn_at_reference=100000,limit_at_reference=100000)
    lgd_inputs=dict(records=[record],design=loss_design,workout_policy=dict(method='closed_only',reference='Observed property sale and closed actual mortgage workout'),uncertainty='none')
    lgd=cr.lgd_backtest(**lgd_inputs).value
    ead=cr.ead_backtest([record],design=loss_design,weighting_policy=dict(method='equal',reference='One actual observed defaulted facility'),uncertainty='none').value
    # Re-evaluate each historical request against its pinned model, then audit it.
    log=[]
    for phase,at,payload in [('APPLICATION','2024-02-01',retail['requests']),('FUNDING','2024-02-03',life['booked']),('DEFAULT','2024-09-02',life['defaults']),('SETTLEMENT','2025-09-30',life['realized_recovery'])]:
        payload=dict(records=payload)
        log.append(dict(event_id=phase,idempotency_key=phase,occurred_at=at,known_at=at,sequence=len(log),payload=payload,payload_hash=fingerprint(payload),revision_of=None))
    snapshots=[dict(snapshot_id=kind,kind=kind,content_hash=fingerprint(value),known_at=at,effective_from=at,effective_until=None)for kind,value,at in [('MODEL',retail['model'],'2023-12-31'),('CALIBRATION',retail['calibrator'],'2023-12-31'),('POLICY',retail['policy'],'2023-12-31'),('SOURCE',life['features'],'2024-02-01')]]
    base=dict(checkpoint_id='final',as_of='2025-09-30',known_at='2025-09-30',event_log_hash=fingerprint(log),snapshot_hashes={s['snapshot_id']:s['content_hash']for s in snapshots},applied_event_ids=[r['event_id']for r in log])
    def event_state():
        state=None
        for event in log:
            change=dict(event_id=event['event_id'],revision='1',occurred_at=event['occurred_at'],known_at=event['known_at'],dependencies=[dict(type='facility',id='facility_A')],payload=event['payload'])
            state=cr.apply_assessment_event(state,change,[dict(assessment_id='mortgage-A',dependency_type='facility',dependency_id='facility_A')],'2025-09-30').value['state']
        return state
    original=dict(base,state=dict(ratings=retail['batch']['rows'],dependency_state=event_state()))
    rerated=cr.rate_subject_batch(retail['requests'],retail['registry'],'synthetic-applications-2024','2024-02-01').value
    replayed=dict(base,state=dict(ratings=rerated['rows'],dependency_state=event_state()),mode='REPLAY')
    replay_inputs=dict(event_log=log,original_checkpoints=[original],replayed_checkpoints=[replayed],snapshot_manifest=snapshots,replay_contract=dict(contract_id='mortgage-historical-execution',version='1',reference='Actual rerating of original requests against retained model and calibrator',ordering='KNOWN_AT_SEQUENCE_EVENT_ID',idempotency='FIRST_IDENTICAL_DELIVERY',corrections='APPEND_REVISION',required_checkpoint_ids=['final'],absolute_tolerance=1e-12,relative_tolerance=0))
    replay=cr.replay_consistency(**replay_inputs).value
    raw=dict(joint_application=joint,collateral=collateral,lineage=lineage,governance=governance,lifecycle=lifecycle,retail=retail,household=household,stress=stress,affordability_inputs=affordability_inputs,affordability=affordability,cohort_inputs=cohort_inputs,cohort=cohort,pool_inputs=pool_inputs,pool=diagnostics,lgd_model=lgdfit.model,closed_workout=closed,lgd_inputs=lgd_inputs,lgd=lgd,ead=ead,replay_inputs=replay_inputs,replay=replay)
    if verify_reference:
        o=data['oracles'];period=stress['periods'][0]
        for key,expected in [('net_income',o['stressed_income']),('expenses',o['stressed_expenses']),('debt_service',o['debt_service']),('residual',o['residual'])]:assert math.isclose(period[key],expected,abs_tol=1e-8)
        assert household['households'][0]['total_budget_debt_service']==900
        assert cohort['decision']['status']=='PASS',cohort['violations']
        assert affordability['decision']['status']!='BLOCKED',affordability
        assert math.isclose(diagnostics['metrics']['overall_count']['brier'],o['pool_brier'],abs_tol=1e-12)
        assert math.isclose(closed['economic_lgd'],o['lgd'],abs_tol=1e-12)
        assert math.isclose(ead['metrics']['ead']['overall']['selected_weighting']['bias'],o['ead_bias'],abs_tol=1e-8)
        assert replay['decision']['status']=='PASS',replay
    return raw if return_artifacts else presentation(raw)


def presentation(raw):
    period=raw['stress']['periods'][0]
    return dict(data_origin='SYNTHETIC',product='residential_mortgage',facility_id='facility_A',
        joint_application=raw['joint_application'],collateral=raw['collateral'],lineage=raw['lineage'],governance=dict(applicability=raw['governance']['applicability'],negative_applicability=raw['governance']['negative_applicability'],policy_action=raw['governance']['policy_explanation']['action']),application=raw['retail']['summary'],household=raw['household'],lifecycle=raw['lifecycle'],
        stressed_budget={k:round(period[k],2)for k in ['net_income','expenses','debt_service','residual']},
        fixed_rate_reset=[c['reset']for c in period['contracts']],
        affordability_status=raw['affordability']['decision']['status'],
        missing_observed_payment_periods=raw['affordability']['missing_payment_periods'],
        cohort_status=raw['cohort']['decision']['status'],cohort_metrics=raw['cohort']['metrics'],
        pool_coverage=raw['pool']['coverage'],pool_brier=raw['pool']['metrics']['overall_count']['brier'],
        closed_workout=raw['closed_workout'],lgd_bias=round(raw['lgd']['metrics']['overall']['ead_weighted']['bias'],8),
        ead_bias=raw['ead']['metrics']['ead']['overall']['selected_weighting']['bias'],
        replay_status=raw['replay']['decision']['status'],
        presentation_precision='Monetary budget2decimals; LGD error8decimals; raw artifacts retained through return_artifacts',
        scope='One actual synthetic funded mortgage; missing payments and nonfunded application outcomes remain unobserved')


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
