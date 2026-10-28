"""Retail arrears, proposed repayment and competing exit from dated input data.

The proposals are conditional scenarios. Actual later receipts are separate
observations and never inserted into a historical forecast.
"""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None, verify_reference=True):
    if data is None:
        data=json.loads(files('creditratingsystem.datasets').joinpath('retail_transitions.json').read_text())
    output={}
    for product,c in data['cases'].items():
        aid=c['account_id']; origin=c['arrangement']['contract']['as_of']
        arrears=cr.compute_days_past_due(**c['delinquency']).value
        cured=cr.compute_days_past_due(**c['cured']).value
        state=cr.assign_delinquency_state(arrears['dpd'],arrears['past_due_amount'],c['opening_claim'],dict(closed=False,confirmed_default=False),data['state_profile']).value
        clean_state=cr.assign_delinquency_state(cured['dpd'],cured['past_due_amount'],c['opening_claim'],dict(closed=False,confirmed_default=False),data['state_profile']).value
        default_control=cr.assign_delinquency_state(0,0,c['opening_claim'],dict(closed=False,confirmed_default=True),data['state_profile']).value
        # Knowledge-time negative control: receipt occurred but was not yet known.
        late_known=deepcopy(c['cured']);late_known['payments'][-1]['known_at']=c['collections']['evaluation_cutoff']
        unconfirmed=cr.compute_days_past_due(**late_known).value
        arrangement=cr.simulate_repayment_arrangement(**c['arrangement']).value
        plan=arrangement['candidate'];dates=c['arrangement']['period_end_dates']
        contract=c['arrangement']['contract']
        # Only this prospective proposal ledger is mapped to conditional EAD.
        events=[]
        for i,row in enumerate(plan['schedule']):
            for sequence,(kind,amount) in enumerate([('INTEREST_PAYMENT',row['paid_interest']),('PRINCIPAL_REPAYMENT',row['paid_principal'])]):
                events.append(dict(event_id=f'proposal-{i}-{sequence}',event_date=row['period_end'],known_at=origin,sequence=sequence,kind=kind,amount=amount))
        exposure_args=dict(contract=dict(facility_id=aid,currency='EUR',as_of=origin,drawn_principal=contract['principal'],credit_limit=contract['principal'],accrued_interest=contract['accrued_interest']),events=events,annual_interest_rates=[0,0,0],period_end_dates=dates,day_count='ACT/365F',allow_overlimit=False,currency='EUR')
        before=cr.exposure_schedule(**exposure_args,default_timing='BEFORE_PERIOD_END_EVENTS').value
        after=cr.exposure_schedule(**exposure_args,default_timing='AFTER_PERIOD_END_EVENTS').value
        affordable=all(x>=0 for x in plan['affordability'][0]['residual_cash'])
        treatment=dict(treatment_id='proposed',version='1',known_at=origin,subject_id=aid,currency='EUR',action='ARRANGEMENT',prediction_reference='Explicit conditional proposal receipts, not causal response estimates',support_status='SUPPORTED',affordability_status='AFFORDABLE' if affordable else 'UNAFFORDABLE',eligibility_status='ELIGIBLE',uptake_probability=.5,offer_cost=0,accepted_upfront_cost=0,rejected_present_value=arrangement['baseline']['npv'],cashflows=[dict(cashflow_id='period-'+str(i),date=row['period_end'],expected_recovery=row['received_payment'],expected_cost=0,discount_factor=c['arrangement']['discount_factors'][i]) for i,row in enumerate(plan['schedule'])])
        profile=dict(profile_id=product+'-proposal',version='1',known_at=origin,currency='EUR',subject_id=aid,baseline_id='contractual',baseline_present_value=arrangement['baseline']['npv'],identification='CONDITIONAL_SCENARIO_NOT_CAUSAL',allowed_actions=['ARRANGEMENT'])
        evaluation=cr.evaluate_treatment_candidates([treatment],profile,as_of=origin).value
        episodes=cr.build_collections_episodes(**c['collections'])
        labels=cr.build_cure_targets(episodes,horizon_days=30,redefault_horizon_days=90,evaluation_cutoff=c['collections']['evaluation_cutoff']).value
        windows=cr.build_observation_windows(**c['windows']).value
        clean_date=c['cured']['as_of']
        panel=[dict(account_id=aid,as_of=origin,state='ARREARS',balance=c['opening_claim'],product=product),dict(account_id=aid,as_of=clean_date,state='CLEAN',balance=max(0,c['opening_claim']-arrears['past_due_amount']),product=product),dict(account_id=aid+'-missing',as_of=origin,state='ARREARS',balance=c['opening_claim'],product=product)]
        roll=cr.estimate_retail_roll_rates(panel,states=['ARREARS','CLEAN','CLOSED'],from_date=origin,to_date=clean_date,group_by=['product'],weighting='count',missing_successor='censor').value
        dep=[dict(assessment_id=aid+'-assessment',dependency_type='account',dependency_id=aid)]
        event=dict(event_id=aid+'-delinquency',revision='1',occurred_at=origin,known_at=origin,dependencies=[dict(type='account',id=aid)],payload=dict(dpd=arrears['dpd'],unpaid=arrears['past_due_amount']))
        impact=cr.apply_assessment_event(None,event,dep,origin).value
        repeated=cr.apply_assessment_event(impact['state'],event,dep,origin).value
        # No occurrence of DEFAULT is inferred from an ordinary late payment.
        history=[dict(entity_id=aid,start=0,stop=14,from_state='arrears',to_state='clean'),dict(entity_id=aid,start=14,stop=60,from_state='clean',to_state='exit'),dict(entity_id=aid+'-control',start=0,stop=100,from_state='arrears',to_state=None)]
        target=dict(event='exit',horizon=60,time_unit='days',probability_measure='PHYSICAL',conditioning='in_arrears_at_landmark')
        meta=dict(target=target,training=dict(data_fingerprint=product+'-observed-state-history',split_id='descriptive_history',known_at=c['collections']['evaluation_cutoff'],data_origin='SYNTHETIC'),model_id=product+'-transitions',model_version='1')
        multistate=cr.estimate_multistate_transitions(history,states=['arrears','clean','exit'],allowed_transitions=[['arrears','clean'],['clean','exit']],initial_state='arrears',**meta)
        curves=cr.predict_event_curves(multistate,[],[0,14,60],feature_order=[],target=target,event_states=['exit']).value
        if verify_reference:
            expected=c['expected']
            assert arrears['dpd']==expected['dpd'] and arrears['past_due_amount']==expected['past_due']
            assert cured['dpd']==0 and cured['past_due_amount']==0
            assert state['state']=='ARREARS' and state['at_risk'] and clean_state['state']=='CLEAN'
            assert default_control['state']=='DEFAULT' and not default_control['at_risk']
            assert unconfirmed['past_due_amount']==expected['past_due'] and unconfirmed['dpd']==44
            assert not arrangement['contract_changed'] and not arrangement['default_status_changed']
            assert math.isclose(plan['npv'],expected['proposed_npv'],abs_tol=1e-9)
            assert math.isclose(plan['remaining_claim'],expected['proposed_remaining_claim'],abs_tol=1e-9)
            assert math.isclose(sum(x['received_payment'] for x in plan['schedule'])+plan['remaining_claim'],c['opening_claim'],abs_tol=1e-9)
            assert affordable==expected['candidate_affordable']
            assert evaluation['candidates'][0]['feasible']==affordable
            assert math.isclose(evaluation['candidates'][0]['expected_present_value'],(expected['proposed_npv']+expected['baseline_npv'])/2,abs_tol=1e-9)
            for b,a,p in zip(before['schedule'],after['schedule'],plan['schedule']):
                assert math.isclose(b['conditional_ead']-a['conditional_ead'],p['received_payment'],abs_tol=1e-9)
            assert labels[0]['cure_at']==expected['cure_date'] and labels[0]['cure_label']==1
            assert labels[0]['redefault_label'] is None and labels[0]['redefault_status']=='CENSORED_EXIT'
            assert [x['target'] for x in windows]==[None,1,0]
            assert roll['groups'][0]['origin_count']==[2,0,0] and roll['groups'][0]['censored_count']==[1,0,0]
            assert roll['groups'][0]['transition_matrix'][0]==[0,1,0]
            assert repeated['status']=='DUPLICATE'
            assert curves['curves'][0]['first_hit_probability']==[0,0,.5]
        output[product]=dict(financial_convention=c['financial_convention'],delinquency=arrears,delinquency_state=state,clean_state=clean_state,confirmed_default_control=default_control,confirmed_cure=cured,unconfirmed_receipt=unconfirmed,arrangement=arrangement,conditional_ead_before=before,conditional_ead_after=after,treatment=evaluation,cure_targets=labels,observation_windows=windows,roll_rates=roll,assessment_event=impact,duplicate_event_status=repeated['status'],multistate=curves)
    return dict(data_origin='SYNTHETIC',products=output,interpretation='Actual observed cure and exit are separate from proposed cashflow scenarios; censored exits are not hypothetical nondefaults.')


if __name__=='__main__':
    print(json.dumps(run(),allow_nan=False))
