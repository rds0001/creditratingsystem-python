"""Product-specific recovery claims with known cash, final settlement and open tails."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('structured_workout.json').read_text())
    data=deepcopy(data);out={};model=data['residual_model']
    for product,case in data['variants'].items():
        ead=case['ead'];eid=case['episode']['episode_id'];date=case['default_date']
        defaults=[dict(episode_id=eid,entity_id=case['subject_id'],default_at=date,known_at=date,ead=ead,currency='EUR')]
        flows=[dict(cashflow_id=f['cashflow_id'],episode_id=eid,cashflow_at=f['cashflow_date'],known_at=f['known_at'],amount=f['amount'],kind='recovery',currency='EUR')for f in case['cashflows']]
        linked=cr.build_recovery_episodes(defaults,flows,cures=[],evaluation_cutoff='2024-03-15').value
        recovery=cr.discount_recoveries(case['cashflows'],case['discount_curve'],case['fx_contract'],default_date=date,evaluation_cutoff='2024-05-01',reporting_currency='EUR')
        observed_closed=all(isinstance(case['episode'][key],str)and case['episode'][key]<='2024-05-01'for key in ['resolution_date','resolution_known_at'])
        closed=cr.workout_lgd(ead,recovery,episode_status='CLOSED'if observed_closed else'OPEN',default_date=date,evaluation_cutoff='2024-05-01',currency='EUR').value
        known=cr.discount_recoveries(case['cashflows'],case['discount_curve'],case['fx_contract'],default_date=date,evaluation_cutoff='2024-03-15',reporting_currency='EUR')
        incomplete=cr.workout_lgd(ead,known,episode_status='OPEN',default_date=date,evaluation_cutoff='2024-03-15',currency='EUR').value
        openargs=dict(episodes=[case['episode']],cashflows=case['cashflows'],fitted_recovery_model=model,X=[[]],valuation_contracts=[case['valuation']],future_cost_cashflows=[],feature_order=[],expected_target=data['target'],stage_spec=data['stage_spec'],as_of='2024-03-15',currency='EUR',extrapolation='parametric')
        projected=cr.estimate_open_workout_lgd(**openargs).value
        unknown=deepcopy(openargs);unknown['valuation_contracts'][0]['tail_discount_factor']=None;unknown['valuation_contracts'][0]['tail_assumption_reference']=None
        unidentified=cr.estimate_open_workout_lgd(**unknown).value
        # Exclusive default/survival states use the conditional actual loss once.
        conditional=None if not observed_closed else cr.event_conditioned_loss([dict(state_id='survive',events=dict(claim_default=False),probability=.5,conditional_loss=0),dict(state_id='default',events=dict(claim_default=True),probability=.5,conditional_loss=closed['economic_loss'])],event_names=['claim_default'],currency='EUR',probability_measure='PHYSICAL').value
        treatment=cr.evaluate_treatment_candidates(**case['treatment_input']).value
        invalid=deepcopy(case['treatment_input']);invalid['treatment_candidates'][0]['eligibility_status']='INELIGIBLE';ineligible=cr.evaluate_treatment_candidates(**invalid).value
        if product=='COUNTERPARTY_EXPOSURE':
            exposure=dict(applicability='NOT_APPLICABLE',method='ead.exposure_schedule',reason='Signed netting, collateral and margin-period closeout are not a drawn-loan schedule. The counterparty_credit_products recipe executes aggregate_netting_set_exposure, simulate_collateral_margin and closeout_exposure_at_default for this mechanism.')
        else:
            exposure=cr.exposure_schedule(**case['exposure_input']).value
        out[product]=dict(linked_known_cashflows=linked,realized_recovery=recovery.value,closed_workout=closed,known_recovery=known.value,incomplete_workout=incomplete,projected_workout=projected,unidentified_tail=unidentified,event_conditioned_loss=conditional,treatment=treatment,ineligible_treatment=ineligible,exposure=exposure)
    if verify_reference:
        mean_positive=math.sqrt(.25*.75)*math.exp((math.log(3)/2)**2/2)
        mean_ratio=(1+mean_positive)/3
        for product,r in out.items():
            c=data['variants'][product];expected=c['ead']-c['past']-c['terminal']
            assert math.isclose(r['closed_workout']['economic_loss'],expected,abs_tol=1e-10)
            assert r['incomplete_workout']['economic_lgd'] is None
            p=r['projected_workout']['episodes'][0]
            assert math.isclose(p['known_discounted_net_cashflows'],c['past'],abs_tol=1e-10)
            assert math.isclose(p['expected_final_loss'],c['ead']-c['past']-(c['ead']-c['past'])*mean_ratio,abs_tol=1e-7)
            assert r['unidentified_tail']['episodes'][0]['expected_final_lgd'] is None
            assert p['unresolved_probability']>0 and any('terminal'in x for x in p['excluded_cashflows'])
            assert math.isclose(r['event_conditioned_loss']['expected_loss'],expected/2,abs_tol=1e-10)
            # Uptake .5, incremental accepted10 less cost2, offercost1: gain3.
            assert r['treatment']['candidates'][0]['scenario_difference_from_baseline']==3
            assert r['ineligible_treatment']['status']=='INFEASIBLE'
            if product!='COUNTERPARTY_EXPOSURE':assert r['exposure']['schedule'][0]['conditional_ead']==c['ead']
    return dict(data_origin=data['data_origin'],version=data['version'],product=data['product'],variants=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
