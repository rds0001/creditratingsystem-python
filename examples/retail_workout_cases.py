"""Independent retail cash ledgers and predictions from a genuinely earlier cohort."""
from copy import deepcopy
from datetime import date,timedelta
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('retail_workout_cases.json').read_text())
    data=deepcopy(data);tr=data['cure_training']
    training_end=date.fromisoformat(tr['calendar_origin'])+timedelta(days=max(r['stop']for r in tr['rows']))
    if training_end>date.fromisoformat(tr['observation_end'])or tr['observation_end']>tr['training']['known_at']:
        raise ValueError('Cure model training spells require complete prior calendar support')
    fitted=cr.estimate_multistate_transitions(tr['rows'],states=['D','C'],allowed_transitions=[['D','C'],['C','D']],initial_state='D',target=tr['target'],training=tr['training'],model_id='retail-prior-cure',model_version='1')
    model=deepcopy(fitted.model);model['collections_contract']=dict(cure_states=['C'],default_states=['D'],supported_actions=['baseline'],supported_initial_states=['D','C'],action_interpretation='observational_conditioning')
    out={}
    for product,case in data['cases'].items():
        ead=case['ead'];eid=case['episode']['episode_id'];default_date=case['default_date']
        linked=cr.build_recovery_episodes([dict(episode_id=eid,entity_id=case['subject_id'],default_at=default_date,known_at=default_date,ead=ead,currency='EUR')],[dict(cashflow_id=f['cashflow_id'],episode_id=eid,cashflow_at=f['cashflow_date'],known_at=f['known_at'],amount=f['amount'],kind='recovery'if f['kind']=='RECOVERY'else'cost',currency=f['currency'])for f in case['cashflows']],cures=[],evaluation_cutoff='2024-03-15').value
        args=dict(cashflows=case['cashflows'],discount_curve=case['discount_curve'],fx_contract=case['fx_contract'],default_date=default_date,reporting_currency='EUR')
        final=cr.discount_recoveries(**args,evaluation_cutoff='2024-05-01');known=cr.discount_recoveries(**args,evaluation_cutoff='2024-03-15')
        resolved=all(isinstance(case['episode'].get(k),str)and case['episode'][k]<='2024-05-01'for k in ['resolution_date','resolution_known_at'])
        closed=cr.workout_lgd(ead,final,episode_status='CLOSED'if resolved else'OPEN',default_date=default_date,evaluation_cutoff='2024-05-01',currency='EUR').value
        incomplete=cr.workout_lgd(ead,known,episode_status='OPEN',default_date=default_date,evaluation_cutoff='2024-03-15',currency='EUR').value
        openargs=dict(episodes=[case['episode']],cashflows=case['cashflows'],fitted_recovery_model=data['residual_model'],X=[[]],valuation_contracts=[case['valuation']],future_cost_cashflows=case['future_costs'],feature_order=[],expected_target=data['target'],stage_spec=data['stage_spec'],as_of='2024-03-15',currency='EUR',extrapolation='parametric')
        estimated=cr.estimate_open_workout_lgd(**openargs).value
        unknown=deepcopy(openargs);unknown['valuation_contracts'][0].update(tail_discount_factor=None,tail_assumption_reference=None)
        unidentified=cr.estimate_open_workout_lgd(**unknown).value
        loss=None if not resolved else cr.event_conditioned_loss([dict(state_id='survival',events={'default':False},probability=.9,conditional_loss=0),dict(state_id='default',events={'default':True},probability=.1,conditional_loss=closed['economic_loss'])],event_names=['default'],currency='EUR',probability_measure='PHYSICAL').value
        allocation=cr.simulate_recovery_waterfall(**case['claim_allocation']).value
        history=cr.build_collections_episodes(**case['collections_input']).value
        value=cr.expected_collections_value(**case['collections_value']).value
        assumptions=case['cure_recovery_assumptions'];cure=cr.predict_cure_and_redefault(current_episode=dict(episode_id=eid,state='D',model_time=0,known_at=default_date,ever_default=True),model=model,action_conditions={k:assumptions[k]for k in ['action_id','assumption_reference']},horizon_grid=[0,30,60],state_recoveries=[[0,0],[0,assumptions['recovery_on_cure']],[0,0]],target=tr['target'],currency='EUR').value
        out[product]=dict(mechanism=case['mechanism'],claim_cost_scope=case['claim_cost_scope'],claim_allocation=allocation,linked_known_cashflows=linked,discounted_recovery=final.value,closed_workout=closed,incomplete_workout=incomplete,estimated_open_workout=estimated,unidentified_tail=unidentified,event_conditioned_loss=loss,collections_episodes=history,collections_value=value,prior_known_cure_and_redefault=cure)
        if verify_reference:
            assert math.isclose(sum(r['gross_recovery']for r in allocation['claim_results']if r['claim_id']in case['owned_claim_ids']),case['terminal'],abs_tol=1e-8)
            expected=ead-.99*case['past']-.95*(case['terminal']-case['cost'])
            assert math.isclose(closed['economic_loss'],expected,abs_tol=1e-8)
            assert incomplete['economic_lgd']is None and unidentified['episodes'][0]['expected_final_lgd']is None
            ratio=(1+math.sqrt(.25*.75)*math.exp((math.log(3)/2)**2/2))/3
            assert math.isclose(estimated['episodes'][0]['expected_final_loss'],ead-.99*case['past']-.95*(ead-case['past'])*ratio+.95*case['cost'],abs_tol=1e-6)
            assert math.isclose(loss['expected_loss'],.1*expected,abs_tol=1e-8)
            assert math.isclose(value[0]['expected_net_recovery'],.25*.95*(case['terminal']-case['cost'])+.75*.95*case['terminal']/2-case['cost']/10,abs_tol=1e-8)
            assert cure['sustained_cure_probability']==[0,.5,.5]and cure['cure_then_redefault_probability']==[0,0,.5]
            assert cure['expected_recovery']==case['terminal']/2 and not cure['causal_effect_identified']
            assert [r['status']for r in history['episodes']]==['CURED','OPEN']and all(r['ever_default']for r in history['episodes'])
    return dict(product=data['product'],data_origin=data['data_origin'],version=data['version'],cure_training=dict(calendar_origin=tr['calendar_origin'],observation_end=tr['observation_end'],known_at=tr['training']['known_at'],training_subjects=sorted({r['entity_id']for r in tr['rows']})),cases=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
