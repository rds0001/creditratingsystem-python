"""One observed default claim, prior cure histories and an unaccepted offered plan."""
from copy import deepcopy
from datetime import date
from importlib.resources import files
import json
import math
import numpy as np
import creditratingsystem as cr


def run(data=None,verify_reference=True,return_artifacts=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('retail_collections_review.json').read_text())
    d=deepcopy(data);target=d['target'];train=d['training'];prior=d['prior'];origin=d['prior_origin'];day=lambda t:(date.fromisoformat(t)-date.fromisoformat(origin)).days
    if max(r['observed_through']for r in prior['coverage'])>train['known_at']or train['known_at']>d['valuation_date']:raise ValueError('Model requires genuinely prior observation support')
    previous=cr.build_collections_episodes(**prior);targets=cr.build_cure_targets(previous,horizon_days=90,redefault_horizon_days=90,evaluation_cutoff=prior['evaluation_cutoff']).value
    times=[];events=[];panel=[];transitions=[]
    for coverage in prior['coverage']:
        aid=coverage['account_id'];label=next(t for t in targets if t['account_id']==aid);stop=day(coverage['observed_through']);history=[r for r in prior['states']if r['account_id']==aid];exit=next((r for r in history if r['state']=='CLOSED'),None)
        t=day(label['cure_at'])if label['cure_label']==1 else day(exit['occurred_at'])if exit else stop;event='cure'if label['cure_label']==1 else'exit'if exit else None
        times.append(t);events.append(event);panel.append(dict(entity_id=aid,start=0,stop=t,event=int(event is not None),covariate_time=0))
        state='D';start=0;changes=[]
        if event=='exit':changes=[(t,'X')]
        elif event=='cure':changes=[(t,'C')]+([(day(label['redefault_at']),'R')]if label['redefault_label']==1 else[])
        for when,new in changes:transitions.append(dict(entity_id=aid,start=start,stop=when,from_state=state,to_state=new));start=when;state=new
        if start<stop:transitions.append(dict(entity_id=aid,start=start,stop=stop,from_state=state,to_state=None))
    meta=dict(feature_order=[],target=target,training=train,model_version='1')
    cause=cr.fit_cause_specific_hazards(panel,np.empty((5,0)),events,causes=['cure','exit'],ties='breslow',model_id='prior-cure-cause',**meta)
    fine=cr.fit_fine_gray(times,events,np.empty((5,0)),cause_of_interest='cure',censoring_spec=dict(estimator='kaplan_meier',independence_assumption='independent_within_censoring_group'),model_id='prior-cure-fine',**meta)
    path=[dict(path_id='first-default',start=0,stop=90,covariates=[],stratum='all',covariate_time=0)]
    curves={name:cr.predict_event_curves(model,path,[15,45,60,90],feature_order=[],target=target).value['curves'][0]for name,model in [('cause_specific',cause),('fine_gray',fine)]}
    aj=cr.estimate_multistate_transitions(transitions,states=['D','C','R','X'],allowed_transitions=[['D','C'],['D','X'],['C','R']],initial_state='D',target=target,training=train,model_id='prior-cure-redefault',model_version='1')
    model=deepcopy(aj.model);model['collections_contract']=dict(cure_states=['C'],default_states=['D','R'],supported_actions=['observed-offer'],supported_initial_states=['D'],action_interpretation='observational_conditioning')
    observed=cr.build_collections_episodes(**d['heldout']);actual=cr.build_cure_targets(observed,horizon_days=60,redefault_horizon_days=60,evaluation_cutoff=d['heldout']['evaluation_cutoff']).value
    missing=deepcopy(d['heldout']);missing['coverage']=[];uncertain=cr.build_collections_episodes(**missing);unknown=cr.build_cure_targets(uncertain,horizon_days=60,redefault_horizon_days=60,evaluation_cutoff=missing['evaluation_cutoff']).value
    action=cr.build_collections_action_history(**d['action_history']).value
    w=d['open_workout'];episode=w['episode'];default=next(r for r in d['heldout']['states']if r['state']=='DEFAULT');asof=d['valuation_date']
    if episode['episode_id']!=default['account_id']+'-first-default'or any(a['account_id']!=default['account_id']for a in d['action_history']['actions'])or episode['default_date']!=default['occurred_at']or any(t['cure_at']is not None and t['cure_at']<=asof for t in actual)or asof!=d['action_history']['actions'][0]['occurred_at']:raise ValueError('Offer and open valuation must precede observed cure on the same default claim')
    residual=episode['residual_exposure_at_reference'];cash=sum(r['amount']for r in w['cashflows']if r['kind']=='RECOVERY'and r['known_at']<=asof)
    if not math.isclose(episode['ead_at_default']-cash,residual):raise ValueError('One residual claim must reconcile prior actual recoveries')
    prediction=cr.predict_cure_and_redefault(dict(episode_id=episode['episode_id'],state='D',model_time=(date.fromisoformat(asof)-date.fromisoformat(default['occurred_at'])).days,known_at=asof,ever_default=True),model,dict(action_id='observed-offer',assumption_reference='Prior observational support, no causal action effect'),[0,30,45,60,75],[[0,0,0,0]]*4+[[.25*residual,residual,.5*residual,.25*residual]],target=target,currency='EUR').value
    p_cure=prediction['sustained_cure_probability'][-1];p_red=prediction['cure_then_redefault_probability'][-1];probs=[p_cure-p_red,p_red,1-p_cure]
    costs=[sum(f['amount']for f in w['future_costs']if f['cashflow_date']==t and f['known_at']<=asof)for t in d['period_end_dates']]
    base=dict(plan_id='existing',payments=d.get('baseline_payments',[residual/2,residual/2,0]),period_interest_rates=[0,0,0],new_fees=[0,0,0],servicing_costs=costs,receipt_fractions=[1,1,1],capitalize_unpaid_interest=False)
    candidate=dict(base,plan_id='deferred-offer',payments=d.get('candidate_payments',[0,residual/2,residual/2]))
    args=dict(contract=dict(currency='EUR',principal=residual,principal_arrears=d.get('principal_arrears',20),accrued_interest=0,fees=0,as_of=asof),baseline=base,candidate=candidate,affordability_paths=[dict(path_id='observed-preoffer-budget',currency='EUR',available_cash=d.get('available_cash',[20,50,50]))],discount_factors=d['discount_factors'],period_end_dates=d['period_end_dates'],currency='EUR')
    arrangement=cr.simulate_repayment_arrangement(**args).value;scenarios=[]
    for sid,q,p in zip(['cure-no-redefault','cure-then-redefault','no-cure'],[1,.5,.25],probs):
        a=deepcopy(args);a['candidate']['receipt_fractions']=[q,q,q];schedule=cr.simulate_repayment_arrangement(**a).value['candidate']['schedule'];flows=[]
        for j,row in enumerate(schedule):flows +=[dict(cashflow_id=sid+'-receipt-'+str(j),kind='RECOVERY',period_end=d['period_end_dates'][j],amount=row['received_payment']),dict(cashflow_id=sid+'-cost-'+str(j),kind='COST',period_end=d['period_end_dates'][j],amount=costs[j])]
        scenarios.append(dict(scenario_id=sid,probability=p,cashflows=flows))
    offered=dict(action_id='offered-plan',currency='EUR',uptake_probability=d['uptake_probability'],accepted_scenarios=scenarios,declined_scenarios=[dict(scenario_id='declined',probability=1,cashflows=[dict(cashflow_id='decline-receipt',kind='RECOVERY',period_end=d['period_end_dates'][-1],amount=d['declined_recovery'])])],offer_cost=d['offer_cost'],support_status='SYNTHETIC_ASSUMPTION',assumption_reference='Explicit cash conditional on prior observational cure paths; uptake and decline cash are assumptions')
    value=cr.expected_collections_value([offered],d['discount_factors'],ead=residual,currency='EUR',period_end_dates=d['period_end_dates']).value[0];declined=deepcopy(offered);declined['uptake_probability']=0
    no_uptake=cr.expected_collections_value([declined],d['discount_factors'],ead=residual,currency='EUR',period_end_dates=d['period_end_dates']).value[0]
    open_value=cr.estimate_open_workout_lgd([episode],w['cashflows'],d['residual_model'],[[]],[w['valuation']],w['future_costs'],feature_order=[],expected_target=d['recovery_target'],stage_spec=d['stage_spec'],as_of=asof,currency='EUR',extrapolation='parametric').value
    out=dict(data_origin=d['data_origin'],prior_cure_targets=targets,prior_first_cure_forecasts=curves,prior_transition_rows=transitions,observed_episodes=observed.value,cure_targets=actual,missing_probation_targets=unknown,action_history=action,landmark_prediction=prediction,exclusive_history_probabilities=probs,repayment_arrangement=arrangement,action_value=value,no_uptake_value=no_uptake,remaining_claim=residual,prior_discounted_recovery=cash,ultimate_assumption_based_lgd=1-(cash+value['expected_net_recovery'])/episode['ead_at_default'],open_workout=open_value,raw_artifacts_available=True)
    if verify_reference:
        assert np.allclose(probs,[1/3]*3)and math.isclose(prediction['expected_recovery'],7/12*residual)
        assert arrangement['baseline']['affordability'][0]['residual_cash'][0]<0 and min(arrangement['candidate']['affordability'][0]['residual_cash'])>=0
        assert np.allclose(curves['cause_specific']['cumulative_incidence']['cure'],[0,math.exp(-.2)*(1-math.exp(-1/3)),math.exp(-.2)*(1-math.exp(-5/6)),math.exp(-.2)*(1-math.exp(-5/6))])
        assert math.isclose(value['expected_net_recovery'],d['uptake_probability']*(7/12*residual-sum(costs))+(1-d['uptake_probability'])*d['declined_recovery']-d['offer_cost'],abs_tol=1e-10)
        assert no_uptake['expected_net_recovery']==d['declined_recovery']-d['offer_cost']
        assert [e['status']for e in observed.value['episodes']]==['CURED','OPEN']and actual[0]['redefault_label']==1
        assert unknown[0]['cure_label']is None and not action['rows'][0]['accepted']and not action['rows'][0]['performed']
        assert action['rows'][0]['preaction_features']['available_cash']==50
        assert not arrangement['contract_changed']and not arrangement['default_status_changed']
    if return_artifacts:out['raw_artifacts']=dict(cause_model=cause.model,fine_gray_model=fine.model,multistate_model=model,action_value_input=offered,arrangement_input=args)
    return out


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
