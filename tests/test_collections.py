"""Dated independent cure and episode-linkage oracles."""
from copy import deepcopy
import pytest
from creditratingsystem import CreditRiskError
from creditratingsystem.collections import (
    build_collections_episodes, build_cure_targets, build_collections_action_history,
    simulate_repayment_arrangement, expected_collections_value,
)


def case():
    states=[dict(account_id='A',state_event_id='e'+str(i),state=s,occurred_at=d,known_at=d)
            for i,(d,s) in enumerate([('2024-01-01','DEFAULT'),('2024-01-10','CLEAN'),
                                      ('2024-02-01','ARREARS'),('2024-02-10','CLEAN'),
                                      ('2024-03-20','DEFAULT')])]
    rule=dict(rule_id='synthetic',version='1',problem_states=['ARREARS','DEFAULT'],clean_states=['CLEAN'],
              terminal_states=['CLOSED'],default_states=['DEFAULT'],probation_days=10,separation_days=15)
    return dict(states=states,actions=[],coverage=[dict(account_id='A',observed_from='2024-01-01',
                 observed_through='2024-04-30',known_at='2024-04-30')],episode_rule=rule,evaluation_cutoff='2024-04-30')


def test_distinct_spells_closed_by_evidence_not_data_endpoint():
    history=build_collections_episodes(**case())
    ep=history.value['episodes']
    assert [e['status'] for e in ep]==['CURED','CURED','OPEN']
    assert [e['ended_at'] for e in ep]==['2024-01-25','2024-02-25',None]
    assert ep[1]['previous_episode_id']==ep[0]['episode_id']
    assert all(e['ever_default'] for e in ep)
    labels=build_cure_targets(history,horizon_days=30,redefault_horizon_days=90,evaluation_cutoff='2024-04-30').value
    assert [r['cure_label'] for r in labels]==[1,1,0]
    assert labels[0]['cure_at']=='2024-01-20'
    assert labels[0]['redefault_label']==1 and labels[0]['redefault_at']=='2024-03-20'


def test_missing_probation_coverage_remains_censored():
    args=case();args['states']=args['states'][:2];args['coverage']=[]
    history=build_collections_episodes(**args)
    assert history.value['episodes'][0]['status']=='OPEN'
    labels=build_cure_targets(history,horizon_days=30,redefault_horizon_days=30,evaluation_cutoff=args['evaluation_cutoff']).value
    assert labels[0]['cure_label'] is None and labels[0]['redefault_label'] is None


def test_relapse_at_probation_boundary_interrupts_sustained_cure():
    args=case();args['states']=args['states'][:3]
    args['states'][2].update(occurred_at='2024-01-20',known_at='2024-01-20')
    history=build_collections_episodes(**args)
    assert len(history.value['episodes'])==1
    labels=build_cure_targets(history,horizon_days=30,redefault_horizon_days=30,evaluation_cutoff=args['evaluation_cutoff']).value
    assert labels[0]['cure_label']==0


def test_termination_censors_cure_and_never_erases_default():
    args=case();args['states']=args['states'][:2]
    args['states'][1]['state']='CLOSED'
    history=build_collections_episodes(**args)
    assert history.value['episodes'][0]['status']=='TERMINATED'
    labels=build_cure_targets(history,horizon_days=30,redefault_horizon_days=30,evaluation_cutoff=args['evaluation_cutoff']).value
    assert labels[0]['cure_label'] is None and labels[0]['cure_status']=='CENSORED_EXIT'


def test_terminal_after_episode_cure_censors_post_cure_risk_set():
    args=case();args['states']=args['states'][:3]
    args['states'][2]['state']='CLOSED'
    history=build_collections_episodes(**args)
    assert len(history.value['episodes'])==1
    assert len(history.value['state_history'])==3
    labels=build_cure_targets(history,horizon_days=30,redefault_horizon_days=90,evaluation_cutoff=args['evaluation_cutoff']).value
    assert labels[0]['cure_label']==1
    assert labels[0]['redefault_label'] is None and labels[0]['redefault_status']=='CENSORED_EXIT'


def test_dated_action_stages_and_future_information():
    args=case();args['actions']=[dict(action_id='offer',account_id='A',action_type='plan',stage='OFFERED',occurred_at='2024-01-05',known_at='2024-01-06'),
        dict(action_id='late',account_id='A',action_type='plan',stage='PERFORMED',occurred_at='2024-01-07',known_at='2024-05-01')]
    history=build_collections_episodes(**args)
    assert [a['stage'] for a in history.value['episodes'][0]['actions']]==['OFFERED']
    changed=deepcopy(history.value);changed['rule']['probation_days']=0
    with pytest.raises(CreditRiskError,match='INTEGRITY_ERROR'):
        build_cure_targets(changed,horizon_days=30,redefault_horizon_days=30,evaluation_cutoff=args['evaluation_cutoff'])


def test_coverage_requires_baseline_and_simultaneous_states_need_resolution():
    args=case();args['coverage'][0]['observed_from']='2023-12-01'
    with pytest.raises(CreditRiskError,match='COLLECTIONS_COVERAGE'):build_collections_episodes(**args)
    args=case();args['states'][1]['occurred_at']='2024-01-01'
    with pytest.raises(CreditRiskError,match='COLLECTIONS_AMBIGUITY'):build_collections_episodes(**args)


def action_case():
    return dict(actions=[dict(action_id='offer',treatment_id='plan1',account_id='A',action_type='plan',stage='OFFERED',occurred_at='2024-01-10',known_at='2024-01-12')],
        features=[dict(feature_record_id='income-before',account_id='A',feature='income',value=2000,observed_at='2024-01-01',known_at='2024-01-02'),
                  dict(feature_record_id='income-after',account_id='A',feature='income',value=5000,observed_at='2024-01-09',known_at='2024-01-11')],
        outcomes=[dict(outcome_id='prior',account_id='A',event_type='payment',occurred_at='2024-01-10',known_at='2024-01-11'),
                  dict(outcome_id='future',account_id='A',event_type='payment',occurred_at='2024-01-20',known_at='2024-01-20')],
        coverage=[dict(account_id='A',observed_from='2024-01-01',observed_through='2024-04-30',known_at='2024-04-30')],
        index_stage='OFFERED',required_features=['income'],horizon_days=30,maximum_feature_age_days=30,evaluation_cutoff='2024-04-30')


def test_action_panel_uses_actual_index_not_late_logging_or_acceptance():
    v=build_collections_action_history(**action_case()).value['rows'][0]
    assert v['preaction_features']==dict(income=2000)
    assert v['accepted'] is False and v['performed'] is False
    assert v['prior_or_same_day_outcome_ids']==['prior']
    assert [r['outcome_id'] for r in v['outcomes']]==['future']
    assert v['outcome_complete'] and not v['causal_effect_identified']


def test_same_day_features_are_not_attested_preaction():
    args=action_case()
    args['features'][1].update(observed_at='2024-01-10',known_at='2024-01-10')
    assert build_collections_action_history(**args).value['rows'][0]['preaction_features']['income']==2000


def test_action_post_outcome_order_visible_and_missing_coverage_not_zero():
    args=action_case();args['coverage']=[]
    args['actions'].append(dict(args['actions'][0],action_id='performed',stage='PERFORMED',occurred_at='2024-01-05',known_at='2024-01-05'))
    v=build_collections_action_history(**args).value['rows'][0]
    assert v['invalid_stage_order'] and not v['descriptive_eligible']
    assert not v['outcome_complete']


def arrangement():
    base=dict(plan_id='base',payments=[500,500,0],period_interest_rates=[0,0,0],new_fees=[0,0,0],
              servicing_costs=[0,0,0],receipt_fractions=[1,1,1],capitalize_unpaid_interest=False)
    return dict(contract=dict(currency='EUR',principal=1000,principal_arrears=200,accrued_interest=0,fees=0,as_of='2024-01-01'),
                baseline=base,candidate=dict(base,plan_id='defer',payments=[0,500,500]),
                affordability_paths=[dict(path_id='tight',currency='EUR',available_cash=[400,400,400])],
                discount_factors=[1,.9,.8],period_end_dates=['2024-02-01','2024-03-01','2024-04-01'],currency='EUR')


def test_deferral_moves_principal_does_not_add_arrears_twice():
    v=simulate_repayment_arrangement(**arrangement()).value
    assert v['baseline']['npv']==950 and v['candidate']['npv']==850 and v['npv_change']==-100
    assert v['candidate']['remaining_claim']==0
    assert sum(r['paid_principal'] for r in v['candidate']['schedule'])==1000
    assert v['candidate']['affordability'][0]['residual_cash']==[400,-100,-100]
    assert not v['contract_changed'] and not v['default_status_changed']
    args=arrangement();args['candidate']=deepcopy(args['baseline'])
    assert simulate_repayment_arrangement(**args).value['npv_change']==0


def test_explicit_interest_capitalization_and_cash_waterfall():
    args=arrangement();args['candidate'].update(payments=[0,1000,1000],period_interest_rates=[.1,.1,0],capitalize_unpaid_interest=True)
    v=simulate_repayment_arrangement(**args).value['candidate']['schedule']
    assert v[0]['capitalized_interest']==100 and v[0]['closing_principal']==1100
    assert v[1]['interest_accrued']==110 and v[1]['paid_interest']==110
    assert v[1]['paid_principal']==890 and v[1]['closing_principal']==210
    assert v[2]['received_payment']==210 and v[2]['unused_payment_capacity']==790


def test_receipt_assumption_does_not_reduce_affordability_obligation():
    args=arrangement();args['candidate']['receipt_fractions']=[1,0,1]
    v=simulate_repayment_arrangement(**args).value['candidate']
    assert v['remaining_claim']==500 and v['npv']==400
    assert v['affordability'][0]['residual_cash']==[400,-100,-100]


def recovery_case():
    period='2024-12-31'
    def scenario(identifier,recovery,cost):
        return dict(scenario_id=identifier,probability=1,cashflows=[
            dict(cashflow_id='recovery',kind='RECOVERY',period_end=period,amount=recovery),
            dict(cashflow_id='servicing',kind='COST',period_end=period,amount=cost)])
    action=dict(action_id='plan',currency='EUR',uptake_probability=.25,
        accepted_scenarios=[scenario('accept',80,10)],declined_scenarios=[scenario('decline',20,0)],
        offer_cost=2,support_status='SYNTHETIC_ASSUMPTION',assumption_reference='analytic-hand-case')
    return dict(actions=[action],discount_factors=[.9],ead=100,currency='EUR',period_end_dates=[period])


def test_uptake_and_offer_cost_integrated_exactly_once():
    args=recovery_case();v=expected_collections_value(**args).value[0]
    # .25*(.9*(80-10)-2) + .75*(.9*20-2) = 27.25.
    assert v['expected_net_recovery']==27.25
    assert v['scenario_min']==16 and v['scenario_max']==61
    assert not v['causal_effect_identified']
    args['actions'].append(dict(args['actions'][0],action_id='same-baseline'))
    out=expected_collections_value(**args).value
    assert out[0]['expected_net_recovery']==out[1]['expected_net_recovery']
    args['actions'][0]['accepted_scenarios'][0]['cashflows'].append(args['actions'][0]['accepted_scenarios'][0]['cashflows'][0])
    with pytest.raises(CreditRiskError):expected_collections_value(**args)
