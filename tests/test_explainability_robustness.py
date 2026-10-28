import pytest
from creditratingsystem._explainability_robustness import assess_counterfactual_robustness


def fixture():
    model=dict(model_id='m',model_version='1',feature_order=['risk','fixed'],output_scale='score',deterministic_frozen=True,target=dict(event='score',horizon=1,time_unit='years',probability_measure='PHYSICAL',conditioning='performing'))
    variants={'nominal':dict(predictor=lambda z:z[:,0],model_contract=model)}
    constraints=dict(contract_id='c',version='1',bounds=dict(risk=[-10,10],fixed=[0,20]),actionable_features=['risk'],immutable_features=['fixed'],integer_features=[],one_hot_groups=[],linear_constraints=[],derived_relations=[])
    target=dict(target_id='t',version='1',kind='MODEL_OUTPUT',operator='<=',value=0)
    scenarios=[dict(scenario_id='down',model_variant_id='nominal',delta=[-.1,.2],weight=1),dict(scenario_id='up',model_variant_id='nominal',delta=[.1,.2],weight=1)]
    return [[-.01,10],[-1,10]],[1,10],variants,scenarios,constraints,target


def test_thin_margin_fails_and_robust_margin_survives_measurement_error():
    r=assess_counterfactual_robustness(*fixture(),candidate_ids=['thin','robust'],nominal_variant_id='nominal').value
    thin,robust=r['candidate_summaries']
    assert thin['nominal']['target_satisfied'];assert thin['scenario_success_fraction']==.5
    assert thin['worst_target_margin']==pytest.approx(-.09)
    assert robust['all_scenarios_succeed'];assert robust['scenario_success_fraction']==1
    assert robust['worst_target_margin']==pytest.approx(.9)
    assert all(c['feasible']for s in r['scenarios']for c in s['candidates'])
    assert r['feature_perturbation_report'][1]['minimum_delta']==.2


def test_failed_model_variant_retained_in_every_denominator():
    args=list(fixture())
    def broken(z):raise ValueError('unavailable variant')
    args[2]['broken']=dict(predictor=broken,model_contract=args[2]['nominal']['model_contract'])
    args[3].append(dict(scenario_id='failure',model_variant_id='broken',delta=[0,0],weight=2))
    r=assess_counterfactual_robustness(*args,candidate_ids=['thin','robust'],nominal_variant_id='nominal').value
    thin,robust=r['candidate_summaries']
    assert thin['scenario_success_fraction']==.25;assert robust['scenario_success_fraction']==.5
    assert robust['unevaluable_weight']==.5;assert robust['scenario_feasibility_lower']==.5;assert robust['scenario_feasibility_upper']==1
    assert robust['worst_target_margin']is None
    assert 'failure'in robust['failed_scenario_ids']
    assert r['scenarios'][-1]['candidates'][0]['error_code']=='EXPLANATION_PREDICTOR'


def test_constraint_failure_retained_and_not_counted_as_unknown():
    args=list(fixture());args[3]=[dict(scenario_id='outside',model_variant_id='nominal',delta=[20,0],weight=1)]
    r=assess_counterfactual_robustness(*args,candidate_ids=['thin','robust'],nominal_variant_id='nominal').value
    for c in r['candidate_summaries']:
        assert c['scenario_success_fraction']==0;assert c['scenario_feasibility_upper']==0;assert c['unevaluable_weight']==0
    assert all('BOUND:risk'in c['findings']for c in r['scenarios'][0]['candidates'])


def test_immutable_action_stays_invalid_even_under_shared_measurement_shift():
    args=list(fixture());args[0][0][1]=11
    r=assess_counterfactual_robustness(*args,candidate_ids=['invalid','robust'],nominal_variant_id='nominal').value
    assert r['candidate_summaries'][0]['scenario_feasibility_upper']==0
    assert all('IMMUTABLE:fixed'in s['candidates'][0]['findings']for s in r['scenarios'])


def test_distinct_frozen_model_variant_and_zero_mass_scenario():
    from copy import deepcopy
    args=list(fixture());contract=deepcopy(args[2]['nominal']['model_contract']);contract['model_version']='2'
    args[2]['adverse']=dict(predictor=lambda z:z[:,0]+.2,model_contract=contract)
    args[3]=[dict(scenario_id='base',model_variant_id='nominal',delta=[0,0],weight=1),dict(scenario_id='alternative',model_variant_id='adverse',delta=[0,0],weight=0)]
    r=assess_counterfactual_robustness(*args,candidate_ids=['thin','robust'],nominal_variant_id='nominal').value
    assert r['candidate_summaries'][0]['scenario_success_fraction']==1
    assert not r['candidate_summaries'][0]['all_scenarios_succeed']
    assert r['candidate_summaries'][1]['all_scenarios_succeed']
    assert r['scenarios'][1]['candidates'][0]['prediction']==pytest.approx(.19)
