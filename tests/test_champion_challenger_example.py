from importlib.resources import files
from pathlib import Path
import json,math,runpy


def run():return runpy.run_path(str(Path(__file__).parents[1]/'examples/champion_challenger.py'))['run']


def test_better_ranking_can_have_worse_calibration_and_brier_then_replay_survives_rollback():
    v=run()();c=v['comparison']
    assert math.isclose(c['differences']['auc'],.25)
    assert math.isclose(c['differences']['brier'],.1375)
    assert math.isclose(c['differences']['calibration_error'],-.225)
    assert c['recommendation']=='REQUIRES_MULTIDIMENSIONAL_REVIEW'
    assert [r['assessment']for r in v['findings']['trace']]==['WATCH','WARNING','WARNING']
    assert v['historical_rows'][0]['value']==[.75]
    assert v['replay']['status']=='REPRODUCED'
    assert all(r['status']=='INSUFFICIENT_INFORMATION'for r in v['fairness'])
    assert all(r['metrics']['selection_rate'] is None for r in v['fairness'])
    assert abs(v['linear_explanation']['reconstruction_residual'][0])<1e-15
    assert math.isclose(v['shapley']['baseline'],6/7)
    assert v['sequential']['validation_false_alarm']['paths']==99


def test_missing_nondefault_coverage_does_not_create_good_outcomes():
    data=json.loads(files('creditratingsystem.datasets').joinpath('champion_challenger.json').read_text())
    import creditratingsystem as cr
    v=run()(data)
    incomplete=cr.build_monitoring_cohorts(v['forecast_inputs']['1'],data['events'],[],target_contract=data['target'],evaluation_cutoff=data['evaluation_cutoff']).value
    rows=incomplete['rows']
    assert sum(r['eligible'] for r in rows)==6
    assert sum(r['status']=='MISSING_OUTCOME'for r in rows)==6
    assert all(r['target'] is None for r in rows if not r['eligible'])
