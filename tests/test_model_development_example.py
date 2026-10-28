"""Independent same-facility PD/LGD and censored-duration reference equations."""
from pathlib import Path
from functools import lru_cache
from copy import deepcopy
from datetime import date
import importlib.util,json,math
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import digamma,expit
_spec=importlib.util.spec_from_file_location('model_development_example',Path(__file__).parents[1]/'examples'/'model_development.py')
_example=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_example)
@lru_cache(maxsize=1)
def result():return _example.run()

def test_same_facility_holdouts_and_independent_pd_null_counts():
    v=result();assert v['relational']['validation']['valid']
    assert all(row['target']==original['outcome']for row,original in zip(v['relational']['windows'],[r for r in v['upstream']['prepared']['rows']if r['selected']]))
    assert abs(v['spiegelhalter']['value']['numerator'])<1e-10
    assert abs(v['spiegelhalter']['value']['variance']-15/8)<1e-10
    assert v['dependent']['value']['null_counts']==[20,22,31,21,10,15,26,22]
    assert v['dependent']['value']['p_value']==7/9
    for name in ['glm','grade','isotonic','scorecard']:
        assert abs(v['holdout']['benchmark']['metrics']['by_model'][name]['brier']-3/16)<1e-10
    assert v['report']['decision']['status']=='PASS'

def test_closed_loss_likelihood_and_open_exclusion_oracles():
    v=result();loss=v['loss'];p=loss['models']['fractional']['parameters']
    assert p['coefficients']==[0.];assert abs(p['covariance'][0][0]-.08)<1e-12
    mixture=loss['predictions']['mixture']
    for key,wanted in [('zero_probability',.3),('one_probability',.3),('interior_probability',.4),('mean_lgd',.5)]:np.testing.assert_allclose(mixture[key],wanted,atol=1e-8)
    m=v['loss_validation']['result'];assert m['coverage']['included_count']==19
    assert m['coverage']['exclusion_counts']=={'OPEN_WORKOUT':1}
    metrics=m['metrics']['overall']['count_weighted']
    assert abs(metrics['bias']-1/38)<1e-10
    assert abs(metrics['mae']-8.3/19)<1e-10
    assert abs(metrics['rmse']-math.sqrt(3.75/19))<1e-10
    # Scalar root of the symmetric beta precision score, independent of multivariate fitter.
    phi=brentq(lambda q:digamma(q)-digamma(q/2)+np.mean(np.log([.1,.2,.8,.9])),.01,1000)
    fitted=loss['models']['mixture']['parameters']['interior_model']['parameters']
    assert abs(math.exp(fitted['precision_coefficients'][0])-phi)<1e-7

def test_censored_weibull_profile_score_and_remaining_value_oracle():
    loss=result()['loss'];model=loss['models']['hurdle'];cutoff=model['training']['known_at'];times=[];events=[]
    for row in loss['hurdle_training_rows']:
        closed=row['resolution_date'] is not None and row['resolution_date']<=row['observation_end']and row['resolution_known_at']<=cutoff
        end=row['resolution_date']if closed else row['observation_end']
        times.append((date.fromisoformat(end)-date.fromisoformat(row['reference_date'])).days);events.append(int(closed))
    t=np.array(times);e=np.array(events);d=sum(e);logs=np.log(t)
    def score(k):
        power=np.exp(k*logs);return d/k+sum(logs*e)-d*sum(power*logs)/sum(power)
    shape=brentq(score,.05,20,xtol=1e-13);logscale=np.log(sum(t**shape)/d)/shape
    timing=model['parameters']['timing_model']['parameters']
    assert abs(timing['coefficients'][0]-logscale)<1e-6
    assert abs(timing['log_time_scale']-1/shape)<1e-6
    # Closed-form conditional Weibull survival of the actual frozen model, not a second API call.
    beta=timing['coefficients'][0];k=1/timing['log_time_scale']
    survival=lambda absolute:math.exp(-math.exp(k*(math.log(absolute)-beta))+math.exp(k*(math.log(16)-beta)))
    s44,s75=survival(44),survival(75);p=model['parameters'];pc=expit(p['cure_gate']['parameters']['coefficients'][0]);positive=expit(p['positive_gate']['parameters']['coefficients'][0]);amount=p['amount_regression'];mean=pc+(1-pc)*positive*math.exp(amount['coefficients'][0]+amount['residual_variance']/2)
    expected=(90-90*mean*((1-s44)*.9+(s44-s75)*.8+s75*.7))/100
    actual=loss['open_valuation']['episodes'][0]
    assert actual['known_discounted_net_cashflows']==10
    assert abs(actual['expected_final_lgd']-expected)<1e-10
    assert abs(actual['unresolved_probability']-s75)<1e-12
    assert loss['open_without_tail']['episodes'][0]['expected_final_lgd'] is None

def test_closed_cash_reconciliation_blocks_fabricated_loss_target():
    from importlib.resources import files
    data=json.loads(files('creditratingsystem.datasets').joinpath('model_development.json').read_text());row=deepcopy(data['workouts'][0]);row['realized_lgd']=.7
    with pytest.raises(ValueError,match='reconcile'):_example.realized_loss(data,row,'2023-03-01')

def test_later_recovery_information_does_not_enter_earlier_hurdle_fit():
    from importlib.resources import files
    data=json.loads(files('creditratingsystem.datasets').joinpath('model_development.json').read_text())
    flow=next(f for f in data['cashflows']if '2018-'in f['episode_id']and f['known_at']>data['residual_training_cutoff'])
    flow['amount']*=.8
    row=next(r for r in data['workouts']if r['episode_id']==flow['episode_id']);row['realized_lgd']=(100-flow['amount'])/100
    changed=_example.loss_development(data)
    assert changed['models']['hurdle']['parameters']==result()['loss']['models']['hurdle']['parameters']
    assert changed['models']['fractional']['parameters']!=result()['loss']['models']['fractional']['parameters']
