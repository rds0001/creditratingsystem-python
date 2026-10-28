"""Observed specialized-credit cohorts and legally closed workout accounting."""
from importlib.resources import files
from copy import deepcopy
from datetime import date
import json
import math
import creditratingsystem as cr


def run(data=None, verify_reference=True):
    if data is None:
        data=json.loads(files('creditratingsystem.datasets').joinpath('specialized_governance.json').read_text())
    contract=dict(forecast_at='2023-01-01',forecast_known_at='2023-01-01',horizon_days=731,outcome_known_at='2025-01-01',evaluation_cutoff='2025-01-02',reference_window=dict(start='2023-01-01',end='2023-12-31'),total_count=22,mature_count=20,censored_count=2,reporting_currency='EUR')
    contract.update(data.get('cohort_contract',{}))
    out=[]
    for journey in data['journeys']:
        observed=journey['observed']
        windows=cr.build_observation_windows(**journey['observation_windows']).value
        cohort=cr.build_monitoring_cohorts(**journey['monitoring'])
        summary=cr.monitor_rating_cohort(cohort,group_by=['cohort'],reference_window=contract['reference_window'],minimum_count=10,uncertainty='iid_homogeneous_binomial').value
        args=deepcopy(journey['recovery_episodes'])
        args['closures']=[dict(closure_id=journey['case_id']+'-legal-close',episode_id=args['default_events'][0]['episode_id'],closed_at=observed['closure_at'],known_at=observed['closure_known_at'],closure_reason='SETTLEMENT')]
        episodes=cr.build_recovery_episodes(**args).value
        recovery=cr.discount_recoveries(**journey['recovery_value'])
        workout_contract=dict(default_date=observed['default_at'],evaluation_cutoff=observed['closure_known_at'],currency=observed.get('currency',contract['reporting_currency']))
        closed=cr.workout_lgd(observed['ead'],recovery,episode_status='CLOSED',**workout_contract).value
        incomplete=cr.workout_lgd(observed['ead'],recovery,episode_status='OPEN',**workout_contract).value
        panel=[dict(row,prior_default=row['label_status']=='prior_default',default_definition='first_default',outcome_known_at=contract['outcome_known_at'],grade='B')for row in windows]
        forecasts=[dict(observation_id=row['observation_id'],pd=journey['reference_pd'],forecast_at=contract['forecast_at'],known_at=contract['forecast_known_at'],default_definition='first_default',horizon_days=contract['horizon_days'],unit='obligor',probability_measure='PHYSICAL',model_id=journey['generation_model_id'],model_version='1')for row in panel]
        counts=cr.default_count_backtest(panel,forecasts,['grade'],unit='obligor',horizon_days=contract['horizon_days'],default_definition='first_default',redefault_policy='exclude_previous_default',evaluation_cutoff=contract['evaluation_cutoff']).value
        days=(date.fromisoformat(observed['settlement_at'])-date.fromisoformat(observed['default_at'])).days
        oracle_pv=(observed['gross_recovery']-observed['direct_cost'])/(1+observed['annual_discount_rate'])**(days/365)
        oracle_lgd=1-oracle_pv/observed['ead']
        if verify_reference:
            assert len(windows)==contract['total_count'] and sum(w['censored']for w in windows)==contract['censored_count']
            assert sum(w['target']or 0 for w in windows)==journey['cohort_defaults']
            assert counts['overall']['observed_count']==contract['mature_count'] and counts['overall']['observed_defaults']==journey['cohort_defaults']
            assert summary[0]['metrics']['n']==contract['mature_count']
            pd=journey['reference_pd'];n=contract['mature_count'];d=journey['cohort_defaults']
            assert math.isclose(summary[0]['metrics']['brier'],(d*(1-pd)**2+(n-d)*pd**2)/n,abs_tol=1e-12)
            assert episodes['episodes'][0]['status']=='closed'
            assert math.isclose(recovery.value['discounted_net_recoveries'],oracle_pv,abs_tol=1e-12)
            assert math.isclose(closed['economic_lgd'],oracle_lgd,abs_tol=1e-12)
            assert incomplete['economic_lgd']is None
        out.append(dict(product=journey['product'],case_id=journey['case_id'],observed_source=observed['source'],windows=windows,cohort_summary=summary,default_backtest=counts,recovery_episodes=episodes,discounted_recoveries=recovery.value,closed_workout=closed,incomplete_workout=incomplete,independent_oracle=dict(discount_days=days,present_value=oracle_pv,lgd=oracle_lgd)))
    return dict(data_origin='SYNTHETIC',scope='Observed monitoring, default counts and closed settlement accounting; simulation labels never treated as observations',journeys=out)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
