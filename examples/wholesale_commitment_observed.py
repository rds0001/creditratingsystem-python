"""Separate 30-day and two-year observed risk sets and closed workout ledgers."""
from importlib.resources import files
from pathlib import Path
import json,runpy
import creditratingsystem as cr

def run(data=None,verify_reference=True,artifact_paths=None):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_commitment_observed.json').read_text())
    engine=runpy.run_path(str(Path(__file__).with_name('specialized_observed_lifecycle.py')))['run'];rows=[]
    for cohort in data['cohorts']:
        for journey in cohort['journeys']:
            f=journey['forecast_contract'];path=artifact_paths[f['model_resource']]if artifact_paths else files('creditratingsystem.datasets').joinpath(f['model_resource'])
            model=cr.load_fitted_model(path)
            if model.diagnostics['sha256']!=f['model_hash']:raise ValueError('Forecast model hash mismatch')
            pd=cr.predict_pd(model,f['features'],feature_order=f['feature_order'],expected_target=f['target'],context={'known_at':cohort['cohort_contract']['forecast_known_at']}).value
            calibrated=cr.apply_calibrator(f['calibrator'],pd,target=f['target'],base_model_version='1',**f['calibrator_arguments']).value[0]
            if abs(calibrated-journey['reference_pd'])>1e-12 or any(abs(x['pd']-calibrated)>1e-12 or x['model_id']!=model.value['model_id']for x in journey['monitoring']['predictions']):raise ValueError('Observed cohort forecast differs from pinned inference')
            horizon=f['target']['horizon']
            if f['target']['time_unit']=='years':
                from datetime import date
                start=date.fromisoformat(cohort['cohort_contract']['forecast_at']);horizon=(start.replace(year=start.year+int(horizon))-start).days
            if horizon!=cohort['cohort_contract']['horizon_days']:raise ValueError('Observation and prediction horizons differ')
        rows.append(engine(cohort,verify_reference=verify_reference))
    return dict(data_origin='SYNTHETIC',scope=data['scope'],cohorts=rows)

if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
