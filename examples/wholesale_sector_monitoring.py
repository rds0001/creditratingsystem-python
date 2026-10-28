"""Frozen sector predictions, longitudinal drift and observed multistate histories."""
from importlib.resources import files
from pathlib import Path
import json,runpy
import creditratingsystem as cr


def run(data=None,verify_reference=True,artifact_paths=None):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sector_monitoring.json').read_text())
    for journey in data['journeys']:
        f=journey['forecast_contract'];path=artifact_paths[f['model_resource']]if artifact_paths else files('creditratingsystem.datasets').joinpath(f['model_resource'])
        model=cr.load_fitted_model(path)
        if model.diagnostics['sha256']!=f['model_hash']:raise ValueError('Forecast model hash differs from frozen source')
        pd=cr.predict_pd(model,f['features'],feature_order=f['feature_order'],expected_target=f['target'],context={'known_at':'2024-03-01'}).value
        calibrated=cr.apply_calibrator(f['calibrator'],pd,target=f['target'],base_model_version='1').value[0]
        if any(abs(x['pd']-calibrated)>1e-12 or x['model_id']!=model.value['model_id'] for x in [row for cohort in journey['cohorts']for row in cohort['predictions']])or abs(calibrated-journey['reference']['cohort_pd'])>1e-12:raise ValueError('Observation forecast differs from pinned model and calibration')
    engine=runpy.run_path(str(Path(__file__).with_name('specialized_monitoring.py')))['run']
    return engine(data,verify_reference=verify_reference)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
