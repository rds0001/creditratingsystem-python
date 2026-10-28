"""Sector-specific treatment and open claims using shared fitted-method engines."""
from importlib.resources import files
from pathlib import Path
import json,runpy


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_sector_risk_review.json').read_text())
    result={}
    for key,example in [('review','specialized_risk_review.py'),('open_workout','specialized_open_workout.py')]:
        engine=runpy.run_path(str(Path(__file__).with_name(example)))['run'];result[key]=engine(data[key],verify_reference=verify_reference)
    return dict(data_origin='SYNTHETIC',**result)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
