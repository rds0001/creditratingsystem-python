"""Separate contractual-horizon portfolios, conditional exposures and observed losses."""
from importlib.resources import files
from pathlib import Path
import json,runpy

def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_commitment_portfolio.json').read_text())
    engine=runpy.run_path(str(Path(__file__).with_name('wholesale_sector_portfolio.py')))['run']
    return dict(data_origin='SYNTHETIC',scope=data['scope'],portfolios=[engine(p,verify_reference=verify_reference)for p in data['portfolios']])

if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
