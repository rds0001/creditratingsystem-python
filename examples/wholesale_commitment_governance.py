"""Contract-dated revolving and leveraged credit histories with frozen inference."""
from importlib.resources import files
from pathlib import Path
import json,runpy

def run(data=None,verify_reference=True,artifact_paths=None,return_audit_artifacts=False):
    if data is None:data=json.loads(files('creditratingsystem.datasets').joinpath('wholesale_commitment_governance.json').read_text())
    engine=runpy.run_path(str(Path(__file__).with_name('wholesale_sector_governance.py')))['run']
    return engine(data,verify_reference=verify_reference,artifact_paths=artifact_paths,return_audit_artifacts=return_audit_artifacts)

if __name__=='__main__':print(json.dumps(run(),allow_nan=False))
