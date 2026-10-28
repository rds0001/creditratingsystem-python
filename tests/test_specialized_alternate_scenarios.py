"""Real local fits, conditional cash ledger and independent stored seal verification."""
from pathlib import Path
from copy import deepcopy
import json,runpy,os
import creditratingsystem as cr

def test_specialized_alternate_fits_and_cash_ledgers():
    root=Path(__file__).parents[1];data=json.loads((root/'src/creditratingsystem/datasets/specialized_alternate_scenarios.json').read_text());extended=os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS','').lower()=='true'
    if not extended:
        data['cases']=data['cases'][:1];sid=data['cases'][0]['case_id']
        data['entity_innovations']=[x for x in data['entity_innovations']if x['channel_id'].startswith(sid+'-')]
        data['parameter_snapshot']['parameters']={sid:data['parameter_snapshot']['parameters'][sid]}
    run=runpy.run_path(str(root/'examples/specialized_alternate_scenarios.py'))['run'];v=run(data,scenario_methods=None if extended else['conditional'],return_audit_artifacts=True)
    assert v['audits']['local_fits']['factor']['diagnostics']['missing_entries']==1
    assert v['audits']['reverse_solver']['feasible']
    for name,audit in v['audits']['executions'].items():
        report=v['report']['alternatives'][name]
        assert cr.content_fingerprint(audit['cube']).value==report['scenario_cube_hash']
        for row in report['transactions']:
            tx=next(t for t in audit['execution']['transactions']if t['transaction_id']==row['case_id'])
            assert cr.content_fingerprint(tx).value==row['transaction_hash']
            changed=deepcopy(tx);changed['risk']['expected_loss']+=1
            assert cr.content_fingerprint(changed).value!=row['transaction_hash']
            assert len(row['ledger_oracles'])==18
