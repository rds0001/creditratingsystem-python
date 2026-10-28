from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import importlib.util,json
import pytest
import creditratingsystem as cr
p=Path(__file__).resolve().parents[1]/'examples'/'retail_owner_journey.py';s=importlib.util.spec_from_file_location('owner_example',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
@lru_cache(maxsize=1)
def full():return m.run(return_artifacts=True)


def test_same_principal_claim_and_independent_financial_oracles():
    out=full();life=out['lifecycle'];summary=out['presentation']['lifecycle'];assert summary['actual_default_ead']==1000 and summary['realized_principal_loss']==400
    assert summary['predicted_lgd']==.4 and summary['portfolio_expected_loss']==400
    assert summary['paired_brier_difference']==-.109375 and summary['override_rate']==0
    assert summary['prior_event_survival']==[1,.5,.5] and summary['first30day_expected_loss']==250
    assert life['account_id']in [r['facility_id']for r in out['application_chain']['products']['RETAIL_SMALL_BUSINESS_FINANCE']['targets']if r['funded']]
    assert summary['terminal_state']['ever_default'] and summary['terminal_state']['remaining_claim']==0 and summary['replay_status']=='PASS'
    assert out['business'][0]['features']['owner_draw_business_outflows']==400
    assert out['business'][0]['features']['cash_after_business_debt']==800


def test_replay_state_and_cash_ledger_mutations_rejected():
    out=full();args=deepcopy(out['lifecycle']['replay_inputs']);args['replayed_checkpoints'][0]['state']['ever_default']=False
    assert cr.replay_consistency(**args).value['decision']['status']=='BLOCKED'
    data=json.loads((Path(__file__).resolve().parents[1]/'src/creditratingsystem/datasets/retail_owner_journey.json').read_text());case=data['products']['RETAIL_SMALL_BUSINESS_FINANCE'];case['lifecycle']['resolution']['principal_written_off']=399
    p=Path(__file__).resolve().parents[1]/'examples'/'retail_owner_lifecycle.py';s=importlib.util.spec_from_file_location('owner_life',p);life=importlib.util.module_from_spec(s);s.loader.exec_module(life)
    with pytest.raises(ValueError,match='settlement'):life.run(case,out['application_chain']['products']['RETAIL_SMALL_BUSINESS_FINANCE'],data['evaluation_cutoff'])
