from copy import deepcopy
from pathlib import Path
import importlib.util,json
import pytest
from creditratingsystem import CreditRiskError,content_fingerprint
BINDINGS={k:content_fingerprint(dict(role=k,scope="INDEPENDENT_COMPONENT_TEST_ONLY")).value for k in ["model","calibrator","policy"]}
s=importlib.util.spec_from_file_location('phases',Path(__file__).parents[1]/'examples/retail_application_phases.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def inputs():
 p=Path(__file__).parents[1]/'src/creditratingsystem/datasets'
 return json.loads((p/'retail_application_journey.json').read_text()),json.loads((p/'retail_application_phases.json').read_text())
def test_real_claims_grace_and_exit_do_not_fabricate_cure():
 source,data=inputs()
 for product,x in data['products'].items():
  r=m.assess(source['products'][product],x,data['evaluation_cutoff'],BINDINGS);a=r['summary']
  assert a['investment_loss']==400 and a['realized_book_mean_loss']==200 and a['remaining_claim']==0 and a['ever_default']
  assert a['dpd']==(0 if product=='RETAIL_EDUCATION_LOAN'else 5)
  assert a['claim_waiver']==(90 if product=='RETAIL_AUTO_LEASE'else 400)
  assert r['episodes_at_account_exit']['episodes'][0]['status']=='open'
  assert r['recovery_episodes']['episodes'][0]['closed_at']=='2025-01-30'
  assert not r['proposal']['contract_changed'] and len(a['phases'])==8

def test_invalid_cash_closure_and_lease_double_count_rejected():
 source,data=inputs();p='RETAIL_UNSECURED_INSTALLMENT';x=deepcopy(data['products'][p]);x['settlement_date']='2025-01-25'
 with pytest.raises(ValueError):m.assess(source['products'][p],x,data['evaluation_cutoff'],BINDINGS)
 p='RETAIL_AUTO_LEASE';x=deepcopy(data['products'][p]);x.update(claim_cash=600,claim_amount=690)
 with pytest.raises(ValueError):m.assess(source['products'][p],x,data['evaluation_cutoff'],BINDINGS)
 p='RETAIL_UNSECURED_INSTALLMENT';x=deepcopy(data['products'][p]);x['proposal_status']='ACCEPTED'
 with pytest.raises(ValueError):m.assess(source['products'][p],x,data['evaluation_cutoff'],BINDINGS)
