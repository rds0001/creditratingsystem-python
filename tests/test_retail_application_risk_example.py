from copy import deepcopy
from pathlib import Path
import importlib.util,json,os
import pytest
s=importlib.util.spec_from_file_location('risk_example',Path(__file__).parents[1]/'examples/retail_application_risk.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def inputs():
 p=Path(__file__).parents[1]/'src/creditratingsystem/datasets'
 return json.loads((p/'retail_application_journey.json').read_text()),json.loads((p/'retail_application_risk.json').read_text())
def test_prior_workout_mean_and_same_application_30day_loss():
 r=m.run(product_ids=['RETAIL_UNSECURED_INSTALLMENT'],return_artifacts=True);v=r['products']['RETAIL_UNSECURED_INSTALLMENT'];raw=r['raw_artifacts']['RETAIL_UNSECURED_INSTALLMENT']
 assert v['application_id']=='RETAIL_UNSECURED_INSTALLMENT-DEFAULT_THEN_EXIT'
 assert v['conditional_lgd']==pytest.approx(.4) and v['thirty_day_expected_loss']==300
 assert raw['loss_result']['lifetime_expected_loss']==pytest.approx(.75*1000*.4,abs=1e-8)
 assert raw['lgd_model']['training']['known_at']<'2024-12-29'
def test_wrong_identity_future_assumption_and_lease_debt_rejected():
 source,data=inputs();p='RETAIL_UNSECURED_INSTALLMENT';r=data['products'][p]
 for key,value in [('account_id','unrelated'),('assumptions_known_at','2025-01-01')]:
  bad=deepcopy(r);bad[key]=value
  with pytest.raises(ValueError):m.evaluate(p,source['products'][p],bad,source['target'])
 p='RETAIL_AUTO_LEASE';bad=deepcopy(data['products'][p]);bad['conditional_ead']=1000
 with pytest.raises(ValueError,match='Claim differs'):m.evaluate(p,source['products'][p],bad,source['target'])
@pytest.mark.skipif(os.getenv('CREDITRATINGSYSTEM_EXTENDED_TESTS')!='true',reason='Six product native/installed acceptance is explicit extended tier')
def test_all_six_forecast_products_and_independent_oracles():
 r=m.run();assert len(r['products'])==6
 assert r['products']['RETAIL_BNPL']['thirty_day_expected_loss']==281.25
 assert r['products']['RETAIL_MICRO_LOAN']['micro']['maximum_feasible_amount']==1000
 assert r['products']['RETAIL_AUTO_LEASE']['conditional_ead']==90
