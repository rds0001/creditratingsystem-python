"""Independent restricted-cash, service-demand and structural-branch oracles."""
from pathlib import Path
import runpy,json,copy
import pytest


def test_wholesale_public_service():
    root=Path(__file__).parents[1];run=runpy.run_path(str(root/'examples/wholesale_public_service.py'))['run']
    data=json.loads((root/'src/creditratingsystem/datasets/wholesale_public_service.json').read_text());r=run(data)['cases']
    assert [x['cash_oracles'][0]['restricted_remaining']for x in r]==[10,10,0]
    assert [x['cash_oracles'][0]['debt_service_gap']for x in r]==[5,5,0]
    assert [x['cash_oracles'][1]['debt_service_gap']for x in r]==[25,15,15]
    assert r[0]['market']is None and r[1]['market']is None
    assert r[2]['market']['asset_value']==100 and r[2]['market']['asset_volatility']==.2
    assert r[2]['market']['physical_pd']!=r[2]['market']['empirical_pd']==.2
    assert r[1]['supported']['pd']==r[1]['reported_fundamental_pd']
    for key,value in [('restricted_purpose','GENERAL_DEBT_SERVICE'),('cashflow_type','REGULATED_TARIFF')]:
        bad=copy.deepcopy(data);bad['cases'][0][key]=value
        with pytest.raises(ValueError,match='incompatible'):run(bad,verify_reference=False)
