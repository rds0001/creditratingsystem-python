"""Independent exclusive-event and discounted exchange oracles."""
from pathlib import Path
import runpy,json,copy,math
import pytest


def test_wholesale_sovereign():
    root=Path(__file__).parents[1];run=runpy.run_path(str(root/'examples/wholesale_sovereign.py'))['run']
    data=json.loads((root/'src/creditratingsystem/datasets/wholesale_sovereign.json').read_text())
    r=run(data)['cases']
    assert [x['reported_pretransfer_pd'] for x in r]==[.04,.08,.12]
    assert all(math.isclose(x['combined']['combined_pd'],p,abs_tol=1e-12)for x,p in zip(r,[.07,.14,.26]))
    assert all(math.isclose(x['loss']['expected_loss'],v,abs_tol=1e-12)for x,v in zip(r,[1.9,5.2,11.8]))
    assert r[0]['applicability']['payments'][0]['events'][0]['event_type']=='PAYMENT_FREEZE'
    assert r[2]['restructuring']['payment_delay_days']==0
    assert math.isclose(r[2]['restructuring']['economic_loss'],3700/121,abs_tol=1e-12)
    assert all(x['double_count_rejected']for x in r)
    bad=copy.deepcopy(data);bad['cases'][0]['currency']='USD'
    with pytest.raises(ValueError,match='incompatible'):run(bad,verify_reference=False)
    bad=copy.deepcopy(data);bad['cases'][2]['legal_claim']='SENIOR_SOVEREIGN_BOND'
    with pytest.raises(ValueError,match='incompatible'):run(bad,verify_reference=False)
