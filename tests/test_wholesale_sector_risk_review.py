"""Real sector fits, independent financial algebra and candidate ordering."""
from pathlib import Path
import runpy,json,copy,math


def test_wholesale_sector_risk_review():
    root=Path(__file__).parents[1];run=runpy.run_path(str(root/'examples/wholesale_sector_risk_review.py'))['run']
    data=json.loads((root/'src/creditratingsystem/datasets/wholesale_sector_risk_review.json').read_text());result=run(data)
    for row in result['open_workout']['journeys']:
        scale=row['reference']['positive_ratio_scale']
        assert math.isclose(row['expected_final_lgd'],1-(1+scale*math.exp(.025))/3,abs_tol=1e-10)
        assert row['known_discounted_net_cashflows']==0
    changed=copy.deepcopy(data);changed['review']['journeys']=changed['review']['journeys'][:1];changed['open_workout']['journeys']=[]
    changed['review']['journeys'][0]['treatment']['treatment_candidates'].reverse()
    shuffled=run(changed)
    assert shuffled['review']['journeys'][0]['treatment']==result['review']['journeys'][0]['treatment']
