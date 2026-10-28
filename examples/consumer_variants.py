"""Consumer evidence, confirmed consolidation and dated early/full repayment."""
from importlib.resources import files
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files("creditratingsystem.datasets").joinpath("consumer_variants.json").read_text())
    snapshots=cr.build_application_snapshot(**data["snapshots"]).value
    c=data["consolidation"];consolidation=[]
    for date in c["cutoffs"]:
        graph=cr.build_customer_household_graph(c["entities"],c["relationships"],as_of=date,information_cutoff=date)
        result=cr.aggregate_household_obligations(c["obligations"],graph,c["coverage"],currency="EUR",period="month",guarantee_budget_policy="exclude")
        consolidation.append(dict(as_of=date,value=result.value))
    prepayment=cr.exposure_schedule(**data["prepayment"]).value
    labels=cr.build_application_targets(**data["labels"]).value
    if verify_reference:
        assert snapshots[0]["missing_features"]==["internal_arrears"]
        assert "internal_arrears" not in snapshots[0]["features"]
        assert snapshots[1]["features"]["internal_arrears"]==0 and not snapshots[1]["missing_features"]
        assert [r["value"]["households"][0]["total_budget_debt_service"]for r in consolidation]==[500,850,350]
        assert [r["conditional_ead"]for r in prepayment["schedule"]]==[0,0]
        assert [r["target"]for r in labels]==[1,1,None]
        assert [r["event_id"]for r in labels]==["first_payment_default","later_default",None]
    return dict(data_origin=data["data_origin"],product=data["product"],snapshots=snapshots,consolidation=consolidation,prepayment=prepayment,labels=labels)


if __name__=="__main__":print(json.dumps(run(),allow_nan=False))
