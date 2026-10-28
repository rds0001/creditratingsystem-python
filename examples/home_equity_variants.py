"""Home equity draw/repayment periods and first/second-lien cash allocation."""
from copy import deepcopy
from importlib.resources import files
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files("creditratingsystem.datasets").joinpath("home_equity_variants.json").read_text())
    exposure=cr.exposure_schedule(**data["exposure"]).value
    unavailable=deepcopy(data["exposure"])
    unavailable["events"].append(data["forbidden_draw"])
    try:
        cr.exposure_schedule(**unavailable)
        restriction="DRAW_SUCCEEDED"
    except cr.CreditRiskError as error:
        restriction=error.code
    collateral={name:cr.combined_loan_to_value(**a).value for name,a in data["collateral"].items()}
    recovery={name:cr.simulate_recovery_waterfall(**a).value for name,a in data["recovery"].items()}
    if verify_reference:
        assert [p["conditional_ead"]for p in exposure["schedule"]]==[62400,50000]
        assert [p["undrawn_commitment"]for p in exposure["schedule"]]==[0,0]
        assert restriction=="EXPOSURE_LIMIT"
        assert all(v["cltv"]==.9 for v in collateral.values())
        for name,expected in [("first_lien",0),("second_lien",.2)]:
            claim=next(c for c in recovery[name]["claim_results"]if c["claim_id"]=="home_equity")
            assert claim["lgd"]==expected
    return dict(data_origin=data["data_origin"],product=data["product"],exposure=exposure,repayment_draw_restriction=restriction,collateral=collateral,recovery=recovery)


if __name__=="__main__":print(json.dumps(run(),allow_nan=False))
