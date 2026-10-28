"""Grace, capitalizing deferral, income-contingent payments and limited guarantee."""
from copy import deepcopy
from importlib.resources import files
import math
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files("creditratingsystem.datasets").joinpath("education_variants.json").read_text())
    rule=data["income_rule"]
    if not 0<=rule["excess_rate"]<=1 or not math.isfinite(rule["monthly_threshold"]) or rule["monthly_threshold"]<0 or any(not math.isfinite(x) for x in rule["income"]):raise ValueError("Invalid supplied excess-income rule")
    contingent=deepcopy(data["contingent"])
    contingent["candidate"]["payments"]=[max(x-rule["monthly_threshold"],0)*rule["excess_rate"]for x in rule["income"]]
    plans={name:cr.simulate_repayment_arrangement(**a).value for name,a in dict(grace_period=data["grace"],deferred_payment=data["deferred"],income_contingent=contingent).items()}
    due=cr.compute_days_past_due(**data["no_due"]).value
    allocation=cr.allocate_guarantee_coverage(**data["guarantee_allocation"])
    protected=cr.apply_guarantee_risk(**data["guarantee_loss"],coverage_allocation=allocation).value
    ineffective=deepcopy(data["guarantee_allocation"])
    ineffective["guarantees"][0]["legally_effective"]=False
    unprotected=cr.apply_guarantee_risk(**data["guarantee_loss"],coverage_allocation=cr.allocate_guarantee_coverage(**ineffective)).value
    joint=cr.assemble_joint_application(**data["joint_application"]).value
    if verify_reference:
        assert due["dpd"]==0 and due["unpaid_due_amount"]==0
        assert plans["grace_period"]["candidate"]["remaining_claim"]==12000
        assert math.isclose(plans["deferred_payment"]["candidate"]["remaining_claim"],12241.2,abs_tol=1e-8)
        assert [row["scheduled_payment"]for row in plans["income_contingent"]["candidate"]["schedule"]]==[0,100]
        assert plans["income_contingent"]["candidate"]["remaining_claim"]==11900
        assert protected["covered_amount"]==3000 and protected["pd"]==.1
        assert math.isclose(protected["expected_loss_after"],960,abs_tol=1e-9)
        assert unprotected["expected_loss_after"]==1200
        assert joint["monthly_income"]==1500
    return dict(data_origin=data["data_origin"],product=data["product"],income_rule=rule,plans=plans,grace_delinquency=due,guarantee=protected,ineffective_guarantee=unprotected,joint_application=joint)


if __name__=="__main__":print(json.dumps(run(),allow_nan=False))
