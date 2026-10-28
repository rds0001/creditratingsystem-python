"""Amortizing and balloon auto loans with capped future-value protection."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files("creditratingsystem.datasets").joinpath("auto_variants.json").read_text())
    schedules={name:cr.build_debt_schedule(**a).value for name,a in data["debt"].items()}
    asset=cr.simulate_asset_values(**data["asset"]).value
    gross=asset["paths"][0]["asset_value"]
    eads={name:v["schedule"][-1]["opening_principal"]+v["schedule"][-1]["interest_due"]for name,v in schedules.items()}
    recoveries={}
    for name,ead in eads.items():
        args=deepcopy(data["recovery"])
        args["claims"]=[dict(scenario_id="sale",default_event_id="default",subject_id="borrower",claim_id="auto",pool_id="car",currency="EUR",ead_at_default=ead,legal_balance=ead,priority=1,deficit_cost_share=1)]
        args["realization_paths"][0]["gross_value"]=gross
        recoveries[name]=cr.simulate_recovery_waterfall(**args).value
        if name=="balloon":
            args["realization_paths"][0]["cost"]=0
            recoveries["balloon_no_cost"]=cr.simulate_recovery_waterfall(**args).value
    debt=eads["balloon"];loss=recoveries["balloon"]["claim_results"][0]["economic_loss"]
    a=deepcopy(data["guarantee_allocation"])
    a["claims"]=[dict(claim_id="auto",entity_id="borrower",amount=debt,currency="EUR",due_at="2023-01-01",known_at="2021-01-01")]
    allocated=cr.allocate_guarantee_coverage(**a)
    net=gross-data["recovery"]["realization_paths"][0]["cost"]
    gap=max(data["net_realization_floor"]-net,0)
    covered=sum(x["covered_amount"]for x in allocated.value["allocations"])
    risk=deepcopy(data["guarantee_loss"])
    for terms in risk["loss_contract"]["states"].values():terms["loss_before_guarantee"]=loss
    denominator=min(covered,max(loss,0))
    risk["loss_contract"]["states"]["borrower_only"]["payout_fraction"]=min(gap,denominator)/denominator if denominator else 0
    protection=cr.apply_guarantee_risk(**risk,coverage_allocation=allocated).value
    a["guarantees"][0]["legally_effective"]=False
    ineffective=cr.apply_guarantee_risk(**risk,coverage_allocation=cr.allocate_guarantee_coverage(**a)).value
    states=protection["states"]
    # Supplied full joint event partition: borrower-only, joint, dealer-only, neither.
    bpd=risk["borrower_risk"]["pd"];gpd=risk["guarantor_risk"]["pd"];joint=risk["joint_default_probability"]
    paths=cr.expected_loss_from_paths(default_indicator=[[1],[1],[0],[0]],lgd_paths=[[states[0]["loss_after_guarantee"]/debt],[states[1]["loss_after_guarantee"]/debt],[0],[0]],ead_paths=[[debt],[debt],[0],[0]],scenario_weights=[bpd-joint,joint,gpd-joint,1-bpd-gpd+joint],discount_factors=[1],scenario_ids=["borrower_only","joint","dealer_only","neither"],time_grid=[2],currency="EUR",probability_measure="PHYSICAL").value
    if verify_reference:
        assert eads==dict(standard_amortizing=10500,balloon=15750)
        assert gross==10000 and loss==6750
        assert recoveries["balloon_no_cost"]["claim_results"][0]["economic_loss"]==5750
        assert math.isclose(protection["expected_loss_after"],435,abs_tol=1e-9)
        assert math.isclose(ineffective["expected_loss_after"],675,abs_tol=1e-9)
        assert math.isclose(paths["expected_loss"],435,abs_tol=1e-9)
    return dict(data_origin=data["data_origin"],product=data["product"],schedules=schedules,asset=asset,ead_before_final_payment=eads,recoveries=recoveries,guarantee=protection,ineffective_guarantee=ineffective,joint_loss_paths=paths)


if __name__=="__main__":print(json.dumps(run(),allow_nan=False))
