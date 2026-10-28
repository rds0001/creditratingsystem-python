"""Separate lease credit claims, cancelled rents and retained residual value."""
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files("creditratingsystem.datasets").joinpath("lease_variants.json").read_text())
    out={}
    for name,c in data["cases"].items():
        claim=c["credit_claim"]
        if name.startswith("early_termination"):
            claim=c["termination_base"]+(c["condition_charge"]if c["customer_bears_condition"]else 0)
            if claim!=c["claim_receivable"]:raise ValueError("Termination claim differs from declared contractual settlement")
        net=c["gross_sale"]-c["realization_cost"]
        if net<0:raise ValueError("Example assumes nonnegative net realization")
        tests=[dict(scenario_id=name,entity_id="lessee",facility_id="lease",date="2023-01-01",currency="EUR",cfads=c["rent_paid"],debt_service_due=c["rent_due"],debt_outstanding=claim,asset_value=net if c["asset_proceeds_applied_to_claim"]else 0,liquidity_buffer_available=0,materiality_met=False,material_days_past_due=0,utp_indicators=[])]
        default=cr.detect_cashflow_default(tests,data["default_policy"],currency="EUR").value
        cash=cr.apply_cash_waterfall([dict(scenario_id=name,date="2023-01-01",currency="EUR",cash_available=net,reserve_target=0,amounts_due=dict(credit_claim=claim))],data["waterfall_policy"],currency="EUR").value
        row=cash["periods"][0]
        credit_loss=claim-row["payments"][0]["total_paid"]
        residual_loss=c["residual_carrying_value"]-row["residual_equity_cash"]
        total=credit_loss+residual_loss
        if not math.isclose(total,claim+c["residual_carrying_value"]-net,abs_tol=1e-10):raise AssertionError("Lease cash bridge fails")
        defaulted=default["first_default_events"][0]["defaulted"]
        credit=cr.expected_loss_from_paths(default_indicator=[[int(defaulted)]],lgd_paths=[[credit_loss/claim if claim else 0]],ead_paths=[[claim]],scenario_weights=[1],discount_factors=[1],scenario_ids=[name],time_grid=[1],currency="EUR",probability_measure="PHYSICAL").value
        # Residual market value is a distinct event, including when credit performs.
        market=cr.event_conditioned_loss([dict(state_id="no_market_loss",events=dict(residual_loss=False),probability=0,conditional_loss=0),dict(state_id="market_loss",events=dict(residual_loss=True),probability=1,conditional_loss=residual_loss)],event_names=["residual_loss"],currency="EUR",probability_measure="PHYSICAL").value
        out[name]=dict(default_assessment=default,cash_allocation=cash,credit_loss=credit_loss,residual_loss=residual_loss,total_economic_loss=total,credit_path=credit,residual_event=market,cancelled_future_payments=c["cancelled_future_payments"],contractual_credit_claim=claim)
    if verify_reference:
        assert not out["operating_lease"]["default_assessment"]["first_default_events"][0]["defaulted"]
        assert out["operating_lease"]["credit_loss"]==0 and out["operating_lease"]["residual_loss"]==4500
        assert out["finance_lease"]["credit_loss"]==4500 and out["finance_lease"]["residual_loss"]==0
        assert out["early_termination"]["contractual_credit_claim"]==2000
        assert out["early_termination_no_condition"]["contractual_credit_claim"]==1800
        assert out["early_termination_no_condition"]["credit_loss"]==800
        assert out["early_termination"]["credit_loss"]==1000 and out["early_termination"]["total_economic_loss"]==11000
    return dict(data_origin=data["data_origin"],product=data["product"],variants=out)


if __name__=="__main__":print(json.dumps(run(),allow_nan=False))
