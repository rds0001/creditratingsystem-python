"""Dated mortgage lifecycle using synthetic analytical cohorts and cash ledgers."""
from importlib.resources import files
from pathlib import Path
import runpy
import json
import math
import creditratingsystem as cr


def run(data=None, evidence_data=None, variant_data=None, risk_data=None):
    if data is None:
        data=json.loads(files("creditratingsystem.datasets").joinpath("mortgage_lifecycle.json").read_text())
    # Shared application/behavioral mechanics; each run fits its supplied cohorts.
    retail_run=runpy.run_path(str(Path(__file__).with_name("retail_slice.py")))["run"]
    lifecycle=retail_run(data)
    evidence=runpy.run_path(str(Path(__file__).with_name("mortgage_evidence.py")))["run"](evidence_data)
    variants=runpy.run_path(str(Path(__file__).with_name("mortgage_variants.py")))["run"](variant_data)
    risk=runpy.run_path(str(Path(__file__).with_name("mortgage_risk.py")))["run"](risk_data)
    delinquency=cr.compute_days_past_due(**data["delinquency"]).value
    assert delinquency["dpd"]==30 and delinquency["past_due_amount"]==900
    arrangement=cr.simulate_repayment_arrangement(**data["arrangement"]).value
    assert arrangement["baseline"]["affordability"][0]["residual_cash"]==[-1800,-1800]
    assert arrangement["candidate"]["affordability"][0]["residual_cash"]==[200,200]
    assert not arrangement["contract_changed"] and not arrangement["default_status_changed"]
    # Period one: interest 600+189 and fees100; proposed1000 pays principal111.
    assert math.isclose(arrangement["candidate"]["schedule"][0]["closing_principal"],94389,abs_tol=1e-8)
    # Period two: interest188.778; principal repaid811.222, balance93577.778.
    assert math.isclose(arrangement["candidate"]["remaining_claim"],93577.778,abs_tol=1e-8)
    recovery=cr.discount_recoveries(**data["realized_recovery"])
    common=dict(default_date=data["realized_recovery"]["default_date"],evaluation_cutoff=data["realized_recovery"]["evaluation_cutoff"],currency="EUR")
    closed=cr.workout_lgd(data["default_ead"],recovery,episode_status="CLOSED",**common).value
    incomplete=cr.workout_lgd(data["default_ead"],recovery,episode_status="OPEN",**common).value
    assert closed["economic_loss"]==32000 and math.isclose(closed["economic_lgd"],32/95)
    assert incomplete["economic_lgd"] is None and not incomplete["complete"]
    behavior=data["behavioral"]
    panel=cr.build_account_month_panel(behavior["accounts"],behavior["statements"],behavior["events"],observation_dates=behavior["observation_dates"]).value["accounts"]
    panel[1]["state"]=evidence["delinquency_state"]["state"]
    vintage=cr.build_vintage_performance(behavior["accounts"],panel,behavior["events"],months_on_book=[0,6,7],evaluation_cutoff="2025-03-01",coverage=behavior["coverage"],default_event="DEFAULT",delinquency_states=["DELINQUENT"],exit_events=["CLOSED"]).value
    assert [v["cumulative_default_rate"] for v in vintage]==[0,0,1]
    assert [v["delinquency_stock_rate"] for v in vintage]==[0,1,0]
    exit_panel=cr.build_account_month_panel(**data["closed_account"]).value
    assert not exit_panel["accounts"][0]["at_risk"]
    return dict(product="residential_mortgage",data_origin="SYNTHETIC",application_and_monitoring=lifecycle,evidence=evidence,variants=variants,risk=risk,delinquency=delinquency,proposed_restructuring=arrangement,realized_recovery=recovery.value,closed_workout=closed,incomplete_workout=incomplete,exit_panel=exit_panel,vintage=vintage)


if __name__=="__main__":
    print(json.dumps(run(),allow_nan=False))
