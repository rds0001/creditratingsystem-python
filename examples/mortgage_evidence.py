"""Mortgage evidence reconciliation and dated payment history."""
from copy import deepcopy
from importlib.resources import files
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files("creditratingsystem.datasets").joinpath("mortgage_evidence.json").read_text())
    data=deepcopy(data)
    graph=cr.build_customer_household_graph(**data["graph"])
    household=cr.aggregate_household_obligations(**data["household"],household_graph=graph).value
    bureau=cr.build_bureau_features(**data["bureau"]).value
    income=cr.reconcile_income_evidence(**data["income"]).value
    data["normalized_income"]["components"][0]["amount"]=income["total_selected_income"]
    normalized=cr.normalize_income_period(**data["normalized_income"]).value
    expenses=cr.estimate_recurring_expenses(**data["expenses"]).value
    payment=cr.build_payment_history_features(**data["payment_history"]).value
    state=cr.assign_delinquency_state(**data["delinquency_state"]).value
    if verify_reference:
        assert household["households"][0]["total_budget_debt_service"]==900
        assert [x["legal_debt_stock"] for x in household["person_legal_liabilities"]]==[100000,100000]
        assert bureau["features"]["total_open_balance"]==5000
        assert income["total_selected_income"]==3000 and income["components"][0]["discrepancy_flag"]
        assert normalized["total_income"]==5000
        assert expenses["recurring_expenses"]==2300 and expenses["debt_service"]==900
        assert payment["features"]["max_dpd"]==30
        assert state["state"]=="DELINQUENT"
    return dict(data_origin=data["data_origin"],household=household,bureau=bureau,income=income,normalized_income=normalized,expenses=expenses,payment_history=payment,delinquency_state=state)


if __name__=="__main__":print(json.dumps(run(),allow_nan=False))
