"""Seven mortgage variants using explicit, replaceable local input contracts."""
from copy import deepcopy
from importlib.resources import files
import json
import math
import creditratingsystem as cr


def run(data=None, verify_reference=True):
    """Use packaged analytical inputs or pass the same schema with your own data.

    Reference verification applies only to the supplied synthetic assumptions.
    This example evaluates product components, not an empirical mortgage PD model.
    """
    if data is None:
        data = json.loads(files("creditratingsystem.datasets").joinpath("mortgage_variants.json").read_text())
    data = deepcopy(data)
    debt = {name: cr.build_debt_schedule(**args).value for name, args in data["debt_cases"].items()}
    shocked = deepcopy(data["debt_cases"]["floating_rate"])
    shocked["base_rate_fixings"][1]["base_rate"] += data["rate_reset_shock"]
    debt["floating_reset"] = cr.build_debt_schedule(**shocked).value
    operating = {name: cr.simulate_operating_cashflows(**args).value for name, args in data["operating_cases"].items()}
    joint = cr.assemble_joint_application(**data["joint_application"]).value
    collateral = cr.combined_loan_to_value(**data["collateral"]).value
    recovery = cr.simulate_recovery_waterfall(**data["recovery"]).value
    out = dict(data_origin=data["data_origin"], version=data["version"], product=data["product"], debt=debt, operating=operating, joint_application=joint, collateral=collateral, recovery=recovery)
    if verify_reference:
        def same(actual, expected):
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10):
                raise AssertionError((actual, expected))
        for name, expected in dict(fixed_rate=[53000,51500], floating_rate=[53000,51500], floating_reset=[53000,52500], interest_only=[3000,103000], partially_amortizing_balloon=[23000,82400]).items():
            for row, value in zip(debt[name]["schedule"], expected): same(row["net_payment_due"], value)
        same(debt["interest_only"]["schedule"][0]["closing_principal"],100000)
        same(debt["partially_amortizing_balloon"]["schedule"][0]["closing_principal"],80000)
        for name,expected in dict(owner_occupied_base=0,owner_occupied_vacancy=0,buy_to_let_base=800,buy_to_let_vacancy=600).items(): same(operating[name]["periods"][0]["cfads"],expected)
        same(joint["monthly_income"],5000);same(joint["monthly_debt_service"],900)
        same(collateral["cltv"],.8)
        claim = next(row for row in recovery["claim_results"] if row["claim_id"]=="mortgage")
        same(claim["economic_loss"],37000);same(claim["lgd"],.37)
    return out


if __name__ == "__main__":
    print(json.dumps(run(), allow_nan=False))
