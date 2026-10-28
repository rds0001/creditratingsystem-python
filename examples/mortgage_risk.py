"""Conditional mortgage exposure, locally fitted severity and joint stress."""
from importlib.resources import files
import math
import json
import creditratingsystem as cr


def run(data=None,verify_reference=True):
    if data is None:data=json.loads(files("creditratingsystem.datasets").joinpath("mortgage_risk.json").read_text())
    poolfit=cr.fit_retail_risk_pools(**data["pool_fit"])
    pool=cr.assign_retail_risk_pool(**data["pool_assignment"],fitted_model=poolfit).value
    exposure=cr.exposure_schedule(**data["exposure"])
    ead=cr.predict_ead(exposure,None,feature_order=[],expected_target=None,reference_dates=[data["exposure"]["contract"]["as_of"]]*2,default_dates=data["exposure"]["period_end_dates"],currency="EUR")
    fit=cr.fit_fractional_lgd(**data["lgd_fit"])
    severity=cr.predict_lgd(fit,[[]],feature_order=[],expected_target=data["lgd_fit"]["target"],default_condition="given_default")
    pd=cr.marginal_pd_from_conditional(data["conditional_pd"]).value.tolist()
    # Exposure is deterministic in this illustrative scenario, so the product
    # equals conditional joint loss; no general LGD/EAD independence is inferred.
    loss=[x*severity.value["mean_lgd"][0] for x in ead.value["conditional_ead"]]
    lifetime=cr.lifetime_expected_loss(pd,loss,data["loss_discount_factors"],[1,2],currency="EUR",probability_measure="PHYSICAL").value
    stress=cr.stress_affordability(**data["stress"]).value
    if verify_reference:
        assert [p["statistics"]["means"]["pd"] for p in poolfit.model["parameters"]["segments"][0]["pools"]]==[.25,.5]
        assert pool["assignments"][0]["score_bin"]==0
        assert ead.value["conditional_ead"]==[95000,90000]
        assert math.isclose(severity.value["mean_lgd"][0],.3,abs_tol=1e-9)
        assert math.isclose(lifetime["lifetime_expected_loss"],1814.094,abs_tol=1e-6)
        for row,expected in zip(stress["periods"],[250,100000*.05/12,100000+100000*.05/12]):assert math.isclose(row["debt_service"],expected,abs_tol=1e-8)
        assert stress["first_liquidity_exhaustion"]=="2024-04-30"
    return dict(data_origin=data["data_origin"],pool_assignment=pool,exposure=exposure.value,ead=ead.value,conditional_lgd=severity.value,marginal_pd=pd,lifetime_loss=lifetime,stress=stress)


if __name__=="__main__":print(json.dumps(run(),allow_nan=False))
