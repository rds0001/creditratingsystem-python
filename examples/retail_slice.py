"""Installed offline retail integration example with transparent synthetic counts.

Run with Python after installing creditratingsystem. No files are written.
This small analytic example is separate from full synthetic reference worlds.
"""
from copy import deepcopy
from importlib.resources import files
import json

import creditratingsystem as cr
from creditratingsystem.rating_system import (
    register_rating_model_version, select_effective_model, route_rating_profile,
    rate_subject_batch, replay_assessment_history,
)


def run(data=None, return_artifacts=False):
    if data is None:
        data = json.loads(files("creditratingsystem").joinpath("datasets/retail_slice.json").read_text())
    product_ids = sorted({row["product_id"] for row in data["applications"]})
    target = data["target"]
    training, calibration = data["training"], data["calibration"]
    fit = cr.fit_binary_glm(training["X"], training["y"], weights=training["weights"],
        link="logit", feature_order=["recent_arrears"], fit_intercept=True,
        target=target, training=training["metadata"], model_id="synthetic-retail-glm", model_version="1")
    predictions = cr.predict_pd(fit, calibration["X"], feature_order=["recent_arrears"], expected_target=target)
    calibrator = cr.calibrate_logistic(predictions.value, calibration["y"], weights=calibration["weights"], metadata=calibration["metadata"])
    model = deepcopy(fit.model)
    model["application_contract"] = dict(product_ids=product_ids,
        population_ids=["synthetic_new_customer"], performance_origin="first_draw_at",
        term_stage="requested", term_features={})
    model["calibration_artifact"] = calibrator.value
    registry = register_rating_model_version(None,
        dict(model=model, dependencies={"method": "native GLM"}, configuration={"fixture": "retail_slice"}),
        dict(model_id=model["model_id"], model_version="1", registered_at="2023-12-31"))
    model_hash = registry.value["entries"][0]["model_hash"]
    route = route_rating_profile(data["route_snapshot"], data["routing_rules"], "2024-02-01", "2024-02-01").value
    assert route["status"] == "ROUTED"
    assignments = cr.assign_model_role(None, registry, dict(event_id="initial", profile_id=route["profile_id"], purpose="application",
        role="champion",slot_id="champion",evaluation_reference="synthetic-analytic-calibration",reason="initial example model",
        required_feature_order=["recent_arrears"],
        model_id=model["model_id"], model_version="1", model_hash=model_hash,
        effective_from="2024-01-01", known_at="2023-12-31"))
    selected = select_effective_model(registry, assignments, route["profile_id"], "application", "2024-02-01", "2024-02-01").value
    assert selected["status"] == "SELECTED"
    snapshots = cr.build_application_snapshot(data["applications"], data["features"], required_features=["recent_arrears"]).value
    requests = [dict(row_id=s["application_id"], known_at="2024-02-01",
        model_id=model["model_id"], model_version="1", model_hash=selected["selected"]["model_hash"],
        recipe=dict(method="application_risk", version="1.0", parameters=dict(target=target, calibrator=calibrator.value)),
        input=dict(snapshot=s)) for s in snapshots]
    batch = rate_subject_batch(requests, registry, "synthetic-applications-2024", "2024-02-01").value
    assert all(r["status"] == "OK" for r in batch["rows"])
    policy = cr.compile_credit_decision_policy(data["policy"])
    decisions, observed_decisions, pds = [], [], []
    decision_artifacts=[]
    dates = dict(as_of="2024-02-01", known_at="2024-02-01")
    for row, economics in zip(batch["rows"], data["economics"]):
        residual = cr.residual_income(economics["net_income"], economics["recurring_expenses"],
            economics["existing_debt_service"], economics["new_debt_service"], period="monthly", currency="EUR")
        dsti = cr.debt_service_to_income(economics["existing_debt_service"] + economics["new_debt_service"],
            economics["net_income"], period="monthly", currency="EUR")
        capacity = cr.assess_affordability(dict(residual=residual.value["residual"], dsti=dsti.value["ratio"]),
            data["affordability_profile"], {"income_verified": True})
        decision = cr.apply_credit_decision_policy(policy,
            dict(row["value"], freshness="CURRENT", **dates),
            dict(status=capacity.value["assessment"], **dates),
            dict(status=economics["eligibility"], **dates),
            dict(status=economics["operations"], **dates), **dates).value
        decision_artifacts.append(decision)
        decisions.append(decision["action"])
        pds.append(row["value"]["pd"])
        # Local recommendation becomes explicitly simulated historical decision
        # data here. Real decisions require the bank's separate process.
        observed = dict(decision_id="decision_"+row["row_id"], application_id=row["row_id"],
            status={"APPROVE":"approved", "DECLINE":"rejected", "REFER":"pending"}[decision["action"]],
            decided_at="2024-02-01", known_at="2024-02-01")
        if row["row_id"] == "A":
            observed.update(accepted_at="2024-02-02", accepted_known_at="2024-02-02")
        observed_decisions.append(observed)
    assert decisions == data["expected_actions"]
    assert all(abs(p - (.12 if i == 1 else .03)) < 1e-8 for i,p in enumerate(pds))
    outcomes = cr.build_application_targets(data["applications"], observed_decisions,
        data["booked"], data["defaults"], [], 365, "2025-03-01").value
    assert [r["target"] for r in outcomes] == data["expected_targets"]
    assert outcomes[-1]["selection_status"] == "approved_not_taken_up"
    # A separately trained account model has its own target and evidence.
    behavior = data["behavioral"]
    bt = behavior["training"]
    bfit = cr.fit_binary_glm(bt["X"], bt["y"], weights=bt["weights"], link="logit",
        feature_order=["dpd"], fit_intercept=True, target=behavior["target"],
        training=bt["metadata"], model_id="synthetic-behavioral-glm", model_version="1")
    bmodel = deepcopy(bfit.model)
    bmodel["behavioral_contract"] = dict(risk_unit="account",event_policy="first_default",
        product_ids=product_ids,allowed_states=["ACTIVE"],maximum_source_age_days=31)
    registry = register_rating_model_version(registry,
        dict(model=bmodel,dependencies={"method":"native GLM"},configuration={"fixture":"behavioral"}),
        dict(model_id=bmodel["model_id"],model_version="1",registered_at="2023-12-31"))
    bhash = next(e["model_hash"] for e in registry.value["entries"] if e["model_id"]==bmodel["model_id"])
    panel = cr.build_account_month_panel(behavior["accounts"],behavior["statements"],behavior["events"],
        observation_dates=behavior["observation_dates"]).value["accounts"]
    assert [s["at_risk"] for s in panel] == [True,True,False]
    breqs = [dict(row_id="behavioral-"+s["as_of"],known_at=s["known_at"],model_id=bmodel["model_id"],
        model_version="1",model_hash=bhash,recipe=dict(method="behavioral_risk",version="1.0",parameters=dict(target=behavior["target"])),
        input=dict(snapshot=s)) for s in panel[:2]]
    bbatch = rate_subject_batch(breqs,registry,"behavioral-2024","2024-08-31").value
    assert all(r["status"]=="OK" for r in bbatch["rows"])
    bpds = [r["value"]["pd"] for r in bbatch["rows"]]
    assert abs(bpds[0]-.01)<1e-8 and abs(bpds[1]-.1)<1e-8
    blabels = cr.build_behavioral_targets(panel,behavior["events"],horizon_months=12,target_event="DEFAULT",
        competing_events=["CLOSED"],exit_treatment="censor",evaluation_cutoff="2025-03-01",coverage=behavior["coverage"]).value
    assert [r["label"] for r in blabels] == [1,1,None]
    # The first account forecast's twelve-month anniversary is 365 days away
    # (2024-02-29 to 2025-02-28); declare this dated monitoring translation.
    monitoring_target = dict(event="default",default_definition="synthetic-first-default",
        unit="account",horizon_days=365,probability_measure="PHYSICAL",default_grade="D")
    forecasts = [dict(forecast_id="behavioral-A-Feb",case_id="A-Feb",entity_id="facility_A",
        cluster_id="customer_A",cohort="2024-02",segment=product_ids[0],model_id=bmodel["model_id"],
        model_version="1",forecast_at="2024-02-29",known_at="2024-02-29",pd=bpds[0],
        target=monitoring_target,prediction_origin="OUT_OF_TIME",origin_grade="P",override=False)]
    mcohort = cr.build_monitoring_cohorts(forecasts,
        [dict(event_id="default_A",entity_id="facility_A",event="default",event_at="2024-09-01",known_at="2024-09-02",default_definition="synthetic-first-default")],
        [dict(entity_id="facility_A",complete_through="2025-03-01",known_at="2025-03-01")],
        target_contract=monitoring_target,evaluation_cutoff="2025-03-01")
    monitoring = cr.monitor_rating_cohort(mcohort,group_by=["segment"],reference_window=dict(start="2024-02-01",end="2024-02-29"),
        minimum_count=30,uncertainty="none").value[0]
    assert monitoring["status"] == "INSUFFICIENT_INFORMATION"
    assert monitoring["metrics"]["n"] == 1
    collections = cr.build_collections_episodes(**data["collections"],evaluation_cutoff="2025-03-01")
    cure = cr.build_cure_targets(collections,horizon_days=365,redefault_horizon_days=365,evaluation_cutoff="2025-03-01").value[0]
    assert collections.value["episodes"][0]["status"] == "OPEN"
    assert cure["cure_label"] is None and cure["ever_default"]
    # A later model registration/promotion does not replace historical bindings.
    later_model = deepcopy(model)
    later_model["model_version"] = "2"
    later_model.pop("calibration_artifact")
    later_registry = register_rating_model_version(registry,
        dict(model=later_model,dependencies={"method":"native GLM"},configuration={"fixture":"later-synthetic-model"}),
        dict(model_id=model["model_id"],model_version="2",registered_at="2025-01-01"))
    later_hash = next(e["model_hash"] for e in later_registry.value["entries"] if e["model_version"]=="2")
    later_assignments = cr.assign_model_role(assignments,later_registry,
        dict(event_id="later-model",profile_id=route["profile_id"],purpose="application",role="champion",slot_id="champion",
             model_id=model["model_id"],model_version="2",model_hash=later_hash,effective_from="2025-02-01",known_at="2025-01-01",
             evaluation_reference="synthetic-administration-only",reason="illustrate historical replay",required_feature_order=["recent_arrears"]))
    assert select_effective_model(later_registry,later_assignments,route["profile_id"],"application","2025-02-01","2025-02-01").value["selected"]["model_version"]=="2"
    assert select_effective_model(later_registry,later_assignments,route["profile_id"],"application","2024-02-01","2024-02-01").value["selected"]["model_version"]=="1"
    replay = replay_assessment_history(requests, later_registry, batch["rows"], "2024-02-01").value
    assert replay["status"] == "REPRODUCED"
    breplay = replay_assessment_history(breqs,later_registry,bbatch["rows"],"2024-08-31").value
    assert breplay["status"] == "REPRODUCED"
    summary = dict(data_origin="SYNTHETIC", actions=decisions, calibrated_pd=pds,
        labels=[r["target"] for r in outcomes], replay=replay["status"],
        behavioral_pd=bpds,behavioral_labels=[r["label"] for r in blabels],behavioral_replay=breplay["status"],
        monitoring_status=monitoring["status"],monitoring_observed_rate=monitoring["metrics"]["observed_rate"],
        collections_status=collections.value["episodes"][0]["status"],cure_label=cure["cure_label"],
        note="One funded observed outcome; no empirical performance claim or hypothetical reject labels.")

    if return_artifacts:
        return dict(summary=summary, model=model, behavioral_model=bmodel, calibrator=calibrator.value,
            registry=later_registry.value, decision_artifacts=decision_artifacts, requests=requests, batch=batch, behavioral_requests=breqs,
            behavioral_batch=bbatch, policy=policy.value, application_targets=outcomes, account_panel=panel,
            behavioral_targets=blabels, observed_decisions=observed_decisions, monitoring=monitoring)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
