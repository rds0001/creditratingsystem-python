"""Independent accounting, information-time, cohort and supplied-draw oracles."""
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
import pytest
import creditratingsystem
from creditratingsystem.core import CreditRiskError
from creditratingsystem.synthetic import (
    generate_macro_reference_world, generate_obligor_panel,
    generate_facility_workouts, generate_retail_reference_world,
    generate_asset_pool_reference_world, generate_system_event_reference_world,
    generate_challenge_cases, validate_synthetic_world,
)


REFERENCE = Path(creditratingsystem.__file__).parent/"datasets"/"synthetic_tiny"
STANDARD = Path(creditratingsystem.__file__).parent/"datasets"/"synthetic_standard"


def rows(world, layer, table):
    path=Path(world)/layer/(table+".jsonl.gz")
    return [json.loads(line) for line in gzip.open(path,"rt")] if path.exists() else []


def spec():
    return json.loads((REFERENCE/"macro/manifest.json").read_text())["specification"]


@pytest.mark.parametrize("name",["macro","obligors","workouts","retail","pools","events"])
@pytest.mark.extended
def test_bundled_tiny_passes_complete_serialized_audit(name):
    result=validate_synthetic_world(REFERENCE/name)
    assert result.provenance["data_origin"]=="SYNTHETIC"
    report=result.value
    assert report["passed"],report["issues"]
    assert report["rows_checked"]>0


@pytest.mark.extended
def test_standard_is_full_offline_world_with_actual_draws():
    manifest=json.loads((STANDARD/"macro/manifest.json").read_text())
    assert manifest["binary_resources"]["scenario_cube"]["shape"]==[16384,60,8]
    assert len(rows(STANDARD/"obligors","observable_inputs","obligors"))==3000
    assert len({x["group_id"] for x in rows(STANDARD/"obligors","observable_inputs","obligors")})==750
    actual=json.loads((STANDARD/"innovations/manifest.json").read_text())
    draw=actual["streams"]["macro_paths"]
    path=STANDARD/"innovations"/draw["file"]
    assert path.stat().st_size==16384*60*10*4
    assert hashlib.sha256(path.read_bytes()).hexdigest()==draw["sha256"]
    assert all(x["period"]<10 for x in rows(STANDARD/"obligors","latent_truth","obligor_period"))


def test_first_macro_state_independent_box_muller_oracle():
    u=(np.fromfile(REFERENCE/"innovations/macro_history.u32le",dtype="<u4",count=10).astype(float)+.5)/2**32
    z0=math.sqrt(-2*math.log(u[0]))*math.cos(2*math.pi*u[1])
    z1=math.sqrt(-2*math.log(u[0]))*math.sin(2*math.pi*u[1])
    innovation=.25*z0+math.sqrt(15/16)*z1
    factor=math.sqrt(1-.85**2)*innovation
    expected_index=100*math.exp(.0015+.002*factor)
    first=rows(REFERENCE/"macro","latent_truth","macro_factors")[0]
    assert first["activity"]==pytest.approx(expected_index,rel=1e-14)
    releases=rows(REFERENCE/"macro","observable_inputs","macro_releases")
    first_revised=next(x for x in releases if x["series"]=="activity" and x["revision"]=="revised")
    assert first_revised["known_at"]=="2000-04-30"
    assert first_revised["value"]==first["activity"]


def test_balance_sheet_option_identity_and_absorbing_default():
    world=REFERENCE/"obligors"
    truth={r["snapshot_id"]:r for r in rows(world,"latent_truth","obligor_period")}
    statements=rows(world,"observable_inputs","financial_statements")
    from statistics import NormalDist
    for r in statements:
        assert r["assets"]==pytest.approx(r["equity"]+r["debt"])
        assert r["cfads"]==pytest.approx(r["ebitda"]-r["tax"]-r["capex"])
        sigma=truth[r["snapshot_id"]]["asset_volatility"];a=r["assets"];d=r["debt"];rate=r["risk_free_rate"]
        d1=(math.log(a/d)+rate+.5*sigma*sigma)/sigma
        expected=a*NormalDist().cdf(d1)-d*math.exp(-rate)*NormalDist().cdf(d1-sigma)
        assert r["market_equity"]==pytest.approx(expected,rel=1e-12)
    for event in rows(world,"observed_outcomes","default_events"):
        assert all(r["known_at"]<event["event_at"] for r in statements if r["obligor_id"]==event["obligor_id"])
    zero=[x for x in truth.values() if x["dgp_stratum"]=="zero_default"]
    assert zero and all(x["true_pd"]==0 and x["potential_default_at"] is None for x in zero)


def test_group_and_forward_time_split_and_no_hidden_truth_in_features():
    for name,table in (("obligors","financial_statements"),("retail","applications")):
        records=rows(REFERENCE/name,"observable_inputs",table)
        for r in records:
            assert r["split"]==(r["entity_split"] if r["entity_split"]==r["time_split"] else "excluded")
            assert not {"true_pd","potential_loan_outcome","latent_frailty"}.intersection(r)
    families={}
    for r in rows(REFERENCE/"retail","observable_inputs","applications"):
        families.setdefault(r["household_id"],set()).add(r["entity_split"])
    assert all(len(v)==1 for v in families.values())


@pytest.mark.extended
def test_workout_discounting_and_explicit_raw_lgd_boundaries():
    world=STANDARD/"workouts"
    flows=rows(world,"latent_truth","future_recoveries")
    truth={x["facility_id"]:x for x in rows(world,"latent_truth","workout_parameters")}
    from datetime import date
    workouts=rows(world,"observed_outcomes","workouts")
    assert workouts
    for r in workouts:
        pv=sum((x["recovery_amount"]-x["cost_amount"])/(1+r["discount_rate"])**((date.fromisoformat(x["event_at"])-date.fromisoformat(r["default_at"])).days/365) for x in flows if x["facility_id"]==r["facility_id"])
        assert truth[r["facility_id"]]["ultimate_discounted_lgd"]==pytest.approx(1-pv/r["ead"])
    cases={x["case_id"]:x for x in rows(world,"latent_truth","analytical_workout_fixtures")}
    assert cases["negative_lgd"]["expected_raw_lgd"]==pytest.approx(-.2)
    assert cases["lgd_above_one"]["expected_raw_lgd"]==pytest.approx(1.2)


def test_facility_origination_cannot_read_future_hazards(tmp_path):
    source=tmp_path/"obligors";shutil.copytree(REFERENCE/"obligors",source)
    records=rows(source,"latent_truth","obligor_period")
    for r in records:
        if r["period"]>0: r["true_pd"] = .999
    path=source/"latent_truth/obligor_period.jsonl.gz"
    with gzip.open(path,"wt") as f:
        for r in records:f.write(json.dumps(r)+"\n")
    manifest=json.loads((source/"manifest.json").read_text());manifest["tables"]["latent_truth/obligor_period"]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest();(source/"manifest.json").write_text(json.dumps(manifest))
    result=generate_facility_workouts(source,spec(),REFERENCE/"innovations",output_directory=tmp_path/"workouts")
    assert rows(result.value["directory"],"observable_inputs","facilities")==rows(REFERENCE/"workouts","observable_inputs","facilities")


def test_retail_actual_labels_cash_identity_and_active_zero_balance():
    world=REFERENCE/"retail"
    events=rows(world,"observed_outcomes","account_events")
    defaults={x["account_id"] for x in events if x["event_type"]=="DEFAULT"}
    for r in rows(world,"observed_outcomes","application_labels"):
        if not r["funded"]: assert r["own_loan_default"] is None and r["account_id"] is None
        else: assert r["own_loan_default"]==int(r["account_id"] in defaults)
    panel=rows(world,"observable_inputs","account_months")
    assert any(r["active_zero_balance"] for r in panel)
    previous={}
    for r in panel:
        assert r["balance"]==pytest.approx(r["opening_balance"]+r["principal_draw"]+r["interest"]-r["payment"]-r["recovery_amount"],abs=1e-8)
        if r["account_id"] in previous: assert r["opening_balance"]==pytest.approx(previous[r["account_id"]])
        previous[r["account_id"]]=r["balance"]
        if r["months_on_book"]==0: assert r["payment"]==r["interest"]==r["principal_draw"]==0
        if r["days_past_due"]>0 and r["state"] in ("ACTIVE","CURED"): assert r["payment"]==0
    assert all(x["known_at"]>next(a["application_at"] for a in rows(world,"observable_inputs","applications") if a["application_id"]==x["pull_id"].removeprefix("bureau-")) for x in rows(world,"observable_inputs","bureau_revisions"))
    assert all("missing_mechanism" not in r for r in rows(world,"observable_inputs","applications"))


def test_pool_uses_actual_source_cash_and_distinct_assets():
    pool=REFERENCE/"pools";members=rows(pool,"observable_inputs","pool_memberships")
    assert len({x["source_asset_id"] for x in members})==len(members)==80
    assert {x["asset_type"] for x in members}=={"accounts","facilities"}
    source={(x["account_id"],x["observed_at"]):x["payment"] for x in rows(REFERENCE/"retail","observable_inputs","account_months")}
    payments=[x for x in rows(pool,"observed_outcomes","servicing_cashflows") if x["component"]=="payment"]
    assert payments
    for r in payments: assert r["amount"]==source[(r["source_asset_id"],r["event_at"])]
    for p in rows(pool,"observable_inputs","pools"):
        tranches=[x for x in rows(pool,"observable_inputs","tranches") if x["pool_id"]==p["pool_id"]]
        assert len(tranches)==4 and sum(x["original_principal"] for x in tranches)==pytest.approx(p["initial_principal"])


def test_hand_authored_event_checkpoint_oracles():
    points=rows(REFERENCE/"events","latent_truth","independent_checkpoints")[:4]
    assert [(p["expected_score"],p["expected_support_capacity"],p["expected_model_version"]) for p in points]==[(110,100,"v1"),(105,100,"v1"),(105,0,"v1"),(105,0,"v2")]
    deliveries=rows(REFERENCE/"events","observed_outcomes","system_events")[:7]
    assert len(deliveries)==7 and len({x["event_id"] for x in deliveries})==6
    assert next(x for x in deliveries if x["event_type"]=="FAILED_BATCH")["valid"] is False


def test_challenges_are_independent_analytical_identities():
    assert generate_challenge_cases().provenance["data_origin"]=="SYNTHETIC"
    cases={x["case_id"]:x for x in generate_challenge_cases().value["cases"]}
    bound=cases["zero_default"]["expected"]["exact_one_sided_upper_pd"]
    assert (1-bound)**100==pytest.approx(.05)
    assert cases["cohort_100"]["expected"]["denominator"]==100
    assert cases["censored_target"]["expected"]["binary_target"] is None
    with pytest.raises(CreditRiskError): generate_challenge_cases(["unknown"])


@pytest.mark.extended
def test_native_generation_chunk_independence_and_all_six_worlds(tmp_path):
    s=spec();draws=REFERENCE/"innovations";worlds={}
    worlds["macro"]=generate_macro_reference_world(s,draws,output_directory=tmp_path/"macro",chunk_rows=7)
    worlds["obligors"]=generate_obligor_panel(worlds["macro"],s,draws,output_directory=tmp_path/"obligors",chunk_rows=7)
    worlds["workouts"]=generate_facility_workouts(worlds["obligors"],s,draws,output_directory=tmp_path/"workouts",chunk_rows=7)
    worlds["retail"]=generate_retail_reference_world(worlds["macro"],s,draws,output_directory=tmp_path/"retail",chunk_rows=7)
    worlds["pools"]=generate_asset_pool_reference_world([worlds["retail"],worlds["workouts"]],s,draws,output_directory=tmp_path/"pools",chunk_rows=7)
    worlds["events"]=generate_system_event_reference_world([worlds["obligors"],worlds["retail"],worlds["workouts"],worlds["pools"]],s,draws,output_directory=tmp_path/"events",chunk_rows=7)
    for name,world in worlds.items():
        assert validate_synthetic_world(world).value["passed"]
        for key,meta in world.value["manifest"]["tables"].items():
            layer,table=key.split("/")
            assert rows(tmp_path/name,layer,table)==rows(REFERENCE/name,layer,table)


@pytest.mark.parametrize("mechanism",["MAR","missing_overlap"])
@pytest.mark.extended
def test_selection_profiles_preserve_potential_truth_and_only_change_policy(tmp_path,mechanism):
    s=spec();s["selection_mechanism"]=mechanism
    result=generate_retail_reference_world(REFERENCE/"macro",s,REFERENCE/"innovations",output_directory=tmp_path/"world")
    actual=rows(result.value["directory"],"latent_truth","application_parameters")
    baseline=rows(REFERENCE/"retail","latent_truth","application_parameters")
    assert [r["potential_loan_outcome"] for r in actual]==[r["potential_loan_outcome"] for r in baseline]
    if mechanism=="missing_overlap": assert any(r["acceptance_probability"]==0 for r in actual)
    if mechanism=="MAR":
        applications={r["application_id"]:r for r in rows(result.value["directory"],"observable_inputs","applications")}
        predictions={r["application_id"]:r for r in rows(result.value["directory"],"frozen_predictions","application_predictions")}
        for r in actual:
            app=applications[r["application_id"]];dti=(app["proposed_installment"]+app["existing_debt_service"])/(app["net_monthly_income"] or 2500)
            expected=1/(1+math.exp(-(2.2-5*predictions[r["application_id"]]["pd"]-.5*dti)))
            if app["application_at"]>="2000-07-01": expected*=.9
            assert r["acceptance_probability"]==pytest.approx(expected)


@pytest.mark.extended
def test_well_specified_ordinary_hazard_equals_frozen_equation(tmp_path):
    s=spec();s["dgp_family"]="well_specified"
    macro=generate_macro_reference_world(s,REFERENCE/"innovations",output_directory=tmp_path/"macro")
    obligor=generate_obligor_panel(macro,s,REFERENCE/"innovations",output_directory=tmp_path/"obligors")
    prediction={r["snapshot_id"]:r["pd"] for r in rows(obligor.value["directory"],"frozen_predictions","obligor_predictions")}
    for r in rows(obligor.value["directory"],"latent_truth","obligor_period"):
        if r["dgp_stratum"]=="ordinary": assert r["true_pd"]==prediction[r["snapshot_id"]]


@pytest.mark.extended
def test_short_cutoff_retains_open_workouts_and_censored_horizons(tmp_path):
    # An explicitly constructed innovation fixture, not outcome-driven reseeding
    # of the bundled stochastic cohort: obligor2 defaults on 2001-04-01.
    import struct
    cutoff="2001-10-01";s=spec();s["observation_cutoff"]=cutoff
    draws=tmp_path/"constructed_draws";shutil.copytree(REFERENCE/"innovations",draws)
    path=draws/"obligor_year.u32le";data=bytearray(path.read_bytes())
    for component in (3,4): struct.pack_into("<I",data,(2*3*16+component)*4,0)
    path.write_bytes(data);manifest=json.loads((draws/"manifest.json").read_text())
    manifest["streams"]["obligor_year"]["sha256"]=hashlib.sha256(data).hexdigest();(draws/"manifest.json").write_text(json.dumps(manifest))
    macro=generate_macro_reference_world(s,draws,output_directory=tmp_path/"macro")
    obligor=generate_obligor_panel(macro,s,draws,output_directory=tmp_path/"obligors")
    workout=generate_facility_workouts(obligor,s,draws,output_directory=tmp_path/"workout")
    assert any(x["status"]=="open" for x in rows(workout.value["directory"],"observed_outcomes","workouts"))
    assert any(x["censored"] for x in rows(obligor.value["directory"],"observed_outcomes","obligor_labels"))
    assert all(x["known_at"]<=cutoff for x in rows(workout.value["directory"],"observed_outcomes","recovery_cashflows"))
    assert rows(obligor.value["directory"],"observed_outcomes","default_events")[0]["event_at"]=="2001-04-01"


def test_budget_invalid_source_and_innovation_hash_rejected_before_output(tmp_path):
    with pytest.raises(CreditRiskError): generate_macro_reference_world(spec(),REFERENCE/"innovations",output_directory=tmp_path/"budget",memory_budget_mb=32)
    assert not (tmp_path/"budget").exists()
    shutil.copytree(REFERENCE/"innovations",tmp_path/"bad_draws")
    path=tmp_path/"bad_draws/macro_history.u32le"
    data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data)
    with pytest.raises(CreditRiskError): generate_macro_reference_world(spec(),tmp_path/"bad_draws",output_directory=tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
    invalid=spec();invalid["start_date"]="1999-01-01"
    with pytest.raises(CreditRiskError): generate_obligor_panel(REFERENCE/"macro",invalid,REFERENCE/"innovations",output_directory=tmp_path/"linked")


@pytest.mark.parametrize("mutation,expected",[("foreign_key","FOREIGN_KEY"),("future","FUTURE_KNOWLEDGE"),("latent","LATENT_LEAKAGE"),("cash","ACCOUNT_CASHFLOW_RECONCILIATION")])
@pytest.mark.extended
def test_audit_detects_rehashed_content_mutations(tmp_path,mutation,expected):
    target=tmp_path/"retail";shutil.copytree(REFERENCE/"retail",target)
    table="applications" if mutation=="latent" else "account_months"
    records=rows(target,"observable_inputs",table)
    if mutation=="foreign_key": records[0]["account_id"]="nonexistent"
    if mutation=="future": records[0]["known_at"]="2099-01-01"
    if mutation=="latent":
        for r in records: r["true_pd"]=.5
    if mutation=="cash": records[0]["balance"]+=100
    path=target/"observable_inputs"/(table+".jsonl.gz")
    with gzip.open(path,"wt") as f:
        for r in records:f.write(json.dumps(r)+"\n")
    manifest=json.loads((target/"manifest.json").read_text());meta=manifest["tables"]["observable_inputs/"+table]
    meta["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest();meta["columns"]=list(records[0]);(target/"manifest.json").write_text(json.dumps(manifest))
    assert expected in {x["code"] for x in validate_synthetic_world(target).value["issues"]}


@pytest.mark.parametrize("kind",["table_traversal","cube_traversal","table_symlink","cube_symlink"])
def test_manifest_resource_escape_rejected_before_read(tmp_path,kind):
    target=tmp_path/"world";shutil.copytree(REFERENCE/"macro",target)
    manifest=json.loads((target/"manifest.json").read_text())
    meta=manifest["tables"]["observable_inputs/macro_releases"] if kind.startswith("table") else manifest["binary_resources"]["scenario_cube"]
    outside=tmp_path/"outside";outside.write_bytes(b"not a resource")
    if kind.endswith("traversal"): meta["file"]="../outside"
    else:
        link=target/"escape";link.symlink_to(outside);meta["file"]="escape"
    (target/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(CreditRiskError) as error: validate_synthetic_world(target)
    assert error.value.code=="SYNTHETIC_WORLD"


def test_short_observation_worlds_real_generation_and_audit(tmp_path):
    s=spec();s['observation_cutoff']='2000-04-01';draws=REFERENCE/'innovations';w={}
    w['macro']=generate_macro_reference_world(s,draws,output_directory=tmp_path/'macro')
    w['obligors']=generate_obligor_panel(w['macro'],s,draws,output_directory=tmp_path/'obligors')
    w['workouts']=generate_facility_workouts(w['obligors'],s,draws,output_directory=tmp_path/'workouts')
    w['retail']=generate_retail_reference_world(w['macro'],s,draws,output_directory=tmp_path/'retail')
    w['pools']=generate_asset_pool_reference_world([w['retail'],w['workouts']],s,draws,output_directory=tmp_path/'pools')
    w['events']=generate_system_event_reference_world([w['obligors'],w['retail'],w['workouts'],w['pools']],s,draws,output_directory=tmp_path/'events')
    for world in w.values():assert validate_synthetic_world(world).value['passed']
    accounts=rows(tmp_path/'retail','observable_inputs','account_months');assert accounts
    for row in accounts:assert row['opening_balance']+row['principal_draw']+row['interest']-row['payment']-row['recovery_amount']==pytest.approx(row['balance'],abs=1e-7)
