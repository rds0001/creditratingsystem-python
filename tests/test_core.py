"""Hand-specified data contracts, interval partitions and integrity failures."""
from copy import deepcopy
import json

import numpy as np
import pytest

from creditratingsystem import (
    CreditRiskError, build_calculation_context, expected_loss, load_fitted_model,
    map_pd_to_grade, serialize_fitted_model, validate_probability, validate_rating_scale,
)


def scale(policy="left_closed"):
    return {"schema_version": "1.0", "scale_id": "example", "version": "1",
            "boundary_convention": policy,
            "grades": [{"grade": "A", "lower": 0, "upper": .1, "representative_pd": .05, "is_default": False},
                       {"grade": "B", "lower": .1, "upper": 1, "representative_pd": .2, "is_default": False},
                       {"grade": "D", "lower": 1, "upper": 1, "representative_pd": 1, "is_default": True}]}


def model():
    return {"schema_version": "1.0", "model_id": "reference-logit", "model_version": "1", "method_id": "classification.fit_binary_glm",
            "feature_order": ["x"], "parameters": {"intercept": 0.0, "coefficients": [1.0]},
            "preprocessing": {}, "target": {"event": "default", "horizon": 12, "time_unit": "months", "probability_measure": "PHYSICAL", "conditioning": "PIT"},
            "training": {"data_fingerprint": "reference", "split_id": "train", "known_at": "2025-01-01"},
            "diagnostics": {"converged": True}, "provenance": {"data_origin": "SYNTHETIC", "parameter_origin": "SYNTHETIC_REFERENCE"}}


def assert_code(code, fn, *args, **kwargs):
    with pytest.raises(CreditRiskError) as error:
        fn(*args, **kwargs)
    assert error.value.code == code


def test_domain_and_shapes():
    assert validate_probability([[0, .5], [1, .3]]).value.shape == (2, 2)
    assert validate_probability([]).value.size == 0
    assert_code("INVALID_PROBABILITY", validate_probability, 1, "open")
    assert_code("INVALID_INPUT", validate_probability, [True, .3])
    assert_code("NON_FINITE_INPUT", validate_probability, np.nan)


@pytest.mark.parametrize("policy,expected", [("left_closed", ["A", "B", "B", "D"]), ("right_closed", ["A", "A", "B", "D"])])
def test_interval_partition(policy, expected):
    result = validate_rating_scale(scale(policy))
    assert map_pd_to_grade([0, .1, .9, 1], result).value == expected
    assert map_pd_to_grade([], result).value == []


def test_bad_scales_are_not_repaired():
    bad = scale()
    bad["grades"][1]["lower"] = .11
    assert_code("INVALID_RATING_SCALE", validate_rating_scale, bad)
    bad = scale()
    bad["grades"][1]["grade"] = "A"
    assert_code("INVALID_RATING_SCALE", validate_rating_scale, bad)
    bad = scale()
    bad["grades"][0]["representative_pd"] = .1
    assert_code("INVALID_RATING_SCALE", validate_rating_scale, bad)


def test_context_hash_order_and_information_date():
    a = build_calculation_context("2025-01-01", "2025-01-15", data_manifest={"a": 1, "b": "é"})
    b = build_calculation_context("2025-01-01", "2025-01-15", data_manifest={"b": "é", "a": 1.0})
    c = build_calculation_context("2025-01-01", "2025-02-15", data_manifest={"a": 1, "b": "é"})
    assert a.value == b.value
    assert c.value["context_hash"] != a.value["context_hash"]
    assert_code("INVALID_DATE", build_calculation_context, "2025-02-29", "2025-03-01")
    assert_code("MISSING_INPUT", build_calculation_context, "2025-01-01")
    assert_code("INCOMPATIBLE_HORIZON", build_calculation_context, horizon=-1, time_unit="years")
    assert_code("INVALID_CONTEXT", build_calculation_context, probability_measure="UNKNOWN")


def test_context_provenance_and_snapshot():
    manifest = {"data_origin": "SYNTHETIC", "parameter_origin": "SYNTHETIC_REFERENCE"}
    ctx = build_calculation_context(data_manifest=manifest, currency="EUR")
    manifest["data_origin"] = "USER_SUPPLIED"
    result = expected_loss(.02, .4, 1000, context=ctx)
    assert result.unit == "EUR"
    assert result.provenance["data_origin"] == "SYNTHETIC"
    assert result.provenance["validation_status"] == "NOT_VALIDATED"
    assert result.to_dict()["value"] == 8


def test_model_roundtrip_and_tamper(tmp_path):
    path = tmp_path / "model.json"
    original = model()
    saved = serialize_fitted_model(original, path)
    loaded = load_fitted_model(path)
    assert loaded.value == original
    assert loaded.diagnostics["sha256"] == saved.value["sha256"]
    assert loaded.provenance["data_origin"] == "SYNTHETIC"
    assert_code("FILE_EXISTS", serialize_fitted_model, original, path)
    envelope = json.loads(path.read_text())
    envelope["model"]["parameters"]["intercept"] = .1
    path.write_text(json.dumps(envelope))
    assert_code("INTEGRITY_ERROR", load_fitted_model, path)


def test_artifact_rejects_unsafe_or_ambiguous_data(tmp_path):
    assert_code("INVALID_PATH", load_fitted_model, "https://invalid.example/model")
    path = tmp_path / "bad.json"
    path.write_text('{"format":"one","format":"two"}')
    assert_code("INVALID_JSON", load_fitted_model, path)
    bad = model()
    bad["parameters"]["intercept"] = float("nan")
    assert_code("INVALID_JSON", serialize_fitted_model, bad, path)
    bad = model()
    bad["schema_version"] = "99"
    assert_code("UNSUPPORTED_SCHEMA_VERSION", serialize_fitted_model, bad, path)


def test_json_object_order_does_not_change_model_hash(tmp_path):
    one = model()
    two = deepcopy(one)
    two["parameters"] = dict(reversed(list(two["parameters"].items())))
    a = serialize_fitted_model(one, tmp_path / "a.json")
    b = serialize_fitted_model(two, tmp_path / "b.json")
    assert a.value["sha256"] == b.value["sha256"]
