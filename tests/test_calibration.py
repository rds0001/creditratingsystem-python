"""Calibration references from exact grouped binomial algebra and hand pooling."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from creditratingsystem.calibration import (
    calibrate_intercept, calibrate_logistic, calibrate_isotonic, apply_calibrator,
)
from creditratingsystem.core import (
    CalcResult, CreditRiskError, build_calculation_context,
    serialize_fitted_model, load_fitted_model,
)


@pytest.fixture
def metadata():
    return {
        "target": {"event": "first_default", "horizon": 12, "time_unit": "months",
                   "probability_measure": "PHYSICAL", "conditioning": "PIT"},
        "calibration": {"start": "2020-01-01", "end": "2021-12-31",
                        "known_at": "2022-01-01", "split_id": "calibration-only",
                        "prediction_origin": "HOLDOUT"},
        "base_model_version": "synthetic-score-v1", "population_definition": "Synthetic examples",
        "outcome_policy": "Fully observed first-default outcomes",
        "weight_type": "frequency", "data_origin": "SYNTHETIC",
    }


def apply(fit, values, metadata, **kwargs):
    return apply_calibrator(fit, values, target=metadata["target"],
                            base_model_version=metadata["base_model_version"], **kwargs)


def error(code, function, *args, **kwargs):
    with pytest.raises(CreditRiskError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code


def test_intercept_odds_double_analytical(metadata):
    # Odds 1/9 multiplied by 2 -> probability 2/11, shift exactly log(2).
    fit = calibrate_intercept([0.1, 0.1], 2/11, metadata=metadata)
    assert isinstance(fit, CalcResult)
    assert fit.value["parameters"]["intercept"] == pytest.approx(math.log(2), abs=2e-12)
    np.testing.assert_allclose(apply(fit, [0.1, 0.2, 0.5], metadata).value,
                               [2/11, 1/3, 2/3], atol=2e-13)
    assert abs(fit.diagnostics["residual"]) <= 1e-12


def test_intercept_fixed_boundaries_and_weighted_target(metadata):
    # Structural 1 contributes 1/4; interior contributes 2/4*p. Target .5 -> p=.5.
    fit = calibrate_intercept([0, 0.2, 1], 0.5, [1, 2, 1], metadata=metadata)
    assert fit.value["parameters"]["intercept"] == pytest.approx(math.log(4), abs=2e-12)
    values = apply(fit, [1, 0.2, 0], metadata).value
    np.testing.assert_allclose(values, [1, 0.5, 0], atol=1e-13)
    assert values[0] == 1 and values[2] == 0


def test_intercept_identity_and_external_target(metadata):
    metadata["calibration"]["prediction_origin"] = "EXTERNAL_TARGET"
    fit = calibrate_intercept([0.25, 0.75], 0.5, metadata=metadata)
    assert fit.value["parameters"]["intercept"] == 0
    assert fit.value["metadata"]["calibration"]["prediction_origin"] == "EXTERNAL_TARGET"


@pytest.mark.parametrize("target", [0, 0.25, 0.75, 1])
def test_intercept_rejects_unattainable_finite_shift_targets(metadata, target):
    error("INFEASIBLE_TARGET", calibrate_intercept, [0, 0.2, 1], target,
          [1, 2, 1], metadata=metadata)


def test_intercept_all_boundaries_not_identifiable(metadata):
    error("NOT_IDENTIFIABLE", calibrate_intercept, [0, 1], 0.5, metadata=metadata)


def test_grouped_logistic_closed_form_and_information(metadata):
    # At logits +/-log(4), independent binomial MLE rates are 1/4 and 3/4.
    fit = calibrate_logistic([0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1], [3, 1, 1, 3], metadata=metadata)
    params = fit.value["parameters"]
    assert params["intercept"] == pytest.approx(0, abs=2e-10)
    assert params["slope"] == pytest.approx(math.log(3)/math.log(4), abs=2e-9)
    np.testing.assert_allclose(apply(fit, [0.8, 0.2, 0.5], metadata).value,
                               [0.75, 0.25, 0.5], atol=1e-9)
    np.testing.assert_allclose(fit.diagnostics["covariance"],
        [[2/3, 0], [0, 2/(3*math.log(4)**2)]], atol=2e-9)
    assert fit.diagnostics["negative_log_likelihood"] == pytest.approx(
        -2*(3*math.log(0.75)+math.log(0.25)), abs=2e-9)
    assert not fit.diagnostics["slope_nonpositive"]


def test_negative_and_zero_slope_are_explicit(metadata):
    p, y = [0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1]
    inverse = calibrate_logistic(p, y, [1, 3, 3, 1], metadata=metadata)
    assert inverse.value["parameters"]["slope"] == pytest.approx(-math.log(3)/math.log(4), abs=2e-9)
    assert inverse.diagnostics["slope_nonpositive"]
    flat = calibrate_logistic(p, y, metadata=metadata)
    assert flat.value["parameters"]["slope"] == 0
    assert flat.diagnostics["slope_nonpositive"]


@pytest.mark.parametrize("p,y,code", [
    ([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], "SEPARATION"),
    ([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1], "SEPARATION"),
    ([0.1, 0.2], [1, 1], "SEPARATION"),
    ([0.5, 0.5], [0, 1], "NOT_IDENTIFIABLE"),
    ([0, 0.2, 0.8, 1], [0, 1, 0, 1], "INVALID_PROBABILITY"),
])
def test_logistic_identifiability_and_boundaries(metadata, p, y, code):
    error(code, calibrate_logistic, p, y, metadata=metadata)


def test_logistic_frequency_weights_equal_literal_replication(metadata):
    p, y, w = [0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1], [3, 1, 1, 3]
    weighted = calibrate_logistic(p, y, w, metadata=metadata)
    repeated = calibrate_logistic(np.repeat(p, w), np.repeat(y, w), metadata=metadata)
    for key in ("intercept", "slope"):
        assert weighted.value["parameters"][key] == pytest.approx(repeated.value["parameters"][key], abs=1e-9)
    np.testing.assert_allclose(weighted.diagnostics["covariance"], repeated.diagnostics["covariance"], atol=1e-9)


def test_weighted_isotonic_hand_pooling_and_input_order(metadata):
    # Violation 1,0 at scores 2,3 pools weights 2+1 and event count 2 -> 2/3.
    fit = calibrate_isotonic([1, 2, 3, 4], [0, 1, 0, 1], [1, 2, 1, 1], metadata=metadata)
    np.testing.assert_allclose(fit.value["parameters"]["values"], [0, 2/3, 2/3, 1])
    np.testing.assert_allclose(apply(fit, [4, 2.5, 1, 3], metadata).value, [1, 2/3, 0, 2/3])
    assert fit.diagnostics["weighted_fitted_mean"] == pytest.approx(3/5)
    assert fit.diagnostics["weighted_outcome_mean"] == pytest.approx(3/5)


def test_isotonic_ties_pool_before_monotonicity(metadata):
    # At score 1: (0*1+1*3)/4=.75, violating next score's zero.
    # Both score levels therefore pool to three events / five observations.
    fit = calibrate_isotonic([3, 1, 2, 1], [1, 0, 0, 1], [1, 1, 1, 3], metadata=metadata)
    np.testing.assert_allclose(fit.value["parameters"]["values"], [0.6, 0.6, 1])
    np.testing.assert_allclose(apply(fit, [1, 3, 1, 2], metadata).value, [0.6, 1, 0.6, 0.6])


def test_isotonic_decreasing_and_support_policy(metadata):
    fit = calibrate_isotonic([1, 2, 3, 4], [1, 0, 1, 0], metadata=metadata,
                             increasing=False, out_of_bounds="clip")
    np.testing.assert_allclose(apply(fit, [0, 2, 3, 5], metadata).value, [1, 0.5, 0.5, 0])
    assert apply(fit, [0, 5], metadata).diagnostics["out_of_support_count"] == 2
    strict = calibrate_isotonic([1, 2], [0, 1], metadata=metadata)
    error("OUT_OF_SUPPORT", apply, strict, [3], metadata)


def test_zero_weight_observations_do_not_set_isotonic_support(metadata):
    fit = calibrate_isotonic([-100, 1, 2, 100], [1, 0, 1, 0], [0, 1, 1, 0], metadata=metadata)
    assert fit.value["parameters"]["knots"] == [1, 2]
    assert fit.diagnostics["zero_weight_count"] == 2
    error("OUT_OF_SUPPORT", apply, fit, [100], metadata)


@pytest.mark.parametrize("weights,code", [
    ([0, 0], "INVALID_WEIGHTS"), ([-1, 2], "INVALID_WEIGHTS"),
    ([0.5, 1.5], "INVALID_WEIGHTS"), ([1], "INCOMPATIBLE_SHAPE"),
    ([True, 1], "INVALID_INPUT"), ([1, np.inf], "NON_FINITE_INPUT"),
])
def test_weight_contract(metadata, weights, code):
    error(code, calibrate_isotonic, [0, 1], [0, 1], weights, metadata=metadata)


@pytest.mark.parametrize("values,outcomes,code", [
    ([], [], "EMPTY_INPUT"), ([0, 1], [0], "INCOMPATIBLE_SHAPE"),
    ([0, 1], [0, 0.5], "INVALID_OUTCOME"), ([0, 1], None, "MISSING_INPUT"),
    ([[0, 1]], [0, 1], "INVALID_INPUT"), ([0, np.nan], [0, 1], "NON_FINITE_INPUT"),
    ([0, 1], [0, None], "MISSING_INPUT"),
])
def test_input_failures(metadata, values, outcomes, code):
    error(code, calibrate_isotonic, values, outcomes, metadata=metadata)


@pytest.mark.parametrize("field", ["target", "calibration", "base_model_version", "population_definition", "outcome_policy", "data_origin"])
def test_required_metadata(metadata, field):
    del metadata[field]
    error("INVALID_CALIBRATION_METADATA", calibrate_intercept, [0.2], 0.3, metadata=metadata)


def test_metadata_does_not_allow_training_reuse(metadata):
    metadata["calibration"]["prediction_origin"] = "TRAINING"
    error("INVALID_CALIBRATION_METADATA", calibrate_isotonic, [0, 1], [0, 1], metadata=metadata)
    metadata["calibration"]["prediction_origin"] = "EXTERNAL_TARGET"
    error("INVALID_CALIBRATION_METADATA", calibrate_logistic, [0.2, 0.8], [0, 1], metadata=metadata)


def test_wrong_target_and_model_version_are_rejected(metadata):
    fit = calibrate_intercept([0.2], 0.3, metadata=metadata)
    for name, value in [("horizon", 24), ("probability_measure", "RISK_NEUTRAL"), ("conditioning", "TTC")]:
        target = deepcopy(metadata["target"])
        target[name] = value
        error("INCOMPATIBLE_TARGET", apply_calibrator, fit, [0.2], target=target,
              base_model_version=metadata["base_model_version"])
    error("INCOMPATIBLE_MODEL_VERSION", apply_calibrator, fit, [0.2],
          target=metadata["target"], base_model_version="different-version")


@pytest.mark.parametrize("kind", ["intercept", "logistic", "isotonic"])
def test_artifact_json_replay_and_provenance(metadata, kind):
    if kind == "intercept":
        fit = calibrate_intercept([0.2, 0.8], 0.4, metadata=metadata)
    elif kind == "logistic":
        fit = calibrate_logistic([0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1], metadata=metadata)
    else:
        fit = calibrate_isotonic([0.2, 0.8], [0, 1], metadata=metadata)
    replay = json.loads(json.dumps(fit.value, allow_nan=False))
    np.testing.assert_array_equal(apply(fit, [0.2, 0.8], metadata).value,
                                  apply(replay, [0.2, 0.8], metadata).value)
    assert fit.provenance["data_origin"] == "SYNTHETIC"
    assert fit.provenance["validation_status"] == "NOT_VALIDATED"
    assert len(fit.value["calibration_input_fingerprint"]) == 64
    assert np.asarray(apply(fit, [], metadata).value).shape == (0,)
    replay["parameter_version"] = "tampered"
    error("INTEGRITY_ERROR", apply, replay, [0.2], metadata)


def test_frozen_inputs_and_fingerprint_order(metadata):
    p = np.array([0.1, 0.2, 0.3])
    original = p.copy()
    before = deepcopy(metadata)
    fit = calibrate_intercept(p, 0.4, metadata=metadata)
    np.testing.assert_array_equal(p, original)
    assert metadata == before
    changed = calibrate_intercept(p[::-1], 0.4, metadata=metadata)
    assert fit.value["calibration_input_fingerprint"] != changed.value["calibration_input_fingerprint"]


def test_nonconvergence_controls(metadata):
    error("NON_CONVERGENCE", calibrate_logistic, [0.2, 0.2, 0.8, 0.8], [0, 1, 0, 1],
          [3, 1, 1, 3], metadata=metadata, max_iterations=1)
    error("INVALID_INPUT", calibrate_intercept, [0.2], 0.3, metadata=metadata, tolerance=0)


def test_context_cannot_silently_change_probability_target(metadata):
    context = build_calculation_context(horizon=24, time_unit="months")
    error("INCOMPATIBLE_TARGET", calibrate_intercept, [0.2], 0.3,
          metadata=metadata, context=context)
    fit = calibrate_intercept([0.2], 0.3, metadata=metadata)
    error("INCOMPATIBLE_TARGET", apply, fit, [0.2], metadata, context=context)


def test_calibration_inside_public_model_bundle_is_lossless(metadata, tmp_path):
    fit = calibrate_intercept([0.1, 0.1], 2/11, metadata=metadata)
    model = {
        "schema_version": "1.0", "model_id": "synthetic-model", "model_version": "1",
        "method_id": "calibration_bundle_example", "feature_order": ["base_pd"],
        "parameters": {"identity": True}, "preprocessing": {}, "target": metadata["target"],
        "training": {"data_fingerprint": "synthetic-training", "split_id": "train",
                     "known_at": "2020-01-01"}, "diagnostics": {},
        "provenance": {"data_origin": "SYNTHETIC", "parameter_origin": "USER_SUPPLIED"},
        "calibration_artifact": fit.value,
    }
    destination = tmp_path / "calibrated_model.json"
    serialize_fitted_model(model, destination)
    loaded = load_fitted_model(destination).value
    np.testing.assert_array_equal(apply(fit, [0.1, 0.2], metadata).value,
                                  apply(loaded["calibration_artifact"], [0.1, 0.2], metadata).value)


@pytest.mark.parametrize("increasing", [True, False])
def test_isotonic_against_independent_scipy_implementation(metadata, increasing):
    from scipy.optimize import isotonic_regression

    # Tied scores and unequal frequencies independently reduce to binomial means.
    scores = np.array([7, 1, 3, 1, 5, 3, 2, 7, 6, 4])
    outcomes = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    weights = np.array([3, 2, 5, 1, 4, 2, 6, 1, 3, 7])
    knots = np.unique(scores)
    mass = np.array([weights[scores == k].sum() for k in knots])
    means = np.array([(weights[scores == k] * outcomes[scores == k]).sum()
                      for k in knots]) / mass
    reference = isotonic_regression(means, weights=mass, increasing=increasing).x
    fitted = calibrate_isotonic(scores, outcomes, weights, metadata=metadata,
                                increasing=increasing)
    np.testing.assert_allclose(fitted.value["parameters"]["values"], reference,
                               rtol=0, atol=2e-15)


def test_rehashed_invalid_artifact_is_still_rejected(metadata):
    from creditratingsystem.core import _fingerprint

    fitted = calibrate_isotonic([1, 2, 3], [0, 0, 1], metadata=metadata)
    invalid = deepcopy(fitted.value)
    invalid["parameters"]["values"] = [1, 0, 1]
    invalid.pop("artifact_hash")
    invalid["artifact_hash"] = _fingerprint(invalid)
    error("INVALID_CALIBRATOR", apply, invalid, [2], metadata)
