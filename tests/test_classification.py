import numpy as np
import pytest
from scipy.special import ndtri
from creditratingsystem.classification import fit_binary_glm, predict_pd
from creditratingsystem.core import CreditRiskError, serialize_fitted_model, load_fitted_model

TARGET = dict(event="default", horizon=1, time_unit="years", probability_measure="PHYSICAL", conditioning="performing_at_observation")
TRAINING = dict(data_fingerprint="independent-two-group-counts", split_id="train", known_at="2025-12-31", data_origin="SYNTHETIC")


def fit(link, x=None, y=None, **kwargs):
    if x is None:
        x = np.array([[0], [0], [1], [1]], float)
    if y is None:
        y = [0, 1, 0, 1]
    options = dict(link=link, feature_order=["group"], fit_intercept=True,
                   target=TARGET, training=TRAINING, model_id="binary-example", model_version="1",
                   weights=[8, 2, 4, 6])
    options.update(kwargs)
    return fit_binary_glm(x, y, **options)


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_grouped_analytic_mle(link):
    links = {"logit": lambda p: np.log(p/(1-p)), "probit": ndtri,
             "cloglog": lambda p: np.log(-np.log1p(-p))}
    rates = np.array([0.2, 0.6])
    expected = links[link](rates)
    result = fit(link)
    np.testing.assert_allclose(result.model["parameters"]["coefficients"],
                               [expected[0], expected[1]-expected[0]], atol=1e-7, rtol=1e-7)
    pred = predict_pd(result, [[0], [1]], feature_order=["group"], expected_target=TARGET)
    np.testing.assert_allclose(pred.value, rates, atol=1e-8)
    expected_ll = 2*np.log(.2)+8*np.log(.8)+6*np.log(.6)+4*np.log(.4)
    assert result.diagnostics["log_likelihood"] == pytest.approx(expected_ll, abs=1e-10)
    # Each fitted group mean has binomial variance p(1-p)/10; transform by link derivative.
    if link == "logit":
        deriv = rates*(1-rates)
    elif link == "probit":
        deriv = np.exp(-expected**2/2)/np.sqrt(2*np.pi)
    else:
        deriv = (1-rates)*(-np.log1p(-rates))
    v = rates*(1-rates)/(10*deriv**2)
    covariance = np.array([[v[0], -v[0]], [-v[0], v.sum()]])
    np.testing.assert_allclose(result.model["parameters"]["covariance"], covariance, rtol=2e-7)
    assert result.provenance["data_origin"] == "SYNTHETIC"


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_intercept_offsets_and_frequency_replication(link):
    result = fit(link, x=np.empty((2, 0)), y=[0, 1], feature_order=[], weights=[7, 3])
    p = predict_pd(result, np.empty((1, 0)), feature_order=[], expected_target=TARGET).value
    np.testing.assert_allclose(p, [.3], atol=1e-8)
    base = fit(link)
    shifted = fit(link, offset=[.3]*4)
    b = np.array(base.model["parameters"]["coefficients"])
    np.testing.assert_allclose(shifted.model["parameters"]["coefficients"], b-[.3, 0], atol=1e-7)
    prediction = predict_pd(shifted, [[0], [1]], feature_order=["group"], expected_target=TARGET, offset=[.3, .3])
    np.testing.assert_allclose(prediction.value, [.2, .6], atol=1e-8)
    doubled = fit(link, weights=[16, 4, 8, 12])
    np.testing.assert_allclose(doubled.model["parameters"]["covariance"],
                               np.array(base.model["parameters"]["covariance"])/2, rtol=2e-7)


@pytest.mark.parametrize("x,y,code", [
    ([[-2],[-1],[1],[2]], [0,0,1,1], "SEPARATION"),
    ([[-1],[0],[0],[1]], [0,0,1,1], "SEPARATION"),
    ([[1],[1],[1],[1]], [0,1,0,1], "RANK_DEFICIENT"),
])
def test_identification_failures(x,y,code):
    with pytest.raises(CreditRiskError) as error:
        fit("logit", x=x, y=y, weights=None)
    assert error.value.code == code


def test_contracts_and_roundtrip(tmp_path):
    result = fit("logit")
    path = tmp_path / "model.json"
    serialize_fitted_model(result, path)
    loaded = load_fitted_model(path)
    np.testing.assert_allclose(predict_pd(loaded, [[0],[1]], feature_order=["group"], expected_target=TARGET).value, [.2,.6], atol=1e-8)
    cases = [(dict(feature_order=["other"], expected_target=TARGET), "FEATURE_MISMATCH"),
             (dict(feature_order=["group"], expected_target={**TARGET,"horizon":2}), "TARGET_MISMATCH"),
             (dict(feature_order=["group"], expected_target=TARGET, offset=[0,0]), "OFFSET_MISMATCH")]
    for kwargs, code in cases:
        with pytest.raises(CreditRiskError) as error:
            predict_pd(result, [[0],[1]], **kwargs)
        assert error.value.code == code
    for kwargs, code in [(dict(weights=[1,.2,1,1]),"INVALID_WEIGHTS"),
                         (dict(weight_type="sampling"),"INVALID_WEIGHT_TYPE"),
                         (dict(max_iter=1),"NON_CONVERGENCE")]:
        with pytest.raises(CreditRiskError) as error:
            fit("logit", **kwargs)
        assert error.value.code == code
