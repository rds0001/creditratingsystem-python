"""Independent algebraic checks for elementary loss and first-default curves.

Expected values come from exact decimal arithmetic or a finite probability
tree, never from another implementation of this package.
"""

from decimal import Decimal, localcontext
from itertools import product

import numpy as np
import pytest

from creditratingsystem import (
    build_calculation_context,
    cumulative_pd_from_conditional,
    expected_loss,
    marginal_pd_from_conditional,
)
from creditratingsystem.core import CalcResult, CreditRiskError


CURVE_METHODS = (cumulative_pd_from_conditional, marginal_pd_from_conditional)


def decimal_curve_oracle(hazards):
    """Evaluate the defining probability tree with 120 decimal digits."""
    with localcontext() as ctx:
        ctx.prec = 120
        survival = Decimal(1)
        cumulative, marginal = [], []
        for hazard in hazards:
            q = Decimal(str(hazard))
            marginal.append(survival * q)
            survival *= 1 - q
            cumulative.append(1 - survival)
        return [float(x) for x in cumulative], [float(x) for x in marginal]


@pytest.mark.parametrize(
    "pd,lgd,ead,expected",
    [(0.02, 0.4, 1000, 8), (0, 0.9, 10000, 0), (1, 0.25, 400, 100)],
)
def test_loss_baseline(pd, lgd, ead, expected):
    assert expected_loss(pd, lgd, ead).value == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "method,hazards,expected",
    [
        (cumulative_pd_from_conditional, [0.1, 0.2], [0.1, 0.28]),
        (marginal_pd_from_conditional, [0.1, 0.2], [0.1, 0.18]),
        (cumulative_pd_from_conditional, [0, 1], [0, 1]),
        (marginal_pd_from_conditional, [0, 1], [0, 1]),
        (marginal_pd_from_conditional, [1, 0.5], [1, 0]),
    ],
)
def test_curve_baseline(method, hazards, expected):
    np.testing.assert_allclose(method(hazards).value, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "pd,lgd,ead,expected",
    [
        (0.2, -0.25, 100, -5),
        (0.2, 1.5, 100, 30),
        ([0, 0.2, 1], [-0.5, 1.25, 2], [10, 100, 50], [0, 25, 100]),
        (0.2, [-0.25, 1.5], [100, 200], [-5, 60]),
        ([0.1, 0.2], 0.5, 100, [5, 10]),
        ([0.2], [-0.25, 1.5], [100], [-5, 30]),
        ([0.02], [0.4], [1000], 8),
        (0.2, 0, 100, 0),
        (0.2, -0.25, 0, 0),
    ],
)
def test_loss_raw_lgd_and_explicit_expansion(pd, lgd, ead, expected):
    result = expected_loss(pd, lgd, ead)
    np.testing.assert_allclose(result.value, expected, atol=1e-12, rtol=1e-12)
    assert np.asarray(result.value).shape == np.asarray(expected).shape


@pytest.mark.parametrize(
    "pd,lgd,ead,expected",
    [
        (1e-308, 1e-308, 1e308, 1e-308),
        ([1e-308, 1e-308], [-1e-308, -1e308], [1e308, 1e-308],
         [-1e-308, -1e-308]),
        (1e-308, 1e308, 1e308, 1e308),
    ],
)
def test_loss_preserves_representable_extreme_scale_product(pd, lgd, ead, expected):
    # Exact decimal algebra cancels opposing powers of ten. A prematurely
    # underflowed intermediate product must not erase the representable result.
    np.testing.assert_allclose(expected_loss(pd, lgd, ead).value, expected,
                               atol=0, rtol=2e-14)


@pytest.mark.parametrize("hazards", [[1e-20, 2e-20, 3e-20], [0.5, 1e-20, 2e-20]])
def test_tiny_hazards_preserve_probability_mass(hazards):
    cumulative, marginal = decimal_curve_oracle(hazards)
    # A broad absolute tolerance would mistakenly accept zero for tiny masses.
    np.testing.assert_allclose(
        cumulative_pd_from_conditional(hazards).value, cumulative, atol=0, rtol=2e-14
    )
    np.testing.assert_allclose(
        marginal_pd_from_conditional(hazards).value, marginal, atol=0, rtol=2e-14
    )


def test_probability_tree_by_exact_enumeration():
    """Independent 64 equally likely paths: hazards 1/2, 1/4, 3/4."""
    counts = [0, 0, 0]
    for draws in product(range(4), repeat=3):
        for period, (draw, threshold) in enumerate(zip(draws, [2, 1, 3])):
            if draw < threshold:
                counts[period] += 1
                break
    marginal = np.array(counts) / 64
    np.testing.assert_array_equal(marginal, [0.5, 0.125, 0.28125])
    np.testing.assert_allclose(
        marginal_pd_from_conditional([0.5, 0.25, 0.75]).value, marginal, atol=1e-15
    )
    np.testing.assert_allclose(
        cumulative_pd_from_conditional([0.5, 0.25, 0.75]).value,
        np.cumsum(marginal), atol=1e-15,
    )


@pytest.mark.parametrize("method", CURVE_METHODS)
@pytest.mark.parametrize("hazards", [[], [0.25], [0, 0, 0]])
def test_curve_shapes_and_zero_hazards(method, hazards):
    result = method(hazards)
    assert np.asarray(result.value).shape == (len(hazards),)
    expected = [] if not hazards else ([0.25] if hazards == [0.25] else [0, 0, 0])
    # log1p/expm1 may round the fractional hazard by one ulp; shape and zeros
    # remain exact, while the nonzero value follows the numerical contract.
    np.testing.assert_allclose(result.value, expected, atol=0, rtol=2e-16)


def test_absorption_does_not_add_first_defaults():
    hazards = [0.25, 1, 0.9, 0, 1]
    cumulative = cumulative_pd_from_conditional(hazards).value
    assert cumulative[0] == pytest.approx(0.25, abs=0, rel=2e-16)
    np.testing.assert_array_equal(cumulative[1:], [1, 1, 1, 1])
    np.testing.assert_array_equal(
        marginal_pd_from_conditional(hazards).value, [0.25, 0.75, 0, 0, 0]
    )


def test_nonmonotone_hazards_yield_monotone_cumulative_pd():
    hazards = [0.3, 0.01, 0.7, 0.05]
    cumulative, marginal = decimal_curve_oracle(hazards)
    actual = cumulative_pd_from_conditional(hazards).value
    np.testing.assert_allclose(actual, cumulative, atol=1e-15, rtol=1e-14)
    assert np.all(np.diff(actual) >= 0)
    assert sum(marginal_pd_from_conditional(hazards).value) == pytest.approx(cumulative[-1])
    np.testing.assert_allclose(
        marginal_pd_from_conditional(hazards).value, marginal, atol=1e-15, rtol=1e-14
    )


def assert_code(code, function, *args):
    with pytest.raises(CreditRiskError) as caught:
        function(*args)
    assert getattr(caught.value, "code", None) == code


@pytest.mark.parametrize(
    "value,code",
    [
        (True, "INVALID_INPUT"), ("0.1", "INVALID_INPUT"),
        (None, "MISSING_INPUT"), (np.nan, "NON_FINITE_INPUT"),
        (np.inf, "NON_FINITE_INPUT"), (-np.inf, "NON_FINITE_INPUT"),
        (0.1 + 0j, "INVALID_INPUT"), ([[0.1]], "INVALID_INPUT"),
        ([], "EMPTY_INPUT"),
    ],
)
@pytest.mark.parametrize("position", range(3))
def test_loss_rejects_invalid_numeric_inputs(value, code, position):
    args = [0.2, 0.4, 100]
    args[position] = value
    assert_code(code, expected_loss, *args)


@pytest.mark.parametrize(
    "args,code",
    [
        ((-0.1, 0.4, 100), "INVALID_PROBABILITY"),
        ((1.1, 0.4, 100), "INVALID_PROBABILITY"),
        ((0.2, 0.4, -1), "INVALID_EAD"),
        (([0.1, 0.2], [0.3, 0.4, 0.5], 100), "INCOMPATIBLE_SHAPE"),
        ((1, 1e308, 2), "NUMERICAL_FAILURE"),
    ],
)
def test_loss_domain_shape_and_overflow(args, code):
    assert_code(code, expected_loss, *args)


@pytest.mark.parametrize("method", CURVE_METHODS)
@pytest.mark.parametrize(
    "value,code",
    [
        (0.1, "INVALID_INPUT"), (None, "MISSING_INPUT"),
        ([None], "MISSING_INPUT"), ([True], "INVALID_INPUT"),
        (["0.1"], "INVALID_INPUT"), ([[0.1]], "INVALID_INPUT"),
        ([np.nan], "NON_FINITE_INPUT"), ([np.inf], "NON_FINITE_INPUT"),
        ([-np.inf], "NON_FINITE_INPUT"), ([-0.01], "INVALID_PROBABILITY"),
        ([1.01], "INVALID_PROBABILITY"), ([0.1 + 0j], "INVALID_INPUT"),
        ([1, -0.1], "INVALID_PROBABILITY"),
    ],
)
def test_curves_reject_invalid_inputs_even_after_absorption(method, value, code):
    assert_code(code, method, value)


@pytest.mark.parametrize(
    "function,args,method_id,unit",
    [
        (expected_loss, (0.02, 0.4, 1000), "losses.expected_loss", "ead_unit"),
        (cumulative_pd_from_conditional, ([0.1, 0.2],),
         "pd_term_structure.cumulative_pd_from_conditional", "probability"),
        (marginal_pd_from_conditional, ([0.1, 0.2],),
         "pd_term_structure.marginal_pd_from_conditional", "probability"),
    ],
)
def test_results_carry_method_and_noninvented_provenance(function, args, method_id, unit):
    result = function(*args)
    assert isinstance(result, CalcResult)
    assert result.method_id == method_id
    assert result.unit == unit
    assert result.status == "OK"
    assert result.context == {"mode": "TIME_INDEPENDENT"}
    assert result.provenance["data_origin"] == "USER_SUPPLIED"
    assert result.provenance["parameter_origin"] == "USER_SUPPLIED"
    assert result.provenance["validation_status"] == "NOT_VALIDATED"


def test_loss_exposes_factorisation_limit():
    assert expected_loss(0.02, 0.4, 1000).limitations


@pytest.mark.parametrize(
    "function,args",
    [(expected_loss, (0.02, 0.4, 1000)),
     (cumulative_pd_from_conditional, ([0.1, 0.2],)),
     (marginal_pd_from_conditional, ([0.1, 0.2],))],
)
def test_synthetic_context_preserves_measure_and_origin(function, args):
    context = build_calculation_context(
        probability_measure="PHYSICAL", conditioning="PIT",
        data_manifest={"data_origin": "SYNTHETIC",
                       "parameter_origin": "SYNTHETIC_REFERENCE"},
    )
    result = function(*args, context=context)
    assert result.context == context.value
    assert result.context["as_of"] is None
    assert result.context["known_at"] is None
    assert result.provenance["data_origin"] == "SYNTHETIC"
    assert result.provenance["parameter_origin"] == "SYNTHETIC_REFERENCE"
    assert result.provenance["validation_status"] == "NOT_VALIDATED"


def test_numeric_array_inputs_are_not_mutated():
    hazards = np.array([0.1, 0.2, 1.0, 0.3])
    original = hazards.copy()
    for method in CURVE_METHODS:
        method(hazards)
        np.testing.assert_array_equal(hazards, original)
    expected_loss(hazards, 0.4, 100)
    np.testing.assert_array_equal(hazards, original)
