"""Finite probability trees and cash allocations are independent loss oracles."""
import itertools
import numpy as np
import pytest
from creditratingsystem._loss_paths import (
    expected_loss_from_paths, lifetime_expected_loss, event_conditioned_loss,
    counterparty_expected_loss, aggregate_portfolio_loss,
)
from creditratingsystem import CreditRiskError

CONVENTION = dict(currency="EUR", probability_measure="PHYSICAL")


def paths(**changes):
    args = dict(default_indicator=[[1, 0], [1, 0], [0, 1], [0, 0]],
                lgd_paths=[[.2, 0], [.8, 0], [0, .5], [0, 0]],
                ead_paths=[[100, 0], [200, 0], [0, 80], [0, 0]],
                scenario_weights=[.25]*4, discount_factors=[1, .9],
                scenario_ids=["a", "b", "c", "d"], time_grid=[1, 2],
                **CONVENTION)
    args.update(changes)
    return args


def test_joint_loss_preserves_conditional_dependence():
    # Four equally likely monetary outcomes: 20,160,36,0. Conditional
    # period-one E[LGD]E[EAD]=.5*150=75 versus E[LGD*EAD]=90.
    result = expected_loss_from_paths(**paths(uncertainty="iid")).value
    assert result["path_losses"] == pytest.approx([20, 160, 36, 0])
    assert result["expected_loss"] == 54
    assert result["period_contributions"] == [45, 9]
    assert result["conditional_independence_loss"] == 46.5
    assert result["conditional_lgd_exposure_covariance_contribution"] == 7.5
    assert result["standard_error"] == pytest.approx(np.sqrt(15632/12))
    lifetime = lifetime_expected_loss([.5, .25], [90, 40], [1, .9], [1, 2], **CONVENTION)
    assert lifetime.value["lifetime_expected_loss"] == result["expected_loss"]


def test_exclusive_states_prevent_default_transfer_double_count():
    states = [dict(state_id=str(i), events=dict(zip(["default", "transfer"], events)),
                   probability=p, conditional_loss=loss)
              for i, (events, p, loss) in enumerate(zip(itertools.product([False, True], repeat=2),
                                                        [.7, .1, .15, .05], [0, 20, 60, 70]))]
    result = event_conditioned_loss(states, event_names=["default", "transfer"], **CONVENTION)
    assert result.value["expected_loss"] == 14.5
    states[-1]["events"] = states[0]["events"]
    with pytest.raises(CreditRiskError, match="Repeated"):
        event_conditioned_loss(states, event_names=["default", "transfer"], **CONVENTION)


def test_counterparty_positive_exposure_and_wrong_way_loss():
    result = counterparty_expected_loss([[1], [0]], [[100], [-20]], [[.4], [.4]], [.5, .5], 1,
        counterparty_id="C", netting_set_id="N", scenario_ids=["bad", "good"], time_grid=[1], **CONVENTION)
    assert result.value["expected_loss"] == 30
    assert result.value["independent_default_exposure_loss"] == 15
    assert result.value["default_exposure_dependence_contribution"] == 15
    assert result.diagnostics["negative_net_values_floored"] == 1


def portfolio(**changes):
    args = dict(facility_loss_paths=[
        dict(facility_id="A", currency="EUR", scenario_ids=["low", "mid", "high"], losses=[0, 10, 30]),
        dict(facility_id="B", currency="USD", scenario_ids=["high", "low", "mid"], losses=[5, 0, 5])],
        portfolio_mapping=[dict(facility_id=f, portfolio_id="P", share=1) for f in ["A", "B"]],
        scenario_weights=[.5, .3, .2], fx_rates=dict(EUR=1, USD=2), scenario_ids=["low", "mid", "high"],
        reporting_currency="EUR", probability_measure="PHYSICAL", quantile_levels=[.75])
    args.update(changes)
    return args


def test_portfolio_aligned_fx_partial_tail_mass_and_additivity():
    # Path losses 0,20,40 with masses .5,.3,.2. Worst quarter comprises
    # all .2 mass at40 and .05 at20: ES=(8+1)/.25=36.
    result = aggregate_portfolio_loss(**portfolio()).value["portfolios"][0]
    assert result["path_losses"] == [0, 20, 40]
    assert result["expected_loss"] == 14
    tail = result["tail_metrics"][0]
    assert tail["value_at_risk"] == 20
    assert tail["expected_shortfall"] == pytest.approx(36)
    assert tail["facility_es_contributions"] == pytest.approx(dict(A=26, B=10))


def test_portfolio_boundary_ties_share_tail_allocation():
    args = portfolio(scenario_weights=[.5, .25, .25])
    args["facility_loss_paths"][0]["losses"] = [0, 10, 0]
    args["facility_loss_paths"][1]["losses"] = [5, 0, 0]
    tail = aggregate_portfolio_loss(**args).value["portfolios"][0]["tail_metrics"][0]
    assert tail["expected_shortfall"] == 10
    assert tail["facility_es_contributions"] == dict(A=5, B=5)


@pytest.mark.parametrize("changes", [dict(default_indicator=[[1, 1]]*4),
    dict(scenario_ids=["a"]*4), dict(scenario_weights=[.5]*4),
    dict(scenario_weights=[1e308]*4), dict(time_grid=[2, 1]),
    dict(discount_factors=[1, -1]), dict(uncertainty="iid", scenario_weights=[.1,.2,.3,.4])])
def test_invalid_path_contracts(changes):
    with pytest.raises(CreditRiskError):
        expected_loss_from_paths(**paths(**changes))


def test_raw_economic_lgd_and_extreme_product():
    args = paths(default_indicator=[[1]], lgd_paths=[[-1e-308]], ead_paths=[[1e308]],
                 discount_factors=1, scenario_weights=[1], scenario_ids=["x"], time_grid=[1])
    assert expected_loss_from_paths(**args).value["expected_loss"] == pytest.approx(-1)


def test_reject_cumulative_probability_mass_and_double_allocation():
    with pytest.raises(CreditRiskError):
        lifetime_expected_loss([.6,.8], [10,10], [1,1], [1,2], **CONVENTION)
    args=portfolio()
    args["portfolio_mapping"].append(dict(facility_id="A", portfolio_id="Q", share=.1))
    with pytest.raises(CreditRiskError):
        aggregate_portfolio_loss(**args)


def test_unrepresentable_dependence_difference_is_controlled():
    with pytest.raises(CreditRiskError,match="NUMERICAL_FAILURE"):
        expected_loss_from_paths([[1],[1]],[[1.7e154],[-5.5e154]],[[1e154],[1]], [.5,.5],1,
            scenario_ids=['a','b'],time_grid=[1],**CONVENTION)
