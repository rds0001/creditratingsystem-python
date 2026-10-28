# Joint loss paths and portfolios

All methods require a reporting currency and an explicit `PHYSICAL` or
`RISK_NEUTRAL` probability measure. They evaluate supplied assumptions; they do
not estimate missing probabilities, exposures, recoveries or dependence.
Period-end times are positive, increasing **years**. Discounts are supplied
factors, not interest rates. Negative economic LGD and loss are permitted;
exposure at default is nonnegative. Inputs must be finite.

`expected_loss_from_paths(default_indicator, lgd_paths, ead_paths,
scenario_weights, discount_factors, *, scenario_ids, time_grid, currency,
probability_measure, uncertainty="none", context=None)` consumes aligned
scenario-by-period matrices. Each scenario has at most one first default.
Weights are nonnegative and sum to one within 1e-12; their residual is reported
and accepted weights normalized. Discounts may be scalar, a period vector, or
a full matrix. The calculation is the weighted sum of
`default_indicator * LGD * EAD * discount`. It preserves correlation, including
stochastic discounting. Output includes scenario losses, period contributions,
default masses and a conditional-mean factorization comparator. The difference
is the contribution from conditional LGD versus discounted-exposure covariance.
It does not establish unconditional independence. `uncertainty="iid"` returns
the sample-mean standard error only for equally weighted paths; independence
is a caller assertion. Model, parameter and time-discretization uncertainty are
excluded. Weighted quadrature scenarios do not imply sampling uncertainty.

`lifetime_expected_loss(marginal_pd, conditional_loss, discount_factors,
time_grid, *, currency, probability_measure, context=None)` evaluates a period
sum. Here `conditional_loss` means the **joint monetary expectation**
`E[LGD * EAD | first default during period]`. PD inputs are marginal first-event
masses, not cumulative probabilities or conditional hazards. Their sum cannot
exceed one. Discounting is deterministic by period; use joint paths for
stochastic dependence. No accounting-standard compliance is implied.

`event_conditioned_loss(states, *, event_names, currency, probability_measure,
context=None)` accepts the complete mutually exclusive Boolean partition of
up to twelve named events. Each state specifies a unique `state_id`, an
`events` mapping, `probability` and monetary `conditional_loss`. All 2^k
combinations, including zero-probability combinations, must be present, and
probabilities must sum to one. A joint default/transfer state is integrated
once. Separate unconditional default and transfer losses cannot be added
without resolving their overlap.

`counterparty_expected_loss(default_indicator, net_exposure_paths,
recovery_paths, scenario_weights, discount_factors, *, counterparty_id,
netting_set_id, scenario_ids, time_grid, currency, probability_measure,
uncertainty="none", context=None)` uses one externally established netting
set. Signed net values are floored at zero; recovery is in [0,1]. Joint
first-default paths produce the loss. A comparator replaces the default/loss
association with the product of default mass and unconditional discounted
loss-given-default exposure at each time. Its difference exposes dependence;
it is not a causal decomposition. No netting across counterparties, margin
closeout, own-default benefit or legal enforceability is inferred.
The [Basel Framework CRE50 definitions](https://www.bis.org/committees/bcbs/basel-framework/standard/cre/50/inforce/2019-12-15/published/2024-07-05)
provide terminology for netting sets, positive exposure, probability measures
and wrong-way risk; this routine is not an implementation of regulatory capital
or the full CVA framework.

`aggregate_portfolio_loss(facility_loss_paths, portfolio_mapping,
scenario_weights, fx_rates, *, scenario_ids, reporting_currency,
probability_measure, quantile_levels, context=None)` aligns each facility's
loss vector by its explicit scenario IDs. Each row has `facility_id`, `currency`,
`scenario_ids`, `losses`. FX rates convert one unit of local currency into
reporting currency; scenario vectors follow the global scenario order. Reporting
currency FX must be one. Mapping rows specify `facility_id`, `portfolio_id`,
`share`; total allocation cannot exceed one. Unallocated shares are reported.
The method aggregates scenario losses **before** computing weighted quantiles.
VaR is the left inverse of the weighted CDF. ES averages the worst `1-alpha`
probability mass, including the necessary fraction at the VaR boundary. Tied
boundary scenarios receive proportional weight. Facility EL and portfolio-tail
ES allocations are additive; standalone facility VaRs are not summed.

Independent numerical examples in `tests/test_loss_paths.py` enumerate a
four-state first-default tree, a default/transfer partition and a three-state
portfolio. The portfolio has losses (0,20,40), probabilities (.5,.3,.2),
75% VaR=20 and ES=36. These are transparent synthetic algebraic checks, not
empirical validation or estimated bank parameters.
