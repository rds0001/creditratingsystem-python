Operating and asset-value paths
===============================

``simulate_operating_cashflows(driver_paths, operating_contract)`` applies an
explicit quantity/price model. Each dated scenario row supplies base_quantity,
base_unit_price, base_unit_variable_cost, base_fixed_cost, volume_multiplier,
price_multiplier, variable_cost_multiplier and fixed_cost_multiplier. All amounts
are totals for the declared period, with one quantity unit and reporting currency.
The contract declares contract_id/version, currency, period_unit, quantity_unit,
initial_working_capital and include_asset_disposals_in_cfads. Rows also require
scenario_id, period_start/end, currency, working_capital_revenue_ratio, cash_taxes,
capex, asset_disposals and other_operating_cash. Periods must be contiguous.

Quantity is base quantity times volume multiplier. Revenue is quantity times
base unit price times price multiplier. Variable cost scales with quantity and
its own cost multiplier; fixed cost uses its own multiplier. Closing working
capital is period revenue times the declared ratio. CFADS equals revenue less
variable/fixed costs, cash taxes, capex and the change in working capital, plus
other operating cash and, only if declared, asset disposals. Tax refunds and
other operating cash may be signed. Reserves, funding and debt service are absent
from CFADS. This is a declared operating model rather than an industry-wide formula.

``simulate_asset_values(asset_parameters, market_paths)`` returns physical values
under P. Parameters explicitly contain asset_id, currency, valuation_date,
initial_value, salvage_value, initial_age_years, economic_life_years,
depreciation_model, idiosyncratic_volatility, probability_measure='PHYSICAL' and
year_basis_days. Straight-line ageing decreases the initial-minus-salvage amount
linearly over remaining life. Geometric-to-salvage ageing multiplies that amount
by exp(-annual_depreciation_rate * elapsed_years); economic life remains a diagnostic
under that asymptotic curve. Salvage is an ageing baseline, not a guaranteed floor.

Every path row contains scenario_id, date, market_index and innovation. Dates are
strictly after valuation; the initial market and idiosyncratic factors are one.
Market indices are levels relative to valuation. Independent-time standard-normal
innovations, with explicitly supplied cross-asset dependence, accumulate as
log_idio += -sigma^2 * dt / 2 + sigma * sqrt(dt) * innovation. The output value is
ageing baseline times market index times exp(log_idio). The mean-one correction
applies under the stated innovation law, not to an arbitrary stress sample.
Ageing, market and idiosyncratic contributions reconcile to the value. Zero
volatility is deterministic. Haircuts, sale costs and liquidation delay belong to
recovery modelling and must not be mistaken for physical asset values.
