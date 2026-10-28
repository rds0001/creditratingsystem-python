Contract counterparty interruptions
===================================

``simulate_contract_counterparties(counterparty_paths, contracts, cure_uniforms,
currency=...)`` draws one absorbing first-default event per scenario/counterparty
from contiguous physical survival-conditional PD periods and supplied uniforms.
Records explicitly contain scenario_id, counterparty_id, period_start,
period_end, conditional_pd, uniform in [0,1) and probability_measure='PHYSICAL'. The
first uniform below conditional PD triggers an event at the period endpoint,
before a payment at that endpoint. Initially all counterparties perform. The
caller supplies the conditional uniform law and cross-counterparty dependence;
contracts of the same counterparty always share its event.

Contracts declare contract_id, counterparty_id, currency,
default_payment_timing='BEFORE_DUE', cure_probability, cure_delay_days,
replacement_enabled, replacement_delay_days, replacement_payment_fraction and
cashflows (date, nonnegative amount). All payment dates belong to the supplied
assessment grid, including its initial date. Cure uniforms are separately supplied
for every scenario/contract pair through scenario_id, contract_id and uniform.

Previously paid cash remains paid. Payments during cure waiting are interrupted;
cure is not treated as free liquidity. A successful cure restores future payments
without inventing collections of missed amounts. Failed cure is followed by the
replacement delay; a replacement then pays the declared fraction of scheduled
cash and is assumed performing for the rest of this horizon. Disabled replacement
requires zero delay and fraction. Instant cure or replacement may preserve payment
while retaining the counterparty event. No independent borrower PD is added.

Outputs preserve separate counterparty events, contract transitions and payment
changes. Feed received contract cash into the appropriate operating revenue
component exactly once; assess borrower default through its own cashflow and
regulatory-reference rules. Detection times are explicit grid approximations,
not inferred intra-period economic times.
