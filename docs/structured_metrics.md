# Tranche event, loss and paired structure metrics

`estimate_tranche_credit_metrics(tranche_paths, target, discount_factors,
confidence_level=.95, context=None)` consumes a bound waterfall artifact.
Its explicit target contains `event` (`TIMELY_INTEREST`,
`LEGAL_MATURITY_PRINCIPAL` or `ULTIMATE_PRINCIPAL`), the full actual calendar-day
`horizon`, `time_unit='days'`, `probability_measure='PHYSICAL'` and the original
scenario `conditioning`. An origin-based positive discount factor is required
for every waterfall payment date. Targets and currency must agree with optional
context. Scenario weights are normalized for statistical aggregation.

The event probability is the weighted frequency of the selected payment
failure. It is never replaced by issuer PD. Principal loss is reported
independently: a zero-coupon tranche can lose capital without any timely-interest
failure. Outputs therefore include expected principal loss outside the selected
event and conditional principal loss given that event; the latter is null when
the event probability is zero. Ultimate loss is available only for complete
runoff. With open runoff, timely-interest/legal-maturity indicators remain
usable, but ultimate loss and discounted shortfall remain null; requesting an
ultimate event raises an error.

For each scenario, discounted contractual cash shortfall is the positive part
of principal promised at legal maturity plus coupons due on the actual opening
outstanding balances, less actual dated principal and interest receipts. The
signed amount and its components are also returned. This definition includes
payment timing; it permits early principal benefit to offset coupon timing
loss within the same scenario. It does not claim market fair value or spread
implied loss. Undiscounted ultimate principal loss is a separate output. For a
fully cured coupon 5 due at discount factor 1 but paid at factor 0.9, with all
principal paid at its legal maturity, timing shortfall is 0.5 even though
ultimate principal loss is zero.

Exact two-sided Clopper–Pearson event intervals and the sample-mean discounted
shortfall standard error apply only to declared IID equally weighted scenarios.
For a single zero-default trial the 95% interval is [0, 0.975]; zero observed
default is not certainty. One trial has no estimated loss MCSE. Fixed weighted
scenarios and dependent/QMC scenarios receive null IID intervals and MCSE.
The engine does not infer independence from distinct identifiers. The exact
binomial construction follows the native statistical definition documented in
[R stats::binom.test](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html).

`stress_tranche_structure(baseline_paths, alternatives, target,
discount_factors, confidence_level=.95, context=None)` compares already evaluated
waterfalls. Each alternative supplies `alternative_id`, `tranche_paths` and an
explicit `assumption`. Scenario IDs, order, weights, origin, currency and the
shared innovation hash must match exactly. Tranche identities must match;
changed notionals are disclosed alongside amount and rate deltas. Structures
may change or different default dependence mappings may transform the same
innovations. Individual uncertainty intervals are not paired-difference
intervals and no independent-sample interval is fabricated.

An independent two-debtor example demonstrates the distinction: two claims of
100 each have identical 50% marginal defaults and zero recovery. With exactly
one debtor defaulting per scenario, senior/junior notionals 180/20 lose 80/20.
With both debtors defaulting together in half the scenarios, expected losses
are 90/10. Total pool expected loss stays 100, while each tranche's event
probability changes from 1 to 0.5. Tranching and dependence affect the risk
distribution even when the pool mean is unchanged; the general mechanism is
discussed by [BIS](https://www.bis.org/publications/securitisations-tranching-concentrates-uncertainty).
These are supplied-assumption stress comparisons, not causal estimates or
optimization guarantees.
