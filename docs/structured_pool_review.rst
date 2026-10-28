Structured pool forecasts and observed servicing
===============================================

Run examples/structured_pool_review.py or run_structured_pool_review() after
sourcing examples/structured_pool_review.R. All inputs are local synthetic data.
The three cases use actual auto-loan collateral, mortgage collateral and trade
invoice contracts. They connect frozen pool paths, independent dated servicing,
ownership allocations and pool cash accounts using the same asset identifiers.

Separate 2020 cohorts estimate retail collateral PD=2/10, LGD=.65 and EAD=100,
known in 2021 before the 2023 holdout. No observed holdout outcome enters fitting.
Two explicit weighted paths represent default and survival; this is a controlled
finite scenario law, not an empirical distribution or confidence simulation.
The target is physical first default before terminal contract exit, conditional
on no default at index. No PIT/TTC interpretation is assigned automatically.

For the auto pool, forecast recovery is 40 conditional on default, hence 8
unconditionally at PD=.2. Independent servicing observes default EAD=100,
recovery=30 and direct costs=5: net recovery=25 and realized LGD=.75. Therefore
unconditional recovery bias is 8-30=-22; conditional LGD bias is .65-.75=-.10.
The mortgage prepays principal100 and interest5. The invoice dilutes10 then pays90;
dilution is not a borrower default. Retail household pooling is explicitly
inapplicable to the commercial invoice; asset-ledger diagnostics cover its risk.

An extra full ownership allocation is detected as double counting. Incomplete
observation coverage excludes the asset from default metrics. An unexplained
one-unit cash edit fails reconciliation. One realized asset per product cannot
establish discrimination, homogeneous pools, or empirical model suitability.
Cash reconciliation does not prove contractual tranche payment priority; that
is exercised by the separate securitization and covered-bond product examples.
