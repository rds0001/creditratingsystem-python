Structured position lifecycle and observed collateral
====================================================

Run examples/structured_journey.py or source examples/structured_journey.R and
call run_structured_journey(). The example reuses the separate governance,
lifecycle, workout and collateral-validation functions with new 2024 ledgers.
The local CSV source supplies three position identities and balances totaling
280 EUR. A later source release is present but excluded by the historical join.
The input contract for this analytical monitor requires x=0; it is a controlled
integration case, not a fitted credit model suitable for deployment.

Each position is observed performing, then has a missed collateral principal
payment, a restructuring proposal, a confirmed default, a closed workout and
assessment retirement. The proposal does not change the contract or erase the
default. Separate historical transition cohorts estimate conditional curves.
The same position receives a January15 forecast and its February1 default in
monitoring; its Brier score is .25. Separate larger illustrative monitoring
cohorts remain labeled component evidence, not outcomes of this position.

Actual settlement controls are independent of forecast scenario paths:

* An 80 EUR senior claim and20 junior claim share25 collateral recovery. The
  explicit zero-coupon senior-first settlement allocates25 to senior and0 to
  junior, giving senior loss55.
* A 100 EUR covered-bond claim receives35 from its cover pool. The residual65
  claim receives50% estate recovery,32.5, giving aggregate67.5 and loss32.5.
  The35 cover recovery is counted once. Estate proceeds are not asset-pool cash.
* A 100 EUR nonrecourse commercial invoice receives debtor payments40 and30,
  giving loss30. No seller guarantee or dilution event is invented.

Retail collateral models use separate, completed historical training cohorts.
The30day position-payment monitor and121day collateral-default forecast have
separate populations and targets. There is no automatic PIT/TTC, horizon or
probability translation between them. A household risk pool does not apply to
commercial invoices; their servicing and ownership allocation are audited.

All component reports are actually recalculated and exactly compared within
each runtime. Replay checkpoints bind model, explicit identity calibration,
policy, source ledger and operational state. Diagnostic confidence intervals
remain in the full report and are compared numerically across languages; they
are not operational decision-state fields. Unrounded fitted parameters remain
in component artifacts. Altering declared covered-bond recovery to100 fails
reconciliation with the independent collateral and estate records.

The2023 economic product examples are separate supplemental contract references,
not the same claims renamed to imply continuity. This example provides three
controlled synthetic histories; it does not establish empirical discrimination,
regulatory approval or suitability for a bank's own data.
