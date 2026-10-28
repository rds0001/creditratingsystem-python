Credit and charge card contracts
===============================

Run ``examples/card_variants.py`` or source ``card_variants.R`` from the R
package's installed examples and call ``run_card_variants()``. Both accept an
explicit data object and allow ``verify_reference=False`` / ``FALSE`` for user
contracts. The default JSON data are small analytical synthetic cases, not
observed customer performance or calibrated population estimates.

The credit-card transactor pays all 1,000 of the statement; the revolver pays the
contractual minimum 50. Both have zero DPD, but full-payer and minimum-payer
features differ, and the remaining drawn exposures are 0 and 950. These examples
set interest to zero explicitly to isolate payment accounting. DPD obligations
for these credit-card contracts are the contractual minimum, not the full
statement balance. Voluntary excess payments are separate from the amount due.

The new account has an observed active zero balance and no established payment
history. Missing coverage stays missing; it is not substituted by a good-payment
record. A separately declared future draw of 200 produces exposure 200. This is
a contractual scenario, not a fitted forecast. A limit cut from 2,000 to 1,000
with unchanged balance 1,000 moves utilization from 50 to 100 percent; the ledger
attributes the change to the limit and does not invent a new purchase.

The monthly charge-card obligation is the full 1,000. Paying only 50 leaves
950 due and, two days after the due date, DPD two. Applying a credit-card minimum
to that contract would be incorrect. The deferred charge contract explicitly
places the due date one month later: before that date unpaid principal remains
1,000 but there is no past-due obligation. These are distinct contracts, not
alternative labels applied to the same ledger.

A separate authorization ledger follows a 200 reservation into a 200 purchase.
Total exposure remains 200, with no additional reservation after settlement.
An expired, unsettled reservation has zero exposure. These separate analytical
reservation cases are not added to the 1,000 statement examples. The granular
``reconcile_card_authorizations`` documentation defines the event, timing and
minor-unit rules. Payments reducing a statement use the account cash ledger;
authorization reversals represent purchase reversals, not arbitrary repayments.

The examples cover the stated variant components and cash bridges. Application
calibration, model lifecycle, collections and portfolio monitoring require their
own linked inputs and evidence; these small ledgers do not establish those
capabilities or empirical model validity.
