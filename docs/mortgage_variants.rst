Mortgage product components
===========================

Run ``python examples/mortgage_variants.py`` after installation. In R, run
``source(system.file("examples/mortgage_variants.R", package="creditratingsystem"))``
and then ``run_mortgage_variants()``. Both examples read the same packaged JSON
contract and perform independent analytical reference checks. Pass a replacement
mapping/list as ``data`` and disable ``verify_reference`` for your own inputs.
Inputs remain local; no external service or fitted proprietary parameters are used.

Seven dimensions have separate evidence: fixed rate, floating rate, interest only,
partial amortization with a balloon, joint borrowers, owner occupancy and rental
use. Rate and repayment dimensions can be combined when the supplied debt contract
is consistent. Usage does not determine contractual amortization or borrower PD.

The annual loan starts at EUR 100,000. At 3%, equal principal repayment gives
payments of 53,000 and 51,500. A two percentage point reset before the second period
raises the floating payment to 52,500. Fixed contractual payments remain unchanged.
A bullet pays 3,000 then 103,000; partial amortization of 20,000 leaves 80,000 and a
final payment of 82,400. These figures use ACT/365F over two non-leap annual periods.

Monthly rent of 1,000 less fixed property costs of 200 gives available property
cash of 800; 20% vacancy reduces it to 600. Owner occupancy generates no rental
cash. Household salary is a different income source. Two borrowers contribute
3,000 and 2,000 monthly salary; their shared 900 obligation is counted once while
both liability records remain present.

A separate default scenario has a 20,000 prior lien and 100,000 mortgage against
150,000 collateral, giving combined LTV 0.8. A single sale for 100,000 incurs costs
of 10,000. Priority allocation pays 20,000 to the prior lien and 70,000 to the
mortgage. A supplied discount factor of 0.9 gives mortgage recovery PV 63,000,
economic loss 37,000 and LGD 0.37. This separate scenario does not imply that the
fully repaid example loan defaults later. Legal priority, sale proceeds and discount
factors are declared scenario assumptions.

These are product component calculations. Empirical PD estimation, lifecycle
validation and suitability for a bank's portfolio require their own dated data and
evidence. The example is not a full mortgage underwriting or regulatory approval.

Dated lifecycle example
-----------------------

``mortgage_lifecycle.py`` and installed ``mortgage_lifecycle.R`` use their own
packaged mortgage data and rerun the shared application and account workflow.
Separate synthetic training/calibration counts identify application PDs 3%/12%
and account PDs 1%/10%. These values are analytical inputs to an integration
example, not population estimates. Only a funded application receives an observed
loan outcome; rejected and untaken offers keep unknown hypothetical outcomes.

A 900 payment due on August 1 remains unpaid on August 31, producing 30 DPD.
A temporary repayment proposal reduces payments from 3,000 to 1,000 against 1,200
available household cash. Period-one interest of 789 and fees of 100 leave principal
94,389 after payment; period-two interest of 188.778 leaves 93,577.778. Principal
arrears are already within principal, not added again. The proposal changes neither
the actual contract nor default status and supplies no causal cure estimate.

The separately observed default exposure is 95,000. Realized allocated proceeds
70,000 discounted by 0.9 yield closed-workout loss 32,000 and LGD 32/95. The same
cashflows in an open workout yield only loss to date, with final LGD unidentified.
A later closure removes the account from the performing risk set. Frozen historical
model replay remains reproducible after a later synthetic model promotion.
The tiny monitoring cohort is reported as insufficient information, not as a
statistically validated mortgage model.

Evidence reconstruction
-----------------------

``mortgage_evidence`` reconstructs one shared mortgage in a dated household graph:
two full legal liabilities remain, but household cash service is counted once.
The coverage declaration is limited to the named bank source; a separate bureau
report contains another 5,000 loan. A claimed monthly salary of 3,300 is reconciled
to documented and transaction-supported 3,000. It is never added three times.
Adding the second borrower's 2,000 gives normalized income 5,000. The declared
living-cost floor raises 2,200 to 2,300, with mortgage payments kept as debt service.
The dated payment history reproduces 30 DPD and a local delinquency bucket.
These tiny reference assumptions are exposed in JSON so each input can be replaced.

Forward conditional loss
------------------------

``mortgage_risk`` supplies a separate forward amortizing scenario. Scheduled
interest payments and 5,000 annual principal reductions leave conditional exposures
95,000 and 90,000 after the period-end events. An intercept-only fractional LGD
fit uses four synthetic closed-workout fractions 0.1, 0.2, 0.4, 0.5, identifying
mean severity 0.3. Conditional annual PD assumptions 0.03 and 0.04 imply marginal
first-default masses 0.03 and 0.0388. With discount factors 0.97 and 0.94 the two-year
expected loss is 1,814.094. Deterministic exposure permits the conditional severity
product here; general dependence requires joint loss paths. The second-year PD is
an explicit scenario assumption, not inferred from the one-year application model.

The separate remaining-three-month bullet stress applies salary -20%, expenses
+10%, collateral -20% and rate +2 percentage points together. Interest is 250 before
the reset and 416.6667 thereafter. The final balloon exhausts the supplied liquid
buffer; no successful refinancing is presumed. The expense floor and every stress
mapping are explicit synthetic conventions, not regulatory thresholds.
