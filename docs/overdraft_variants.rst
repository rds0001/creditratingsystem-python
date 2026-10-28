Overdraft contracts and observed salary cashflows
================================================

``examples/overdraft_variants.py`` and the installed R example
``run_overdraft_variants()`` accept explicit local input data. Their default
resource contains three synthetic contract variants with independent cash and
payment oracles. They cover component behavior, not full model validation.

An authorized line starts with drawn balance 800 and pays an expense of 100:
closing exposure is 900 within a 1,000 limit. The contractual repayment date is
later, so this draw creates no installment arrears. A separate unauthorized
excess case pays expenses of 400 and reaches 1,200. Its supplied demand contract
makes the excess 200 due on January 2; on January 3 it has 200 overdue and DPD one.
The excess does not create this deadline automatically: the input contains it.

The salary-linked pair starts with the same 800 drawn balance and the same 400
expense. Actual salary of 1,000 reduces closing exposure to 200. If that salary
was not received, exposure remains 1,200; the example never inserts an expected
salary as an actual receipt. Both contracts in this comparison allow 1,500 and
have a future repayment date, so missing salary alone does not create DPD or a
default event. Cash pressure and contractual delinquency remain separate.

Transaction features, supplied daily balances, positive bank exposure snapshots
and the signed cash bridge reconcile. These cases have one account, complete
capture, no FX, transfers or noncash postings. Other situations require the
appropriate explicit reconciliation inputs rather than reuse of this example's
simple cash identity. The obligations are contractual demands, not forecasts.

Two separately dated historical default episodes illustrate CCF: (1,200-800) /
(1,500-800) = 4/7 when reference undrawn is positive; when reference drawn 1,200
already exceeds limit 1,000, no positive denominator exists and CCF is undefined.
It is not replaced with zero. These historical default episodes are not default
labels assigned to the current, performing salary scenarios.
