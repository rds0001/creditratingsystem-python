Retail arrears, repayment proposals and exit
==========================================

Run ``examples/retail_transitions.py`` after installation, or pass your own
versioned input object to ``run(data, verify_reference=False)``. In R, source
the installed ``examples/retail_transitions.R`` and call
``run_retail_transitions(data, verify_reference=FALSE)``. The packaged JSON
declares all monetary amounts, observation and knowledge dates, payment
obligations, realized receipts, remaining claims and proposed schedules.

Twelve product cases use different contractual obligations: mortgage instalment,
interest-only home-equity line, unsecured instalment, first post-grace education
payment, automobile balloon, finance-lease payment, card minimum, charge-card
full balance, demanded overdraft excess, BNPL instalment, weekly microcredit and
small-business working capital. These cases supplement the separate product
variant examples; they do not replace those variants or establish real-world
model performance. Operating-lease asset residual value is not a credit claim.

An explicit synthetic state profile maps the reconstructed arrears to a
delinquency bucket. It does not turn that bucket into default. A separately
confirmed default event takes precedence even at zero DPD; confirmed cure of
arrears alone does not override such an event. Materiality inputs here are
analytical example parameters and not a regulatory threshold recommendation.

The recorded principal and accrued interest already reflect earlier cash
allocations. For the automobile balloon, the earlier 5,000 receipt is explicitly
allocated to principal; 750 accrued interest remains. Remaining principal
arrears are part of principal, not an additional debt. Prospective interest is
explicitly zero for these arithmetic examples. Dates use actual elapsed days.

Every case reconstructs a 30-day late payment. A separate later receipt clears
the arrears; withholding its knowledge date correctly leaves the old arrears
visible. Ten days of observed clean probation establishes cure. An observed
account transfer/closure subsequently censors the redefault window. Closure
ends observation and does not assert that an unpaid claim was forgiven or fully
recovered. A separate defaulted control and a fully observed nondefaulted control
produce distinct targets. No default is inferred merely from ordinary arrears.

Repayment proposals do not amend contracts or default status. A proposal's
interest/principal allocation is reused in its conditional exposure schedule:
exposure immediately before minus exposure immediately after each scheduled
receipt equals that receipt, with interest paid only once. All proposal cash
flows are assumptions known at the proposal date, separate from later actual
cure receipts. Neither the proposal nor the historical cure proves causality.
The treatment evaluator applies its explicit 50 percent uptake assumption once
and checks affordability independently. Several catch-up proposals fail that
gate despite initially reducing payment amounts.

Observed roll rates retain the missing successor as censored. Thus one cured
account out of one observed successor gives conditional cure frequency one,
while the original risk set still contains two accounts. The Aalen-Johansen
example places half the original population in the exit state by day 60; it is
a descriptive estimate from its supplied observation history, not a forecast
available at the earlier landmark. The full results preserve cash ledgers,
knowledge controls, censoring and event deduplication evidence.
