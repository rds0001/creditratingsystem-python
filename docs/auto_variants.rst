Auto loan variants
==================

Run ``auto_variants.py`` or installed ``run_auto_variants()`` in R. A 20,000 loan
at 5% has either equal annual principal repayments of 10,000 or a 5,000 first
repayment followed by a 15,000 balloon. Default immediately before the final
payment therefore exposes 10,500 or 15,750, including accrued contractual interest.
All default timing and valuation dates are explicit inputs.

A physical car value starts at 20,000, ages linearly over ten years and receives
an explicit two-year market index of 0.625, giving value 10,000. Repossession and
sale costs of 1,000 reduce net recovery to 9,000. Balloon loss is 6,750; with zero
costs it is 5,750. The same proceeds are allocated only once against the claim.

A declared net realization floor of 12,000 has a capped dealer guarantee of 3,000.
Payment is limited to the floor gap, remaining loss and allocated cap. Borrower PD
0.1, dealer PD 0.2 and supplied joint-default mass 0.02 give protected expected
loss 0.08 x3,750 +0.02 x6,750 =435. An ineffective guarantee leaves expected loss
675. The joint-default state retains the unpaid guarantee loss; no dealer promise
is treated as risk-free. A separate complete four-state path integration reproduces
435. These synthetic contract scenarios are not empirical estimates or legal
interpretations; bank data, calibrated states and legal evidence remain explicit.
