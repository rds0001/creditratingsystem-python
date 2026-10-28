Microcredit dates, seasonal affordability and group support
==========================================================

Run ``examples/micro_variants.py`` or the installed R example
``run_micro_variants()``. Default inputs are analytical synthetic contracts;
callers can supply their own data and disable the example-specific assertions.
The three component cases are independent declared contracts, not a fitted
population or a claim that every microloan shares the same terms.

The individual weekly loan disburses 400, with zero interest explicitly supplied,
and four installments of 100 on January 8, 15, 22 and 29. Only January 8 is paid.
At January 20 the outstanding due amount is 100 and DPD is five. A monthly
bucket would conceal the actual January 15 contractual deadline.

The irregular-income example is a separate monthly contract. In each quarter it
receives net income 0, 0 and 900; each month has expenses 100 and debt service 100.
Residual cash is -200, -200 and 700 despite a positive annual total of 1,200.
Annual-average affordability cannot substitute for the dated liquidity profile.
No seasonal future income is treated as an actual repayment of the weekly loan.

The support graph contains group support relationships but no cross-default
agreement. An observed member A default therefore leaves member B and the group
outside that default group. Economic association alone does not merge legal
default units. The example maps supplied events through explicit default-group
assignments; it does not infer a legal agreement or fabricate a group event.

Two separate claims of 80 share one effective guarantee capped at 100, with
explicit priority allocating 80 and 20. It cannot guarantee both claims for 80.
Under the supplied borrower PD 0.10, guarantor PD 0.20 and joint default
probability 0.02, and zero payment on joint default, expected losses after the
payment guarantee are 1.6 and 6.4. Borrower PDs remain 0.10. These probabilities
are transparent scenario assumptions, not empirical estimates. Shared capacity
is accounted for once across both claims, before conditional payoff evaluation.

These references establish the variant accounting and timing behavior. They do
not replace origination fit, local policy approval, collections estimation or
complete lifecycle and portfolio validation.
