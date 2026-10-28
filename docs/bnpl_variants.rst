Purchase plans, refunds and observation horizons
================================================

The Python ``examples/bnpl_variants.py`` and installed R
``run_bnpl_variants()`` examples take user-supplied data or the packaged
analytical synthetic contracts. The examples expose component behavior;
application calibration and full lifecycle evidence are separate.

A pay-in-four plan of 100 has weekly installments of 25. After the first 25 is
paid, 75 remains outstanding; on January 10 its unpaid January 8 installment is
25 overdue at DPD two. A pay-later contract for the same 100 is due January 22:
on January 10 its exposure is 100 but DPD is zero. The date and amount of the
contractual obligation determine the difference.

Two overlapping 100 plans each pay their first 25, and an effective merchant
refund reduces one future installment by 10. Total remaining debt is therefore
200 - 50 - 10 = 140, with 65 due in the next ten days at the example cutoff.
Plan identities and explicit installment allocations prevent double counting.
A full, effective return reducing a 100 obligation to zero creates no overdue
amount; a dispute or refund request alone would not cancel that debt.

The DPD adapter consumes remaining installment obligations after the BNPL ledger
has allocated payments and effective returns. It deliberately passes no further
payment credit to the DPD function, preventing a second subtraction. This is a
current net-obligation snapshot, not a reconstruction of pre-refund daily DPD.
Historical DPD must be recomputed with the event knowledge available at each
historical cutoff. No default flag is inferred from merchant-return status.

The completed short plan pays all four installments and closes on January 22.
Its active count and remaining debt are zero at January 25. A separately defined
20-day first-default observation ends January 21 and is fully observed with no
default. The annual first-default observation of that same facility is censored
at its actual January 22 exit and has no binary annual label. There are no
invented post-closure active months or unobserved annual good labels.
