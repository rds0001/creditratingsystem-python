Small-business cashflows and legal units
======================================

The Python ``examples/small_business_variants.py`` and installed R
``run_small_business_variants()`` examples accept explicit local contracts. Their
reference inputs are independent synthetic component cases, not a business PD
model or a replacement for application and portfolio validation.

January sales of 200 less operating costs 60 and tax 10 leave operating cash 130.
After capex 20, cash before business debt service is 110; debt service 30 leaves
80. The owner takes 40, recorded as a business debit and matching household
credit. This internal transfer is not additional external income. Subtracting
separate household debt service 10 gives combined cash 70. Financing inflow 100
is retained separately and is not treated as operating earnings.

For the sole trader, the supplied legal-identity evidence says the owner record
and business accounting record represent the same legal person. Explicit mutual
identity edges therefore yield one default group. For the micro-company the
owner has only limited ownership: the supplied records remain two default units.
Ownership alone does not establish a personal guarantee or cross-default.
These conclusions depend on the input legal evidence, not on a company-name
heuristic or a universally assumed jurisdictional rule.

The revolving working-capital example supplies a separate prospective February
line calendar, after the January observed cash window. Opening drawn principal
20 increases by a declared 30 draw and decreases by a 10 repayment; the limit
changes from 100 to 120. Closing conditional EAD is 40 and undrawn commitment 80.
Interest is explicitly zero in this accounting oracle. A fixed installment
amortization schedule cannot reproduce those independent draw and limit events.
These future events are scenario assumptions already known at their snapshot,
not secretly supplied future observations.

A separately signed owner guarantee covers term and line claims of 100 each.
Its shared aggregate cap 150 and declared priority allocate 100 and 50, never
100 to both. Limited ownership and this optional payment guarantee remain
separate legal facts; guarantee coverage does not merge default units. Full
origination, lifecycle, collections and empirical risk-model acceptance require
additional linked evidence.
