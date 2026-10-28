# Sector cash rights, capital stress and claim priority

Python `run(data=None, verify_reference=True)` in
`examples/wholesale_sector_cashflows.py` and native R
`run_wholesale_sector_cashflows(data=NULL, verify_reference=TRUE)` accept the
schema in `wholesale_sector_cashflows.json`. Default execution covers eleven
variants with granular operating cashflows, liquidity waterfalls, contract
counterparty cashflows and recovery waterfalls. Reference verification controls
independent scalar cash and recovery assertions.

Bank funding outflows differ from nonbank wholesale maturities; no deposits or
bank capital ratio is invented for the nonbank. The holding receives only six
units of permitted subsidiary distributions: its 200 units of restricted
subsidiary capital cannot fund its own coupon. Its available cash after expenses
is five against eight due, leaving three unpaid. These are explicit cash rights,
not notional ownership support.

Life cashflow obligations split into surrender, guarantee and expense payments.
Nonlife uses claim development and expenses. The reinsurer separately receives
50 from a named retrocession counterparty in the performing scenario and zero
when that counterparty interrupts payment. Counterparty interruption does not
automatically classify the insurer itself as defaulted. Operational arrears and
financial-credit arrears remain distinct; neither is automatically a regulatory
or contractual default without a separate event definition and duration test.

The instantaneous asset-market/reserve stress bridge deducts each revaluation
loss once and each noncash reserve increase once. Life, nonlife and reinsurer
capital falls to 10, 105 and 98, respectively. This stock revaluation is separate
from the dated liquidity ledger: reserve strengthening is not also paid as a
cash expense, and unrealized asset losses are not separately deducted from cash.
It is not a complete projected balance sheet with all future earnings movements.

Region, municipality and agency receive different own revenues and transfers.
A 50% transfer cut changes only the transfer line. The resulting financial cash
shortfalls are 25, zero and 30; the agency also leaves 45 of operating bills
unpaid, for a total cash shortfall of 75. Ownership alone never supplies a guarantee.

Multilateral and supranational promises are financing, excluded from operating
CFADS. Their contracts include explicit trigger state, notice date, a minimum
31-day notice period, due dates, member identity and currency conversion. In the
call-authorized state, payments arrive only in June/July; no May cash is invented.
The untriggered state schedules zero, even if the members remain solvent.
Multilateral promises of EUR40 and USD30 at the declared contractual .9 EUR/USD
conversion yield EUR67 in total; the body's two promises yield EUR35. Fixed stress
states interrupt the first member or both members; joint stress produces zero
funding. Each promise is scheduled once, and no parallel parent-support PD benefit
is applied to these same payments. These are explicit stress scenarios, not an
estimated joint probability law or a currency-market quote.

Recovery waterfalls use separate, explicitly declared synthetic claim ranks and
unrestricted realization proceeds. They distinguish senior policyholder or other
prior claims from the financial-credit claim. The hand credit recovery is
`min(max(proceeds-prior_claim,0),credit_claim)`; legal priorities are assumptions
of these cases and do not purport to encode any jurisdiction's insolvency law.

Routing requires product, variant, borrower class, legal claim and economic
cashflow type together. Removing the legal claim produces no applicable route.
Full ledgers, capital bridge, contract paths, recovery results and independent
oracles are returned. All inputs are synthetic and open. This extends the sector
fundamental components with required economic contract cases; dated lifecycle and
portfolio acceptance remain separate work.
