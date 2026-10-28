# Contractual tranche waterfall

`run_tranche_waterfall(asset_pool_paths, structure_profile, context=None)` takes
a bound `AssetPoolPaths` and a versioned capital structure. Python and native R
return a bound `TrancheWaterfallPaths` with dated tranche cashflows, cash account
ledgers and separate ending-state payment/default measures.

The profile declares `structure_id`, `version`, `currency`, increasing
`payment_dates`, `tranches`, `fees`, `initial_reserve`, `reserve_target`,
`liquidity_limit`, `principal_mode` (`SEQUENTIAL` or `PRO_RATA`),
`principal_for_interest` (`ALWAYS`, `TRIGGER_ONLY` or `NEVER`), Boolean
`excess_interest_to_principal`, `trigger_seniority`, nonnegative
`oc_threshold` and `ic_threshold`, and `terminal_policy`
(`REQUIRE_COMPLETE` or `KEEP_OPEN`). Each tranche has unique `tranche_id`,
unique positive integer `seniority` (lower is senior), positive `notional`,
`legal_maturity` and a `coupon_fractions` mapping for every payment date.
Coupon fractions already incorporate contractual day count and scenario rate
assumptions. They multiply opening outstanding principal, with no interest on
arrears. Fractions after legal maturity must be zero; no penalty claim is
invented. Additional deal `fees` map every payment date and must exclude pool
servicing/workout costs already supplied. All legal maturities and the final
pool date must be payment dates within the pool horizon. Optional context uses
the actual calendar-day horizon and matching physical conditioning/currency.

Pool principal and gross recovery enter the principal cash account; pool
interest enters the interest account. Actual pool workout and servicing costs
become senior fee claims. Between payment dates the accounts accumulate cash.
At a payment date, additional fees are added and the following order applies:

1. Compute trigger ratios before support usage. OC is performing pool principal
   plus pending net modeled recovery, held pool reinvestment cash and waterfall
   cash/reserve, divided by outstanding protected tranches. IC is interest
   cash after senior fees, floored at zero, divided by protected current coupon
   and prior arrears. Protected tranches have seniority no greater than the
   declared trigger seniority. A zero denominator means no binding ratio and
   is returned as null. This OC valuation is a declared contractual proxy,
   not a market valuation or regulatory asset-eligibility conclusion.
2. Pay senior fees from interest, principal, reserve, then undrawn liquidity.
   Pay tranche coupon plus arrears in seniority order, from interest, optionally
   principal under the declared diversion rule, reserve and liquidity. Unpaid
   senior fees block tranche payments. Any remaining interest arrears block
   all tranche principal, an explicit cross-tranche payment restriction.
3. Repay liquidity from remaining interest then principal before debt principal.
   Liquidity is a senior repayable claim: a draw can fund interest and then be
   repaid from principal even when *direct* principal-to-interest diversion is
   prohibited. The facility does not create recovery or absorb credit loss.
4. Refill reserve from remaining interest up to its target. Transfer remaining
   interest to principal if declared; otherwise distribute it to equity.
5. Allocate available principal sequentially or pro rata to current balances.
   A breached trigger forces sequential allocation. Surplus principal reaches
   equity only after tranche principal, interest, fees and liquidity are paid.

Initial reserve is separately funded external support, not repeated pool cash.
Every date reports opening cash plus incoming cash plus liquidity draws less
fees, tranche payments, liquidity repayment, equity and closing cash. Reserve
movements are internal transfers. Cash conservation is checked independently
of credit loss attribution.

For a pool paying principal 40 then 60 and interest 5 then 3, tranches of 80
and 20 each with 5% coupon receive sequential principal 40/0 then 40/20 and
interest 4/1 then 2/1. Under pro rata they instead receive principal 32/8 then
48/12 and final interest 2.4/0.6. If the pool defaults and later recovers 40
with workout cost 5, zero-coupon tranches recover senior principal 35; ultimate
principal losses are 45 senior and 20 junior.

`REQUIRE_COMPLETE` rejects any scenario with outstanding pool assets, retained
reinvestment cash or pending recovery. On the final date it releases reserve,
permits final available cash to settle interest, prohibits new liquidity draws
and reports residual tranche principal as ultimate write-down. Final reserve
support is used once. `KEEP_OPEN` reports unpaid principal but returns null
ultimate loss and zero write-down. A timely-interest failure or legal-maturity
principal shortfall remains flagged after later cure; neither is automatically
an ultimate capital loss. Ending interest arrears and principal loss remain
separate. Full legal documentation, tax, swap waterfalls, post-maturity penalty
interest and discretionary managerial trades require explicit additional
contracts and are not inferred.
