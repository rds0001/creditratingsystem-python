# Structured asset-pool foundations

A structured asset pool is a set of assigned contractual claims, not a retail
rating pool. These methods preserve asset identity, debtor/default-group
relationships, dated ownership, contractual payments, currency and explicit
eligibility assumptions. They do not determine legal enforceability or statutory
securitization/covered-bond eligibility.

## Build a dated credit asset pool

`build_credit_asset_pool(asset_contracts, pool_assignments, eligibility_profile,
*, pool_id, as_of, information_cutoff, context=None)`

The profile is versioned (`profile_id`, `version`, `known_at`, `currency`). It
must be known by `information_cutoff`, which cannot follow `as_of`. It adds
`product_kind`, `pool_mode` (STATIC or REVOLVING), explicit nullable
`revolving_until`, `allowed_asset_classes`, integer `maximum_dpd`, Boolean
`exclude_disputed` and `eligible_recourse` (NONE and/or SELLER).

Product routes have concrete compatible claim classes:

| Product kind | Eligible claim classes before additional profile restrictions |
|---|---|
| ABS | CONSUMER_LOAN, AUTO_LOAN, LEASE |
| RMBS | RESIDENTIAL_MORTGAGE |
| CMBS | COMMERCIAL_MORTGAGE |
| CLO | CORPORATE_LOAN |
| MORTGAGE_COVER | RESIDENTIAL_MORTGAGE, COMMERCIAL_MORTGAGE |
| PUBLIC_SECTOR_COVER | PUBLIC_SECTOR_CLAIM |
| TRADE_RECEIVABLES | INVOICE |

A mismatched profile route is rejected. Individual claims outside the declared
profile class have an eligibility exclusion. These routes do not themselves
supply product PDs, recoveries or legal permissions. Loan schedules, including
amortization, balloon and bullet structures, remain explicit contractual inputs.
STATIC cannot declare a reinvestment end. REVOLVING needs an exclusive future
end date; acquisitions at that endpoint are outside the window.

The tape contains exactly one local snapshot per `asset_id`, with:

- `asset_class`, explicit nullable `invoice_id`, `obligor_id`, `default_group_id`,
  `seller_id`, `servicer_id`, `currency`;
- `origination_date`, `observed_at`, `known_at`, `balance_date`;
- nonnegative `principal_balance`, `schedule`, integer `overdue_days`, Boolean
  `disputed`, `dilution_reserve`, `recourse`, `collateral_allocations`.

Origination precedes observation, which precedes knowledge; the balance date
cannot follow observation. Only cutoff-known observations are considered. Current
pool balances must be dated exactly `as_of`; stale balances are not rolled forward
by assuming scheduled payments actually occurred. A future-originated claim cannot
appear in the historical initial pool. The caller supplies the intended latest
local tape; duplicate asset IDs are rejected rather than choosing an unseen
revision policy.

Invoice claims need a unique invoice identity; duplicate invoices cannot be
financed through different asset IDs. Other asset types cannot carry invoice IDs.
One obligor uses one explicit default group. Several obligors may share a declared
group, but one obligor cannot be split into independent groups. Seller, servicer,
borrower and collateral references are distinct roles, not automatic default
contagion edges.

Each schedule row has a strictly increasing `date`, nonnegative `principal_due`
and `interest_due`, after the balance date. Total scheduled principal must equal
the observed balance within numerical tolerance. No residual principal or assumed
refinancing is invented. Dilution reserve lies between zero and balance and is
kept separate from the gross contractual schedule. Each collateral allocation
has `collateral_id`, `allocation_fraction`; per-collateral shares summed over the
supplied visible tape cannot exceed one. This supports explicitly allocated
collateral, not an unstated priority waterfall on multiply pledged whole values.

Assignments contain `assignment_id`, `pool_id`, `asset_id`, `known_at`,
`effective_from`, explicit nullable `effective_until`, positive `share` at most
one, nonnegative `purchase_price` and integer `purchase_priority`. Validity is
half-open. All cutoff-known assignments across **all supplied pools** are checked
at interval boundaries: simultaneous transfer shares of one asset cannot exceed
one. A transfer ending on the next transfer's start date does not overlap.

Only eligible assignments active at `as_of` contribute to `reference_principal`
(share times current face balance). Future eligible assignments are separate
`reinvestment_candidates` for REVOLVING pools only, strictly before the exclusive
reinvestment end. They are pre-existing observed warehouse claims with declared
future purchase terms, not future-originated claims inserted into the initial
pool. Expired, unavailable, stale, disputed, aged, incompatible and static-pool
future assignments have explicit exclusion reasons. The canonical CreditAssetPool
artifact retains the tape, active assignments, pipeline, profile, input hash and
content binding for subsequent scenario replay.

## Static tranche loss

`tranche_loss_from_pool(pool_loss, attachment, detachment, *, pool_notional,
loss_unit='fraction', context=None)`

For nonnegative pool loss fraction L and 0<=A<D<=1, tranche loss fraction is
`min(max(L-A,0),D-A)/(D-A)`. Tranche notional is pool notional times `(D-A)`;
absolute loss multiplies that tranche notional, not the full pool notional.
`loss_unit='amount'` first divides by a positive reference pool notional.
Scalar or one-dimensional losses are accepted; output arrays preserve their
order. Negative loss inputs, invalid A/D, zero notional and nonrepresentable
normalizations are rejected. Pool losses above notional can represent costs;
contractual tranche loss remains capped at tranche notional.

At L=.15, A=.10, D=.20 and pool100, tranche notional is10, loss fraction.5 and
loss amount5. This is a static sequential capital-loss formula, not a timed
interest-payment test, tranche PD or a replacement for the dynamic waterfall.
The role of attachment/detachment and concentrated nonlinear uncertainty is
illustrated by the [BIS tranching analysis](https://www.bis.org/publications/securitisations-tranching-concentrates-uncertainty)
(reviewed 2026-10-13). No regulatory capital model is implemented by this formula.

## Receivables borrowing base

`receivables_borrowing_base(invoices, borrowing_profile, *, as_of,
information_cutoff, context=None)`

The versioned common profile adds integer `maximum_overdue_days`, Boolean
`exclude_disputed`, `allowed_recourse`, `advance_rates` mapping NONE/SELLER to
fractions, `default_group_cap`, optional overrides in `group_caps`,
`additional_reserve`, `reserve_scope='ADDITIONAL_NOT_IN_INVOICE_DILUTION'` and
`coverage_status` (COMPLETE, PARTIAL or UNKNOWN). The recourse labels describe
contractual rights; seller recourse is not treated as risk-free.

Invoice rows identify `invoice_id`, `debtor_id`, `debtor_group_id`, `seller_id`,
`servicer_id`, `currency`, `issue_date`, `due_date`, `balance_date`, `known_at`,
`outstanding`, `disputed`, `dilution_reserve`, and `recourse`. Duplicate invoices
and splitting one debtor among concentration groups are rejected. Overdue days
are `max(as_of-due_date,0)`. Balances must match the snapshot date and knowledge
cutoff. Dispute, aging, currency, recourse, unknown and stale-date exclusions
remain separate.

Eligible face value first subtracts that invoice's dilution reserve. The
concentration denominator is the **fixed total of these eligible net faces before
any concentration caps**. Each group's retained amount is the lesser of its
eligible total and its cap fraction times that fixed denominator. Retention is
allocated proportionally to that group's invoices. This declared convention does
not claim that the retained group's share of the smaller post-cap pool remains
below the same percentage; no hidden endogenous fixed-point problem is solved.

After concentration, apply each invoice's recourse-specific advance rate. Sum
advances, subtract the additional nonoverlapping reserve and floor at zero.
For net invoices80+100 in groupG and100 in groupH, a50% fixed-denominator cap
retains140 ofG and100 ofH. An80% advance gives192; additional reserve5 yields187.

COMPLETE coverage returns the total borrowing base. PARTIAL/UNKNOWN returns a
known subtotal and null total, rather than treating missing receivables as zero.
The result includes per-invoice exclusions, dilution, cap deductions, group
totals, applied reserve and a canonical content hash. No actual account limit
or collateral transfer is changed.
