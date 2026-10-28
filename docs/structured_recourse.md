# Covered-bond dual recourse

`covered_bond_dual_recourse(cover_pool, cover_pool_paths, bond_contract,
issuer_paths, context=None)` combines a typed mortgage or public-sector
`CreditAssetPool`, its matching `AssetPoolPaths`, and externally supplied issuer
and legal assumptions. Both languages enforce the source hashes and share the
same physical scenario IDs and innovation binding.

The economic distinction follows the [European Commission's covered-bond
explanation](https://finance.ec.europa.eu/banking/banking-regulation/covered-bonds_en):
bondholders have issuer recourse and, after issuer failure, preferential access
to cover assets plus a residual ordinary estate claim. The function's precise
timing, segregation and extension rules below are supplied modeling conventions,
not a determination of national insolvency law or regulatory eligibility.

The bond contract supplies `bond_id`, `version`, `currency`, `legal_profile_id`,
positive `principal`, `original_maturity`, explicit nullable `extended_maturity`,
dated `interest_schedule` and `extension_interest_schedule`, `extension_rule`
(`NONE` or `ISSUER_DEFAULT_AND_DECLARED_TRIGGER`),
`cover_cash_convention='SEGREGATED_UNTIL_ISSUER_DEFAULT'`,
`estate_rule='RESIDUAL_AFTER_COMPLETE_COVER'`, and `innovation_hash` matching the
pool paths. Original coupons fall after origin through original maturity;
extension coupons fall strictly later through the extended maturity.

Each issuer scenario supplies `scenario_id`, explicit nullable
`issuer_default_date`, dated `issuer_capacity`, Boolean `extension_triggered`,
`estate_payment_date`, `estate_recovery_fraction` in [0,1] and nonnegative
`estate_capacity`. Capacity is available on the stated date, not an automatically
carried issuer cash account. Default at origin is supported; dates before origin
are rejected. Issuer default applies before that day's issuer payment.
Extension activation requires the declared rule and issuer default by original
maturity. It moves principal to extended maturity and adds only the explicitly
supplied extension coupons. Inactive extension adds no coupons.

Cover principal, interest and recoveries, net of dated pool servicing/workout
costs, accumulate in a segregated account. Before issuer default only issuer
capacity pays due bond claims. After default cover cash pays due claims; interest
ranks before principal. Residual estate settlement occurs only after the full
cover cash calendar and all possible bond maturity dates. Incomplete cover
runoff is rejected. Estate payment is the remaining claim times the supplied
recovery fraction, capped by estate capacity and the remaining claim. A fully
covered claim obtains no additional estate gain. Unused cover cash and unfunded
cover costs remain separate outputs.

For a claim of 100 and net cover recovery 35, a 50% residual estate recovery
pays 32.5 and leaves shortfall 32.5. Estate capacity 10 instead leaves shortfall
55. Full cover of 108 pays only 100 and leaves 8 unused. If a public-sector cover
pool supplies 108 and an activated extension adds coupon 5, the total capped
payment is 105, with 3 remaining. The original-date principal deferral is 100,
while effective-maturity shortfall can still be zero.

Outputs include source-separated dated payments, cover-account conservation,
contractual claim reconciliation, original-date deferral/shortfall,
effective-maturity principal shortfall, timely-interest failure and ultimate
principal/interest shortfalls. A solvent issuer paying late can have a maturity
shortfall that subsequently cures. No penalty coupon, automatic issuer recovery
uplift, pre-default cover distribution or competing early estate recovery is
inferred. Optional context must match currency, physical conditioning and the
actual final settlement horizon in days.
