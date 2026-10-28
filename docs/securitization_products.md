# Securitization product contracts

Run `python examples/securitization_products.py` after installing the package.
The R equivalent is `run_securitization_products()` from the installed
`examples/securitization_products.R`. Both load the same local synthetic
`securitization_products.json`. Pass a replacement data object to `run(data,
verify_reference=False)` or the corresponding R function; reference assertions
apply only to the supplied analytical inputs.

Six separately defined cases exercise actual pool and tranche cashflows:

| Case | Contractual behavior | Independent first-period principal oracle |
|---|---|---:|
| ABS | Auto-loan claim prepays after coupon | 100 |
| RMBS | Residential mortgage prepays after coupon | 100 |
| CMBS | Commercial mortgage amortizes 20 and leaves balloon 80 | 20 |
| CLO | Corporate-loan principal finances a dated warehouse purchase | 0 distributed |
| Revolving pool | Principal purchases eligible assets and retains unused cash | 0 distributed |
| Static pool | Amortizes without new acquisitions | 40 |

Both revolving cases purchase a remaining claim of 50 for 50 and retain another
50 until the revolving endpoint. Warehouse payments before acquisition never
belong to the pool. Each case constructs its own typed eligibility contract,
simulates assets, allocates the tranche waterfall, computes ultimate-principal
risk, compares a shared-innovation debtor-default stress, and composes a dated
senior-issue assessment. The base cases repay all principal. The separate credit
stress has recovery 40, workout cost 5 and unpaid contractual coupons 10,
leaving aggregate tranche principal loss 75. Static attachment boundaries
independently give tranche loss fractions 0, 0.5 and 1.

A subsequent account-bank example takes cash already collected in the first
modeled period and holds it for later remittance. It uses that later origin;
future unearned asset payments are not classified as held cash. A separate
50% account-bank recovery stress loses half the held amount. It is reported as
a distinct conditional scenario and is not automatically added to the
origination tranche PD or loss.

Each product case also executes a deliberately invalid contract and checks its
specific controlled error: incompatible asset eligibility, inconsistent
balloon principal, an expired reinvestment window or overlapping assignments.
These are small deterministic reference examples, not calibrated market
frequencies or institutional ratings. The examples demonstrate product and
recipe mechanics; the complete longitudinal governance, monitoring and workout
workflow is documented separately when available. No lifecycle completeness
is inferred from merely changing a product label.
