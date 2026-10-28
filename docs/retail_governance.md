# Retail application, account review and assessment history

Run `examples/retail_governance.py` or the installed R `run_retail_governance()` example. Both accept the packaged data object, an optional product-ID selection and `verify_reference=False` for replacement data. The example has an explicit calendar: application January 1, performing assessment January 31 and review February 28, 2025; mature account monitoring ends January 31, 2026. Supplied features, payment receipts, statement balances, default events and coverage are inputs. The underlying public APIs accept bank-specific dates and records independently of this example calendar. Keep IDs, targets, dates and frozen model contracts consistent when replacing data.

The twelve cases use distinct contractual capacity calculations. Each budget and its debt service use the same declared frequency; annual, monthly and weekly amounts are never mixed.

| Product | Contract checked | Independent residual-income reference |
|---|---|---|
| Residential mortgage | Annual interest 3% on 120,000 plus principal 6,000; fixed declining principal means the first annual payment is the largest | 6,000 annually |
| Home equity line | Drawn 40,000 at 6%, two interest-only months and full repayment in month three | 1,600; 1,600; -38,400 monthly |
| Unsecured installment | New payment 350 while old payments 300 and 200 remain open at the cutoff | 250 monthly |
| Education | Two grace months followed by a payable 600 installment; repayment income and expenses explicitly supplied | 500; 500; 600 monthly |
| Auto loan | Principal repayments 5,000 and 15,000 with interest on the remaining balance, giving payments 6,000 and 15,750 | 7,000; -2,750 annually |
| Operating auto lease | Rent 300; the lessor's residual asset value 12,000 is not a customer payment obligation | 200 monthly |
| Credit card | Minimum payment max(25, 5% of 1,000) = 50 | 350 monthly |
| Charge card | Full statement balance 1,000; no revolving minimum substituted | -600 monthly |
| Overdraft | Explicit known demand for 800, not an assumed amortizing repayment | -300 monthly |
| BNPL | Parallel plans aggregate to weekly dues 100, 100 and 50 | -50; -50; 0 weekly |
| Micro loan | Weekly dues 100 with seasonal income 0, 0, 900 and 0 | -200; -200; 700; -200 weekly |
| Small business finance | Owner draw 40 plus other household income 100, private expenses 100, existing private service 10 and new service 15 | 15 monthly |

The business case separately reconciles sales 200 minus operating costs 60, tax 10, capex 20, business debt service 30 and owner draw 40 to retained business cash 40. Business sales do not become private income, and the owner transfer is not counted twice. Consolidation gives no premature benefit for a settlement known only after the application. The mortgage example covers the largest fixed-rate annual payment; floating-rate, collateral, guarantee and other product variants retain their separate companion examples. This is an explicit synthetic contract set, not advice on loan terms or regulatory underwriting limits.

Capacity is separate from PD. A supplied policy requires nonnegative residual income in every assessed period and verified income and terms. Missing evidence prevents approval. Credit and charge cards have the same illustrative PD but different decisions because their payment obligations differ. A positive average microloan surplus does not erase individual liquidity deficits. No guarantee, asset sale or assumed refinancing is silently treated as household cash.

The frozen application model was fitted to prior-2024 synthetic frequency counts: 10 defaults among 500 clean observations and 50 among 500 arrears observations, yielding PDs 0.02 and 0.10. It explicitly declares a pooled population covering all twelve products and a common conditional arrears effect. That declaration is an illustrative assumption, not evidence that products share empirical coefficients. The independent behavioral model has 5 defaults among 500 observations at zero DPD and 10 among 100 at 30 DPD. The complete training counts and fitted models are packaged; normal examples do not refit. Application risk is conditional on funding; account risk is conditional on being active. Neither model uses household affordability as a substitute for PD.

A separately frozen, product-stratified pool artifact assigns application and performing records to supported score bands. Its 365-day performing-facility target and very small historical training rows are explicit. Assignment supplies a segmentation label only: it does not replace either application or behavioral PD with a pool estimate, and it does not equate their conditioning populations.

The already-funded historical account is distinct from the current application, including when that application is declined. Its supplied contractual due is the product-specific reference payment above; its observed half-payment produces 28 DPD at the February month end. The statement's DPD and overdue amount must reconcile to the dated payment ledger. The account's illustrative behavioral PD at 28 DPD is `1 / [1 + 99 / 11^(28/30)]`. A bureau release known January 2 cannot change the January 1 application snapshot. Unknown history is not imputed as clean history.

Each case constructs typed subjects, routing, applicability, a frozen application batch, a composite assessment, a performing assessment, pool assignments and current/stale checks. The payment-delay event marks the relevant dependency. A separate restructuring proposal remains proposal-only and changes neither the contract nor default status. The later exit supersedes the reviewed historical account assessment. Registering a new application model does not alter pinned application or behavioral replay.

Monitoring includes only the supplied already-funded account. Its June default is a dated observed event, not a result inferred from the decision or simulated acceptance. The twelve-month January 31 anniversary is explicitly 365 days here. One mature account yields `INSUFFICIENT_INFORMATION`; it is not a validation claim. Feature drift uses a complete four-row permutation reference. Sequential calibration drift uses a deliberately degenerate zero residual reference solely to exercise event/statistic plumbing. Its alarm has no market-calibrated significance and must not be used as a deployment threshold. Real monitoring requires an independently justified dependence/null design and a sufficiently large cohort.

Default package tests run twelve inexpensive payment-budget oracles and two complete card/charge journeys, plus chronology and double-counting rejection cases. The separate full installed example executes all twelve products, with strict Python/native-R output and artifact-hash parity.
