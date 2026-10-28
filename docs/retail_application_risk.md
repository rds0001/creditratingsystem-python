# Prior-known loss forecasts for the six application accounts

`examples/retail_application_risk.py` and the native R example `run_retail_application_risk()` supplement the original six application journeys. Each product uses its own `DEFAULT_THEN_EXIT` application, customer, household and proposed facility identifiers. That descriptive identifier selects an illustrated account; its subsequently observed default or settlement is never a fitting input. These forecasts are illustrative synthetic forecasts, not independent predictive validation.

The original product-specific training/calibration cohorts supply the application PD. The optional `application_chain` argument reuses the unmodified raw original journey; otherwise only the original prior fitting helper and dated application snapshot are run. An independent two-cell binomial calibration oracle checks the fitted application probability. Each product has separate explicitly synthetic 2022 closed recovery episodes: realized, known cash is discounted first, closed workout LGD is calculated, and a genuine intercept-only fractional model estimates its mean. The two observations illustrate arithmetic and conditional fitting; they do not establish population precision. Live 2025 recovery outcomes never enter this estimator.

The frozen exposure scenario assumes funding on 2024-12-31 and no interim principal collection before default in the following 30 days. Contractual obligations are not forgiven: this is an explicit exposure scenario, not an inferred payment forecast or full repayment schedule. Schedule replay produces conditional EAD1000 for the five principal contracts. The operating lease has one unpaid rent claim90, no customer residual liability and no borrower1000 debt. Each conditional mean LGD is applied to its matching claim denominator.

`lifetime_expected_loss` is evaluated over this **30-day window only**. Its one marginal first-default mass is the original 30-day application PD. The function name does not establish a full-contract lifetime forecast; no annualization, constant-hazard extrapolation or additional default mass is invented. Zero discounting is an explicit synthetic assumption. Independent oracles are:

| Product | Application PD | Conditional EAD | Prior mean LGD | 30-day loss |
|---|---:|---:|---:|---:|
| Unsecured installment | .75 | 1000 | .40 | 300 |
| Auto loan | .375 | 1000 | .30 | 112.50 |
| Auto lease rent claim | .25 | 90 | .70 | 15.75 |
| Education | .50 | 1000 | .50 | 250 |
| BNPL | .625 | 1000 | .45 | 281.25 |
| Micro loan | .75 | 1000 | .60 | 450 |

Auto scenarios independently depreciate a prior1000 vehicle over one year to900 and apply a2/3 market index: value600, liquidation cost50 and capital loss450. For a loan this is a secured-principal scenario. For the lease the recovery allocation is against the lessor's owned investment and proceeds entitlement; it creates no tenant liability. Neither auxiliary asset loss is added to the separately estimated credit loss. The scenario is not the later observed600 disposal or its realized400 investment loss.

BNPL's two proposed plans have remaining400 and600, each paid in two installments. Their near-term contractual service is500, debt1000. The same applicant's conditional-on-funding household graph counts the facility once. A separate, completely matured2023 ten-account prior has two defaults; a genuinely fitted grouped logit hazard gives.20 over30 days. This benchmark does not silently replace the application PD. No censored partial period is treated as a complete noncase.

Micro's separate named expert population has elicited Beta(2,8) and disjoint complete conditionally-iid observations2/10, giving Beta(4,16), mean.20. This opinion does not replace the application PD. The discrete frontier illustrates explicit alternative30-day offers500/1000/1500 with transported mean.20, LGD.60 and fixed concentration denominator10000. EL60/120/180 and the declared affordability cap1000 leave maximum feasible amount1000. Constant risk across those three offers is a declared scenario assumption, not causal price/limit inference or approval of the original ten-month contract.

The compact output displays expected loss to cents. `return_artifacts=True` (R `TRUE`) retains the original unrounded loss results, fitted models, prior workout calculations and other raw artifacts. Both language implementations check unrounded independent losses with absolute tolerance1e-8 before presentation; no seal or model is rounded. The BNPL oracle281.25 accommodates only underlying finite optimizer precision. Monetary presentation is not evidence of identical independently fitted model hashes.

Default tests fit one representative product and reject identity, knowledge-date and lease-debt mutations. `CREDITRATINGSYSTEM_EXTENDED_TESTS=true` runs all six; the installed example also runs all six. No external service or private calibration is required.
