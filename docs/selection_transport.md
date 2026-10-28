# Selection and transport on actual application journeys

Run `examples/selection_transport.py` or source `examples/selection_transport.R`
and call `run_selection_transport()`. The four products are installment, education,
credit card and sole-trader finance. Every result retains application and funded
facility identifiers from its originating journey. Different product horizons
remain distinct: 30 days for the first, second and fourth; 365-day first default
for the card. The original card model remains conditional on funding and retains
its funded-only population contract.

The first three product chains have five submitted applications: two observed
funded outcomes (0 and 1), one declined application, one approved offer without
uptake, and one funded immature account. The card retains its three actual funded
accounts (0, 1, and a censored early exit), supplemented by two separately dated
unbooked applications. Their affordability decisions follow documented income,
expenses and contractual minimum payment. No credit outcomes are supplied for them.

`build_selection_dataset` reconstructs stages. `selection_bias_sensitivity`
independently replays bounds, inverse observation weights and log-odds sensitivity.
The two observations among five applicants imply bounds [0.2, 0.8]. The declared
constant observation propensity 0.4 gives effective sample size 2; it is a MAR
scenario, not an estimated or validated observation process. MNAR shifts -1, 0, 1
apply only to missing outcomes and use each frozen model's conditional predictions.
A second scenario explicitly has zero support for missing outcomes and returns
`NOT_IDENTIFIED_ZERO_SUPPORT`. Neither trimming nor imputation proves transport.

The pseudo-label negative control assigns labels to unbooked cases only in a
separate validation call and confirms that factual sample size is unchanged.
It never changes the actual cohort or provides validation truth. Prior training
and calibration remain chronologically separate from these held-out accounts.
An observed default before closure stays a default; an early closure without
complete follow-up stays censored.

Source lineage, original policy decisions, cohort validation and actual replay
are reused by calling the originating example helpers. Policy backtests execute
for all four products. The card uses the explicit version1.1 first-draw target
contract and retains first_default unchanged. Actual approval, uptake, booked
amounts and policy decisions reconcile: four offers and three funded accounts
among five applicants. Complete365day net profit and application costs remain
unknown; no invented offer-uptake or profitability estimates are supplied.
The factual credit-outcome selection test remains distinct from this incomplete
economic ledger. No causal policy value is claimed.

For own data, call Python `audit` / native R `st_audit` with dated records,
a genuinely prior fitted model and calibrator, actual training entities, cutoff
and named sensitivity assumptions. Each record declares application/customer/
facility IDs, application/decision/actual-draw dates, offered/funded/observed,
observed own-contract outcome or null, policy version and frozen PD. Original
funded horizons start at the actual draw; unbooked scenario horizons explicitly
assume funding at application. Model contracts and economic records must be
adapted to the bank's own population. `return_artifacts=True` / `TRUE` retains
unrounded models, prediction designs, scenarios and validation inputs. Compact
reported PD/score summaries use eight decimals; fitted parameters are never rounded.

Default tests use a tiny real prior intercept fit with the independent PD=0.5,
Brier=0.25 and bounds [0.25,0.75] oracles. The full installed example separately
executes all four real chains. A portable two-origin replay verifies the frozen
models and calibrators without fitting them again. These synthetic examples
establish implementation and lineage behavior, not empirical bank acceptance.
