# Market-implied credit

These eight methods consume local, dated inputs. They price and calibrate under
`RISK_NEUTRAL`, conventionally denoted Q. The historical bridge separately estimates
`PHYSICAL` first-default probabilities from actual issuer outcomes. A spread,
upfront, clean price, hazard rate and physical PD are different quantities.
A conditional market forecast or historical bridge is not automatically a TTC PD.

The pricing engine has deterministic piecewise-constant hazards and instantaneous
forward discount rates. Integrals split at every hazard, discount and cashflow
boundary. Default and recovery settle continuously at default. This is a declared
reduced-form model, **not an ISDA-equivalence implementation**: supply contractual
payment dates yourself. There is no holiday calendar, automatic contract rolling,
ISDA step-in convention, intraday quote ordering, stochastic-rate correlation or
endogenous bond option exercise in this engine.

## Common contracts

Python contracts are dictionaries with array fields as lists. Native R uses named
lists for objects and unnamed lists for JSON arrays, including empty/single arrays.
All date fields are ISO calendar dates (`YYYY-MM-DD`). The information clock is
calendar-day resolution; use a separately constructed cutoff-eligible daily input
when intraday ordering matters. Public functions accept optional `context`; its
measure, currency, valuation date and knowledge cutoff must be compatible where
applicable. Numeric inputs must be finite.

Both `hazard_curve` and `discount_curve` contain:

| Field | Meaning |
| --- | --- |
| `as_of`, `known_at` | Valuation origin and input vintage; knowledge cannot follow valuation. |
| `currency` | Explicit matching currency identifier. |
| `time_basis` | Exactly `ACT/365F`: actual calendar days divided by 365. |
| `knots` | Strictly increasing positive interval ends in years from origin. |
| `rates` | One rate per interval; final rate continues after the final knot. |
| `extrapolation` | Exactly `FLAT_LAST_RATE`, explicitly authorizing that continuation. |
| `probability_measure` | Required `RISK_NEUTRAL` for hazard curves. |

Hazards are nonnegative. Discount forward rates may be negative. Discount factors
are `D(t)=exp(-integral r)` and survival is `S(t)=exp(-integral lambda)`.
Very large exponents fail explicitly instead of overflowing or silently producing
zero survival. Curve fitting records the latest quote/discount input knowledge
in the fitted hazard vintage; an earlier known discount curve can price a later
eligible quote at the same valuation origin.

Instrument contracts share `as_of`, `currency`, `probability_measure=RISK_NEUTRAL`
and `default_settlement=AT_DEFAULT`. Calibration additionally requires
`instrument_type` (`CDS` or `BOND`), `issuer_id` and `seniority` on every instrument.
Each calibration fits a single issuer and seniority.

## Local quote snapshot

`build_credit_quote_snapshot(quotes, instruments, discount_curve, snapshot_contract, context=None)`

`instruments` is a nonempty array with unique `instrument_id` and complete
instrument contracts. Each quote has unique `quote_id`, `instrument_id`,
`quote_type`, `unit`, `value`, `observed_at`, `known_at`, and optional paired
`bid`, `ask`. Knowledge cannot precede observation. Supported conversions are:

| `quote_type` | Accepted units | Normalized unit |
| --- | --- | --- |
| `CDS_SPREAD` | `BASIS_POINTS`, `FRACTION` | Annual coupon fraction |
| `CDS_UPFRONT` | `PERCENT_NOTIONAL`, `FRACTION` | Fraction of notional |
| `BOND_CLEAN_PRICE` | `PER_100_PAR` | Price per 100 outstanding par |

`snapshot_contract` requires `as_of`, `known_at`, `currency`, integer
`maximum_age_days >= 0`, `stale_policy` (`EXCLUDE`, `KEEP_FLAGGED`) and
`bid_ask_policy` (`REQUIRED`, `OPTIONAL`). It returns all normalized rows with
`eligible` and findings. Future information, currency mismatch, crossed/outside
bid-ask quotes and conflicting values for the same instrument/type/vintage are
ineligible. Stale rows are either excluded or explicitly retained with a flag.
Different seniorities remain separate and appear in `cross_seniority_groups`.
The snapshot is canonically hashed. It never downloads market data.

The calibration `quotes` argument accepts eligible normalized snapshot rows
(`snapshot.value['quotes']` filtered by `eligible`) or equally explicit local
rows. Each needs `quote_id`, `eligible=True`, `observed_at`, `known_at`,
`instrument`, `quote_type`, `normalized_unit`, `value`. Quote knowledge must not
exceed valuation. Eligibility assertions supplied without a snapshot remain the
caller's data-preparation contract; calibration cannot attest to unseen quotes.

## CDS cashflows and bootstrap

`cds_expected_cashflow_legs(hazard_curve, discount_curve, cds_contract, context=None)`

`cds_contract` additionally requires `recovery` in [0,1], nonnegative annual
`coupon`, `protection_start`, `maturity`, `settlement_date`, signed
`accrued_premium`, signed `upfront`, and `premium_mode`. Positive upfront and
settlement-accrued amounts are paid by the protection buyer; negative amounts are
rebates received by the buyer. All results are per unit notional.

* `CONTINUOUS`: premium integrates `D(t) S(t)` over the protection interval.
  Settlement accrued premium must be zero. This provides the exact flat oracle
  `par_spread=(1-recovery)*hazard`.
* `SCHEDULED`: requires `day_count` (`ACT/360` or `ACT/365F`), boolean
  `accrual_on_default`, and contiguous `schedule` rows with `accrual_start`,
  `accrual_end`, `payment_date`. Accrual end cannot exceed maturity; the last end
  equals maturity. The first period can start before valuation. Coupon payment
  can lag accrual end: survival is tested at accrual end and the surviving coupon
  is then owed at payment, with no additional post-maturity default risk.
  Default accrual integrates actual linear accrued fraction, including any
  pre-valuation portion, over the covered default interval.

Protection PV is `(1-R) integral D S lambda`. The premium and accrued-default
annuities are separate. `fair_upfront=(protection-coupon*risky_annuity)/D(settlement)
-accrued_premium`; the residual discounts the difference from supplied upfront.
`par_spread` excludes settlement accrued cash, so it is the ratio of protection
PV to total risky annuity, not an implicit rescaling of a supplied accrued quote.
The payment/accrual conventions are deliberately explicit.

`bootstrap_cds_hazard_curve(quotes, discount_curve, calibration_contract, context=None)`

The contract requires fixed `recovery`, positive `maximum_hazard` and positive
`price_tolerance`. Maturities must be positive and distinct. Spread quotes set
coupon to the normalized spread and upfront to zero, and require zero settlement
accrued premium. Upfront quotes retain the instrument's fixed coupon and accrued
settlement amount. A bounded scalar root is solved for each next hazard segment.

The returned sealed `MarketCreditCalibration` includes `feasible`, the fitted or
partial `hazard_curve`, quote residuals, survival, solver bracket residuals and
boundary statuses. A missing nonnegative bracket or unmet residual tolerance is
reported; negative hazards are never clipped. Partial curves are not accepted by
basis/identifiability methods. Local knot sensitivities identify their fixed-prior-
knots scope; `curve_sensitivities` additionally inverts the full quote/hazard
Jacobian for recursive quote and recovery effects. Full sensitivities are withheld
at a parameter boundary or deficient rank. A perfectly fitted zero spread with
100% recovery does not identify a hazard.

## Bonds and hazard calibration

`risky_bond_expected_cashflows(hazard_curve, discount_curve, bond_contract, context=None)`

The contract requires `recovery`, fixed `liquidity_spread`,
`recovery_convention` (`PAR`, `TREASURY`, `MARKET_VALUE`),
`cashflow_assumption` (`CONTRACTUAL_NONOPTIONAL`, `EXOGENOUS_OPTION_SCENARIO`),
positive `outstanding_principal`, nonnegative `accrued_interest`,
`settlement_date=as_of` and nonempty `schedule`. Each schedule row has strictly
increasing future `payment_date`, nonnegative `coupon_amount`, `principal_amount`.
Principal payments must sum to outstanding principal. Optional cashflows require
an additional `option_scenario_id`; an externally supplied scenario is not an
option price. Bond prices are at same-day settlement, with no hidden funding or
forward-settlement adjustment.

* `PAR`: recovery at default is R times remaining outstanding principal. The
  principal decreases only at scheduled amortization dates.
* `TREASURY`: recovery is R times the then risk-free value of all remaining
  promised coupon/principal payments. This differs from recovery of par.
* `MARKET_VALUE`: recovery is R times pre-default value. The deterministic
  valuation discounts promised cashflows at `r + liquidity + (1-R)*lambda`.

Survival-weighted promised coupon and principal PVs remain separate from recovery
PV under all conventions. For market-value recovery, recovery PV is the exact
residual between the reduced-form value and survival-weighted promised payments.
Liquidity is an exogenous discount wedge applied to both promised and default
cashflows. Treasury's replacement claim uses the risk-free curve without that
wedge. Output separately reports the risk-free, credit-only and credit-plus-
liquidity prices and their additive price effects. Clean price is dirty price
minus supplied accrued interest; `clean_price_per_100` divides by outstanding par.
No credit/liquidity attribution is identified merely by this arithmetic.

`calibrate_bond_hazard_curve(quotes, discount_curve, calibration_contract, context=None)`

Each bond quote additionally requires strictly positive `price_standard_error`
in price-per-100 units. The contract requires `hazard_knots`, `initial_hazards`,
positive `maximum_hazard`, fixed `recovery`, fixed `liquidity_spread`, positive
integer `max_evaluations`, and
`weight_interpretation=KNOWN_GAUSSIAN_PRICE_STANDARD_ERRORS`. Native bounded
nonlinear optimization minimizes the sum of squared price residuals divided by
those known errors. Python uses bounded least squares; R uses native L-BFGS-B
with a numerical pricing Jacobian. Solver iteration counts need not match.

Outputs retain raw/weighted residuals, fitted prices, objective, convergence,
Jacobian, column-scaled rank/condition, active boundaries and covariance when
available. Intervals are untruncated local 95% Wald approximations conditional
on fixed recovery/liquidity, model correctness and supplied Gaussian price errors.
They are withheld at boundaries or deficient rank; they are neither global
profile intervals nor evidence that recovery or liquidity was learned from data.
Overparameterized fits remain flagged instead of claiming unique identification.

## Identifiability and basis

`diagnose_market_credit_identifiability(calibration, alternative_assumptions, diagnostic_contract, context=None)`

`calibration` is a sealed feasible CDS/bond calibration value.
`alternative_assumptions` is a nonempty array with unique `scenario_id`, `recovery`
and, for bonds, `liquidity_spread`. Every alternative actually refits hazards.
`diagnostic_contract` requires nonnegative `equivalence_tolerance`, one positive
`quote_standard_errors` value per quote, two-element `recovery_bounds`, and for
bonds two-element `liquidity_bounds`. Bounds must contain the reference and all
alternatives. The objective is squared standardized quote residuals, optionally
plus independent recovery information.

Optional `independent_recovery_observation` requires `value`, positive
`standard_error`, nonempty `evidence_id`, and eligible `known_at`. Its independent
Gaussian measurement assumption is explicit. The joint Jacobian includes hazard,
recovery and (bond) liquidity columns plus the independent observation row.
Profiles report actual alternative hazards/objectives, failed feasibility and
near-equivalent sets. This is a **finite declared scenario profile**, not a search
proving the global identified set. The normal equivalence tolerance is a supplied
objective tolerance, not an automatically inferred confidence threshold.

`compare_cds_bond_credit_basis(cds_calibration, bond_calibration, comparison_contract, context=None)`

The comparison contract requires fixed `recovery`, `bond_liquidity_spread`,
`funding_treatment=NO_ADDITIONAL_FUNDING_WEDGE`, integer
`maximum_quote_age_days`, and a full `comparison_cds_contract`. The method checks
issuer, seniority, currency, curve valuation/vintage, exact discount conventions,
recovery, liquidity, quote freshness and maturity coverage. Mismatches return
`comparable=False`, explicit findings and null basis, rather than an unmarked
number. Matching curves are repriced with the same CDS schedule. The sign is
CDS par spread minus bond-implied equivalent CDS par spread. Output includes both
spreads, basis in fractions/basis points and survival difference. A matched
synthetic world gives zero basis. This is a convention-conditional market
comparison, not a pure default signal or trading recommendation.

## Historical physical-PD bridge

`fit_market_to_physical_pd_bridge(observations, feature_order, bridge_contract, folds, *, model_id, model_version, context=None)`

Each observation contains unique `observation_id`, `entity_id`, `as_of`,
`label_known_at`, `complete_through`, `market_curve_as_of`, `market_curve_known_at`,
`market_curve_hash`, `features_known_at`, `features` mapping exactly the declared
feature names, binary `outcome`, `default_date` (null for no observed default),
`initial_state=PERFORMING`, and `outcome_origin=OBSERVED_ISSUER_DEFAULT`.
Market features/curve vintages must be available by the prediction date; target
windows must be fully mature and available by fit cutoff. A default must occur
within the declared target horizon. Feature hashes document supplied lineage;
the method does not reconstruct unseen historical quote books.

The bridge contract requires:

* `target`: `event=first_default`, positive `horizon`, `time_unit` in days/months/
  years, `probability_measure=PHYSICAL`, `conditioning=initially_performing`.
  Date windows use integer days or an integer number of calendar months, clamping
  calendar month ends.
* `feature_probability_measure=RISK_NEUTRAL`, `market_feature_contracts`: one
  object per feature with `probability_measure=RISK_NEUTRAL`, positive `horizon`,
  `time_unit` and nonempty `definition`. Feature horizons may differ from physical
  target horizons because the learned conditional model is explicit.
* `outcome_origin=OBSERVED_ISSUER_DEFAULT`, nonempty `target_population`,
  `population_selection=MARKET_QUOTED_ISSUERS`,
  `transformation=IDENTITY_PREDECLARED_FEATURES`, `known_at`, `split_id`,
  `link` (`logit`, `probit`, `cloglog`), and optional `data_origin`.

`folds` contains nonempty unique zero-based `train` and `validation` index arrays.
Training and validation entities are disjoint, validation rows occur in at most
one fold, and every training label is known strictly before the earliest
validation prediction date. Each fold fits an actual Bernoulli GLM; no full-fit
predictions are substituted. Warmup rows have null out-of-time predictions.
The final model fits all eligible observed rows for subsequent prediction.
Out-of-time Brier/log loss and entity-cluster Brier uncertainty use only real
observed outcomes. Native GLM coefficient covariance is also retained. Separation,
rank deficiency and nonconvergence fail explicitly.

The frozen model uses common `ModelArtifact` serialization and generic
`predict_pd` dispatch, requiring matching features and the PHYSICAL target. An
estimated conditional historical association is not a universal Radon–Nikodym
measure change, a causal premium decomposition, an unquoted-population estimate,
or a justified TTC estimate. Selection into the quoted population remains an
explicit limitation. There is no automatic identity Q-to-P conversion.

## References and checks

The recovery distinctions and reduced-form valuation follow the definitions in
[Darrell Duffie, Defaultable Term Structure Models with Fractional Recovery of Par
(1998)](https://www.darrellduffie.com/uploads/1/4/8/0/148007615/duffiedefaultabletermmodels1998.pdf).
[ISDA CDS Standard Model documentation](https://cdsmodel.com/documentation.html)
illustrates the broader contractual conventions necessary for standard-model
reproduction; this implementation makes no such equivalence claim.

Default tests include actual fits, independent closed-form prices, independent
accrual quadrature, amortization conservation, inverse quote calibration, recursive
sensitivity refits, nonidentification with and without independent recovery data,
zero-basis matching, temporal/entity leakage and frozen model interchange.
`CREDITRATINGSYSTEM_EXTENDED_TESTS=true` adds the recovery/liquidity/recovery-
convention calibration grid. Default CI avoids that exhaustive grid.
