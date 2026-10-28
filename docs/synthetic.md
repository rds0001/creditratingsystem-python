# Synthetic reference worlds

The eight functions in `creditratingsystem.synthetic` create and audit offline
credit datasets. Python and native R consume the same **actual innovation bytes**;
matching random seeds is not the interchange contract. All observations are
synthetic. Neither the fixed DGP parameters nor its realized performance estimates
represent a bank portfolio, a regulatory calibration, or a recommended policy.

## Public interfaces

All generators return a `CalcResult` whose `value` contains `directory` and
`manifest`. Generation writes to the caller's absent or empty directory. There is no implicit installed-resource write, download, model fit, or
choice of another innovation stream to obtain more convenient outcomes.

```python
generate_macro_reference_world(specification, innovations, *, output_directory,
                               chunk_rows=1000, memory_budget_mb=256)
generate_obligor_panel(macro_world, specification, innovations, *,
                      output_directory, chunk_rows=1000, memory_budget_mb=256)
generate_facility_workouts(obligor_world, specification, innovations, *,
                          output_directory, chunk_rows=1000, memory_budget_mb=256)
generate_retail_reference_world(macro_world, specification, innovations, *,
                               output_directory, chunk_rows=1000,
                               memory_budget_mb=256)
generate_asset_pool_reference_world(worlds, specification, innovations, *,
                                   output_directory, chunk_rows=1000,
                                   memory_budget_mb=256)
generate_system_event_reference_world(worlds, specification, innovations, *,
                                     output_directory, chunk_rows=1000,
                                     memory_budget_mb=256)
generate_challenge_cases(case_ids=None, context=None)
validate_synthetic_world(world, check_hashes=True, context=None)
```

R has the same argument names and defaults; it does not require a Python runtime.
`world`, `macro_world` and `obligor_world` accept a generator result, its value, or
the directory containing a generated manifest. `worlds` is a list of such worlds.
Linked sources must have the same tier, start date and observation cutoff. Pool
sources contain actual funded accounts or wholesale facilities; event sources
contain existing obligors, accounts, facilities or pools.

`specification` is a mapping/named list with required fields:

| Field | Meaning |
|---|---|
| `tier` | `tiny`, `standard`, or `stress`; sizes below are fixed |
| `dgp_family` | `well_specified` or `misspecified` |
| `start_date` | ISO `YYYY-MM-DD` world origin |
| `observation_cutoff` | ISO last permissible knowledge date |
| `selection_mechanism` | Required for retail: `MAR`, `MNAR`, or `missing_overlap` |

`innovations` is the directory of a `creditratingsystem-u32le-v1` manifest and its
binary streams. `chunk_rows` is an integer from 1 through 50,000.
`memory_budget_mb` is an explicit planning budget of at least 32 MiB. A conservative
stream/index estimate is checked before creating output; it is not an operating
system process-memory limit. The manifest records the estimate and requested
budget. Scenario paths are written one at a time, tables as gzip JSONL, and R reads
innovations through bounded blocks. No account-by-scenario Cartesian product is
materialized. An insufficient innovation stream or budget produces an error;
there is no hidden subsampling. Interrupted directories can be inspected and a
fresh empty directory selected for another execution.

`case_ids` optionally selects unique challenge identifiers. `context` is the
standard calculation context; generation itself derives dates from the explicit
DGP specification. `check_hashes=False` disables only file checksum verification,
not row, time, identity or economic consistency checks.

## Layers and dates

Each world has five separate directories: `observable_inputs`, `latent_truth`,
`frozen_predictions`, `decision_history`, and `observed_outcomes`. Hidden asset
values, frailty, conditional PDs, unobserved recoveries and potential outcomes live
only in `latent_truth`. A rejected/unfunded applicant never receives an observed
own-loan outcome. Positive funded labels come from actual account default events,
including the reporting lag; early closures are competing nondefaults. Immature
unresolved horizons have null labels. Potential default before a loan exists is
not an observation.

Statements have an explicit observation date and a 90-day knowledge lag. Default
events have a 14-day reporting lag; recoveries have a seven-day lag. Macro releases
have initial/revised knowledge lags of 30/90 days. Bureau corrections delivered
45 days after application cannot enter the original prediction. An observation
cutoff filters observable data; future DGP values remain segregated as truth.

Economic groups and households receive stable train/calibration/test partitions
using group modulo 10 (60/20/20). Time partitions use forward 60/20/20 windows.
Only matching entity/time cells are eligible; other cells are marked `excluded`.
This deliberately sacrifices rows to avoid overlapping households/groups between
training and out-of-time evaluation. Users must still apply their chosen training
cutoff and matured-target eligibility.

## DGP and economic contracts

The macro history has eight correlated AR(1) factors, shared-normal loading 0.25,
and explicit rate/index transformations. The misspecified family adds a persistent
adverse regime. The binary forecast cube stores standardized factors after the
final history date in **path, month, factor** order, float64 little endian.

Wholesale statements satisfy assets = debt + equity and explicitly reconcile
EBITDA, depreciation, EBIT, taxes, capex, CFADS and debt service. Observable equity
and equity volatility are generated independently from latent asset values using
the European-call structural equations. This use of equity as a contingent claim
follows [Merton's original corporate-debt paper](https://dspace.mit.edu/entities/publication/5189ad6f-77e1-410e-8b71-29347b2ceb71).
The well-specified ordinary stratum uses the fixed observable logistic equation;
explicit zero/low-default strata are separate. The misspecified family adds group
frailty and nonlinear terms. First defaults end subsequent financial statements.
External opinions have independent noise; dated overrides have reasons and expiry.

Operating assets include property leases and tenant concentration, aircraft/ship
charters, project offtake, utilization, maintenance, CFADS, construction completion,
and distinct lessee/lessor/SPC identities. Country transfer draws are independent
of base-default innovations; the foreign-currency sensitivity table records base,
transfer and union events separately from ordinary borrower labels. Support edges
identify scope, capacity and validity. These are transparent engineering scenarios,
not market-calibrated sector simulators.

Facility utilization, conversion at default and recoveries share the borrower's
time-invariant origination leverage driver, never a future-period PD. Gross
recoveries and costs occur at 90, 360 and 900 days;
open workouts are distinguished from completed workouts. Discounted LGD remains
unclipped. Constructed boundary ledgers with LGD -0.2 and 1.2 are explicitly
separate from stochastic cohort frequencies.

Retail spans mortgages, installment and auto loans, cards, overdrafts, BNPL,
small-business credit and retail leasing. Households, persons, co-borrowers,
net-income observations, expenses, bureau tradelines and transactions have stable
links. Internal transfers and reversals are identified; historical transactions
become available to the lender when the application extract is delivered.
MCAR/MAR/MNAR mechanism labels live in latent truth; the observable application
contains only the actual missing-income flag. The frozen demonstration score uses observable income
(fixed 2,500 fallback when missing) and bureau data (0.5 fallback for no-hit),
without fitting or reading latent frailty. Selection changes between policy
versions; MNAR acceptance additionally depends on latent frailty. Take-up and
funding are separate stages, with an explicit funded-capacity limit.

Month-on-book zero is the opening observation; scheduled payments start at month
one. Monthly accounts reconcile opening balance + draws + accrued interest - payments
- recoveries = closing balance. Floating mortgages reset after 12 months; auto and
leasing contracts have explicit balloons. Missed installments produce arrears,
active undrawn accounts remain in the panel, and first defaults, cures, repeated
episodes, prepayments, contractual closures and collections are dated separately.
Collections actions and recoveries are a fixed synthetic process, not an estimated
causal treatment effect or optimized intervention policy.

Pools partition existing assets without duplication and retain time-valid
memberships. Observed servicing copies source payment/recovery/cost/default-EAD
ledgers. Separate contractual cashflows are a no-default/no-prepayment projection.
The four tranche notionals partition original assigned principal; the pooled book
uses forward additions, and final aggregate notionals become known only after the
last assigned origination. Original assigned principal is not current outstanding
EAD. Tranche waterfall valuation is a separate calculation, not an observed label.

System-event deliveries have separate event and delivery IDs, effective and
knowledge dates, late corrections, duplicates, failed batches, guarantee expiry
and a model switch. Expected checkpoints are authored from a closed-form schedule,
independently of the event-replay implementation. They are constructed engineering
cases, not random operational incident-frequency estimates.

## Reference sizes and offline use

| Component | Tiny | Standard | Stress |
|---|---:|---:|---:|
| Wholesale obligors / years | 64 / 3 | 3,000 / 10 | 20,000 / 15 |
| Macro history months | 36 | 240 | 360 |
| Forecast paths / months / factors | 512 / 24 / 8 | 16,384 / 60 / 8 | 131,072 / 120 / 8 |
| Retail applications / origination months | 320 / 12 | 50,000 / 36 | 500,000 / 84 |
| Households / persons | 160 / 240 | 20,000 / 30,000 | 200,000 / 300,000 |
| Funded account capacity / months | 160 / 24 | 24,000 / 60 | 240,000 / 120 |
| Pools / asset capacity | 2 / 80 | 12 / 30,000 | 100 / 100,000 |

Actual survivor, funded, recovery and pool rows depend on supplied draws and
cutoffs. These capacities never force acceptance or realized default counts.
Manifests record actual rows, file hashes, versions, dependencies and runtime.

The package contains `datasets/synthetic_tiny` (all six coupled worlds and actual
draws) and `datasets/synthetic_standard` (full wholesale/macro/workout worlds and
actual draws). R resources are under `system.file("extdata", package =
"creditratingsystem")`. Every stored path is relative to its world or innovation
manifest. Resources can be read directly as ordinary gzip JSONL and binary files.

```python
from importlib.resources import files
from creditratingsystem.synthetic import validate_synthetic_world
root = files("creditratingsystem").joinpath("datasets", "synthetic_tiny")
report = validate_synthetic_world(str(root.joinpath("retail")))
assert report.value["passed"]
```

For new large experiments, the installed `synthetic_innovations.py` and native
`synthetic_innovations.R` resource scripts write exactly matching actual innovation
files with a documented SHA-256 counter construction. Invoke either with a new
output directory, `standard` or `stress`, and `--retail`. The optional experiment ID
selects a new experiment before outcomes are generated. This portable construction
differs from the fixed PCG64 reference draws already bundled; both use the same
file-consumption protocol. Large-tier authoring writes the complete declared
streams and requires substantial disk space; it never downloads missing files.

## Audit and analytical challenges

`validate_synthetic_world` checks table hashes, row/column counts, primary and local
foreign keys, observation/knowledge dates, truth leakage, funded-only outcomes,
financial identities, account/contract cash reconciliation, facility limits, pool
asset uniqueness, tranche totals and split eligibility. External-source existence
is checked by the composition generators; dependency hashes remain in manifests.
The audit reports conditional PD sums and realized events descriptively. Dependent
cohorts are not silently treated as independent binomial samples.

`generate_challenge_cases` contains complete separation, zero defaults,
Spiegelhalter's zero-variance p=0.5 case, singular covariance, support cycles,
transfer extremes, censoring, raw LGD boundaries, a hand-authored 100-account
cohort and a future feature. The zero-default upper bound solves
`(1-p)^100 = 0.05`; exact-binomial methodology is documented by
[NIST](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm).
Passing these synthetic engineering checks is not evidence of predictive accuracy,
calibration adequacy, legal compliance or causal identification on bank data.

System-event worlds include version registries for model, policy and calibration,
plus complete copied source snapshots with independently audited content hashes.
Checkpoint versions are demonstration fixtures, not fitted models. Pool servicing
also carries source lifecycle events with their original reporting dates. A separate
constructed open-workout row exercises censoring even when a fixed tiny cohort
realizes no wholesale defaults; it is excluded from stochastic default counts.
Independent language calculations may differ by a floating-point unit; exact
snapshot hashes are checked against their own source, and identical frozen
sources must produce identical hashes.
