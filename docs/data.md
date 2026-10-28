# Local data and observation contracts

All six functions return `CalcResult`; inspect `.value`, `.diagnostics` and
`.provenance`. The native R functions use the same names and `$value`. Inputs are
lists of named rows (Python dictionaries; R lists). Python additionally accepts a
pandas DataFrame without importing pandas itself; R accepts data.frames. Outputs
are lists of rows, including explicit `None`/`NULL` cells, rather than silently
coercing mixed columns. Dates are ISO `YYYY-MM-DD` strings and horizons are **calendar
days**, never implicitly months or years. Keys and IDs must be nonempty strings; preserve
leading zeroes. These functions neither infer identities nor fetch network data.

## Typed local import

`read_local_dataset(path, schema, encoding="utf-8", delimiter=",", decimal_mark=".")`
reads `.csv` or a JSON array of row objects. Schema version is `"1.0"`; `columns`
is a mapping of exact column names to `{type, nullable?, unit?}`. Types are
`string`, `integer`, `number`, `boolean`, and `date`. Numeric columns require a unit
string, e.g. `EUR`, `days`, `dimensionless`. `nullable` defaults to false; JSON null
and empty CSV cells become missing values, never zero. Boolean literals are
`true`/`false`. Integer strings contain signed digits only and must be within
±9,007,199,254,740,991; larger identifiers belong in string fields. Decimal and
scientific number syntax is supported; grouping separators are not. JSON numeric
literals follow JSON's decimal point, while explicit decimal-comma conversion
applies to text cells. `primary_key` optionally names a string key column or lists unique, nonmissing
string key columns. Empty tables require `allow_empty: true`. Columns must match exactly.
CSV headers and all JSON object keys must be unique. URI paths and UNC paths are rejected.

```python
from creditratingsystem.data import read_local_dataset
schema = {"version": "1.0", "columns": {
    "entity_id": {"type": "string"},
    "ead": {"type": "number", "unit": "EUR"}},
    "primary_key": ["entity_id"]}
# result = read_local_dataset("exposures.csv", schema)
```

Diagnostics include row count, schema version and units. Provenance includes a
SHA-256 of original file bytes, so identical bytes share a fingerprint across
languages. Imports are user-supplied data, not empirically validated datasets.
Supported adapters are CSV and JSON; no Excel/Parquet parser is implied.

## Relational validation

`validate_credit_dataset(tables, contract_version="1.0")` accepts any subset of
these canonical tables; referenced parents must nevertheless be supplied:

| Table | Unique key | Other required fields |
|---|---|---|
| `obligors` | `entity_id` | — |
| `observations` | `entity_id, observed_at` | `available_at` |
| `facilities` | `facility_id` | `entity_id, ead` |
| `defaults` | `episode_id` | `entity_id, default_at` |
| `recoveries` | `cashflow_id` | `episode_id, cashflow_at, amount, kind` |

EAD and amounts are finite and nonnegative; recovery `kind` is `recovery` or
`cost`, with positive cost amounts. Observations, facilities and defaults refer
to obligors; recoveries refer to defaults. Dates are checked, availability cannot
precede observation, cure/closure cannot precede default, and recovery cannot
precede its default. Duplicate keys and absent required fields are reported.
Unknown tables or versions raise an error; row-level problems return
`{valid, violations, row_counts}`. Every violation identifies table, **one-based**
row number, field and code. Additional table-specific business rules, currencies,
retail party hierarchies and default materiality require their own contracts.

## Bitemporal feature join

`join_as_of(left, right, entity_keys=("entity_id",), as_of="as_of",
known_at="known_at", observed_at="observed_at", available_at="available_at",
release_id="release_id", suffix="_right")`
requires left entity keys, economic as-of date and information cutoff; right rows
require matching keys, observation date, availability date and globally unique
release ID. A candidate satisfies both `observed_at <= as_of` and
`available_at <= known_at`. Select the latest observation date, then the latest
available revision of that observation. Dates tied at both levels are ambiguous
and raise an error: row order is never a priority rule. Multiple sources should
be separated in the keys or explicitly reduced before calling. All right columns
except entity keys are returned, with the suffix on names also present on the
left. Further collisions raise an error. Unmatched rows remain with missing right
fields; the selected release ID is included. This is a fact/revision join, not
an effective-from/to interval join.

## First-default performance windows

`build_observation_windows(ratings, default_events, exits, horizon_days,
evaluation_cutoff, entity_key="entity_id")` uses:

- Ratings: unique `observation_id`, entity ID, `as_of`; optional
  `performance_start` (default as-of), `followup_end` (default evaluation cutoff),
  `funded` (default true), and optional `outcome_available_at` describing when
  complete outcome ascertainment becomes available. Performance cannot begin before the prediction.
- Defaults: unique `event_id`, entity ID, `default_at`, `known_at`.
- Exits: entity ID, `exit_at`, `known_at`.

Performance is `(performance_start, performance_start + horizon_days]`; an event
at its end counts. Only events known by evaluation cutoff are observable. An
event known before it occurs is invalid. Follow-up ends at the earliest horizon,
evaluation cutoff, supplied follow-up end or exit. Default on the exit date wins
the tie. Prior default at or before origin makes a first-default observation
ineligible. A known in-window default yields target 1; a complete observed window
yields 0. Premature exit, incomplete follow-up and a not-yet-started window have
missing targets. A supplied `outcome_available_at` after the evaluation cutoff
yields `outcome_pending` with a missing target unless a default is already observed.
This distinguishes calendar maturity from completed outcome ascertainment. Unfunded/rejected applications have missing targets regardless
of simulated or external events. Added columns are `performance_start`,
`outcome_end`, `observed_until`, `target`, `censored`, `eligible`, `label_status`,
and matched `event_id`. `eligible` describes the at-risk funded population;
censoring can still prevent use in ordinary binary fitting.

The user must supply a complete event feed **through followup_end** at the stated
knowledge cutoff. Reporting gaps require a shorter follow-up end; absent reports
cannot prove a mature nondefault. This API targets first default and excludes
previous defaulters. Cure/re-default targets require an explicit episode-based
model, not reuse of a first-default zero. Early repayments are treated as
censoring, not hypothetical future good credit performance.

## Workout episodes

`build_recovery_episodes(default_events, recovery_cashflows, cures=None,
reopening_days=90, evaluation_cutoff=None, *, closures=None)` requires defaults with unique
`episode_id`, `entity_id`, `default_at`, `known_at`, nonnegative `ead`, and
`currency`; no two defaults may occur for the same entity on the same date.
Cashflows require unique `cashflow_id`, `episode_id`, `cashflow_at`, `known_at`,
nonnegative `amount`, `kind` (`recovery`/`cost`), and matching `currency`.
Cures have unique `episode_id`, `cure_at`, `known_at`, and optional `closed_at`. A closure also
requires `closed_known_at >= closed_at`; it takes effect only once that knowledge
date is reached. Cure knowledge alone never makes a future closure available.

Optional `closures` records terminate a legal workout without asserting borrower
cure. Each has unique `closure_id` and `episode_id`, `closed_at`, `known_at`
(on or after closure), and `closure_reason`: `SETTLEMENT`, `WRITE_OFF`, or
`LEGAL_TERMINATION`. Closure cannot precede its default and applies only when
known at the evaluation cutoff. Multiple terminal records for one merged workout,
including simultaneous legacy cure closure, fail with `DATA_TERMINAL_CONFLICT`.
A legal closure preceding a later default in the same continuous workout fails;
legal termination alone never resets borrower default state or supplies a cure.
Explicit closure outputs additionally retain `closure_id` and `closure_reason`.
Day-count parameters accept finite integral JSON numbers (e.g. `90.0`), but reject
booleans, strings, fractional and nonfinite values.

Defaults and cures are filtered by knowledge cutoff. A repeated default belongs
to the previous workout if there is no known cure or it occurs within the stated
number of days **after the preceding default's cure date**, inclusive. Otherwise
it starts a new workout. Reopening preserves EAD at the first default rather than
adding exposures. Each supplied episode ID maps to exactly one workout and each
cashflow to exactly one supplied default ID. Unknown references, duplicate IDs,
pre-default recoveries and currency mismatches fail. Unknown-at-cutoff cashflows
remain excluded. Explicit closure applies to the latest default in a workout;
a reopened workout resets closure. Cure alone and data-end alone never imply
closure. A cashflow after explicit closure fails and requires revised source data
or an explicit reopening.

Output contains `episodes` and allocated `cashflows`. Episode fields include IDs,
first default date/EAD, default and cashflow ID lists, positive recoveries and
costs, net recovery, currency and open/closed status. Cashflows retain dates,
workout ID and actual calendar days since first default for subsequent discounting.
Totals are **undiscounted** amounts; no LGD, discount rate or recovery-cap assumption
is silently applied. This contract treats supplied default/cure events as an
already adjudicated event history; it does not derive sustained cure from DPD.

## Grouped chronological split

`make_grouped_temporal_splits(panel, train_end, calibration_end, test_end,
embargo_days=0, entity_key="entity_id", group_key="group_id",
time_key="observed_at", outcome_end_key="outcome_end")`
requires unique `observation_id`, string entity ID, observation and outcome-end
dates; group IDs are optional strings. Boundaries strictly increase. Candidate
intervals are `(-infinity, train_end]`, `(train_end, calibration_end]`, and
`(calibration_end, test_end]`. All entity/group connections are combined
transitively: a borrower changing households connects both groups. Each connected
component is reserved for its **latest** candidate partition, including records
beyond test end. Earlier rows of that component are excluded, not moved through
time. This prioritizes untouched future cohorts over training sample size.

Training/calibration outcomes plus embargo must end by their respective upper
boundary. Calibration/test observation dates must be strictly later than their
lower boundary plus embargo; test outcomes must end by test end. Output contains
one assignment for every observation, counts, and verified `group_overlap=false`.
Excluded rows name `outside_test_window`, `group_isolation`, `outcome_overlap`, or
`embargo`, in that precedence order. Empty partitions are possible and remain
visible in counts; no automatic random reassignment repairs them. Fit feature
transformations only after this split. This group-isolated design tests unseen
connected entities; it is not a repeated-customer operational backtest.

## Controlled failures and scope

All failures use `CreditRiskError.code` in Python or condition `$code` in R.
Codes: `DATA_SCHEMA`, `DATA_DATE`, `DATA_DATE_ORDER`, `DATA_DUPLICATE`,
`DATA_REMOTE_PATH`, `DATA_FORMAT`, `DATA_READ`, `DATA_EMPTY`, `DATA_NULL`,
`DATA_TYPE`, `DATA_AMBIGUOUS_JOIN`, `DATA_COLUMN_COLLISION`, `DATA_FOREIGN_KEY`,
`DATA_AMOUNT`, `DATA_CURRENCY`, `DATA_CLOSED_EPISODE`. Validation report codes
are `MISSING_FIELD`, `INVALID_KEY`, `DUPLICATE_KEY`, `INVALID_AMOUNT`, `INVALID_DATE`, `DATE_ORDER`,
`INVALID_KIND`, `FOREIGN_KEY`, `BEFORE_DEFAULT`.

Small/medium in-memory tables are supported. Joins and event preparation use
explicit scans; they are not an indexed database engine. These are deterministic
data preparation rules, not an empirical validation or regulatory certification.
