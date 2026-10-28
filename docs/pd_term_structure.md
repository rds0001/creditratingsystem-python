# Default curves and rating migration

`conditional_pd_from_cumulative` reverses an ordered, nondecreasing cumulative
first-default curve. The conditional probability is the increase in cumulative PD
divided by survival at the preceding endpoint. After survival becomes zero, that
conditional quantity is undefined: the result uses null and an explicit
`undefined_after_full_default` mask. It does not invent a zero conditional risk.
Empty curves remain empty. Existing `cumulative_pd_from_conditional` and
`marginal_pd_from_conditional` provide the forward cumulative and unconditional
period-mass calculations.

`hazard_to_pd(hazard,interval_lengths)` expects one nonnegative annual hazard and
one positive year fraction per interval. Integrated hazards determine conditional
PD, cumulative PD, survival and unconditional marginal default masses. A hazard
is a rate, not a probability: constant intensity lambda gives cumulative PD
`1-exp(-lambda*T)`. Stable exponential-minus-one evaluation preserves tiny risks.

## Estimation from a bank's dated observations

`estimate_transition_matrix` implements two explicitly different estimands:

- `cohort`: endpoint transition counts divided by observed origin counts.
- `duration`: exact state-change counts divided by time at risk in the originating
  state; diagonal entries are negative total outward intensity.

Provide ordered `states`, `default_state`, `estimator`, `knowledge_date` and
`observation_scheme`, plus record lists (R also accepts data frames):

| Input | Fields |
|---|---|
| ratings | entity_id, at, known_at, state |
| observation_windows | entity_id, start, end, complete_until, known_at |
| exits (optional) | entity_id, at, known_at |

Dates are ISO calendar dates. One observation window per entity is accepted per
call. Resolve competing ratings/revisions at the same date upstream; duplicate
dates are rejected. State at window start is the latest known rating at or before
start. Missing initial states are excluded with a reason. Known rating changes
then update the state on `(start,end]`. Default is an absorbing event; observed
cures out of the declared default state are rejected under this contract.
Represent default occurrences as rating events in `default_state`.

The window's `complete_until` declares how far observation is complete, and its
`known_at` must establish that completeness by the knowledge date. Future-known
ratings and exits are ignored and counted in diagnostics. Completeness itself is
a user assertion: inconsistent reporting systems require upstream reconciliation.
Events and exits cannot be known before their occurrence. Exit on the same date
as a rating change includes that change, then censors subsequent observation.

Cohort estimation requires a positive integer `cohort_period_days` and equal
window lengths. A borrower censored before the endpoint is excluded unless
absorbing default has already occurred. This is a **complete-case estimator**;
informative exit can bias it. It does not treat an exit as survival or default.
`observation_scheme='state_carried_until_change'` explicitly states how ratings
remain effective; `exact_event_times` is also accepted.

Duration estimation requires `observation_scheme='exact_event_times'`: all actual
rating transitions must be observed, not merely periodic snapshots. Exposure
ends at the earliest of window end, complete observation cutoff and exit. Calendar
days are divided by explicit `days_per_year` (default 365.25). Counts/exposure
are the maximum-likelihood annual generator under homogeneous Markov dynamics.
For off-diagonal intensity q, the relevant log likelihood is
`N_ij log(q_ij) - q_ij * exposure_i`; differentiation gives `N_ij/exposure_i`.
Periodic panel observations require a different likelihood and are rejected for
this duration estimator.

Nondefault origins without a denominator raise `EMPTY_TRANSITION_ROW`. An
explicit `empty_row_policy='identity'` substitutes a no-migration row (zero
intensities for duration) and lists every unidentified state. This is an
assumption, not an estimated zero rate. Default's absorbing row is imposed even
if no defaults appeared in the sample.

## Projection

`project_transition_matrix(matrix,horizon,matrix_kind=...,states=...,
default_state=...,step_length=...,time_unit='years')` uses origin rows and
destination columns. A discrete `transition` matrix requires nonnegative
stochastic rows and a positive `step_length` in years. Horizon must be an integer
number of those periods; projection is an integer matrix power. The routine
does not silently take a fractional matrix power or infer a generator.

A `generator` contains annual intensities, nonnegative off-diagonals, nonpositive
diagonals and zero row sums. Supply no discrete step. Projection is `exp(Q*T)`,
using SciPy or native R Matrix matrix-exponential routines. Default's generator
row must be exactly zero; its transition row must be exactly absorbing.

The result includes the projected matrix, cumulative `pd_by_origin` and
`survival_by_origin`. Projection at zero is identity, including a PD of one for
an already-defaulted origin. Nonnegative stochastic output is checked, with only
tiny numerical noise corrected after validation. The dynamics are assumed Markov
and time-homogeneous; these assumptions are not established by the function.

PIT/TTC factor conditioning is documented separately in [ttc.md](ttc.md).
It is not implied by a migration estimate or historical average.

Matrix-exponential implementation:
[R Matrix documentation](https://stat.ethz.ch/R-manual/R-devel/library/Matrix/html/expm-methods.html).
For continuous-time state models and the distinction between exact events and
panel observations, see [Jackson (2011)](https://www.jstatsoft.org/article/view/v038i08).
