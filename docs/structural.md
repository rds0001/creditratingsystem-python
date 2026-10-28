# Structural credit risk

These functions expose asset-based valuation, terminal default and first-passage
default separately. All horizons and grids are in **years**, volatility is per
square-root year, and rates are continuously compounded per year. Values use one
explicit common currency. Scalar analytic methods deliberately reject arrays;
portfolio callers can retain their own obligor identifiers while applying them.
`context` is optional, but any measure, horizon or time-unit fields must agree.

## Equity valuation and asset inference

`merton_equity_value(A,D,sigma,r,T,payout=0,currency=...)` returns equity value,
delta, d1 and d2. Under the pricing measure, equity is a terminal European call:
`A exp(-qT) Phi(d1) - D exp(-rT) Phi(d2)`. A continuous payout yield q reduces
asset growth. This is the value of the terminal residual claim; intermediate
payouts received by equity are not included. Zero debt, zero volatility and zero
maturity have explicit deterministic limits. Delta at a deterministic payoff kink
uses the declared convention one half; d1 and d2 are null in degenerate cases.

`infer_merton_assets(E,sigma_E,D,r,T,asset_bounds=...,volatility_bounds=...,
currency=...,payout=0)` solves that price equation jointly with
`sigma_E E = sigma_A A delta`. Bounds are strictly positive ordered pairs. The
routine solves in logarithmic parameters and checks both scaled residuals against
`tolerance` (default 1e-9). Failure raises `NONCONVERGENCE`; a boundary-constrained
best fit is not presented as a solution. This inversion supplies no estimate of
the physical expected return. A single debt face remains a modeling choice.

## Terminal and path-dependent events

`distance_to_default(A,B,sigma,physical_drift,horizon,payout=0)` computes the
standardized physical terminal log-distance. At zero horizon/volatility or zero
barrier it returns a null standardized distance plus a named boundary status,
not an infinite JSON value.

`merton_default_probability(A,D,sigma,drift_or_rate,horizon,measure=...,payout=0)`
requires `PHYSICAL` or `RISK_NEUTRAL`. Supply the physical expected asset return
for P and the risk-free rate for Q. It evaluates terminal assets at or below debt.
At zero volatility or time the event uses the deterministic endpoint, including
equality. Positive GBM assets cannot hit a zero barrier. Q PD is not a historical
default forecast. The inputs alone do not establish PIT or TTC conditioning.

`first_passage_pd(A,barrier_spec,sigma,drift,horizon,measure=...,payout=0)`
uses continuous monitoring and includes a default at the starting time.
`barrier_spec` is `{kind: 'constant', initial_level: B}` or
`{kind: 'exponential', initial_level: B, growth_rate: g}`. For positive B below A,
set `x=log(A/B)` and `m=drift-q-sigma²/2-g`. The passage probability is
`Phi((-x-mT)/(sigma sqrt(T))) + exp(-2mx/sigma²) Phi((-x+mT)/(sigma sqrt(T)))`.
Log-CDF evaluation avoids ordinary exponential-times-tiny-tail overflow.
This event includes paths that cross and later recover; it differs from terminal
default. Deterministic paths and initial barrier breaches are handled separately.

## Shared-innovation asset paths

`simulate_structural_paths(asset_dynamics,barrier_schedule,time_grid,innovations,
measure=...,jump_counts=None,jump_innovations=None,bridge_uniforms=None)` takes a
paths-by-intervals standard-normal matrix and one barrier level at each time.
The strictly increasing time grid starts at zero. Dynamics contain `asset_value`,
`volatility`, `drift`, and optional `payout`, `jump_intensity`, `jump_log_mean`,
`jump_log_sd`. Counts follow Poisson(intensity × interval length); jump normals
are independent standard normals giving the aggregate conditional log jump
`count*jump_log_mean + sqrt(count)*jump_log_sd*jump_normal`. Drift is the expected
asset return before payout; log steps subtract the compensator
`intensity*(exp(jump_log_mean+jump_log_sd²/2)-1)` as well as sigma²/2.

All randomness is supplied by the caller. Python/R comparisons therefore share
actual draws, without assuming identical random number generators. Draw hashes
are included. The methods cannot verify independence or distributional suitability.
Zero intensity requires zero counts. Terminal lognormal increments are exact at
the chosen grid; jump event times within intervals are not modeled.

Optional independent uniforms in (0,1) add Brownian-bridge crossing decisions for
pure diffusion against positive, log-linearly interpolated barrier levels. Given
positive endpoint log-distances x,y, the crossing probability is
`exp(-2xy/(sigma²*dt))`. With jumps, this correction is rejected; grid monitoring
can miss crossings. `default_time_upper_bounds` reports the end of the first
crossing interval, not the unknown exact hitting time. After default, survival
remains zero for that path although unabsorbed hypothetical asset paths continue.
Reported survival standard errors are plug-in IID Monte Carlo estimates. They
exclude model, calibration and discretization uncertainty.

## Empirical distance calibration

`calibrate_distance_pd_mapping(distances,outcomes,metadata=...,weights=None,
evaluation_distances=None,bootstrap_indices=None,confidence_level=.95,
out_of_bounds='error')` estimates a decreasing, left-constant isotonic default
map from mature physical observed outcomes. It uses the common calibration sample
and target contract; see [calibration.md](calibration.md). The resulting frozen
`calibration_artifact` works with `apply_calibrator`. Larger DD cannot increase PD.
No external vendor default-frequency table is supplied or implied.

Optional bootstrap indices are a matrix with one complete row resample per row,
**zero-based in both languages**. At least two resamples and unit observation
weights are required. Each sample is independently refitted and clipped to its
own score support when evaluated. Percentile quantiles use linear interpolation
(R type 7). These are pointwise empirical bootstrap summaries, not guaranteed
confidence coverage for isotonic estimation. Without resamples, bounds are null.
Fit diagnostics explicitly say that out-of-sample validation has not occurred;
use a separate time-separated test population for validation.

## References and checks

[Merton (1974)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1974.tb03058.x),
equations 3b and 10–12, motivates equity valuation and volatility inversion.
[Black and Cox (1976)](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1976.tb01891.x),
section III and equation 7, supplies the moving-boundary first-passage setting.
Tests independently integrate discounted payoffs and hitting-time densities,
check deterministic limits, reconstruct asset parameters and compare bridge
simulation against the reflection-principle probability.
