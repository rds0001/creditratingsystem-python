# Specialized state histories and monitoring

Python `run(data=None, verify_reference=True)` in
`examples/specialized_monitoring.py` and native R
`run_specialized_monitoring(data=NULL, verify_reference=TRUE)` accept the schema
in `specialized_monitoring.json`. The default executes five actual multistate
estimations, five feature drift tests, and five sequential PD drift calculations.
Reference verification controls the hand-oracle assertions only.

Each historical product cohort contains 20 borrowers. Between one and five
borrowers default at year one, cure at year two and default again at year three.
The remaining borrowers stay performing through the observation endpoint.
The multistate estimator uses all observed dated transitions. Its first-hit
default probability stays at `d/20` after year one, whereas default-state
occupancy is `d/20`, zero, `d/20` at the three dates. This checks why a cure cannot
erase an already observed first default. These are observed historical cures,
separate from legal settlement closures in the workout examples.

The feature example compares two observed reference DSCR values with two later
DSCR values reduced by an explicitly product-specific amount. All 24 row
permutations are supplied. The independent Wasserstein oracle is that common
reduction. Exchangeable rows and no multiple-testing adjustment are explicit
method assumptions for each separate two-sample illustration.

PD monitoring builds three disjoint observed cohorts per product. Their forecasts
use a prespecified synthetic two-year 10% benchmark, dated before the outcomes;
they are not backcasts of the later fitted macro model and do not claim such a
model existed historically. All 20 outcomes per cohort are mature at its cutoff.
Residuals are computed from observed default rate minus mean recorded forecast.
The independent two-sided EWMA statistic is the absolute value of the recursion
`z_t=.5*z_(t-1)+.5*(d_t/20-.1)`, starting at zero. The complete supplied null paths
are a declared synthetic monitoring reference, not an empirically estimated null
or a claim of live-portfolio false-alarm control. Dependence assumptions and
calibration date are retained by the method's output.

Output includes complete state curves, feature/drift statistics, cohort source
counts and residuals, and independent EWMA values. Defaults, cures, benchmark
forecasts and null paths are all explicit synthetic records. These examples
provide state-history and monitoring evidence; alternative macro/dependence
recipes remain separate acceptance work.
