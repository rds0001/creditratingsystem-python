Retail monitoring and the same facilities in portfolio risk
===========================================================

``examples/retail_portfolio_cases.py`` and the native R equivalent consume the
identical packaged ``retail_portfolio_cases.json``. Twelve retail product cases
bind actual holdout observations, dated default/recovery records, borrower and
household links, original exposures, frozen risk forecasts and prior-only pool
training. Product exposures and facility commitment conventions are explicit:
revolving lines have an undrawn reference amount; amortizing loans and leases
have no invented unused commitment for a CCF calculation.

The miniature cohorts are deliberate analytical cases, not empirical banking
calibrations. They share a six-unit reference design with product-specific IDs,
amounts, strata and facility conventions. Product contractual variants are
covered by their corresponding component examples; these examples establish
portfolio and monitoring phase integration.

Observed monitoring
-------------------

Three of six held facilities default. The original exposure weights are
``[100, 300, 200, 400, 100, 100]`` times the product scale. First-pool observed
count PD is 3/4, while its exposure-weighted PD is 600/1000. Two closed
workouts with losses .6 and .2 have exposure-weighted LGD .3; the third open
workout is excluded from final LGD but contributes its actual default EAD.
Frozen LGD .4 therefore has bias +.1, and forecast EAD 150 versus mean actual
EAD 200 has bias -50, in product-scaled units. These observed estimates never
replace the frozen PD/LGD/EAD inputs of the portfolio simulation.

All horizons mature before calibration metrics. An earlier cutoff retains
known default events while excluding every immature forecast from performance.
Removing negative-outcome completeness leaves that outcome unknown. Vintage
counts retain original cohort denominators; the next month remains immature.
Default-count backtesting is descriptive, without an independence-based
p-value. Separate LGD and EAD targets are explicitly conditioned on default.

Pool boundaries are fitted to independent 2023 training records and frozen
before the 2024 holdout. Household relationships affect concentration and
bootstrap clusters; they do not automatically propagate default. Supplied
whole-household resamples quantify conditional evaluation uncertainty. Replicates
without an identifiable metric remain explicit. Independent-unit intervals
computed by the descriptive pool estimator are not retained for the connected
household cohort. Tiny samples do not establish pool homogeneity or general
predictive performance.

Score drift uses prior-frozen numeric bins and descriptive PSI. Counts
``[2,2,4,2,0]`` versus ``[4,0,0,2,0]`` include the missing-value bin. Each cell
receives the declared .5 smoothing count; PSI is calculated from the resulting
shares and carries no uncalibrated significance claim.

Portfolio rating recipe
-----------------------

The same held facilities enter all ten portfolio recipe methods: preparation,
default simulation, migration, loss distribution, concentration, ES allocation,
paired incremental risk, asymptotic benchmark, factor stress and typed risk
assessment composition. Borrowers have separate legal default groups; household
context alone creates no cross-default edge.

The reference DGP explicitly assumes independent Bernoulli defaults with
origin-frozen probabilities .1 for four facilities and .5 for two, and supplied
conditional LGD .4. All 64 event states are enumerated with exact product
probabilities. Decimal probabilities are stored to twelve places, beyond the
seven places needed by this finite rational design, to avoid multiplication-order
noise in cross-language innovation hashes. No fitted or observed outcome is used
to choose these probabilities.

A physical three-state migration model has two performing grades with those
same default probabilities and absorbing default. Complementary uniforms produce
the same default events. Supplied terminal marked values equal remaining EAD on
survival and recovery value on default, so marked losses reconcile the direct
portfolio loss paths. Physical transition probabilities are not a pricing measure.

Expected shortfall allocates fractional probability mass at the quantile and
reconciles all position/household contributions. Removing one held facility uses
paired loss changes on the same scenarios. An independent two-name oracle uses
actual BNPL facilities with equal conditional loss L and PDs .1 and .5: losses
0, L, 2L have masses .45, .50, .05. EL is .6L and ES at .8 is 1.25L. Removing
the first facility changes EL by -.1L and ES by -.25L.

The zero-correlation asymptotic benchmark matches analytical EL. Dependence stress
uses the same systematic/idiosyncratic coordinates and fixed recovery paths.
Three common-factor cells crossed with the 64 idiosyncratic cells exactly
integrate default thresholds at factor loadings zero and one. Their endpoints
preserve the marginal PDs but increase tail concentration. This is **not** a
generic quadrature rule for intermediate loadings or an IID normal simulation
sample. It supplies no simulation confidence interval.

Typed portfolio assessment retains EL and EAD in EUR; assessment EL is rounded
to cents under an explicit presentation convention. Distribution calculations
remain unrounded. ES is never relabeled PD or assigned a grade without a separate
calibration scale. Frozen content hashes bind inputs and models; they do not
authenticate legal evidence.

Default tests run the twelve small cases, the two-name tail oracle, invalid
training roles, duplicated facility exposure and split-household failures.
Python/native R comparisons cover all returned values and content hashes.
