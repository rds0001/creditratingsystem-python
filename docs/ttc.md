# Cycle conditioning and observed long-run default rates

`pit_ttc_transform` implements a declared Gaussian latent-factor model. It requires
an explicit horizon, PHYSICAL probability measure and a standard-normal factor
specification with factor ID, reference period and model version. A positive
systematic factor represents a good economic state. For asset correlation rho:

    PIT(z) = Phi((Phi_inverse(TTC) - sqrt(rho) * z) / sqrt(1-rho))
    TTC = Phi(sqrt(1-rho) * Phi_inverse(PIT(z)) + sqrt(rho) * z)

TTC averages conditional PD over the specified standard-normal distribution.
Evaluating PIT at z=0 generally does not recover TTC. This operation neither
estimates factor sensitivities nor determines whether a bank's observed rating
is PIT, TTC or hybrid. A shared horizon and default event interpretation are
essential. Physical and risk-neutral probabilities cannot be substituted.
Correlation must be in [0,1); endpoints 0 and 1 for PD are preserved. Finite
precision can saturate a conditional result at an endpoint; diagnostics report
loss of inverse information. Scenario metadata supplied in context is retained.

`estimate_long_run_pd` accepts explicit selected cohort IDs, knowledge date,
weighting, uncertainty, dependence assumption, representativeness assessment and
target. Each cohort supplies `cohort_id`, `period_start`, `period_end`, `known_at`,
`defaults`, `denominator`, `mature`, `default_definition`. Periods are start-inclusive
and end-exclusive, one calendar year long. A February 29 start ends February 28
of the following year. Mature complete outcomes must be known at the knowledge
date. Upstream event construction must establish complete denominators and handle
censoring, selection and reporting delays. Counts alone cannot verify that work.

Annual-mean weighting averages cohort default rates equally; pooled-obligor
weighting divides summed defaults by summed denominators. These are distinct
estimands: 1/10 and 2/100 give 6% annual mean and 3/110 pooled rate. Neither is
necessarily TTC. A representative full cycle, consistent populations and suitable
conditioning must be justified separately. `representativeness` contains
`assessment` (representative, not_representative or unknown), `rationale` and
`reference_period`; the result marks this as an unverified user assessment.

Uncertainty is explicit: `none` returns null bounds; `year_t` uses the sample
standard deviation of annual rates and a Student-t critical value, intersected
with [0,1], requiring annual-mean weighting, at least two nonoverlapping years
and `independent_years`. This interval is approximate. `clopper_pearson` is an
exact binomial interval requiring pooled weighting, nonoverlapping periods and
`independent_homogeneous_obligors`. Repeated obligors, macro dependence or varying
probabilities invalidate that assumption. The method rejects incompatible
settings; it cannot infer independence from aggregate counts.

Both functions return CalcResult. Long-run `value` has `pd`, `lower`, `upper`;
diagnostics preserve cohort selection, rates, counts, assumptions and target.
The long-run target contains `event`, `default_definition`, `horizon=1`,
`time_unit='years'`, `probability_measure='PHYSICAL'` and `conditioning`.
No regulatory acceptance or customer-specific calibration is implied.

The Gaussian conditional-default model is explained in the
[BIS explanatory note](https://www.bis.org/bcbs/irbriskweight.pdf).
The exact interval follows the method documented by
[R stats::binom.test](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html).
Independent checks integrate conditional PD over the Gaussian factor, use the
analytic zero-default binomial bound and the df=1 Cauchy quantile for the t interval.
