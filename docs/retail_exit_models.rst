Retail default and competing exit models
=======================================

The ``retail_exit_models`` example fits cause-specific hazards, a Fine-Gray
subdistribution model, grouped cloglog hazards and a lognormal time-to-exit
model in both Python and native R. Pass your own input object to ``run`` or
``run_retail_exit_models`` with ``verify_reference`` disabled to use other
observed cohorts. The independent uncensored lognormal likelihood check still
applies; censored AFT input requires using the granular public API directly.

The twelve product profiles have explicit day units and 7-, 30- or 90-day
time scales. All development observations are synthetic historical records
known before the 2024 application and collections examples. Different methods
use explicitly separate cohorts and estimands; no fitted exit model is used
to relabel a censored default outcome. The examples demonstrate estimation
mechanics and event definitions, not product-specific empirical calibration.

Eight observed subjects include one default and two competing exits at the
first time point. Joint cause-specific integration gives survival exp(-3/8),
default incidence (1-exp(-3/8))/3 and twice that competing-exit incidence.
Treating competing exits as if they remained exposed would produce a different
estimand. The Fine-Gray cohort independently gives retained weighted risk sets
3.75 and 2.75 at its default times. Two defaults in ten fully observed grouped
intervals give survival 0.8, with square-root survival at half an interval under
the explicitly constant cloglog intensity assumption.

Six fully observed lognormal exit durations have log deviations
(-1.5,-1,-0.5,0.5,1,1.5) from the log time scale. Hence the exact maximum
likelihood location is the log scale, and the log-time standard deviation is
sqrt(7/6). Each runtime checks its fitted likelihood against this independent
closed-form maximum within 1e-7 and its survival probabilities within 1e-5.
Independent optimizer outputs are retained, not rounded into apparent equality.
Cross-language acceptance permits absolute 1e-5 only on the enumerated fitted
lognormal coefficients and predictions; all other results remain subject to
the strict default comparison. No content-hash exceptions are used.

Default tests cover one case for each time scale. The installed acceptance
example executes all twelve. No large simulation is required by these tests.
