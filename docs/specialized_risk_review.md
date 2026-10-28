# Specialized finance: competing exits and conditional arrangements

Python `examples/specialized_risk_review.py` exposes
`run(data=None, verify_reference=True)`; the native R example exposes
`run_specialized_risk_review(data=NULL, verify_reference=TRUE)`. The optional input
is the dictionary/list schema in `specialized_risk_review.json`. Both examples
execute all five actual fits and all ten candidate cashflow valuations.
`verify_reference=False` disables only the independent synthetic assertions.

Each product has an independent historical cohort of 20 entities observed during
2018–2020 and available before 2023 origination. Defaults and competing exits at
one year have distinct counts by product; remaining entities are right censored at
two years. A cause-specific proportional-hazards model with no covariates is
actually fitted using Breslow ties. It is a baseline-only fit, not an asserted
relationship between synthetic features and default. The independent oracle uses
`h_default=d/20`, `h_exit=e/20` and total mass `1-exp(-(d+e)/20)`. Incidence allocates
this mass proportionately to the two hazards, leaving the remaining survival
probability. It deliberately does not treat an exit as a default or estimate a
one-cause default curve by silently removing competing events.

Each March 2024 offer compares continuing operations with an orderly asset sale.
Recovery forecasts, asset-specific delay, acceptance probability, direct cost,
upfront cost and rejection value are all explicit in the packaged resource. The
financial oracle is `u*(discounted accepted net cashflow - upfront cost) +
(1-u)*rejected value - offer cost`, using a declared 5% effective annual curve.
The assumptions differ for real estate, projects, aircraft, ships and equipment;
these examples do not supply market estimates for any of them.

The output preserves training metadata, full competing-event curves, full
candidate evaluation and seals, and the independent analytical values. This is
conditional scenario comparison with `CONDITIONAL_SCENARIO_NOT_CAUSAL`
identification: historical treatment selection is not repaired, missing
counterfactuals are not supplied, and no arrangement is executed. Outcomes are
not relabelled to validate the preferred alternative. Portfolio review, residual
workout estimation and alternate macro methods remain separate acceptance work.
