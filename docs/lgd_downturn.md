# Declared adverse-scenario LGD comparison

`estimate_downturn_lgd` applies one frozen conditional LGD model to matched baseline
and adverse covariates. Supply `feature_order`, `expected_target`, separate optional
precision designs, and one nonnegative aggregation weight per named portfolio unit.
Weights normalize using a scaled sum, so equally large finite balances do not cause
an intermediate overflow. Zero-weight units do not trigger the support policy.

The required `downturn_definition` records `definition_id`, `version`,
`selection_basis`, `baseline_scenario_id`, `adverse_scenario_id`, `known_at`, `as_of`,
ordered unique `unit_ids`, and `extrapolation_policy='error'|'report'`. Policy and
model must be known at evaluation; context as_of must match if supplied. Selection
of an economically adverse condition is caller policy evidence. The function does
not search observed losses for a favorable or unfavorable period.

Outputs include baseline/adverse means, their signed difference, unit-level LGDs,
normalized weights and separate marginal support flags. Adverse-minus-baseline
can be negative if the fitted model implies that response. No floor or standard
surcharge is added. Identical scenarios give exactly zero difference. Marginal
support diagnostics do not establish joint support or causal identification.

Uncertainty is `NOT_ESTIMATED` by default. Optional `model_draws` must contain at
least two compatible frozen artifacts from the same family, target and feature
contract. The same draw evaluates both scenarios, preserving paired differences.
Declare `uncertainty_contract={'source':'bootstrap'|'posterior'|'user_supplied',
'interpretation':...}`. Linear empirical quantiles (R type7) at the selected
`interval_level` summarize baseline, adverse and paired-difference draws. They
retain the supplied inferential interpretation; the function does not turn two
illustrative scenarios into a justified statistical confidence interval.

An intercept/group fractional model with group means0.25 and0.75 gives a difference
of0.5 when all units move from group0 to group1. Reversing those scenarios gives
−0.5. This method implements a declared model comparison, not regulatory downturn
LGD certification or model approval.
