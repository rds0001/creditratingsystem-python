Sampling-aware simulation risk
===============================

``estimate_simulation_risk(path_records, sampling_contract, confidence_level=.95)``
requires one subject and unique scenario trials. Records explicitly contain
scenario_id, subject_id, initial_default, defaulted, horizon_complete, weight,
ead_at_default and economic_loss. Nondefaults have zero default EAD and loss.
Initial defaults are excluded from the initially performing risk set; incomplete
positive-weight remaining trials are rejected, not discarded.

The contract declares contract_id/version, subject_id, currency, probability_measure
(PHYSICAL/RISK_NEUTRAL), horizon, time_unit (days/months/years),
conditioning='initially_performing', event='first_default' and sampling_design.
IID_UNWEIGHTED requires unit trial weights and permits exact two-sided
Clopper-Pearson PD intervals. FIXED_WEIGHTED_SCENARIOS and DEPENDENT_OR_QMC have
normalized scenario means but no IID uncertainty. Kish effective weight count is
only a concentration diagnostic. Independence cannot be established from row IDs.

With zero defaults in n IID trials, the two-sided upper confidence limit is
1 - ((1-confidence_level)/2)**(1/n), which is positive. A zero plug-in PD standard
error must not be read as certainty. Loss standard error is a sample moment under
the IID assumption, with no claim of reliable rare-tail coverage.

Expected loss uses the joint monetary loss on each path. Conditional mean EAD
times conditional mean LGD generally differs from conditional mean loss. The
output therefore also reports EAD-weighted LGD; when defined,
PD * mean(EAD | default) * EAD-weighted-LGD equals expected loss. Zero-EAD defaults
leave the individual LGD mean undefined. Raw losses, negative LGDs and LGDs above
one are retained. An empty risk set has an explicit status and no invented PD.

Reference: Clopper and Pearson (1934), DOI 10.1093/biomet/26.4.404;
https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html
