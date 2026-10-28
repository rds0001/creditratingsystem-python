Mortgage phase diagnostics on the original facility
==================================================

Run examples/mortgage_phase_review.py or source the native R example and call
run_mortgage_phase_review(). This supplements mortgage_household_review on its
actual facility_A, original model/calibration/policy and actual held-out ledger.
It does not create another mortgage with the same display label. Use
return_artifacts=True (TRUE in R) to retain original upstream artifacts, frozen
prior cure/exit models and exact scenario inputs. Native optional upstream,
house_data and life_data inputs support replay of previously retained native
artifacts; the default loads and executes the installed upstream example.

An August31 payment delay makes the February assessment stale. The actual
August delinquent and September default endpoints produce one delinquency-to-
default roll observation; unseen origin rows stay unsupported. Descriptive PSI
compares actual assignment score against the eight prior mortgage training rows.
One mature annual default is compared with its previously frozen pool PD0.25.
EWMA uses smoothing0.5 and explicit supplied one-step Bernoulli-null paths known
before monitoring. These finite reference paths define the calculation; they
are not independent prospective false-alarm validation or empirical evidence
from a large population. The original observation is a single linked household.

Two separate prior2023 histories support the fitted multistate cure model:
one cures at30 and redefaults at60; the other remains in default. Their dates
are checked to precede the actual September2024 default. Transfer of this
synthetic reference to the mortgage is an explicit modeling assumption, not a
claimed empirical mortgage calibration. A hypothetical cure-state receipt is
0.8*95000; its probability0.5 gives expected receipt38000. This is a forecast
at default, not observed cure. Actual CLEAN recorded January1 has complete
coverage only through March1, less than90-day probation; the actual cure target
stays unknown. Historical default is retained and no later cure is invented.

Four independent synthetic2022 histories provide a separate first-default/exit
reference: one default and one exit at30, two observed through60. Cause-specific
hazard increments are0.25 each. Proper joint integration gives both CIFs
0.5*(1-exp(-0.5)), with survival exp(-0.5). These are alternative first events,
not independent cumulative probabilities added without survival. Native fits
and frozen predictions use the declared sixty-day target and calendar support.

The actual recovery ledger has default EAD95000, sale recovery70000 on
August31 2025 known September1, economic discount0.9 and closure September15.
The final recovery-episode linkage includes that cash; the October1 2024 known
view excludes it. Actual discounted recovery63000 gives loss32000. Multiplying
this realized conditional severity by the original pool PD0.25 gives8000 as
an explicitly retrospective event-conditioned diagnostic, not an ex-ante loss
forecast that knew the eventual sale.

The actual interim cash ledger is incomplete: default EAD95000 and September
statement principal94500 do not identify a500 cash recovery. The proposal also
has separate interest600 and fees100. Neither a principal adjustment nor a
cash payment is invented. The open-recovery calculation therefore is explicitly
CONDITIONAL_SCENARIO_ONLY, not identified actual open LGD. It declares hypothetical
residual95000, no supplied interim cash, a prior residual-terminal model and
flat0.95 future economic discount including the unresolved tail. It never uses
the later sale amount or observed0.9 discount. Its independent mean-ratio oracle
is r=(1+sqrt(0.25*0.75)*exp((log(3)/2)^2/2))/3, giving loss95000-0.95*95000*r.
Removing the tail valuation makes final LGD unknown rather than zeroing the
unresolved probability. This conditional output cannot cure missing actual
cashflow completeness or serve as an empirically validated current estimate.

The same unaccepted plan_A is evaluated under declared conditional monetary
scenarios: accepted expected recovery38000, accepted cost100, declined19000,
uptake0.25 and offer cost10. Both treatment evaluation and collections integration
produce23715 expected net value. Uptake applies once. These values are assumed
terminal-stage recovery economics, not the two contractual1000 installments
and not additional observed payments. The original proposal affordability is
retained; no accepted treatment, contractual change or causal effect is inferred.

The original approval and actual closure transition the assessment from DRAFT
to FINAL to SUPERSEDED. Portfolio preparation retains one gross100000 funding
position and collateral solely in conditional LGD, with no duplicate collateral
deduction or inferred group default. Concentration is deliberately one household.
Default-count and pool-parameter estimates use the same mature facility outcome
and actual closed workout. They are small-sample diagnostics, not refitting of
the original scoring model or a population performance claim.

Tests check prior-only fits, missing-probation cure, exact CIF and monetary
identities, local reexecution of original scenario inputs, and rejection of a
future-known cure model. Default verification remains small; there is no long
MCMC, large simulation or stress sample in this example. Original artifacts and
seals are retained, and financial comparisons use absolute1e-10/relative1e-12
without rounding sealed reports. See survival, retail_workout_cases and
mortgage_household_review for the underlying method contracts and references.
