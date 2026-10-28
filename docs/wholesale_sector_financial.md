# Financial, insurance and public-sector reference components

Run `examples/wholesale_sector_financial.py`, or source the R counterpart and call
`run_wholesale_sector_financial()`. Python `run(data=None,
verify_reference=True, return_fitted_models=False)` and R
`run_wholesale_sector_financial(data=NULL, verify_reference=TRUE,
return_fitted_models=FALSE)` accept the dictionary/list schema in
`wholesale_sector_financial.json`. With `return_fitted_models=True`/`TRUE`, the
result contains `report` and `raw_native_fit_details`, retaining all raw fitted
preprocessors, scorecards, holdout calibrators and unrounded predictions.
`fit_sector_case_models(case, target)` also returns the three fitted objects.
Reference verification disables only the independent synthetic assertions.

The eleven distinct cases cover banks, nonbank lenders, financial holdings,
life insurers, nonlife insurers, reinsurers, regions, municipalities, public
agencies, multilateral lenders and supranational bodies. Their respective driver
ratios use risk-weighted assets/core capital, funding leverage, holding-company
double leverage, technical reserve coverage, premium/capital, catastrophe
exposure/capital, debt/revenue, debt/cash surplus, budget dependence, loans/paid-in
capital and debt service/committed revenue. Each input includes its economic
meaning and common valuation/currency basis. These single-driver components
exercise granular methods; they do not claim to be a calibrated complete bank,
insurance or sovereign supervisory rating system.

Statements are selected by observation and knowledge date. The later revision is
unavailable at the original assessment date and becomes eligible only in April.
Financial ratios feed a training-only standardizer and an actual weighted
scorecard. Two independent training groups have specified observed default
frequencies; an independent holdout cohort fits a logistic calibrator. The raw
reference probability follows linear interpolation of the two group log odds at
the standardized ratio. Calibration declares a normalized score-equation tolerance
of `1e-9`; raw native predictions are checked against the analytical oracle within
`1e-7`. No calibrator is silently replaced with an identity function.

The reference reporting convention rounds calibrated PD to six decimal places
**before** support, constraints and audit artifacts. This is explicit output
precision, not a change to fitting or a claim that native fitted coefficients or
model hashes are identical. The opt-in raw results preserve those differences.
Saved raw models from both languages are separately loadable and replayable; the
standard core serialization functions retain schema and content integrity.

Support is a declared, allocated single-sponsor cash rescue, not an automatic
parent guarantee or diversified sovereign support calculation. Rescue succeeds
only in a permitted state with enough available capacity. The holding and public
agency have insufficient allocated capacity; the reinsurer blocks transfer of
funds. The hand oracle is `p-(p-joint_default)*conditional_rescue_success`.
Callable member capital is never treated as paid-in capital. The holding has an
explicit grade floor for its structural position; other bands are nonbinding.
Representative PD changes only when that declared grade constraint binds.

A local, dual-control qualitative grade override is active during March and
expired on April 1. It changes only the grade: continuous PD remains visible and
unchanged. Declared author/approver labels are local evidence, not authentication
or executed credit authority. Every case also demonstrates routing failure when
financial evidence is unverified. Full support, constraint and override envelopes
are returned, with their content seals and cutoff dates.

All statements, group counts, reporting conventions and support assumptions are
open synthetic parameters. This component covers the eleven required economic
variants; full dated lifecycle/portfolio evidence and other sector recipes remain
separate acceptance work.
