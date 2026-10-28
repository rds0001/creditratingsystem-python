Lifetime risk with competing account exits
=========================================

Run ``examples/lifetime_risk_review.py`` or source the corresponding native R
example and call ``run_lifetime_risk_review``. The packaged JSON contains a small,
independent synthetic history. It demonstrates calculation and validation; it
is not an empirically calibrated bank model or a release approval.

The prior five-account cohort starts on 2022-01-01: prepayment at day 15,
administrative censoring at 30, defaults at 45 and 60, and another account
observed through 60. Model availability is 2022-03-03, before all 2024 forecasts.
Both cause-specific Cox hazards and Fine–Gray are genuinely fitted to those
records. Intercept-only designs provide independently checkable estimates.
Cause-specific prepayment hazard is 1/5; default hazards are 1/3 and 1/2.
Joint integration gives default CIF exp(-1/5)*(1-exp(-1/3)) at day 45 and
exp(-1/5)*(1-exp(-5/6)) at day 60. Survival plus both CIFs equals one.
Fine–Gray retains the exited account with IPCW: increments are 1/3.75 and
1/2.75. Its fitted CIF is an alternative estimate, never added to Cox CIF.

The target is first account default before terminal contractual prepayment.
The held-out prepaid account has observed terminal exit history and explicit
coverage of that target through day 61. Retaining it as a noncase in the
subdistribution validation does not imply continued exposure, later issuer
nondefault, or unpaid hypothetical contractual continuation. Another held-out
account really is administratively censored at day 30. Independent administrative
censoring has frozen survival .75 after day 30, with prior support through 90;
this explicit synthetic DGP is separate from Fine–Gray's fitted censoring law.
Brier validation uses IPCW and divides by all six accounts, not observed weight.
Supplied bootstrap draws resample whole accounts. Four illustrative draws are
not a reliable bank confidence interval; non-estimable rank intervals stay null.

Amortization is known at origination. Default immediately before period-end
repayment gives conditional EAD 100, 90, 70. The default-conditional fractional
LGD fit to .2, .4, .6 gives .4. The recipe records its prediction to twelve decimal
places to make replay inputs identical across languages. Deterministic exposure
and constant conditional LGD justify their product here; general joint EAD/LGD
dependence needs its conditional product expectation. Only differences of CIF
enter expected loss. The loss API uses years, so days are divided by 365 exactly
once. Reusing cumulative probabilities as period masses demonstrably overstates
loss even when their sum remains less than one.

Three independent held-out default balances have EAD errors of -10 each.
Two finalized workouts have realized LGD .2 on EAD100 and .6 on EAD80; the
count-weighted bias is zero and EAD-weighted bias is 4/180. The third workout is
explicitly OPEN and is excluded even if an invalid purported final label is
present. Its actual dated recovery ledger is supplied separately. A genuinely
prior frozen recovery hurdle estimates the remaining cash flow; no later cash
flow becomes a March observation. Removing the explicit residual-tail discount
assumption makes final LGD unidentified. The finite tail is an assumption-bound
estimate, not a completed outcome or a statistically established confidence band.

The example executes all ten lifetime-risk methods, with real supporting LGD
fit and contractual exposure scheduling. Tests include temporal leakage,
unsupported terminal noncase history, exclusion of open loss labels, analytic
CIFs, IPCW Brier, EAD/LGD arithmetic and full native-language output parity.
Default tests use this tiny real fit. The installed acceptance call is the same
complete recipe, not an isolated function smoke test.

Method references: Fine and Gray (1999), A proportional hazards model for the
subdistribution of a competing risk, JASA 94:496–509, DOI10.1080/01621459.1999.10474144;
Graf et al. (1999), Assessment and comparison of prognostic classification
schemes for survival data, Statistics in Medicine18:2529–2545,
DOI10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5.
The independent arithmetic above is derived from the declared synthetic cohort.
