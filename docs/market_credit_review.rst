Market quotes, identification and observed physical defaults
==========================================================

The ``market_credit_review`` examples run the nine public market-review methods
as one calculation: dated quote normalization, CDS/bond pricing, both hazard
calibrations, recovery identification diagnostics, matched credit basis, a fitted
historical bridge, and a typed risk assessment. Local quotes and instruments,
discount assumptions, observed issuer outcomes and forward folds are replaceable
inputs. No market feed or network connection is used.

The synthetic one- and three-year quotes come from independent flat-curve
formulas: discount rate .03, Q hazard .02, recovery .4. Continuous CDS spread is
.6 * .02 = .012 (120 basis points). A bond pays 105 once at maturity T and recovers
40 at default; its value is 105 exp(-.05T) + .8(1-exp(-.05T))/.05. Both calibrators
start separately and recover .02 from the supplied prices. A future quote is
retained visibly as ineligible, not incorporated into the fitted curve.

Two CDS quotes identify the two hazard parameters conditional on recovery. They
do not independently identify a third recovery parameter: recovery .2 gives
hazard .015, and recovery .6 gives hazard .03 at the same spreads. A separately
supplied synthetic recovery observation adds information and changes the rank.
The matched CDS/bond basis is zero in this constructed common-contract world.
None of these examples asserts that recovery is known from real quotes alone.

The physical bridge uses 300 distinct synthetic issuers across 2018, 2020 and
2022. Within each cohort, Q-PD .01 has one observed default among 50 issuers;
Q-PD .05 has five among 50. This yields independently known fitted physical
probabilities .02 and .10. Two strictly forward folds evaluate 200 mature
observations; 100 warmup observations have no invented out-of-time prediction.
The independent Brier value is (.02*.98 + .10*.90)/2 = .0548. Reused issuer IDs
across folds and future-known market releases are rejected.

For the current Q-PD 1-exp(-.02), the fitted bridge gives approximately .029986
physical PD. The assessment retains separate Q and P components, with separate
probability measures and explicitly different interpretations. This conditional
association applies to the declared quoted-issuer population. It is not a
universal change of probability measure or an empirical claim about real loans.

By default the example returns a financial report with the complete quote
snapshot and pricing quantities, fitted hazards/bridge coefficients, identification
findings and typed assessment components. ``return_artifacts=True`` (R: ``TRUE``)
also returns both full calibration artifacts, the fitted bridge and full locally
sealed assessment, including backend diagnostics. Fresh optimizer artifacts may
have different numerical diagnostics and content hashes. The financial report
uses strict ordinary numerical parity; foreign frozen artifacts are independently
replayed in each language, and each supplied calibration content seal is verified.
Those are separate checks, not a claim of identical fresh optimizer artifacts.
