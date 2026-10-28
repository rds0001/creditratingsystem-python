Education loan variants
=======================

Run ``education_variants.py`` or installed ``run_education_variants()`` in R.
All payment, interest and guarantee conventions are explicit synthetic inputs.
The JSON contract can be replaced with locally approved bank assumptions.

A genuine grace period has no due payment and therefore no DPD from nonpayment.
The interest-free example leaves principal12,000 unchanged. In the distinct
capitalizing deferral, two supplied monthly effective rates of1% increase the
claim to12,000 x1.01 x1.01 =12,241.2. This is not implied by payment relief itself.

The supplied income-contingent rule requires10% of positive monthly income above
2,000. Income1,500 requires zero payment; income3,000 requires100. At the supplied
zero interest rate, principal becomes11,900. The example explicitly applies this
rule before passing the contractual payments to the cash ledger; no national
student-loan law or automatic cancellation is assumed.

A parent guarantees at most3,000 of the12,000 claim. Borrower PD0.1, guarantor PD0.2
and explicitly supplied joint default probability0.02 yield unprotected expected
loss1,200. Full payout in borrower-only default and no payout on joint default give
protected loss0.08 x9,000 +0.02 x12,000 =960. Borrower PD is unchanged. An ineffective
guarantee leaves loss1,200. The parent's4,000 income is not treated as borrower
income merely because the parent is a guarantor; the student's1,500 stands alone.
These are separate product variant and protection scenarios, with empirical
calibration and full lifecycle validation remaining separate requirements.
