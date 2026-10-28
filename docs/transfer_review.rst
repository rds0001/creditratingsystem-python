Dated transfer aggregation
=========================

Run ``examples/transfer_review.py`` or source ``examples/transfer_review.R`` and
call ``run_transfer_review()``. The bundled inputs are declared synthetic
probabilities and payment-route assumptions, with no inferred jurisdictional law.
For own data, call Python ``assess(data, known_at=...)`` or R
``assess_transfer_review(data, known_at=...)`` with the resource's explicit schema.

The performing physical one-year base PD is 10%. A payment freeze occurs with
probability20%; conditional unmitigated default is50%, and a prior liquidity
arrangement prevents20% of those failures. Transfer-default PD is therefore8%.
The dependence sensitivities use independence, nesting and the lower Frechet
intersection. The corresponding union PDs are17.2%,10%,18%. Four disjoint states
have losses0,60,20,70EUR (neither, obligor-only, transfer-only, both), giving
expected losses7.52,6.8,7.6EUR. Joint loss is an explicit conditional amount,
not an average of standalone losses. The stated amounts include any contractual
EAD and recovery assumptions; there is no second multiplication by PD or LGD.

An exemption first known in March cannot change the original January result.
With March knowledge the exemption removes this specified transfer mechanism;
base expected loss remains6EUR. A missing payment route yields insufficient
information, not zero risk. Equal currencies do not remove a payment freeze,
but do remove the supplied conversion-restriction case. These are explicit
policy assumptions, not a legal opinion. Grades use unrounded union PD.

The example preserves the original assessment and verifies exact replay hashes.
Already adjusted input PD and reuse of the same liquidity effect at the loss
layer are rejected. Upper/lower intersections are sensitivity assumptions, not
empirical estimates of event dependence. A mapped default grade is not an
observed default event. The stress test also covers a nonzero Frechet lower bound.
