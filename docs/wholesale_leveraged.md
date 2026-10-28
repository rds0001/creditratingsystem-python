# Synthetic leveraged loan component

Execute `examples/wholesale_leveraged.py`, or source the installed R example and call `run_wholesale_leveraged()`. Packaged `wholesale_leveraged.json` specifies every synthetic input. Native scorecard fitting, separate holdout calibration, observed event-curve estimation, debt scheduling, default detection and discretionary sponsor support are executed.

Three independently routed facilities have debt/EBITDA 6 (sponsor backed), 8 (high leverage), and 6 (covenant lite). Their training rates are 0.20 at leverage 4 and 0.40 at 8 over two years. Missing verified loan terms prevents routing. Training, calibration and prediction dates are separated; hidden truth is not used for estimation.

The loans pay 10% simple annual interest, ACT/365F, with principal due at 2026-01-01. The first coupon includes 366 days; the second includes 365. Debt 120 with positive annual CFADS 18 leaves maturity shortfall 114. Debt 160 with CFADS 20 leaves shortfall 156. No refinancing commitment or sponsor cash is assumed in these paths. Asset values 95 and 125 respectively are below the outstanding claims. The declared dual cashflow/value test identifies default at maturity, not at the earlier coupon.

The maintenance covenant threshold is leverage 5. A breach is monitored but is not automatically UTP or payment default. Covenant-lite has no maintenance test; it still owes full interest and principal. Discretionary sponsor rescue is a separate probabilistic calculation: capacity 130 covers the declared 114 requirement, but willingness 0.5 and execution probability 0.8 prevent interpretation as a guarantee or booked cash.

An independent ten-obligor observed vintage has one default at year one and one at year two. Its Kaplan–Meier survival is 0.9 and 0.8. That cohort curve is distinct from each facility's calibrated PD and from the deterministic unfunded stress path. All targets are PHYSICAL/PIT; no TTC or risk-neutral conversion is inferred.

This is component/variant acceptance, not completion of all eight lifecycle phases. Python/R compare every output at absolute tolerance 1e-12 and relative tolerance 1e-10 with no field exceptions.
