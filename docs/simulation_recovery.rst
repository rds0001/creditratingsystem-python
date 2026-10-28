Recovery on shared default events
=================================

``simulate_recovery_waterfall(default_events, claims, realization_paths,
recovery_policy)`` consumes supplied ultimate realizations on the same first-default
identities used for PD and EAD. It does not draw a second default or independent
collateral scenario. Each scenario/subject has one first-default event; each claim
belongs to one pool and each collateral asset can be realized once per scenario.
Represent partial sales by distinct, nonoverlapping asset identities. Zero recovery
requires an explicit zero-valued realization, not missing rows.

Claims declare EAD, legal balance, priority and their share of costs which cannot
be paid from proceeds. ``LEGAL_CAP`` consumes dated proceeds in ascending rank;
equal ranks share pro rata to remaining legal balances. Caps persist across dates.
Surplus after all claims is reported. ``UNCAPPED_PRO_RATA`` is an explicit single-rank
participation entitlement with fixed legal-balance weights; it is not inferred from
a normal capped loan. Legal entitlement and EAD are different quantities.

Each realization supplies gross value, fraction in [0,1], cost and positive discount
factor anchored to the shared default date. Values and costs are in the policy
currency. Proceeds pay costs first; only any deficit is separately funded using the
explicit creditor shares. Costs cannot be subtracted again in LGD. Discount each
claim's net cash at its actual date. The cash identity is realized proceeds plus
creditor deficit funding equals costs plus gross creditor recoveries plus surplus.

Economic loss is EAD minus discounted net recovery. LGD is undefined at zero EAD;
negative losses or LGD above one are retained. Negative rates can imply discount
factors above one. The caller supplies the curve and legal priority evidence.
``ULTIMATE_COMPLETE`` asserts the path includes all recoveries and costs. Open
workouts belong to separate censored-workout estimation methods, not this final
realization calculation. The method does not authenticate completeness.

Small independent fixtures verify senior/junior allocation (80 proceeds,10 costs,
60 senior and10 junior), discounting, cost deficits, equal-rank cumulative caps,
surplus, raw LGD, shared event anchors and rejection of duplicate collateral.
