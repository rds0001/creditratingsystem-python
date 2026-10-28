Joint mortgage household and observed account review
===================================================

Run ``examples/mortgage_household_review.py`` or the native R function
``run_mortgage_household_review()`` after sourcing its installed example file.
The executable composes the existing mortgage lifecycle with household, cohort,
affordability, frozen-pool, closed-workout LGD/EAD and historical replay checks.
All data are synthetic analytical references, not empirical suitability evidence.

The six original applications retain their actual selection states. Only
application A funds facility_A on 3 February 2024. Its first default is observed
on 1 September 2024. The joint household has two legal full-recourse borrowers;
the mortgage's monthly 900 obligation is counted once in its economic budget.
It is distinct from the existing unsecured monthly 100 obligation. Joint
liability does not propagate an account default to the co-borrower.

The illustrative fixed mortgage coupon is determined from principal 100000,
240 monthly payments of 900. The budget is projected from 1 February to the next
monthly anniversary, 1 March. A 20% net-income reduction and 10% expense rise give
4000 income, 2530 expenses, 1000 debt service and 470 residual. Neither fixed-rate
contract resets under the supplied rate shock. This scenario is not observed
payment performance. The February/March payment outcome is deliberately missing;
the affordability validator reports missing capture rather than a zero default.
Unsecured debt is not treated as a property claim. Alternative mortgage contracts
inside the reused lifecycle remain separately identified contractual examples.

The frozen risk-pool model is fitted on an earlier cohort. The funded held-out
facility receives PD 0.25; its observed default gives Brier 0.5625. The separate
closed-workout LGD model uses four prior losses 0.1, 0.2, 0.4, 0.5 and predicts 0.3.
Observed EAD 95000 and actual sale 70000 discounted by 0.9 give recovery 63000,
loss 32000 and LGD 32/95. A disclosed conservative frozen drawn-EAD forecast 100000
has bias 5000. PD, default-EAD and completed-workout denominators stay separate;
one default establishes neither discrimination nor pool homogeneity.

The original model/calibrator and policy snapshots remain pinned after model
replacement. Replay actually reapplies public dependency events and rerates
historical requests. Tests reject invented unfunded labels, inconsistent income
evidence and a changed replayed PD. The compact report displays budget amounts
to 2 decimals and LGD bias to 8 decimals; ``return_artifacts=True`` in Python or
``return_artifacts=TRUE`` in R exposes unrounded models, inputs and audit outputs.
Native optimizers need not have identical diagnostics or fitted artifact hashes.
Both origins' unchanged artifacts are separately replayed in both runtimes.

Local CSV inputs retain all six application IDs and requested amounts. The
historical source join excludes later releases with unavailable driver values.
Ledger controls and source lineage are audited before the selected feature
values enter the original application model. Unsupported product routing is
explicitly tested. These content controls do not authenticate external sources.
