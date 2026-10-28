Retail workout claims and prior-known cure forecasts
===================================================

Run ``examples/retail_workout_cases.py`` or the native R equivalent with the
packaged ``retail_workout_cases.json``. Twelve independent synthetic claim
ledgers cover mortgage, home-equity line, unsecured instalment, education,
auto loan, auto lease, credit card, charge card, overdraft, BNPL, micro loan and
small-business finance. These are phase integration examples, not a claim that
one case covers every contractual product variant or provides empirical bank
calibration.

The ledgers separately identify default EAD, interim recovery, creditor workout
cost and terminal realization, including dates of occurrence and availability.
Cash is discounted with explicit synthetic factors .99 for interim receipts
and .95 for later cash. The reference final loss is
``EAD - .99 * interim - .95 * (terminal - cost)``. All costs are direct creditor
costs outside the illustrated estate allocation, so they are charged once.

The home-equity claim is junior to an independent 40,000 mortgage claim and
receives 30,000 of 70,000 estate proceeds. The residential mortgage is senior
to a distinct junior creditor. Unsecured estates use separate equal-ranking
creditors. BNPL has two purchase claims sharing one default event; their
recoveries sum to 150. Vehicle realization, lease residual proceeds, salary
receipts, social support and owner/firm separation remain explicit mechanisms.
Social support is not converted into a legally enforceable guarantee.

Recovery episodes link only cash known at the evaluation date. The open March
valuation excludes the eventual May settlement and April actual cost. It uses
a frozen residual recovery model fitted to separate 2023 observations and a
separately declared March-known future cost schedule. The model is intentionally
a tiny synthetic common reference; the product-specific amounts are supplied
inputs. A missing unresolved-tail discount assumption yields no final LGD.
A late-known resolution likewise cannot establish a closed realized loss.
Event-conditioned loss applies the default probability exactly once.

Collections history records an observed cured spell followed by a new open
default; cure never erases the prior default. Action valuation includes uptake,
declined outcomes, offer costs and discounting once, without claiming a causal
treatment effect.

A separate multistate model is actually fitted to two earlier accounts observed
January--April 2023 and available on April 2, 2023. It predicts February 2024
cases. Calendar support is checked against the elapsed-time training records.
One of two accounts cures at day 30 and redefaults at day 60, producing first
cure probabilities ``[0, .5, .5]`` and ordered cure/redefault probabilities
``[0, 0, .5]``. State occupancy is not confused with first-hit events. Recovery
on cure is a product-specific, explicit monetary assumption, not an empirically
estimated causal effect. The transition interpretation retains the model's
Markov and support assumptions. Future-known fitted models are rejected.

Default tests exercise all twelve cases plus late resolution, future model,
wrong currency and duplicated claim failures. Frozen Python/native R output
replays include the full cash, claim, collections and cure results. The tiny
training cohort is an analytic oracle, not sufficient evidence for deployment.
