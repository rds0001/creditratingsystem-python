Actual historical replay and late source corrections
===================================================

Run ``examples/structured_replay_review.py`` or source its native R counterpart
and call ``run_structured_replay_review()``. Inputs reuse the published local
``structured_governance.json`` contracts for tranche, covered bond, receivables
and counterparty positions. The example provides a source-correction review of
these positions; their product cashflows and workout assumptions remain in the
associated product examples. Pass a replacement configuration to ``run(data)``
or ``run_structured_replay_review(data)`` for bank-owned cases.

A declared synthetic logit model maps feature0 to PD50%. A correction to log(3)
is observed for the same January1 snapshot but first becomes known January16;
the restated PD is75%. The original prediction remains50%. The correction is
retrospective information, never part of the original forecast. No additional
post-calibration is applied: a versioned identity contract records that fact.
This example tests historical execution, not predictive suitability of those PDs.

The original and corrected requests are executed, their assessments and separate
policy decisions preserved, and the original model is then joined by version2.
Both historical requests are actually recomputed against the pinned version1.
``replay_consistency`` receives those actual checkpoints, the original delivery
log and content hashes for model, calibration contract, policy and source input.
The validator itself does not execute a replay engine. No replay output is made
by copying the expected state. Tampering with a replay PD produces a blocked
validation result.

The event projector treats identical repeated delivery as a duplicate, defers
the unknown correction, and applies it once its knowledge date is reached.
Its projection identifies affected assessments; explicit new calculation creates
the restatement. Batch order does not change outputs. An invalid pinned model
hash remains a visible row error and does not discard successful rows. An early
corrected request is unavailable. Historical source, assessment and event-state
hashes remain unchanged after later model registration.
