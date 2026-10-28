# Actual account phases and legal settlement

`examples/retail_application_phases.py` and native `run_retail_application_phases()`
reuse the six actual application journeys. Optional `application_chain` accepts
an already executed raw chain; no additional model is fitted. The installed full
example executes original application models and all six phase extensions.
`assess` / `rap_assess` accepts a case, explicitly dated extension ledger, cutoff
and original model/calibrator/policy hash bindings for own inputs.

The packaged extension is explicitly new synthetic documentary evidence on the
same funded account IDs. It preserves original economic totals. Account closure
on January 25 is an operational exit, not legal extinguishment. Principal recovery
600 and final waiver 400 occur January 30 and become known January 31. The recovery
episode is still open at the account-exit cutoff; legal closure occurs only after
the payment. A waiver is not a cash recovery and default history never disappears.

Operating lease is different: the customer's unpaid 90 rent claim is waived under
a separate settlement; tenant rent recovery is zero. The lessor's 600 asset-disposal
receipt against 1000 investment implies 400 investment loss. These are distinct scopes
and are never added as if they were independent exposures. The 600 sale is not a
rent payment or a recovery of an invented 1000 loan against the tenant. Customer
residual liability remains zero. The education grace period has zero scheduled
payment and DPD 0; its later observed default is not inferred from arrears.

January 15 restructuring is an unaccepted scenario only. The zero-receipt baseline
is explicitly a stress comparison, not a replacement for the original contractual
schedule; candidate payments are zero followed by two equal claim installments.
It changes neither contract nor default status. Original product terms remain in
the output, including balloon, grace, overlapping plans and weekly installments.

All eight lifecycle phase events run through the public dependency reducer twice,
with an additional identical proposal delivery. The account state also folds the
actual ordered event payloads. `replay_consistency` verifies the separate executions
against original model, calibration, policy and source snapshots. Portfolio output
is the descriptive distribution of two realized investment losses [0,400], with
mean 200; it is not an estimated future portfolio loss or a new LGD forecast.

Fast independent tests cover all six claim ledgers without fitting, using explicit
component-test snapshot bindings. Full installed examples bind actual prior models
and policies. Tests reject premature legal closure, use of asset sale as tenant
payment and substitution of an accepted restructuring. These examples demonstrate
implementation, accounting and chronology, not empirical bank approval.
