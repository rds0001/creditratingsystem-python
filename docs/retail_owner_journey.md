# Sole-trader application, facility and settlement

`examples/retail_owner_journey.py` / `run_retail_owner_journey()` executes an
own-data-capable sole-trader slice of the SME journey across all eight phases.
It uses five dated applications, three genuinely funded accounts, independent
2018 training and 2020 calibration, and the same facility identifiers throughout.
The originating mechanics are shared with the retail application example;
models and supplied operating cash flows belong specifically to this product.

Business cash after current business debt is 800 in the normal case. A documented
owner transfer is 400 out of the business and 400 into the household. Separate
household income is 800. The transfer is matched once, financing is excluded from
income, and the evidence type is OTHER_NET_INCOME. Household income is1200,
expenses400. Sole-trader legal identity is explicitly documented; ownership
alone never creates a default equivalence. The declined application's scaled
cash flows support only60 household income and therefore insufficient capacity.

A real1000 principal claim misses its100 installment, reaches five days past due
on January15, receives a repayment proposal that is never accepted, defaults on
January20 and settles on January25. A dated January24 receipt recovers600 of
principal; a documented full-and-final agreement writes off400. An actual closure
alone would not prove extinguishment. The20 of separately received interest is
bank income in policy profitability and is excluded from this explicitly
principal-only LGD claim. Unaccepted proposal cash flows never replace actual
payments, EAD, default or recovery. The closure does not erase the earlier default.

Prior closed workouts [.2,.4,.6] identify an intercept-only fractional LGD mean
0.4. The full prior dataset and unrounded fitted model are retained. A frozen
schedule gives actual default EAD1000. The original December31 book contains
the two then-funded facilities; the January31 immature facility is excluded from
that earlier portfolio. Independent owner defaults are an explicit scenario
assumption. Exact enumeration gives portfolio EL400. The first30-day default
loss of the defaulting applicant is250 under its frozen PD0.625 and prior LGD0.4;
the `lifetime_expected_loss` primitive is used for this declared30-day window,
not misrepresented as the entire loan lifetime.

A Kaplan-Meier curve uses actual prior first-draw and outcome dates: half of the
2018 funded sample defaults17 days after draw; the rest remains observed through
30days. Predicted survival is [1,.5,.5] at [0,17,30]. The independent intercept
PD benchmark is0.5. On the two genuinely mature live accounts, the product model
has Brier0.140625 versus0.25, difference−0.109375. This tiny synthetic comparison
is not empirical bank validation. Review retains PDs without override, and the
empty actual override ledger is checked under a distinct post-review stage.

Lineage, actual funded cohort, policy outcomes, affordability and historical
application replay execute in the originating helper. The companion reexecutes
public dependency events for proposal, default, settlement and exit, checks
idempotent duplicate delivery, and compares actual terminal replay states.
A separate terminal assessment has zero remaining contractual EAD supported by
the settlement ledger; ever-default remains true.

For own data pass `data`, `applications_path` and `releases_path`. The JSON
source tables have the same declared dated-input schema as the application
example. `include_lifecycle=False` selects the origination/performing subset
used by the selection journey. `return_artifacts=True` / `TRUE` returns every
unrounded model, source graph, validation input and original/replayed state.
Public economic primitives remain callable independently; the paired lifecycle
helper accepts `case`, the actual originating artifact and an evaluation cutoff.

Normal tests check independent transfer conservation, prior fitted LGD and cash
loss arithmetic. The installed full example executes the actual chain and all
required checks. Portable replay reuses both runtimes' frozen models without
refitting. No empirical, causal uptake or regulatory approval is claimed.
