# Dated exposure and realized conversion

`exposure_schedule(contract, events, annual_interest_rates, *, period_end_dates,
day_count, default_timing, allow_overlimit, currency, context=None)` projects one
explicit facility snapshot through dated contractual events. The contract contains
`facility_id`, `currency`, ISO `as_of`, nonnegative `drawn_principal`, `credit_limit`
and `accrued_interest`. These balances already include activity through `as_of`.

Each event supplies a unique `event_id`, `event_date`, `known_at`, nonnegative
integer `sequence`, nonnegative `amount`, and one of `DRAW`,
`PRINCIPAL_REPAYMENT`, `INTEREST_PAYMENT`, `LIMIT_CHANGE`. Events follow the
snapshot, fall within the supplied horizon, and must already be known at the
snapshot. They are explicit scenario assumptions, not behavior learned from later
observations. Sequence resolves simultaneous events; duplicate date/sequence pairs
are rejected. `LIMIT_CHANGE.amount` is the new total limit, not a change amount.

Interest is simple annual interest on drawn principal, accrued between each
successive event and period end. `annual_interest_rates` has one rate per period;
`day_count` is explicitly `ACT/365F` or `ACT/360`. Accrued interest is not capitalized.
Principal and interest payments are allocated separately and cannot exceed their
corresponding outstanding balance. Negative rates and fees are outside this
specific schedule contract. Known fees or other products need separately defined
cashflow treatment rather than implicit balance changes.

`default_timing` is `BEFORE_PERIOD_END_EVENTS` or `AFTER_PERIOD_END_EVENTS`.
Both values are returned for every period, and the selected value becomes
`conditional_ead`. A bullet repayment on the endpoint can therefore leave exposure
before the payment and zero principal after it. Interest earned through that date
is included in both values until an explicit interest payment occurs.

Draws above the commitment require `allow_overlimit=True`. An initial overlimit
balance or a later limit cut is retained and reported; no artificial negative
undrawn commitment is created. Rows expose opening/closing principal, interest,
limit, undrawn commitment, overlimit amount and both endpoint exposures. The event
ledger reconciles each balance change. A canonical `schedule_hash` binds the
result, with a separate input fingerprint in diagnostics.

`realized_ccf(reference_drawn, reference_limit, ead_at_default, *, reference_date,
default_date, reference_known_at, default_known_at, evaluation_cutoff, currency,
minimum_undrawn=0, context=None)` computes

```text
CCF = (EAD at default − reference drawn) / (reference limit − reference drawn)
```

The reference must be frozen before default: reference knowledge ≤ reference date
< default date ≤ default knowledge ≤ evaluation cutoff. Currency and the exposure
component definitions must agree. Negative CCF and CCF above one remain unchanged.
Zero or negative undrawn commitment gives missing CCF with `NO_POSITIVE_UNDRAWN`.
A positive denominator at or below the explicit monetary `minimum_undrawn`
threshold gives `BELOW_MINIMUM_UNDRAWN`. The result retains numerator, denominator,
reference balances, actual EAD and reconstructed EAD where defined.

```python
from creditratingsystem.ead import realized_ccf

result = realized_ccf(
    40, 100, 70, reference_date="2023-12-01", default_date="2024-01-01",
    reference_known_at="2023-12-01", default_known_at="2024-01-02",
    evaluation_cutoff="2024-01-31", currency="EUR",
)
assert result.value["ccf"] == 0.5
```

These methods do not infer cancellation rights, future borrower draw behavior,
legal netting or automatic adjustments for changes in credit limits.
