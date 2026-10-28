Card authorizations and booked exposure
======================================

``reconcile_card_authorizations(authorizations, events, policy, *, as_of,
known_at, currency, context=None)`` reconciles a supplied authorization ledger.
It returns a ``CalcResult`` containing each reservation, booked balance,
account totals, ordered event history, excluded future or late-known records,
and fingerprints of inputs and output. It does not infer other account debts.

Each authorization supplies ``authorization_id``, ``account_id``, ``approved_at``,
``known_at``, ``sequence``, ``currency``, ``amount`` and ``expires_at``.
Each event supplies ``event_id``, ``authorization_id``, ``account_id``,
``occurred_at``, ``known_at``, ``sequence``, ``currency``, ``amount`` and ``kind``.
A SETTLEMENT additionally declares Boolean ``authorization_complete``; a REVERSAL
names ``reverses_settlement_id``; a CANCELLATION must have amount zero.
Sequences resolve events on the same account and date. Events must follow their
original authorization. Observation and knowledge dates are filtered separately.

The explicit policy contains ``policy_id``, ``version``, ``known_at``,
``expiry_boundary='EXCLUSIVE'``, Boolean ``late_settlement_allowed``,
``authorization_conversion_factor`` between zero and one, and positive
``minor_unit`` (for example 0.01). All amounts must be whole minor units.
Internally amounts are converted to integer units before any capture or refund;
0.30 refunded by 0.10 and 0.20 therefore leaves exactly zero. Per amount and
account aggregate the supported bound is 2**40 minor units. Conversion admits
only binary floating-point representation error, using a tolerance in minor
units of max(1e-9, min(1e-3, 8 * 2**-52 * abs(amount/minor_unit))). This is not a
business rounding rule. Scenario exposures may contain fractional minor units
when the explicit conversion factor is fractional.

A settlement replaces the corresponding reservation. With an authorization of
200 and a partial settlement of 80, booked exposure is 80 and the remaining hold
is 120; factor one gives total scenario exposure 200. Complete settlement of 80
releases the unused hold and leaves 80. A refund of 40 against the partial
settlement reduces booked exposure to 40 but does not restore the consumed hold.
Expiry removes an unsettled hold at ``as_of >= expires_at``; it does not forgive
booked debt. Late settlement requires the declared policy permission. Completion
or cancellation closes the authorization to further settlement. Additional
captures require a new authorization; overcapture is rejected.

The conversion factor is a supplied scenario assumption, not an estimated CCF.
Contract rules must be supplied by the user. No card-network convention,
national legal rule or actual credit action is inferred. Payments against the
whole account and installment conversions need their own account cash ledger.
