Cashflow default events
=======================

``detect_cashflow_default(path_records, default_policy, currency=...)`` returns
facility tests, dated subject assessments and first-default events. Cashflow,
debt service, outstanding debt and asset values use the same reporting currency.
CFADS and debt service must cover the same period; outstanding debt and asset
values refer to the same assessment date and valuation perimeter. The caller
supplies these aligned quantities, rather than mixing enterprise and facility values.

The economic double test is DSCR below its declared threshold AND LTV above its
declared threshold, on the same facility, scenario and date. Thresholds and their
strict/inclusive comparisons are explicit policy inputs; no default calibration
is implied. Zero debt service has no DSCR breach. Positive debt and zero assets
have an LTV breach without representing infinity in a JSON result.

The Basel-reference event is independently established unlikeliness to pay (UTP)
OR material past-due arrears exceeding the applicable day threshold. Value coverage
never cancels either trigger. ``dual_test_as_utp`` declares whether the economic
intersection contributes a UTP signal under the supplied policy, provided a
debt-service shortfall remains after explicitly available liquidity buffers.
Buffers must be legally and operationally available by the payment date, net of
prior claims; hypothetical collateral proceeds or uncommitted refinancing do not
qualify. A buffer never cancels another independently established UTP signal. The intersection
alone is not a universal Basel definition. An empty ``utp_indicators`` list means
an explicit assessment found no other UTP indicators; it must not stand for missing
assessment. The supplied facilities and UTP evidence must cover the declared
assessment subject, including relevant obligations outside the simulated facility.

The policy records its ID/version, reference basis ``BASEL_CRE36_REFERENCE``,
assessment level, DSCR/LTV thresholds/comparisons, day threshold/comparison,
past-due/materiality/UTP rule references, dual-test role,
``economic_joint_timing='SAME_DATE'`` and
``default_state='ABSORBING_FIRST_DEFAULT'``. The reference day comparison is
strictly greater than 90. A different threshold through 180 requires an explicit
applicable supervisory exception declaration; facility scope requires declared
eligible retail treatment. These declarations do not certify legal eligibility.

Rows contain scenario_id, entity_id, facility_id, date, currency, cfads,
debt_service_due, debt_outstanding, asset_value, materiality_met,
material_days_past_due, liquidity_buffer_available and utp_indicators. Materiality is determined under the
referenced applicable rule. Its continuous counter must advance with calendar
time or reset; ordinary delinquency days are not interchangeable with that counter.
Obligor assessment aggregates triggers across facilities on a common date grid.

Default is absorbing in this first-event calculation. ``first_default_date`` is
the first detection-grid endpoint, not an inferred exact intra-period event date.
``last_nondefault_date`` supplies the preceding endpoint. Initial defaults are
left censored, flagged explicitly, and must be excluded from an initially performing
PD risk set. Cure and redefault need separate episode logic. Downstream PD, EAD and
LGD calculations should reuse the same event identity and detection convention.

Reference: Basel Framework CRE36.68--36.75,
https://www.bis.org/committees/bcbs/basel-framework/standard/cre?allChapters=true&chapter=36

Tests use the four economic-test states, threshold equality, 90/91-day arrears,
UTP despite high collateral, scope propagation and censoring. They require only
small deterministic paths and run in the default test profile.
