# Specialized portfolio and observed loss review

Python `run(data=None, verify_reference=True)` in
`examples/specialized_portfolio_review.py` and native R
`run_specialized_portfolio_review(data=NULL, verify_reference=TRUE)` accept the
schema in `specialized_portfolio_review.json`. The default runs five distinct
specialized positions. Reference verification controls hand-oracle assertions.

The example prepares a gross-exposure portfolio with explicit borrower, facility
and claim identities, currency, target horizon, default-link policy and empty
cross-default edges. The absence of contractual cross-default links does not
assert statistical independence. Concentration is measured by sector and checked
against `sum((EAD_i/total_EAD)^2)`; no capital formula is inferred.

Two dated exposure schedules per facility accrue one year's 10% simple interest
on original principal and allocate a 20% principal payment at the period end.
A default immediately before that payment has EAD `1.1*principal`; immediately
after it has EAD `.9*principal`. This isolates an explicit same-day convention.
It is a contractual timing sensitivity, not an assertion that the observed
servicing history paid its missed coupon or changed its contract.

LGD and EAD forecasts are the recorded conditional quantities from the original
specialized simulation examples, carrying their original common scenario hash
and PIT sampling contract. Realized outcomes are from separate dated servicing
and legally closed settlement records. The loss review uses a 731-day out-of-time
forecast horizon, physical first-default target, declared currency and closed-only
workout policy. It never substitutes estimated open-workout LGD for realized LGD.
EAD bias is the equal-facility mean forecast-minus-realized difference; LGD bias
weights that difference by observed default EAD. Both are checked by independent
scalar arithmetic. These five realized defaults demonstrate accounting and
validation contracts; uncertainty is explicitly `none`, and no statistical power
or suitability for a live bank portfolio is claimed.

Output includes the prepared portfolio and seals, full concentration output,
both exposure ledgers, LGD/EAD backtests, original forecast provenance and hand
oracles. All contractual values, scenario forecasts and observed settlements are
openly specified synthetic inputs. This example supplies portfolio/loss review
coverage, not a complete acceptance claim for every product recipe or macro model.
