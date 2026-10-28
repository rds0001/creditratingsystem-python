# Fund finance borrowing bases, liquidity and credit score

Run `examples/wholesale_fund_finance.py` or source the corresponding R example and call `run_wholesale_fund_finance()`. Python `run` and the R function accept `data`, `verify_reference` (default true) and `return_fitted_models` (default false). The last option returns the report plus full raw fitted preprocessors, scorecards, calibrators and predictions. Inputs and parameters in `wholesale_fund_finance.json` are synthetic and openly specified. These three reference cases demonstrate economic components; they do not constitute a complete monitored fund lifecycle.

| Structure | Borrowing-base rights | Initial / stressed base | Stress liquidity gap |
| --- | --- | --- | --- |
| NAV backed | Eligible NAV after asset haircuts, 50% advance | 36 / 21.6 | 5.4 |
| Subscription line | Eligible, uncalled investor commitments, 90% advance | 36 / 0 | 25 |
| Levered fund | Eligible NAV, 70% advance | 70 / 42.7 | 23.3 |

Amounts are EUR in a common synthetic scale. The subscription base excludes a conditional ineligible investor entirely. Its performing investor has 80 committed less 30 paid in, with 80% eligibility haircut. That investor's stress failure removes the borrowing base; NAV is not silently substituted. NAV stress revalues each asset before computing the borrowing base. Margin due is drawn debt minus the stressed base, floored at zero. Redemptions are separate liquidity claims. Operating distributions net of operating costs form CFADS; financing margins and redemptions enter the dated cash waterfall separately. Priority is operating cost, margin, then redemptions. Available cash is opening cash plus distributions minus operating cost; the gap equals unpaid claims.

An independent grouped-binomial logit interpolation checks each genuine weighted scorecard fit. A separate held-out calibration sample estimates logistic calibration. The preprocessor is fitted only on training data. Raw predictions must agree with the analytical probability within 1e-7 before the declared six-decimal reporting convention is applied. Raw fits remain available through `return_fitted_models`; independently fitted Python/R model hashes are not asserted equal. Standard public `serialize_fitted_model` / `load_fitted_model` support frozen replay, and `apply_calibrator` verifies the calibrator seal.

The two aligned underlying-asset scenarios have explicit physical weights 0.8 and 0.2. Stress losses are 38, 38 and 39, respectively; expected asset losses are therefore 7.6, 7.6 and 7.8. These underlying portfolio losses are separate from the fund's empirical credit PD and from its liquidity gap. The examples make no independence or empirical frequency claim for the declared scenario law. Routing requires product, variant, borrower class, legal claim and compatible borrowing-base rights; swapping the base yields no route, and incompatible input contracts are rejected even when reference checks are disabled.
