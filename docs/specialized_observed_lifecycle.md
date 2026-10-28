# Observed specialized-finance cohorts and workouts

Run `examples/specialized_observed_lifecycle.py` or source the corresponding R
example and call `run_specialized_observed_lifecycle()`. Python `run(data=None,
verify_reference=True)` and the R counterpart accept the packaged
`specialized_governance.json` dictionary/list, or user records with that schema.
`verify_reference=False` disables only the synthetic reference assertions.

The five products have distinct observed default counts (2, 3, 1, 2, 4 out of 20)
and different dated settlements. A further lost-followup borrower and prepaid
borrower are censored in each cohort. An active borrower with zero drawn balance
remains in the denominator. The recorded original two-year PD forecasts are
compared with these separate servicing observations; simulated defaults are not
validation labels. The forecast interval is 731 calendar days from 2023-01-01.

The example calls observation-window construction, monitoring-cohort construction,
cohort metrics, and default-count backtesting. The hand Brier oracle is
`[d*(1-p)^2 + (20-d)*p^2]/20`. Binomial uncertainty explicitly assumes independent,
homogeneous borrowers for this synthetic cohort; this is not a portfolio dependence
claim. Missing outcomes are not classified as performing loans.

Each workout has an explicit legal settlement closure, without a fabricated cure.
Recovery episode aggregation, recovery discounting, and closed/open workout LGD
are executed separately. The independent financial oracle discounts gross recovery
less direct costs at 5% effective annual interest for actual days/365, then computes
`1 - present_value/EAD`. Open-workout final LGD remains missing. Legal closure is
known only on its recorded knowledge date. Neither settlement nor an exit from the
observation set implies a return to performing borrower status.

The output contains full observation windows, cohort metrics, default backtest,
allocated cashflows, discounted values, closed/open LGD, and independent financial
oracles for each product. All input amounts, dates, forecast and censoring records
are openly specified synthetic parameters. These examples demonstrate observed
monitoring and closed settlement accounting; residual-recovery estimation,
treatment selection, competing-event fitting and portfolio review require their
separate acceptance examples. They do not constitute empirical calibration of a
bank portfolio or endorsement of a decision policy.

The reusable example accepts an optional data-level `cohort_contract` with explicit forecast/knowledge dates, horizon, outcome/evaluation cutoffs, reference window, total/mature/censored counts and reporting currency. Each observed workout may declare its own currency. Omitted fields retain this example's original defaults; the seven-sector example demonstrates a distinct 365-day cohort without backdating its forecasts.
