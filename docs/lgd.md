# Recovery valuation and workout LGD

`discount_recoveries(cashflows, discount_curve, fx_contract, *, default_date,
evaluation_cutoff, reporting_currency, context=None)` values realized recovery and
cost records that both occurred and became known by the cutoff. Each cashflow has
a unique `cashflow_id`, ISO `cashflow_date`/`known_at`, currency, nonnegative amount,
and kind `RECOVERY`, `DIRECT_COST` or `INDIRECT_COST`. Costs reduce net recovery.
Future and not-yet-known records are returned as exclusions, not interpreted as
zero future recovery. Duplicate records and cashflows before default are rejected.

The discount curve explicitly supplies `anchor_date=default_date`, `known_at`, a
nonempty economic `discount_basis` description and `points`, each containing
`cashflow_date` and nonnegative `discount_factor`. Exact dates must match included
cashflows. No interpolation or default interest rate is applied. If supplied, the
anchor-date factor must equal one. The curve must be known by the evaluation date.

The FX contract supplies `reporting_currency`, `known_at`, `conversion_basis`
(`cashflow_date` or `default_date`) and `rates` with currency/date/rate. Every used
currency/date pair needs an explicit positive rate, including an identity rate of
one for the reporting currency. A rate means reporting-currency units per unit of
cashflow currency. FX knowledge must meet the evaluation cutoff. These supplied
contracts define the valuation convention and are retained through a canonical
input fingerprint; the function does not infer market prices or a regulatory rate.

Each included contribution is amount × FX × discount factor with the sign implied
by its cashflow kind. The result exposes gross recovery, costs, net recovery,
individual contributions and exclusions. Power-of-two product scaling avoids
avoidable intermediate overflow; a nonfinite final value is a controlled error.

`workout_lgd(ead_at_default, discounted_net_recoveries, *, episode_status,
default_date, evaluation_cutoff, currency, context=None)` gives loss = EAD − net
recovery and the raw economic loss-to-EAD ratio. A recovery `CalcResult` can be
passed directly; its currency, default date and cutoff must match. A scalar net
recovery can instead be supplied with those explicit valuation declarations.

`episode_status='CLOSED'` makes this ratio the observed final `economic_lgd`.
`'OPEN'` reports `economic_lgd_to_date` but leaves final `economic_lgd` missing.
It does not assume that all remaining claims are lost or that recovery is complete.
At zero EAD the monetary loss remains available while the ratio is missing with
`ZERO_EAD`. Recoveries above EAD can produce negative LGD; costs can produce LGD
above one. Raw economic values are never silently clipped.

```python
from creditratingsystem.lgd import workout_lgd

result = workout_lgd(
    100, 80, episode_status="CLOSED", default_date="2024-01-01",
    evaluation_cutoff="2024-06-30", currency="EUR",
)
assert result.value["economic_lgd"] == 0.2
```

The completeness and valuation declarations are caller evidence. Neither function
turns a date-only snapshot into a fitted recovery forecast or a regulatory approval.
