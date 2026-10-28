# Expected loss and first-default probabilities

Three elementary calculations are available through the `creditratingsystem`
top-level API. Each returns a `CalcResult`; use `.value` for the number or
one-dimensional result vector. The equivalent native R functions return an S3
`CalcResult` with `$value` and the same method identifiers.

These calculations do not estimate a rating model or calibrate a probability.
The examples below use explicitly synthetic inputs to illustrate their algebra.

## Expected default loss

```python
from creditratingsystem import build_calculation_context, expected_loss

# Synthetic algebraic illustration, monetary amounts in a common user-chosen unit.
context = build_calculation_context(data_manifest={
    "data_origin": "SYNTHETIC", "parameter_origin": "SYNTHETIC_REFERENCE",
    "dataset_id": "bootstrap_handcrafted",
})
loss = expected_loss(pd=0.02, lgd=0.4, ead=1000, context=context)
assert loss.value == 8.0
```

`expected_loss(pd, lgd, ead)` calculates `pd * lgd * ead` over the horizon already
represented by `pd`. The result is undiscounted and uses the monetary unit of
EAD, identified as `ead_unit` unless the supplied context explicitly names a
currency; the function does not infer a currency or perform currency conversion.

PD must lie in the closed interval `[0, 1]`, and EAD must be nonnegative. LGD is
an economic loss ratio and may be negative or greater than one. For example,
`expected_loss(0.2, -0.25, 100).value` is `-5`: excess recoveries remain visible.
There is no implicit clipping of input or output values.

The product assumes deterministic or jointly specified conditional-on-default
LGD and EAD. In general,
`E[LGD * EAD | default]` differs from
`E[LGD | default] * E[EAD | default]` when dependence remains. A product of
separately estimated marginal means therefore needs its own justification.

Scalar or length-one inputs expand to the common length of longer input vectors.
All longer vectors must have identical lengths. All-length-one inputs return a
scalar; other inputs return a one-dimensional vector. For example:

```python
losses = expected_loss([0.2], [-0.25, 1.5], [100], context=context)
# losses.value contains [-5.0, 30.0].
```

Empty inputs, matrices and unequal non-singleton vector lengths are rejected.

## Conditional, marginal and cumulative PD

```python
from creditratingsystem import (
    cumulative_pd_from_conditional,
    marginal_pd_from_conditional,
)

conditional_pd = [0.1, 0.2]  # Synthetic chronological period hazards.
cumulative = cumulative_pd_from_conditional(conditional_pd, context=context)
marginal = marginal_pd_from_conditional(conditional_pd, context=context)
# cumulative.value: [0.1, 0.28]; marginal.value: [0.1, 0.18].
```

Each input `q[t]` is the probability of first default in period `t`, conditional
on survival to that period's start. Periods must be supplied in chronological
order. Starting from `S[0] = 1`, survival is
`S[t] = S[t-1] * (1 - q[t])`. Cumulative PD is `1 - S[t]`; marginal PD is the
unconditional first-default mass `S[t-1] * q[t]` in that period. Marginal masses
sum to the terminal cumulative PD for a nonempty sequence.

A unit hazard exhausts survival. For `[0.25, 1, 0.9]`, cumulative PDs are
`[0.25, 1, 1]` and marginal PDs are `[0.25, 0.75, 0]`. The later formal hazard
creates no additional first defaults. The functions do not model cures or
repeated defaults.

Both functions return vectors, including for one period or an empty sequence.
Python requires a one-dimensional sequence, so write `[0.1]` for one period;
native R uses `c(0.1)`. No annualization, interpolation, change of probability
measure, PIT/TTC conversion or discounting takes place. Small probabilities are
computed without subtracting nearly equal cumulative values to obtain marginal
masses.

## Results and input failures

`CalcResult` carries `value`, `unit`, `method_id`, `context`, `status`,
`provenance`, `diagnostics` and `limitations`. With no supplied context, the
context mode is `TIME_INDEPENDENT` and data provenance is `USER_SUPPLIED`.
No observation or knowledge date is
invented. A supplied context preserves the caller's interpretation of the inputs.

All numeric inputs must be finite real values; booleans, strings, complex
numbers and matrices are invalid. The following stable error codes distinguish
failure reasons in Python exceptions and native R conditions:

| Code | Meaning |
|---|---|
| `INVALID_INPUT` | Invalid type or dimensionality |
| `MISSING_INPUT` | Missing required value, including `None` or native R `NA` |
| `NON_FINITE_INPUT` | NaN or positive/negative infinity |
| `INVALID_PROBABILITY` | PD outside `[0, 1]` |
| `INVALID_EAD` | Negative EAD |
| `EMPTY_INPUT` | Empty expected-loss input |
| `INCOMPATIBLE_SHAPE` | Unequal non-singleton expected-loss vector lengths |
| `NUMERICAL_FAILURE` | Finite inputs produce an unrepresentable loss |

The complete input sequence is checked even after a unit hazard. An invalid
later hazard is an error rather than being hidden by absorption.

The examples are mathematical demonstrations. They do not establish model
performance on a bank's data.
