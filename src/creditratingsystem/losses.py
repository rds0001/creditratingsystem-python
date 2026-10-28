"""Loss calculations with explicit conditioning and shape conventions."""
import numpy as np

from .core import _context_value, _fail, _numeric, _probability, _result


def expected_loss(pd, lgd, ead, *, context=None):
    """Return PD * LGD * EAD for fixed or suitably conditioned inputs.

    Length-one inputs expand; all other vectors must have equal length. Raw
    economic LGD may be negative or exceed one. This product is not the general
    expectation of dependent random exposure and loss severity.
    """
    arrays = [_probability(pd, name="pd"), _numeric(lgd, "lgd"), _numeric(ead, "ead")]
    if any(x.ndim > 1 for x in arrays):
        _fail("INVALID_INPUT", "Expected scalar or one-dimensional inputs")
    if any(x.size == 0 for x in arrays):
        _fail("EMPTY_INPUT", "Loss inputs must not be empty")
    lengths = {x.size for x in arrays if x.size != 1}
    if len(lengths) > 1:
        _fail("INCOMPATIBLE_SHAPE", "Only length-one expansion is allowed")
    if np.any(arrays[2] < 0):
        _fail("INVALID_EAD", "Exposure must be nonnegative")
    # Multiplying raw operands can underflow before a large exposure restores the
    # scale. Mantissas stay bounded; only the final exponent can overflow/underflow.
    parts = [np.frexp(x.reshape(-1)) for x in arrays]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        value = np.ldexp(parts[0][0] * parts[1][0] * parts[2][0],
                         parts[0][1] + parts[1][1] + parts[2][1])
    if not np.all(np.isfinite(value)):
        _fail("NUMERICAL_FAILURE", "Loss arithmetic produced a non-finite value")
    ctx = _context_value(context)
    return _result(float(value[0]) if value.size == 1 else value, "losses.expected_loss",
                   ctx.get("currency") or "ead_unit", ctx,
                   diagnostics={"conditioning": "fixed_or_jointly_conditioned_inputs"},
                   limitations=("Not the product of arbitrary marginal means for dependent LGD and EAD.",))

from ._loss_paths import (
    expected_loss_from_paths, lifetime_expected_loss, event_conditioned_loss, counterparty_expected_loss, aggregate_portfolio_loss
)
