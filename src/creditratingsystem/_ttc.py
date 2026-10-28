"""Explicit Gaussian-factor PIT/TTC mapping; no universal conversion."""
from collections.abc import Mapping
import numpy as np
from scipy.special import ndtr, ndtri
from .core import _context_value, _fail, _numeric, _probability, _result


def _physical_horizon(horizon, time_unit, probability_measure, context):
    h = _numeric(horizon, "horizon")
    if h.ndim or h <= 0 or time_unit not in ("days", "months", "years"):
        _fail("INCOMPATIBLE_HORIZON", "A positive common horizon and explicit time unit are required")
    if probability_measure != "PHYSICAL":
        _fail("INCOMPATIBLE_MEASURE", "This method requires the PHYSICAL probability measure")
    ctx = _context_value(context)
    metadata = {"horizon": float(h), "time_unit": time_unit, "probability_measure": "PHYSICAL"}
    for key, value in metadata.items():
        if ctx.get(key) is not None and ctx[key] != value:
            _fail("TARGET_MISMATCH", f"Context {key} conflicts with the explicit method contract")
    return ctx, metadata


def pit_ttc_transform(pd, systematic_factor, asset_correlation, *, direction,
                      factor_distribution, horizon, time_unit, probability_measure,
                      context=None):
    """Condition or uncondition a Gaussian latent default threshold.

    Positive factor values are good economic states. TTC is the average over
    the declared standard-normal factor distribution, not PD evaluated at zero.
    Scalar inputs expand; other inputs must be equal-length one-dimensional vectors.
    """
    if direction not in ("ttc_to_pit", "pit_to_ttc"):
        _fail("INVALID_INPUT", "direction must be ttc_to_pit or pit_to_ttc")
    if not isinstance(factor_distribution, Mapping) or factor_distribution.get("family") != "standard_normal":
        _fail("UNSUPPORTED_FACTOR_DISTRIBUTION", "Explicit standard_normal factor distribution required")
    for key in ("factor_id", "reference_period", "model_version"):
        if not isinstance(factor_distribution.get(key), str) or not factor_distribution[key].strip():
            _fail("INVALID_FACTOR_SPEC", f"factor_distribution.{key} is required")
    ctx, metadata = _physical_horizon(horizon, time_unit, probability_measure, context)
    p = _probability(pd)
    z = _numeric(systematic_factor, "systematic_factor")
    rho = _numeric(asset_correlation, "asset_correlation")
    arrays = (p, z, rho)
    if any(a.ndim > 1 for a in arrays):
        _fail("INVALID_INPUT", "Inputs must be scalars or one-dimensional vectors")
    if any(a.size == 0 for a in arrays):
        _fail("EMPTY_INPUT", "Transformation inputs must not be empty")
    n = max(a.size for a in arrays)
    if any(a.size not in (1, n) for a in arrays):
        _fail("INCOMPATIBLE_SHAPE", "Only scalar expansion or equal lengths are supported")
    p, z, rho = (np.full(n, float(a.reshape(-1)[0])) if a.size == 1 else a.copy() for a in arrays)
    if np.any((rho < 0) | (rho >= 1)):
        _fail("INVALID_CORRELATION", "Asset correlation must lie in [0,1)")
    output = p.copy()
    active = (rho > 0) & (p > 0) & (p < 1)
    q = ndtri(p[active])
    with np.errstate(over="ignore"):
        latent = ((q - np.sqrt(rho[active]) * z[active]) / np.sqrt(1-rho[active])
                  if direction == "ttc_to_pit" else
                  np.sqrt(1-rho[active]) * q + np.sqrt(rho[active]) * z[active])
    output[active] = ndtr(latent)
    numerical_boundary = active & ((output == 0) | (output == 1))
    source, destination = ("TTC", "PIT") if direction == "ttc_to_pit" else ("PIT", "TTC")
    diagnostics = {**metadata, "direction": direction, "source_philosophy": source,
        "destination_philosophy": destination, "factor_distribution": dict(factor_distribution),
        "systematic_factor": z.tolist(), "asset_correlation": rho.tolist(),
        "factor_sign": "POSITIVE_IS_GOOD", "input_factor_conditioning": "integrated_standard_normal" if source == "TTC" else "given_factor",
        "output_factor_conditioning": "integrated_standard_normal" if destination == "TTC" else "given_factor",
        "numerical_boundary_count": int(numerical_boundary.sum()),
        "inverse_information_loss": bool(numerical_boundary.any()), "zero_factor_is_ttc": False}
    return _result(float(output[0]) if n == 1 else output,
        "pd_term_structure.pit_ttc_transform", "probability", ctx, diagnostics=diagnostics,
        limitations=("Conversion depends on the declared Gaussian latent-factor model and shared horizon; factor and correlation suitability are not estimated here.",
                     "Finite-precision output at zero or one cannot recover an interior input through the inverse mapping."))
