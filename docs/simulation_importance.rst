Gaussian importance sampling for default events
==============================================

``importance_sample_defaults(z, proposal_shift, evaluator, importance_contract)``
uses supplied trial-by-coordinate standard-normal innovations. It evaluates
``x = z + shift`` and the complete joint density ratio
``exp(-dot(z, shift) - dot(shift, shift)/2)``. Correlated financial drivers may be
constructed inside the evaluator from these independent base innovations.
Every random coordinate must be included; hidden random calls invalidate the
contract. The callable is local caller code identified by a declared version,
not serialized executable model content.

The evaluator returns one binary ``defaulted`` and finite ``economic_loss`` per
trial, with zero default loss for nondefaults. It must implement the declared
initially-performing first-default target, horizon, conditioning, currency and
physical or risk-neutral measure. Default detection, including cashflow and
independent default indicators, belongs to that evaluator; importance sampling
does not change the event definition or supply regulatory eligibility.

The main estimates are ordinary means of density-weighted outcomes. IID sample
standard errors use these contributions. Deterministic or dependent/QMC inputs
are allowed with explicit ``DEPENDENT_OR_QMC`` design and no IID standard error.
The proposal shift must be frozen independently of the evaluated IID sample;
tuning on those same draws invalidates the ordinary IID justification. Finite
second moments are needed for meaningful standard errors; this routine does not
establish their existence for arbitrary caller loss functions.

A finite-sample probability estimate can exceed one and is retained. The separate
self-normalized estimate is generally biased. Weight concentration diagnostics
are not a guarantee of tail precision. No exact target-PD upper bound is inferred
from zero proposal defaults. Numerical overflow or underflow of density ratios
is rejected instead of clipping or silently removing paths.

The fast analytical tests cover identity proposals, full multidimensional normal
density ratios, a Gaussian tail integral using 2,048 midpoint quantiles, zero
proposal defaults and non-clipped estimates. There is no fitted synthetic
performance benchmark.

Reference: Art B. Owen, *Monte Carlo theory, methods and examples*, Chapter 9,
https://artowen.su.domains/mc/Ch-var-is.pdf (ordinary and self-normalized importance
sampling). This package contains an independent implementation, not that text.
