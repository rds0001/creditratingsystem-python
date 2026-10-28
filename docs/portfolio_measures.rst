Portfolio loss and concentration measures
========================================

``credit_loss_distribution`` accepts finite signed monetary loss paths, aligned
scenario IDs and nonnegative weights. It reports the weighted mean, distribution
standard deviation, mass function, left-inverse-CDF quantiles and expected shortfall.
Zero-weight paths never define an extreme quantile. Weights are normalized explicitly;
no IID sampling claim or confidence interval follows merely from those weights.

For a discrete distribution ES uses exactly the upper ``1-alpha`` mass, including
only the needed fraction at the VaR atom. Boundary ties share this fraction in
proportion to their scenario probabilities. At alpha zero ES is the mean; at alpha
one it is the maximum-loss limit. For loss0 with probability0.9 and loss100 with
probability0.1, the 80-percent VaR is0 and ES is50. Conditioning naively on losses
at or above VaR would give a different answer.

``allocate_expected_shortfall`` uses that same portfolio tail mass for every
position and optional group. Contributions sum to ES, including negative hedge
contributions. At an atom this is a declared allocation convention, not a claim
that the risk measure has a unique derivative. ``incremental_credit_risk`` compares
base and changed paths on the same scenarios. Addition/removal/scaling are supplied
as signed monetary changes; VaR jumps are finite differences, not derivatives.

``credit_concentration_measures`` requires an explicit nonnegative weight field,
unit and grouping dimensions. It returns HHI, reciprocal-HHI effective counts,
entropy, exponential-entropy effective counts and largest shares. Four equally
weighted positions have HHI0.25 and effective count4; grouping two of them together
increases HHI to0.375. A zero denominator is explicitly non-estimable. These are
portfolio descriptors, not legal connected-client findings or exposure limits.

``asymptotic_factor_loss`` supplies a physical homogeneous Gaussian benchmark:
``conditional_PD(y) = Phi((Phi_inverse(PD)-sqrt(rho)*y)/sqrt(1-rho))``.
Loss rate is this conditional PD times fixed LGD. The supplied PD is unconditional
under the standard-normal factor law; rho is latent asset correlation, not default
correlation. Negative factors are adverse. With zero correlation the conditional
loss is constant PD times LGD. The benchmark assumes infinitely granular identical
exposures and does not replace finite-name, multifactor or regulatory capital models.
An unconditional factor law is not automatically an empirically calibrated TTC law.

References: Rockafellar and Uryasev, *Conditional Value-at-Risk for General Loss
Distributions*, https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf;
BIS, *Securitisations: tranching concentrates uncertainty*,
https://www.bis.org/publications/securitisations-tranching-concentrates-uncertainty.
Independent implementations; reference texts are not distributed with the package.
