# Dependence and shared innovations

`validate_dependence_matrix(matrix, variable_ids, matrix_kind=..., repair_policy='none')`
checks ordered dimensions, symmetry, correlation diagonals or covariance variances,
and eigenvalues. Relative PSD tolerance scales with the matrix magnitude, including
small covariance units. Singular PSD matrices are valid inputs, but not valid for
all likelihood/inverse operations. Without requested repair the returned matrix is
unchanged. Eigenvalue diagnostics use its symmetric part. Explicit `spectral`
repair symmetrizes, truncates negative eigenvalues and, for correlations, rescales
the diagonal. Its absolute and Frobenius changes are reported. This is not a
nearest-correlation algorithm.

`fit_copula` requires aligned training rows, named variables, target/training
metadata, and an explicit family, input kind and tie policy. Raw observations
become average ranks divided by N+1. Supplied uniforms must be strictly inside
(0,1). Default-style tied data cannot silently masquerade as continuous data:
`tie_policy='error'` rejects them; `average` labels the continuous approximation.
Constant columns are unidentified.

The estimator inverts empirical Kendall tau. Gaussian and t correlations use
sin(pi*tau/2), with matrix repair only when explicitly requested. The t fit then
profiles the **joint** copula likelihood over positive degrees of freedom within
`df_bounds`; alternatively `df` freezes a user-supplied value. Boundary solutions
are marked `BOUND_LIMITED`. This is a two-stage estimate with fixed rank
correlations during profiling, not a joint maximum-likelihood fit. Exchangeable
Clayton uses theta=2*tau/(1-tau), Gumbel theta=1/(1-tau), where tau is the average
of pairwise estimates. These Archimedean implementations support nonnegative
association, theta>=0 for Clayton and theta>=1 for Gumbel; negative association
requires another family. Pairwise tau ranges disclose departures from
exchangeability. Perfect rank dependence cannot supply an invertible likelihood.

Diagnostics include pairwise composite log likelihood and analytic lower/upper
tail dependence, separately from the t profile likelihood. Composite likelihood
is not the joint likelihood and does not supply an IID standard error. No
uncertainty estimate is fabricated. The established [native R copula fitting
reference](https://search.r-project.org/CRAN/refmans/copula/html/fitCopula.html)
distinguishes rank inversion from likelihood methods; [statsmodels rank fitting](https://www.statsmodels.org/stable/generated/statsmodels.distributions.copula.api.GaussianCopula.fit_corr_param.html)
provides the Gaussian Kendall relationship. The [copula package paper](https://www.jstatsoft.org/article/view/v021i04)
describes the elliptical and Archimedean families used for independent diagnostics.

`simulate_copula` takes a frozen model and independent open uniforms with columns
in the frozen variable order. Gaussian draws use an ordered semidefinite
Cholesky transform of normal quantiles. The t family additionally consumes one
uniform per row for a common chi-square scale. Positive Clayton consumes one
uniform for gamma frailty; Gumbel above independence consumes two for a positive
stable frailty (angle and exponential draw). These auxiliary inputs must be
provided explicitly. Clayton theta0 and Gumbel theta1 return the independent
uniforms unchanged and reject unnecessary auxiliary inputs. Numerical boundary
counts and matrix-factorization residuals are reported. Normal CDF rounding may
produce exact zero or one; this is not silently remapped to an arbitrary epsilon.

`joint_default_probability` treats defaults as lower-tail events U<=PD and
requires either a bivariate frozen model or an explicit intersection. Event IDs
and order must match the frozen model exactly. Gaussian and t intersections use
deterministic one-dimensional conditional-distribution quadrature, with its
absolute error estimate reported. Archimedean intersections use their CDFs.
Zero/one marginal probabilities and perfect Gaussian correlations have direct
boundary solutions. Frechet bounds are enforced. Zero latent correlation in a
finite-df t model does **not** mean independent default events. A physical or
risk-neutral interpretation belongs to the declared target; these functions do
not convert between measures.

`make_shared_innovations(scenario_ids, time_grid, factor_ids, stream_spec, seed,
output_directory=...)` writes local float64 little-endian streams and a checksum
manifest. The output directory must be absent or empty. Each stream declares a
unique `stream_id` and `distribution` (`uniform` or `normal`); the seed is a
nonempty string. The canonical SHA256 key contains format version, seed,
scenario ID, numeric time, factor ID, stream ID and lane number. Its first32bits
produce `(integer+0.5)/2^32`. Normal draws use Box-Muller cosine with two addressed
lanes. Thus addressing, rather than a language-specific RNG seed, determines a
value. Batch size, scenario order, added scenarios and stream order do not alter
an existing address. Uniform files are bit-identical in Python/R; normal
transcendental arithmetic is verified numerically, and a shared frozen file gives
exact replay. Manifest text formatting may differ; each manifest carries the
hash of its own bytes. No global RNG state is touched. Files are ordered scenario,
time, factor (last axis fastest), so reordering scenarios reorders file rows while
retaining the value associated with every address.

Analytic tests include the Gaussian/t central-quadrant arcsine identity, Clayton
and Gumbel CDF values, singular comovement, independence limits and the
small-positive-Clayton limit. Shared fixture tests cover all four independent
fits and bidirectional frozen simulations. Marginal and Kendall checks use a
fixed finite innovation sample and stated tolerances; simulation checks are not
empirical evidence about a bank's portfolio.
