# Proper-prior Bayesian PD estimation

`fit_bayesian_pd` estimates a Bernoulli logit or probit posterior with independent
proper normal coefficient priors. The intercept is first in `prior_mean` and
`prior_sd`, followed by `feature_order`. Positive finite prior SDs are mandatory.
Single-class samples, including zero defaults, remain estimable under the prior.
The posterior is conditional on the declared likelihood, independent-row
assumption and priors. Repeated forecasts of one obligor do not become independent
observations through a prior. Input features are already prepared training values;
no scaling is learned by this estimator.

The sampler performs actual symmetric random-walk Metropolis updates. It accepts
explicit independent standard-normal innovations and open-interval uniform draws,
so native R and Python can replay the same chain without backend random state:

```python
rng = np.random.default_rng(17)
sampler_spec = {
    "initial_positions": np.zeros((4, coefficient_count)),
    "proposal_cholesky": np.eye(coefficient_count) * 0.3,
    "normal_innovations": rng.normal(size=(4, 10000, coefficient_count)),
    "uniforms": rng.uniform(size=(4, 10000)),
    "warmup": 2000,
    "thin": 1,
    "innovation_contract": "INDEPENDENT_STANDARD_NORMAL_AND_UNIFORM",
}
```

Supply the corresponding numeric matrices/array (chain, step, coefficient) or
nested lists in R. Proposal Cholesky is lower triangular with positive diagonal;
there is no hidden adaptation. Initial positions and innovation fingerprint are
recorded. All retained draws are frozen in the artifact; predictions never rerun
the sampler. A poorly chosen proposal can mix slowly; increasing draws cannot
replace examining diagnostics. Thinning is supported for explicit storage choices,
not presented as a cure for dependence.

Diagnostics include per-chain acceptance, rank-normalized and folded split-Rhat,
Geyer initial-positive/monotone paired-sequence bulk and tail ESS, and raw-quantity
ESS for posterior-mean MCSE. ESS is conservatively capped by draw count. Chains are
split into equally long first/last halves, discarding the middle draw if needed.
Rhat uses average ranks, the Blom normal-score transform and the larger folded or
unfolded value. Zero within-chain variation yields unavailable Rhat and zero ESS.
`DIAGNOSTICS_PASSED` requires Rhat <=1.01 and both bulk and tail ESS >=400 for every
coefficient. An insufficient or failed fit returns its diagnostic status and draws
for investigation; it does not silently claim convergence. Even passed diagnostics
cannot prove convergence from finite chains. Credible intervals and MCSE require a
representative stationary sample; inspect the status before using them.

`predict_pd` returns the posterior **mean of event probabilities**, not the inverse
link applied to mean coefficients. Its diagnostics include equal-tail PD credible
intervals at `credible_level`, posterior SD of latent PD, Monte Carlo standard
error of mean PD, prediction-specific chain diagnostics and fit status. The
individual Bernoulli predictive variance is mean_PD*(1-mean_PD). A credible
interval describes uncertainty in conditional PD, not a fractional outcome for one
borrower. `predict_model_score` exposes the same posterior mean with an explicit
probability-score scale.

Optional `group_spec` supplies all observed `levels`, a training `labels` vector
and positive `prior_sd` for normally distributed group intercepts. The group prior
mean is zero; its SD is fixed rather than estimated. This implements partial
pooling with posterior group uncertainty conditional on that SD. All declared
levels must be observed. Predictions then receive `X={"X": matrix, "groups": labels}`
in Python or `list(X=matrix, groups=labels)` in R. Previously unseen groups are
rejected; the library does not substitute a zero effect and call it integrated
prediction. Posterior parameters are intercept, slopes, then group effects in the
declared level order.

Sparse-default inference can be strongly prior-sensitive. Refit explicitly with
substantive alternative prior means/SDs and compare posterior predictions and
intervals; no empirical-Bayes choice is inferred from a heldout test set. All fits
use the common immutable target, split and information-cutoff metadata. External
calibration and out-of-time validation are still required for a banking application.

Sources: [Stan posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html),
[rank-normalized Rhat and ESS](https://mc-stan.org/rstan/reference/Rhat.html),
and [mean ESS/MCSE interpretation](https://mc-stan.org/posterior/reference/ess_basic.html).
Independent checks integrate the intercept-only posterior by one-dimensional
quadrature for both links, examine low-default prior sensitivity and deliberately
nonmixing chains, and replay shared innovations in native R/Python. Frozen public
JSON posterior predictions agree to floating-point precision.

Default tests retain small genuine fits and frozen predictions, including both Bayesian links, MLP, GRU and LSTM. Full posterior recovery against quadrature and the optional independent Torch autograd oracle run when `CREDITRATINGSYSTEM_EXTENDED_TESTS=true`. All-parameter finite-difference gradient checks run by default. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` for test runs.
