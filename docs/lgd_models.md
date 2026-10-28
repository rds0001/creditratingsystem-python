# Conditional LGD regression

All three estimators require an explicit physical target with
`conditioning='given_default'`, identity-preprocessed predictor columns, and
training metadata including `workout_population='closed_defaulted_episodes'`,
`data_fingerprint`, `split_id` and `known_at`. The caller must construct mature,
closed episodes and predictors available at the stated observation date. These
matrix estimators cannot verify row-level chronology from a metadata declaration.
Economic LGD outside [0,1] is never clipped: bounded-response estimators reject it.

`fit_fractional_lgd` estimates a logit conditional mean by Bernoulli quasi
likelihood, allowing exact zero and one targets. HC0 uses the sandwich covariance;
HC1 additionally multiplies it by N/(N−k). Integer frequency weights represent
independent replicated rows, so the sandwich meat uses w, not w². These are not
exposure, survey, cluster or importance weights. Rank deficiency, complete/quasi
separation and failed convergence are controlled errors. Constant interior
responses can have zero empirical robust covariance. A conditional mean alone
does not identify quantiles. See [Papke and Wooldridge (1996), equations 5–9](https://academicweb.nd.edu/~rwilliam/ndonly/readings/CDA/Fractional/Fractional_Response_Models_1996.pdf).

`fit_beta_lgd` uses logit mean and log precision with distinct design matrices.
With µ=logit⁻¹(Xβ) and φ=exp(Zγ), the density is Beta(µφ,(1−µ)φ).
The precision design always includes an intercept; omitted design means constant
precision. Endpoints and a perfectly explained interior response are rejected.
BFGS uses the analytic score; reported covariance inverts the positive-definite
observed likelihood information. This covariance assumes independent observations
and the correctly specified beta distribution. Mean and precision uncertainty
are jointly retained. [Cribari-Neto and Zeileis (2010)](https://www.jstatsoft.org/article/view/v034i02)
describe variable mean/precision beta regression.

`fit_zero_one_inflated_lgd` requires
`component_spec={'boundary_origin':'structural','gate_link':'logit'}`.
It estimates boundary probability b, probability o of one conditional on a boundary,
and an interior beta distribution. Thus P(0)=b(1−o), P(1)=bo,
P(interior)=1−b, and E(LGD)=bo+(1−b)µ. Likelihood contributions split over
these observed components. All three components need positive-frequency support;
separation or nonidentifiability in any component fails the full fit. Rounded or
censored endpoints require a different observation model. The mixture follows the
[zero-and-one-inflated construction of Ospina and Ferrari, section 3](https://arxiv.org/pdf/0705.0700),
with separately fitted regression designs.

`predict_lgd` requires matching `feature_order`, `expected_target`, and
`default_condition='given_default'`. Beta precision predictors must also match.
Optional strictly increasing `quantile_levels` are interior probabilities;
fractional-only models reject quantile requests. Inflated-beta quantiles account
for both endpoint atoms. Predictions expose marginal training-range violations,
including component-specific ranges; this is not a joint-support test. Saved
models are declarative JSON artifacts with coefficients, covariance, target,
training and input fingerprints; no callbacks or executable objects are needed.

```python
import numpy as np
from creditratingsystem.lgd import fit_fractional_lgd, predict_lgd

target = dict(event='economic_loss', horizon=5, time_unit='years',
              probability_measure='PHYSICAL', conditioning='given_default')
training = dict(data_fingerprint='user-training-v1', split_id='train',
                known_at='2025-12-31', data_origin='USER_SUPPLIED',
                workout_population='closed_defaulted_episodes')
fit = fit_fractional_lgd(np.empty((4, 0)), [0, .2, .8, 1],
    feature_order=[], link='logit', covariance='HC0', target=target,
    training=training, model_id='lgd-mean', model_version='1')
prediction = predict_lgd(fit, np.empty((1, 0)), feature_order=[],
    expected_target=target, default_condition='given_default')
assert prediction.value['mean_lgd'] == [.5]
```

These estimators do not supply downturn selection, regulatory calibration,
open-workout censoring corrections or model approval. Native R uses the same
likelihoods with its own optimizer. Iteration counts can differ across runtimes;
frozen artifacts replay in either language without refitting.

Beta fits report `effective_score_tolerance=max(sqrt(tolerance),1e-7)`;
this normalized score threshold determines numerical acceptance. Native R reports
`gradient_evaluations`, whereas SciPy reports `iterations`. Prediction preserves
training provenance and rejects conflicting context horizon, units, measure,
conditioning or model version, plus currency when declared in the frozen target.
LGD ratios without a declared target currency do not establish a monetary currency.
