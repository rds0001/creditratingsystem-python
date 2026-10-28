# Frozen classification models

These functions fit training data supplied by the caller. Artifacts contain only
numbers, ordered feature names, preprocessing, target, provenance and diagnostics.
They never retain a backend estimator or use prediction rows to estimate scaling,
category sets, smoothing parameters or coefficients. Training metadata identifies
the data fingerprint, split and information cutoff; callers must construct the
training sample with outcomes already observed by that cutoff. Repeated forecasts
of one obligor are not independent observations for the reported likelihood and
sandwich diagnostics. Temporal/group validation and probability calibration require
separate, mature holdout data; training fit does not establish calibration.

All fits require `target`, `training`, `model_id`, `model_version`, and
`feature_order`. The target contains event, positive horizon, time_unit,
probability_measure and conditioning. Frozen predictions require the same ordered
features and exact `expected_target`. `predict_model_score` returns a numeric score
with its scale, orientation and model lineage. A score is not a calibrated PD.
`predict_pd` can use model-based class posteriors; it rejects raw SVM artifacts.
Public `serialize_fitted_model`/`load_fitted_model` transfer artifacts between Python
and native R without backend objects. No fitting backend is called for inference.

| Function | Estimand and controls | Diagnostics and limits |
|---|---|---|
| `fit_discriminant_pd` | `variant='lda'/'qda'`, covariance `mle` or `unbiased`, explicit additive `covariance_ridge`, optional positive class `priors=[P(0),P(1)]` | Gaussian Bayes posterior; pooled within-class LDA scatter denominator n or n-2, QDA per-class denominator n_k or n_k-1. At least two rows per class, no silent dropped features; Cholesky and condition-number check. Ridge units are squared supplied-feature units. |
| `fit_naive_bayes_pd` | `variant='gaussian'/'categorical'/'mixed'`, explicit `feature_types`, strictly positive additive `alpha`, Gaussian `variance_floor` added to MLE variance | Conditional independence assumption, class counts and all fitted counts/probabilities. No empirical calibration or parameter-uncertainty intervals implied. |
| `fit_svm_score` | linear or RBF, explicit positive C and RBF gamma; train-only population-standard-deviation scaling, positive optional class weights | Signed LIBSVM margin, higher means default; portable support vectors, signed dual coefficients and intercept. Native R verifies dual KKT conditions. R rejects a non-NULL max_iter because e1071 does not expose an iteration cap; Python can enforce an optional cap. |
| `fit_firth_logit` | Bernoulli log likelihood + half log determinant of expected information; full rank, optional intercept | Python exact adjusted score with BFGS, native R brglm2 MPL_Jeffreys a=1/2; finite separated estimates. Allows an entirely nondefault sample when design has full rank. Stored inverse expected information is a local approximation, not a penalized profile interval. |
| `fit_penalized_glm` | logistic ridge/lasso/elasticnet; explicit decreasing lambda_path, selected_lambda, elastic-net l1_ratio, scale and case weights | Unpenalized intercept; weighted mean negative likelihood + lambda[(1-alpha)/2 sum(beta²)+alpha sum(abs(beta))]. Frozen weighted scaling; full objective and KKT residual at every lambda. No automatic holdout selection; no post-selection confidence intervals. |
| `fit_ordinal_model` | cumulative logit/probit, explicit ordered_categories from low to high risk | P(Y<=j|x)=F(threshold_j-x beta). Strictly ordered thresholds, category probabilities, joint sandwich Wald parallel-slopes diagnostic. Default PD requires explicit terminal default_categories at the target horizon. |

Naive Bayes tables contain finite numeric Gaussian columns and nonempty string
categorical columns. In Python pass rows or a DataFrame; in R pass a data frame or
list of rows to preserve mixed types. Missing values are Python None/NaN or R
NULL/NA. `missing_policy='error'` rejects them; `'marginalize'` omits the corresponding
likelihood factor; `'category'` reserves a separate categorical missing bin and
rejects missing Gaussian values. `unknown_policy='reserved'` reserves a distinct
unknown-category bin even if never seen during training; `'error'` rejects new
categories. Both reserved bins are counted in the Laplace denominator. Gaussian
missingness marginalization does not claim missing-at-random is testable from data.

Penalties act on fitted standardized slopes when scale=True, on supplied units
otherwise. Zero-variance scaling factors become one without deleting columns;
ridge can identify otherwise redundant slope coefficients, whereas an unpenalized
zero-lambda design must be full rank and nonseparated. A zero lambda reproduces
ordinary logistic MLE. Alpha endpoints are exposed as ridge and lasso; elasticnet
requires an interior l1_ratio. Case weights normalize by their sum and need not be
integers; they do not automatically define replicated-sample uncertainty.

Ordinal covariance uses slopes followed by the first threshold and log threshold
spacings. The native R threshold covariance is transformed into that convention.
The intercept-only ordinal estimate equals observed category frequencies. For
nonzero features, the proportional-slope diagnostic fits each cumulative binary
regression and combines their row-level influence functions, retaining cross-
threshold correlation. It is an asymptotic IID Wald diagnostic, not a test of
causal effects or proof of model adequacy. Separated threshold fits or singular
contrast covariance produce NOT_ESTIMABLE. `predict_model_score` provides category
probabilities in diagnostics even without a declared default mapping.

## Sources and verification

Primary mathematical/backend sources: [scikit-learn Gaussian discriminants](https://scikit-learn.org/stable/modules/lda_qda.html),
[MASS native discriminants](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/lda.html),
[scikit-learn naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html),
[LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/),
[brglm2 Jeffreys fitting](https://github.com/ikosmidis/brglm2/blob/main/R/brglmFit.R),
[glmnet elastic-net algorithms](https://www.jstatsoft.org/article/view/v106i01),
[statsmodels OrderedModel](https://www.statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html),
and [MASS polr](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/polr.html).

Tests use independent normal-density Bayes calculations, MASS/sklearn reference
fits, categorical hand probabilities, an exact two-point SVM margin, the exact
intercept-only Firth estimate (events+1/2)/(n+1), finite-difference Jeffreys gradients,
weighted row replication and zero-penalty MLE equivalence. Public model JSON is
replayed both ways for all 13 stage-one variants; frozen predictions agree at
floating-point precision, independently fitted predictions within 3e-8 on the
recorded deterministic fixtures. These fixtures establish numerical contracts,
not performance on a bank's portfolio.
