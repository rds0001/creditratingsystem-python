# Smooth and tree PD models

These estimators share the frozen target, feature-order, training metadata and
serialization contract described in classification_models.md. Predictions contain
no fitted backend objects. Calibration and temporal or obligor-group holdout
validation remain separate steps; model probabilities are not evidence of
calibration. All input predictors must be finite and available at their forecast
origin. The caller declares the data split and information cutoff.

`fit_gam_pd` fits Bernoulli logistic additive models. Every feature appears exactly
once in `linear_features` or `smooth_specs`. A smooth declares `feature`, two
`boundary_knots`, increasing `interior_knots`, `degree` 1–3, nonnegative `smoothing`,
`monotonicity` none/increasing/decreasing and `extrapolation` error/constant/linear.
Knots and penalties are supplied training choices, never selected on prediction
rows. An intercept plus B-splines with first coefficient fixed to zero identify
the smooth. For monotone terms the remaining spline coefficients are cumulative
nonnegative increments (sign reversed for decreasing). Constrained fitting
therefore guarantees the derivative sign throughout the knot domain; no sorted
postprocessing or grid-only repair is used. The objective is mean Bernoulli
negative log likelihood plus one half of each smoothing parameter times the
squared second differences of the full spline coefficients. This is a coefficient
P-spline penalty, not an integrated squared derivative penalty. L-BFGS-B solves
both native implementations; projected KKT residuals certify stationarity. A linear
program checks for separating recession directions in the penalty null space
(relative eigenspace tolerance 1e-10), respecting monotone coefficient bounds.
Positive smoothing alone does not eliminate separation in unpenalized affine
directions; such fits are rejected. EDF
is the trace of penalized-information inverse times likelihood information on
locally free coefficients. It is conditional on the active constraints and supplied
smoothing choices, not a posterior uncertainty interval. With no smooths the model
reduces to ordinary logistic regression. Extrapolation can error, hold the boundary
value constant, or continue its tangent; the choice is frozen in each term.

`fit_random_forest_pd` constructs actual weighted CART trees from an explicit
`sampling_plan`, a list of records with zero-based `sample_indices` and
`node_features`. Repeated row indices are bootstrap draws; distinct indices form a
subsample. `node_features` maps paths `root`, `L`, `R`, `LL`, etc. to ordered distinct
zero-based candidate features. Each visited, eligible split node requires its
candidate draw; unvisited entries have no effect. For example:

```python
sampling_plan = [
    {"sample_indices": [0, 1, 1, 3, 4],
     "node_features": {"root": [1], "L": [0], "R": [1]}}
]
```

The library does not call a deterministic repeated tree a random sample: the
caller supplies the random design, which is recorded in diagnostics. At each
node, all midpoints of distinct values of the sampled features are evaluated with
weighted Gini or Bernoulli entropy (`criterion='gini'/'log_loss'`). Splits must meet
`min_samples_leaf` counting bootstrap rows and positive `min_weight_leaf`; depth
is bounded by `max_depth`. Candidate order, then increasing threshold, breaks gain
ties at 1e-14. Terminal probability is weighted default count divided by leaf
weight. Forest prediction is the arithmetic mean of leaf probabilities, including
exact zero/one values. No clipping, implicit resampling weights, calibration or
unreported class adjustment occurs. OOB probabilities use only trees for which
the original row is absent. Counts, missing OOB coverage and weighted OOB Brier are
explicit; OOB does not separate repeated borrowers or guarantee temporal validity.

`fit_gradient_boosting_pd` uses CPU XGBoost histogram Bernoulli loss with no row or
column subsampling. Explicit controls cover depth, iterations, learning rate,
L2 leaf penalty, minimum child Hessian weight, minimum split gain, histogram bins,
and one -1/0/1 monotonicity constraint per feature. Initial probability is the
training default fraction. Portable trees store already-shrunk leaf margins and
strict-less-than split tests. Features are cast to float32 at inference to preserve
the fitting backend's boundary routing. Nonrepresentable inputs are rejected.
The artifact contains only tree dictionaries and an initial logit, not XGBoost
binary files. Version differences can affect fitted trees; replay of the same
artifact is the cross-language exact contract.

Early stopping requires an explicit `inner_validation` record containing X, y,
unique row_ids disjoint from `train_row_ids`, `role='inner_validation'`, a distinct
split_id and known_at no later than the model information cutoff. A final test set
must not be labeled inner validation. Disjoint IDs cannot establish independence
between different records of one borrower; callers must construct the split at
the correct grouping level. Loss paths record all evaluated rounds and the selected
best prefix, which is the only prefix frozen for prediction. Validation with one
observed class is allowed. No validation data silently join the training objective.

Sources: [SciPy B-spline recurrence and derivatives](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html),
[R native spline basis](https://stat.ethz.ch/R-manual/R-devel/library/splines/html/splineDesign.html),
[P-spline monotonic constructions](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/smooth.construct.ps.smooth.spec.html),
[scikit-learn CART impurity and sample weights](https://scikit-learn.org/stable/modules/tree.html),
[XGBoost tree objective](https://xgboost.readthedocs.io/en/stable/tutorials/model.html),
and [XGBoost monotonic constraints](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html).

Independent checks include the cubic Bernstein basis identity, linear GAM versus
GLM MLE, analytic spline derivatives and active monotonic constraints, hand-weighted
leaf/OOB calculations, an independent sklearn CART fixture, and the one-step Newton
leaf update with explicit learning-rate shrinkage. Seven public model JSON fixtures
are replayed both directions: largest frozen difference 1.2e-16, independent GAM
fit difference 1.2e-7; controlled forest and boosting fixtures agree exactly.
