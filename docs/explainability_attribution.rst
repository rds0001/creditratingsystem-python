Frozen linear and Shapley attribution
====================================

Both methods require a frozen model contract with ``model_id``, ``model_version``,
ordered ``feature_order``, ``output_scale`` (probability, log_odds, score or currency),
``deterministic_frozen=True`` and a target containing event, horizon, time_unit,
probability_measure and conditioning. The predictor is a local deterministic
callable taking a numeric row matrix and returning one finite prediction per row.
Probability outputs must lie in [0,1]. No model fitting occurs during explanation.
The user supplies an explicit nonoverlapping partition of all transformed features
into named groups. Grouping changes the attribution game.

``explain_linear_credit_prediction`` requires coefficients, intercept, identity/
logit/probit/cloglog link and transformed_feature_lineage (raw input names for each
transformed feature). Inputs already use the frozen model's transformations.
Relative to an explicit baseline, feature j contributes beta_j*(x_j-baseline_j).
Row offsets are compared to the separate baseline_offset. Contributions and group
sums reconstruct the linear predictor; response and response difference are
reported separately. Logit contributions are log-odds contributions, not additive
percentage-point PD changes. A baseline is a supplied reference, not an assertion
that the reference borrower is feasible or representative.

``estimate_credit_shapley_values`` explains one record using either exact grouped
coalitions or a supplied permutation plan. The masking contract supplies masking,
reference_id and reference_version. INTERVENTIONAL_EMPIRICAL uses the ``background``
object with values (rows by features) and nonnegative weights. It replaces observed
coalition features by the record while retaining all other background columns.
This deliberately breaks dependence between observed and masked features.
SUPPLIED_CONDITIONAL_DISTRIBUTIONS instead uses ``coalition_backgrounds`` in the
masking contract. Each requested coalition supplies values and weights conditional
on its observed feature values, which must equal the record exactly. Keys are
comma-separated zero-based indices of alphabetically sorted group IDs; ``EMPTY``
names the empty coalition. The complete coalition must reproduce the prediction.
Conditional distributions require independent justification by the user.

Exact enumeration needs 2**number_of_groups coalitions within ``max_coalitions``
(default 4096). Each group's weighted marginal contribution uses the standard
Shapley subset weight |S|!*(m-|S|-1)!/m!. Permutation mode takes a list of complete
group-ID permutations and averages their telescoping marginal contributions.
FIXED_PERMUTATIONS provides no sampling standard error. IID_RANDOM_PERMUTATIONS
reports the sample standard deviation divided by sqrt(number of permutations),
when at least two permutations are available. This quantifies only declared
permutation sampling, not model, reference-distribution or population uncertainty.
Every evaluated coalition is cached, counted and subject to the explicit budget.
The result includes the reference mean, prediction, group contributions, local
accuracy residual, evaluated coalition values and model/reference hashes.

For f(x,y)=x*y at (2,3) relative to (0,0), exact contributions are (3,3).
The two permutation paths contribute (0,6) and (6,0). With equally likely perfectly
correlated (0,0)/(1,1), f=x at (1,1) has conditional contributions (.25,.25) around
baseline .5; interventional attribution would give no contribution to unused y.
This distinction is intentional. Attributions are not causal interventions,
credit approval reasons or evidence of real-world model validity.
