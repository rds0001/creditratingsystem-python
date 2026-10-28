Feasible local ICE paths
========================

``compute_credit_ice`` evaluates a supplied ordered finite grid for selected
features of one record, using the same frozen prediction contract as Shapley
attribution. Grid rows contain values in varying_features order; other features
remain at their recorded values. U-shaped responses and every adjacent crossing
are retained. A grid need not be monotone, so turn_indices describes changes
along the supplied path rather than a derivative with respect to a sorted axis.
No threshold location between grid points is inferred.

The feasibility contract declares contract_id, version, bounds for every feature,
a partition into actionable_features and immutable_features, integer_features,
one_hot_groups, linear_constraints and derived_relations. Empty constraint lists
are explicit. Bounds are finite and closed; immutable and integer equality is
exact. One-hot groups require binary values summing to one. Linear constraints
have constraint_id, coefficients (feature-to-coefficient map), operator <=, >= or
==, and rhs. Derived identities have constraint_id, output_feature, exactly two
input_features, and operation product or ratio. A zero ratio denominator fails
feasibility. Linear and derived equalities use a relative numerical tolerance of
1e-10. Constraints encode user knowledge; passing them does not establish
empirical support or causal feasibility.

Every infeasible row receives findings and a null prediction before any predictor
or policy callback is invoked. The baseline is checked too. Adjacent intervals
containing an invalid endpoint are NON_ESTIMABLE and are not interpolated. Optional
scalar thresholds have threshold_id, value and operator <= or >=. Their matched
states and each crossing are returned separately from rating and policy results.

A rating_scale may be supplied only for probability output. The package's frozen
``map_pd_to_grade`` interval convention then determines each grade and adjacent
grade change. An optional policy_evaluator(record, prediction) must replay a full
frozen policy and return a sealed DecisionArtifact under ``decision``, together
with explained_input_hash of the object containing record and prediction. Its
policy_contract declares policy_hash, source_mode REPLAY_FULL_POLICY, and
fixed_source_hashes (a source-to-content-hash map for any risk, affordability,
eligibility or operations inputs that must remain fixed). Hash mismatches and
unknown actions are rejected. Each complete returned decision and action change
is preserved. Input bindings are caller assertions, not independently authenticated
proof of how an arbitrary callback was written. This local replay never executes
an actual credit action.
