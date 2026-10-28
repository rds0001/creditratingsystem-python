Stored evidence and deterministic reason codes
==============================================

``rank_credit_reason_codes`` consumes stored contributions, an explicit reference
basis and a versioned reason dictionary. Each contribution has a unique feature_id
and a finite scalar contribution. The reference names attribution_id, model_id,
model_version, reference_id, explained_quantity, output_scale, baseline, prediction
and adverse_direction (HIGHER or LOWER). Contributions must reconstruct prediction
minus baseline on that same scale within numerical tolerance. Probability values,
log odds, scores, currency and general linear predictors remain distinct.

The dictionary has dictionary_id, version, model_reasons and policy_reasons.
Each model reason specifies reason_code, description, nonnegative integer priority
and feature_ids. Feature sets may leave contributions unmapped, but cannot overlap
or contain unknown features. All transformed features representing one underlying
fact should be assigned to one reason. Their signed contributions are summed
before testing adversity; offsetting effects can therefore suppress a group.
The output reports suppressed groups and every original backing contribution.
Adverse model groups rank by descending net adverse magnitude, ascending priority,
then code. An optional positive max_reasons retains omitted reasons separately.
Coverage reports mapped feature counts and mapped absolute original contribution
mass; this is attribution coverage, not model validity or causal coverage.

An optional sealed DecisionArtifact contributes matched policy rules separately.
Each policy dictionary reason binds its code to explicit rule_ids. Matched rules
without this binding are rejected; multiple rules under one code are consolidated
without inventing additional explanations. Selected status, action and source
names are preserved from the recorded rule trace. Policy groups rank by priority
then code, and are never numerically added to model contributions. Unselected
matched rules remain visible. Recorded findings are returned separately.
Content hashes detect changes, but do not authenticate the supplied evidence.
These descriptive reasons are not automatically a legally sufficient notice or
evidence that changing a feature causes a different real credit outcome.
