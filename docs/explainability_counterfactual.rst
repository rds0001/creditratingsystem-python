Finite feasible counterfactual search
=====================================

``find_feasible_credit_counterfactual`` enumerates a supplied finite matrix of
candidate records with distinct candidate_ids. The baseline and candidates use
the frozen model feature order. It reuses the ICE feasibility contract: explicit
bounds, immutable/actionable partition, integers, one-hot groups, linear
constraints and product/ratio identities. Infeasible candidates receive findings
and no model or policy evaluation. All evaluated failures remain in the report.

The cost_contract names cost_id, version, metric, feature_weights and
feature_scales. Both feature maps cover every model feature. Weights must be
nonnegative with at least one positive weight; scales must be positive. With
normalized changes d_j=(candidate_j-record_j)/scale_j, WEIGHTED_L1 computes
sum_j weight_j*abs(d_j), WEIGHTED_L2 computes sqrt(sum_j weight_j*d_j**2), and
WEIGHTED_CHANGE_COUNT sums the weights of exactly changed features. Zero weights
are explicit cost assumptions. Costs are not automatically monetary or empirical
intervention costs.

The versioned target contains target_id, version and kind. MODEL_OUTPUT targets
supply operator <=, >= or INTERVAL and a scalar value or two closed interval
endpoints. POLICY_ACTION targets supply actions and require the full policy
callback and binding contract documented for ICE. A low model prediction alone
does not satisfy an approval target when affordability or eligibility blocks it.

Candidates are visited by ascending cost and then candidate ID. max_candidates
limits the number of candidate records examined, including infeasible records.
If all supplied candidates were examined, the result is FEASIBLE_SOLUTION or
NO_FEASIBLE_SOLUTION, with the best feasible witness when one exists. If any
remain, status is SEARCH_LIMIT even when a feasible witness was found; all
unevaluated IDs are reported. No completeness or optimality outside the supplied
finite domain is claimed. Baseline feasibility and prediction, each candidate's
changes, cost, prediction, optional sealed policy decision and target result are
retained. The domain and contracts are content-hashed.

For target 4-x-2*y <= 0 on the integer square [0,3]^2 starting at (0,0), costs
abs(x)+2*abs(y) have minimum 4, with the candidate-ID tie rule selecting (0,2)
when IDs are x-y. Weighted Euclidean cost sqrt(x*x+2*y*y) instead selects (2,1)
with cost sqrt(6). Making y immutable leaves no feasible solution on that domain.
These are model-conditioned examples, not promises that real actions cause
approval or improve a borrower's outcome.
