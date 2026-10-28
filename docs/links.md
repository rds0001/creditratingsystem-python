# Relationships, support, guarantees and clearing

These functions accept local records and return `CalcResult`. They do not require
an enterprise graph, database or server. Dates are ISO calendar dates. A supplied
knowledge cutoff can describe a retrospective view; historical decisions must
use their actual information cutoff. Risk inputs identify the entity, continuous
`pd`, `stage`, `target` and distinct `applied_effects`. Default targets explicitly
specify event, positive horizon, time unit, probability measure and conditioning.
Entity/exposure identifiers may differ across counterparties; event semantics must
agree. Calculation context cannot contradict the target.

`build_entity_graph(entities, edges, relationship_policy, as_of=..., known_at=...)`
selects known edges valid in `[valid_from, valid_until)`. Entities have unique
`entity_id`; edges have unique `edge_id`, `source_id`, `target_id`, `edge_type`,
validity and knowledge dates, and `evidence_reference`. An ownership edge also
has `ownership_fraction`. A versioned policy declares the permitted edge types.
The content-bound graph reports excluded edges, repeated relationships and cycles
by type. Ownership, control, support, guarantee and cross-default are distinct.

`resolve_parent_links(graph, control_policy)` uses explicit control edge types
and an explicit ownership threshold with an inclusive/exclusive flag. It returns
candidate parents, edge traces and ultimate parents. Multiple eligible parents
or a cycle remain unresolved. It does not multiply indirect ownership shares or
infer support from a parent relationship.

`assign_default_groups(graph, default_link_policy, facility_mappings)` requires
selected edge types, `direction_policy='mutual_only'|'undirected_explicit'`, and
Boolean `transitive`. Mutual-only requires a reverse relationship. Transitive
policy forms connected equivalence classes; a nontransitive policy rejects
non-clique overlapping groups as unresolved. Facility mappings identify a facility
and its entity. A common group never silently assigns a common PD.

`apply_parent_support(standalone_risk, parent_risk, support_contract,
joint_default_probability=...)` accepts distinct standalone risks. It decomposes
child default into parent-alive mass `p-j` and joint-default mass `j`, checking
Fréchet bounds. Each state explicitly supplies available capacity, permission,
conditional willingness, success given willingness and funding, and evidence.
Funding must cover the stated rescue amount. The contract also declares total
capacity, reservations elsewhere and the allocated capacity; an arbitrary funding
ratio is never a rescue probability. Support in parent default needs its own
permitted and funded terms. Output stage `PRE_TRANSFER` preserves the child's
target and records the effect. Cross-beneficiary budgets are caller evidence.

`allocate_guarantee_coverage(guarantees, claims, allocation_policy, as_of=...,
known_at=..., currency=...)` reserves each shared, non-reinstating nominal cap once.
Claims identify entity, amount, currency, known-at and future due date. Guarantees
identify guarantor, cap, consumed cap, coverage fraction, covered claim IDs,
validity, knowledge, legal-effectiveness flag and evidence. The versioned policy
supplies complete guarantee and claim priority orders. Each allocation is bounded
by remaining cap, remaining uncovered claim and original claim times coverage
fraction. Claim maturity must be within guarantee validity. FX conversion and
legal conclusions are explicit inputs. The sealed result retains original claims.

`apply_guarantee_risk(borrower_risk, guarantor_risk, allocation, claim_id,
loss_contract, joint_default_probability=...)` preserves borrower PD and changes
monetary loss. The loss contract requires a new effect ID, `stage='PRE_GUARANTEE'`,
currency, assumption reference, `valuation_date` equal to the claim due date, and
borrower-only/joint-default conditional losses and payout fractions. In each
state payment is `min(allocated cap, max(loss,0))*payout_fraction`. No payout is
inferred from guarantor PD alone. Previously guarantee-adjusted inputs are rejected,
including when a caller renames the effect. All protection on this claim must have
one guarantor; several guarantors need a joint event model. Discount state losses
separately. Nominal caps must not be compared directly with present-value losses.

`apply_rating_constraints(base_result, rating_scale, constraints,
representative_pd_policy=...)` validates base PD/grade consistency and intersects
all supplied bounds. Grade order is increasing risk. A `grade_bounds` rule declares
best/worst allowed grade; `notch_bounds` declares reference grade and nonnegative
integer maximum improvement/deterioration. Each rule has an ID and evidence.
An empty intersection raises a conflict. Nonbinding bounds preserve continuous PD
exactly. `preserve_pd` changes only grade and reports any resulting PD/grade
inconsistency. `representative_on_binding` changes PD to the selected grade's
representative only when the original grade is excluded. Original values and
binding rules remain available.

`apply_override(base_result, request, policy, as_of=..., known_at=...)` produces a
sealed original/current/history artifact. Requests record ID, target, value,
reason, author, approver, recording date and nonempty validity interval. A
versioned policy declares target paths and probability, grade-set or bounded
numeric domains, with optional dual control. Audit/event fields cannot be targets.
Expired, future-valid and not-yet-known requests are recorded without applying.
Duplicate request IDs fail. Chained adjustments retain original values and hashes;
a different evaluation cutoff requires replay from the preserved original, so an
expired past override cannot silently persist. Only the requested field changes.
The artifact records declared evidence; it does not authenticate users or approve
real loans.

`solve_contagion_network(liability_matrix, external_assets, external_liabilities,
entity_ids=..., currency=..., model_assumptions=..., asset_scenarios=...)` implements
static Eisenberg–Noe clearing. Debtors are rows, creditors columns, diagonal zero.
Explicit assumptions are `model='eisenberg_noe'`, `seniority='single_pari_passu'`,
`bankruptcy_costs='none'`, `outside_creditor='sink'`. Outside obligations share the
same seniority and outside creditors do not recycle payments. With total nominal
liabilities b and relative liability matrix Π, payments satisfy
`p=min(b, external_assets + transpose(Π)*p)`.

SciPy HiGHS/native R lpSolve maximize the sum of payment fractions over feasible
payments, selecting the greatest clearing. Node-relative fixed-point residuals
are checked. A monotone iteration from zero supplies a lower bound. A small step
alone does not certify uniqueness: the returned greatest/lower gap is explicit.
An exact second fixed point with a nonzero gap reports `MULTIPLE_CLEARINGS`;
otherwise an unresolved gap reports `NOT_ESTABLISHED`. The numerical enclosure
status is tolerance-dependent. A closed zero-asset reciprocal cycle can have
both zero and full-payment clearing. Default classification uses the same stated
relative tolerance. Extreme amount dynamic ranges below 1e-10 of the largest
amount are rejected rather than silently discarded. Supplied asset scenarios
produce actual payment/default sensitivity while holding liabilities fixed.

The clearing formulation follows Eisenberg and Noe (2001), *Systemic Risk in
Financial Systems*, DOI [10.1287/mnsc.47.2.236.9835](https://doi.org/10.1287/mnsc.47.2.236.9835),
particularly its fixed-point and increasing-objective characterizations.
It is not a group-PD estimator, multi-seniority waterfall or dynamic insolvency model.
