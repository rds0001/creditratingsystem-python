# Local credit decision policies

Risk estimates, affordability, eligibility and operational findings are separate
inputs. The three initial decision methods compile a supplied policy, evaluate
it at explicit dates, and explain the recorded evaluation. They do not execute
loans or modify accounts. Policy approval references are user assertions.

`compile_credit_decision_policy(specification)` accepts specification schema version `1.0` or `1.1`,
`policy_id`, `version`, `approval_reference`, `known_at`, `effective_from`,
`effective_until`, `target`, `fields`, `rules`, `default_action` and
`missing_action`. The effective interval includes its start and excludes its end.
The target declares `event`, positive `horizon`, `time_unit` (days/months/years),
`probability_measure=PHYSICAL`, and `conditioning` (for example PIT or TTC).
Conditioning is compared exactly at evaluation; it is never converted silently.

Fields use `risk.name`, `affordability.name`, `eligibility.name` or
`operations.name`. Every declared field is required. A field has `type=number`
or `type=category`; categories enumerate distinct string `values`. Numerical
fields contain finite binary64 values. Each rule declares a unique `rule_id`,
nonnegative integer `priority`, `action`, `reason_code`, and a `when` array.
Predicates contain `field`, `op`, `value`; numerical operators are `eq`, `ne`,
`lt`, `le`, `gt`, `ge`, and category operators are `eq`, `ne`. Predicates within
a rule are conjunctive; an empty array matches all cases. Exceptions are explicit
higher-priority rules, with smaller priority numbers evaluated first.

The compiler checks satisfiability over the declared category sets and finite
binary64 numbers, including exact endpoints and adjacent floating-point values.
Impossible rules fail. Equal-priority overlapping rules with different actions
fail. Overlap across priorities is recorded; a rule fully shadowed by one earlier
rule is identified. Joint coverage by several earlier rules is not claimed to be
exhaustively diagnosed. Equal-priority rules prescribing the same action resolve
by rule ID for deterministic reason selection. There is no executable rule text.

An unmatched case must explicitly REFER, DECLINE or BLOCKED. Missing/invalid
required information must explicitly REFER or BLOCKED. Rules can prescribe
APPROVE, REFER, DECLINE or BLOCKED. Specification `1.0` approval also requires
 `affordability.status=AFFORDABLE`, `eligibility.status=ELIGIBLE` and
`operations.status=CLEAR`. A rule cannot waive these gates; failure changes an
otherwise APPROVE result to REFER and records the failed gate.

`apply_credit_decision_policy(policy, risk, affordability, eligibility,
operations, as_of=..., known_at=...)` accepts four source objects (or CalcResult
objects containing those values). Each source includes `as_of` and `known_at`;
knowledge cannot precede its observation. Risk additionally includes the exact
policy `target` and `freshness=CURRENT`. Future, stale, missing or incompatible
inputs are recorded and use the configured missing action. Policy dates are
checked independently. Callers obtain freshness from their own dated assessment
contract; this function does not infer freshness from a universal age threshold.

Results contain the local action, immutable-content policy hash, four input
hashes, decision dates, findings and every rule's predicate outcomes. Missing
predicate results are null, not favorable values. Rule selection records the
executed table rule even if a subsequent approval gate forces REFER; the final
action and gate finding remain explicit. Canonical content hashes detect changes
but are not digital signatures or proof of who approved a policy.

`decompose_policy_decision(decision, reason_dictionary)` returns the stored
matched-rule trace, sources and descriptions supplied by the caller. The selected
rule and nonselected matches are distinguished. Gate/data findings remain
separate from rule reason codes, so an eligibility decline is not attributed to
the PD model. Unknown descriptions fail rather than inventing an explanation.

These methods do not yet provide policy tuning, limit or pricing optimization,
treatment evaluation or policy robustness analysis. Those require their own
statistical and cashflow contracts.

## Versioned investment and counterparty approval contracts

Specification schema `1.1` adds the mandatory `approval_contract` object. Policy
and decision artifact envelopes remain schema `1.0`; the specification version
controls these additional semantics. Existing `1.0` specifications retain their
exact behavior and content hashes. An `approval_contract` supplied with `1.0`
is rejected instead of silently ignored.

The contract contains exactly `domain`, `affordability` and `required_gates`.
Domains are `RETAIL_CREDIT`, `WHOLESALE_CREDIT`, `INVESTMENT` or
`COUNTERPARTY_LIMIT`. For required affordability, use `{"mode":"REQUIRED"}`.
Only the investment and counterparty domains permit `NOT_APPLICABLE`, requiring
nonempty `reason`, `reference` and a nonnegative integer `maximum_age_days`.
For example:

```json
{"domain":"COUNTERPARTY_LIMIT",
 "affordability":{"mode":"NOT_APPLICABLE",
  "reason":"CSA exposure, not household budget assessment",
  "reference":"scope-v1","maximum_age_days":30},
 "required_gates":[{"field":"operations.legal_status","value":"VERIFIED"}]}
```

The separate runtime affordability object remains mandatory and dated. For
nonapplicability it must have status `NOT_APPLICABLE`, the matching `domain`
and `applicability_reference`, and an exclusive `valid_until` date. Observation
age up to and including `maximum_age_days` is accepted; older evidence, a
mismatching binding, missing validity, expired validity or future knowledge
prevents approval. These are explicit bank-supplied applicability conclusions,
not a domain inferred from the product name or an automatic legal opinion.

Every additional gate references a declared categorical field and one of its
allowed values. Gates are distinct and conjunctive. Category fields remain
required even when no decision rule mentions them. Eligibility `ELIGIBLE`,
operations `CLEAR`, current risk and matching physical risk target remain
mandatory; no policy domain can waive them. Credit domains continue to require
`AFFORDABLE`. An otherwise approving rule with a failed status gate becomes
`REFER`, regardless of priority. Invalid/missing inputs use the existing
`missing_action`. A `1.1` decision records `approval_domain` and an
`approval_gate_trace` with field, required value, actual value and passed flag.
These checks describe local recommendations and do not authorize a trade.
