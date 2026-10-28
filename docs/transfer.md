# Transfer events and payment restrictions

The four functions distinguish applicability, event occurrence, payment default and economic loss. Inputs are bank-supplied assumptions, not interpretations of current law. A sovereign grade alone does not establish transaction exposure or a probability.

`assess_transfer_applicability(payments, event_policy, as_of=..., known_at=..., horizon_end=...)` matches payment route jurisdictions and settlement currency against dated, explicitly named conversion restrictions, payment freezes and cross-border blocks. Payment dates use the inclusive forecast window; policy validity is `[effective_from, effective_until)`. Policy events and exemptions must be known at the knowledge cutoff. Equal settlement/source currency eliminates a conversion restriction but does not eliminate a payment freeze. Missing route or currency information yields `INSUFFICIENT_INFORMATION`. A matched policy interval does not assert that the event has occurred. Policies identify their own version and each exemption supplies a reason and evidence reference.

`transfer_event_default_probability(event_probability, event_to_default_probability, mitigation_contract, event_id=..., target=...)` computes `q = event_probability * failure_given_event * (1 - mitigation_success)`. Success must explicitly condition on the event **and** the unmitigated failure. The mitigation contract carries a unique economic effect ID and assumption reference. A zero success probability explicitly represents no effective mitigation. The result records gross default mass and avoided mass separately.

The common `target` declares `exposure_id`, positive `horizon`, `time_unit` (`days`, `months`, `years`), `probability_measure` (`PHYSICAL` or `RISK_NEUTRAL`) and `conditioning`. All input probabilities concern this same target. The functions neither estimate conditional probabilities from bank history nor infer horizon conversion.

`combine_default_transfer(pre_transfer_risk, transfer_risk, dependence=..., joint_probability=...)` requires a base object with `stage='PRE_TRANSFER'`, `pd`, `target` and `applied_effects`, plus the transfer-default result. It enforces identical targets and disjoint effect IDs. With base PD `p`, transfer PD `q` and intersection `j`, the four state masses are:

| State | Probability |
|---|---|
| Neither | `1-p-q+j` |
| Default only | `p-j` |
| Transfer only | `q-j` |
| Joint | `j` |

`max(0,p+q-1) <= j <= min(p,q)` is mandatory. `dependence='explicit'` requires the intersection; `dependence='independent'` computes `j=p*q` and rejects an additional supplied intersection. The union is `p+q-j`. Already adjusted base stages are rejected. Independence is an assumption, not a fitted dependence model.

`transfer_loss_components(combined_risk, conditional_losses, currency=..., applied_loss_effects=...)` consumes an explicit monetary conditional loss for each of the four states. Each can represent the joint discounted expectation of LGD and exposure, including recovery timing and cashflow restrictions calculated separately. Negative economic losses are permitted. There is no independence assumption between LGD and EAD within a state, and the joint-state loss is never averaged from standalone losses. The expected loss is the sum of state probability times conditional loss. An effect used in PD mitigation cannot be applied again at loss level under the same ID. Users must identify economic effects consistently; undeclared duplication cannot be inferred.

For example, `p=q=0.1` under independence gives masses `(0.81,0.09,0.09,0.01)`. With corresponding losses `(0,60,20,70)`, expected loss is `7.9` currency units. Complete nesting instead yields union PD `0.1`; a zero transfer probability preserves the base PD and base loss. Analytic tests cover these cases, conditional mitigation boundaries, infeasible intersections and historical exemptions. Python and native R replay the same frozen finite-event fixtures.

Applicability is a due-date route-screening model: it uses the declared payment due date, not separately timed conversion, instruction and settlement legs. For leg-level analysis, supply separate payment rows for the distinct legs with their own dates and jurisdictions. A knowledge cutoff later than `as_of` produces a retrospective view; historical backtesting must use the knowledge cutoff available at the original decision.
