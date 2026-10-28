# Synthetic corporate component example

Run `examples/wholesale_corporate.py` with the installed Python package, or source the installed R example and call `run_wholesale_corporate()`. Both read packaged `wholesale_corporate.json` and fit native models. Every input is synthetic; amounts use one illustrative EUR scale.

Weighted training rates are 0.10 at net debt/EBITDA 2 and 0.30 at 4. Logit interpolation at leverage 3 is 0.1791287847. A separate dated calibration cohort provides an identity calibration oracle. Assessment information is available at 2024-01-01; hidden asset truth is excluded from training.

Equity price and volatility are generated from assets 100, asset volatility 0.20, debt 80 and risk-free rate 0.03. Structural inversion must recover the asset inputs. Physical drift 0.10 gives a terminal default probability distinct from its risk-neutral counterpart. First passage with zero log-asset drift has a reflection-principle oracle. Empirical distance calibration maps the structural signal to the observed default target before pooling. Linear, log-odds and conditional regime-mixture pooling have separate algebraic checks. These are PIT examples; no forecast average is labelled TTC.

SME and large-corporate profiles require different statement/listing evidence. Missing statements or audited evidence refers the case. Missing EBITDA remains missing. Senior, subordinated and secured instruments share issuer standalone PD. Proceeds 100, senior face 80 and subordinated face 40 give LGDs 0 and 0.50. Separate secured collateral proceeds 70 against face 80 give LGD 0.125. Instrument routes require declared verified legal evidence; absent eligibility produces no route. These flags are synthetic inputs, not a legal review.

Discretionary sponsor support uses joint default 0.01 and willingness times success 0.40: supported PD = standalone PD - 0.40 * (standalone PD - 0.01). Capacity is explicit; this is not a guarantee.

This covers the corporate component recipe and five variant routes. Full lifecycle acceptance remains separate and pending. Native Python/R numeric comparison uses absolute tolerance 1e-8 and relative tolerance 1e-7. Independent floating-point fits may produce different signal fingerprints; underlying values are compared without rounding.
