# Fast checks and extended acceptance

The default pytest suite uses analytical cases, small real model fits and short
observation windows. It exercises all implemented method families without running
large-data audits or statistical recovery experiments. Backend tests use one
thread. No test is skipped merely because it runs on a particular hosting service.

Run `python -m pytest tests --durations=15` for the default suite. To additionally
run expensive statistical, exhaustive-variant and full-data checks, set
`CREDITRATINGSYSTEM_EXTENDED_TESTS=true` before the same command. The `extended`
marker identifies opt-in checks; some estimator tests use the same explicit
environment switch. Optional independent-oracle software is separate from runtime
requirements. Mandatory finite-difference and analytical tests remain enabled.

Routine CI should run the default suite on supported platforms. Full acceptance
and large stress-tier generation should run in separately selected jobs. Track
individual test timings and total elapsed time; do not weaken tolerances or remove
method coverage to meet a budget. Large synthetic datasets remain available
offline and need not be regenerated for every code change.

Run extended checks from an editable GitHub checkout (`python -m pip install -e ".[test]"`).
The standard reference panel used by those checks is retained in the repository
and is not included in PyPI distributions; see [synthetic data](synthetic.md).
