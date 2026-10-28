# creditratingsystem

[![Minimal tests](https://github.com/rds0001/creditratingsystem-python/actions/workflows/minimal-tests.yml/badge.svg)](https://github.com/rds0001/creditratingsystem-python/actions/workflows/minimal-tests.yml)

GitHub Actions installs the package and runs `tests/test_core.py` on Python 3.11
and Linux. This is a minimal core check; the full test suite is separate.

A Python library for granular credit risk modelling with local user data.

See [getting started](docs/getting_started.md) for local installation, typed CSV
input and a workflow for fitting and validating models with your own data.

Development version; product integration and final package acceptance are in progress.
Available methods include train-only preprocessing and monotone binning,
scorecards, binary GLMs, frozen PD calibration, Merton and first-passage models,
Gaussian-factor PIT/TTC conversion, observed long-run PD, and discrete and
continuous-time transition curves. Survival includes Cox, AFT, competing risks,
Fine-Gray and multistate estimates; collections projects ordered cure and redefault.
Dated workout recovery and exposure schedules retain economic loss and raw CCF. Joint loss paths preserve default, LGD,
exposure and discount dependence; portfolio tails use fractional boundary mass.

Consumer methods cover application snapshots, affordability, credit-policy
rules, dated account panels, behavioral predictions, roll rates and vintage
performance. Collections methods preserve open episodes, probation and
post-cure censoring, reconstruct pre-action evidence and compare repayment
cashflows. Model registries, role histories and pinned batch recipes support
historical replay. Monitoring reports cohort maturity, missing outcomes,
feature/calibration drift and explicit challenger/group comparisons.

The installed retail example connects training, separate calibration, application,
booking, behavioral prediction, monitoring, collections and historical replay.
Its tiny synthetic cohort deliberately reports insufficient monitoring evidence.
Assumptions, applicable models and numerical conventions are documented per method.

```python
from creditratingsystem import expected_loss
expected_loss(0.02, 0.4, 1000).value  # 8.0
```

Each calculation returns its value, units, method identity, context, provenance,
diagnostics and limitations. Probabilities use decimal fractions. User-supplied
rating scales and parameters remain explicit. Calculations do not download data
or send user information to external services.

Classification, LGD/EAD and selection-bias estimators are implemented, alongside
cashflow, structured-credit and market-implied methods. Product examples retain
their own dated contracts, outcomes and applicability limits.
Synthetic evidence does not establish performance on a bank's own portfolio.

Author: RiskDataScience GmbH. Responsible person: Dr. Dimitrios Geromichalos.
Contact: riskdatascience@web.de.

Licensed under the GNU Affero General Public License, version 3 only
(`AGPL-3.0-only`); see [LICENSE](LICENSE).

Bundled synthetic data include six connected tiny reference worlds and a standard
wholesale/macro/workout panel. Native offline generators also create larger retail,
asset-pool and system-event worlds from shared innovation files. Observable inputs,
latent truth, frozen predictions, decisions and outcomes are stored separately.
See [synthetic data](docs/synthetic.md) for tier sizes and generation contracts.

See [test tiers](docs/testing.md) for quick default checks and explicit extended acceptance.
