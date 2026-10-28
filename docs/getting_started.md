# Working with your own credit data

The library runs locally. Supply your own data, target definition, rating scale,
model parameters and policy. The synthetic examples provide executable reference
contracts and deliberately small independent checks; their fitted parameters are
not a production calibration.

## Install a local copy

Python dependencies are listed in `pyproject.toml`; R dependencies are listed in
`DESCRIPTION`. With those dependencies and build tools installed locally:

```sh
python -m pip install --no-index --no-build-isolation ./creditratingsystem-python
R CMD INSTALL ./creditratingsystem-r
```

For a controlled environment, install into your chosen Python environment or R
library and retain the dependency versions alongside the fitted artifacts.
Installation does not require credentials, a database or an online service.
The current local verification baseline is Python 3.10.12 and R 4.1.2 on Linux.
The package metadata requires Python 3.10 or later and R 4.1.2 or later;
other interpreter versions and operating systems need their own package checks.

## Start with an explicit data contract

For example, a local `exposures.csv` can contain:

```csv
facility_id,ead
0001,100000
0002,25000
```

Preserve IDs as strings, declare the currency and keep missing amounts distinct
from zero. The following imports the same typed rows in either language:

```python
from math import isclose
from creditratingsystem import read_local_dataset, expected_loss

schema = {
    "version": "1.0",
    "columns": {
        "facility_id": {"type": "string"},
        "ead": {"type": "number", "unit": "EUR"},
    },
    "primary_key": ["facility_id"],
}
rows = read_local_dataset("exposures.csv", schema).value
# Illustrative supplied one-year PD and conditional LGD, not estimated here.
loss = expected_loss(0.02, 0.40, rows[0]["ead"])
assert isclose(loss.value, 800.0, rel_tol=1e-12)
```

```r
library(creditratingsystem)
schema <- list(
  version = "1.0",
  columns = list(
    facility_id = list(type = "string"),
    ead = list(type = "number", unit = "EUR")
  ),
  primary_key = list("facility_id")
)
rows <- read_local_dataset("exposures.csv", schema)$value
loss <- expected_loss(0.02, 0.40, rows[[1]]$ead)
stopifnot(isTRUE(all.equal(loss$value, 800, tolerance = 1e-12)))
```

Inspect `diagnostics`, `limitations`, `context` and `provenance` as well as
`value`. A successful calculation does not by itself validate the input model.
Use `build_calculation_context` to bind observation date, knowledge date,
currency, horizon, probability measure and conditioning where applicable.

## Develop and validate a model

1. Reconcile entity, household, application, account and episode identifiers.
   Use `join_as_of` to retain only releases known at each prediction date.
2. Build targets with the applicable performance origin and observation window.
   An unfunded application or incomplete follow-up has no observed good label.
3. Separate training, calibration and final evaluation, including connected
   entities and outcome windows. Fit preprocessing only on the training split.
4. Fit the selected estimator, apply a separately fitted calibrator, and retain
   their complete versioned artifacts. Declare physical versus risk-neutral
   probabilities and PIT versus TTC assumptions explicitly.
5. Evaluate discrimination, calibration, stability and the product's loss and
   exposure targets on appropriate held-out observations. Preserve unsupported
   or unidentifiable results instead of replacing them with numerical zeros.
6. Evaluate affordability and eligibility separately from PD. Bind a local
   policy decision to the exact model, calibration and information snapshot.
7. Monitor mature outcomes and replay historical decisions using the original
   artifacts. A later champion model must not replace a historical prediction.

## Choose an executable reference

The `examples` directory contains paired Python and native R workflows. Each
loads explicit synthetic contracts and checks independently specified outcomes.

| Need | Example stem |
|---|---|
| Fit, calibration, held-out validation and selection sensitivity | `retail_development`, `model_development` |
| Application, booking and subsequent monitoring | `retail_application_journey` |
| Joint mortgage borrowers, collateral and workout | `mortgage_household_review` |
| Cards, overdraft and home-equity lines | `revolving_lifecycle` |
| Corporate and specialized cashflow risk | `wholesale_corporate_lifecycle`, `specialized_asset_finance` |
| Receivables, securitization and covered bonds | `structured_journey` |
| Low-default bounds, including ordered rating grades | `low_default_bound` in the calibration reference |

Run a Python example from its source distribution with
`python examples/retail_application_journey.py`. For R, source the installed
example and call its named entry point:

```r
source(system.file("examples", "revolving_lifecycle.R",
                   package = "creditratingsystem"))
result <- run_revolving_lifecycle()
```

Tiny synthetic worlds are bundled for all six domains; larger generation is an
explicit task. Ordinary tests use small fits and analytical cases. Extended
statistical checks and stress generation are separately selected operations.
Method documentation specifies each input contract, model assumption and scope.
