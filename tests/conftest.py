
# Expensive acceptance checks are explicit and never selected by host identity.
import os
import pytest

def pytest_collection_modifyitems(config, items):
    if os.environ.get("CREDITRATINGSYSTEM_EXTENDED_TESTS", "").lower() == "true":
        return
    marker = pytest.mark.skip(reason="Extended acceptance test: set CREDITRATINGSYSTEM_EXTENDED_TESTS=true")
    for item in items:
        if "extended" in item.keywords:
            item.add_marker(marker)
