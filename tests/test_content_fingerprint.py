"""Caller recipe binding semantics and fixed cross-language reference."""
import pytest
from creditratingsystem import content_fingerprint, CreditRiskError


def test_recipe_content_binding_is_order_independent_only_for_objects():
    digest='e2581cdf698074aae7c4881573e93dfc654f1fd8fb256e83a9f2b7966b345018'
    assert content_fingerprint({'a':1,'b':[2]}).value==digest
    assert content_fingerprint({'b':[2.0],'a':1.0}).value==digest
    assert content_fingerprint([1,2]).value!=content_fingerprint([2,1]).value
    assert content_fingerprint(True).value!=content_fingerprint(1).value
    assert content_fingerprint({}).value!=content_fingerprint([]).value
    assert content_fingerprint(-0.).value==content_fingerprint(0).value


def test_recipe_binding_rejects_nonportable_values():
    for value in (float('nan'),float('inf'),{'':1},{'non-ascii-ä':1},2**53):
        with pytest.raises(CreditRiskError):content_fingerprint(value)
