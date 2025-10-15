import pytest
from txgraffiti2025.forms import SubstructurePredicate

def test_substructure_basic(df_objects):
    pred = SubstructurePredicate(lambda obj: getattr(obj, "has_even", lambda: False)(), object_col="object")
    m = pred.mask(df_objects)
    # Only even-valued objects True; None -> False
    assert m.to_list() == [False, True, False, True, False]

def test_substructure_missing_col_raises(df_basic):
    pred = SubstructurePredicate(lambda obj: True, object_col="missing")
    with pytest.raises(KeyError):
        pred.mask(df_basic)

def test_substructure_on_error_false(df_objects):
    pred = SubstructurePredicate(lambda obj: 1/0, object_col="object", on_error="false")
    m = pred.mask(df_objects)
    assert m.sum() == 0
