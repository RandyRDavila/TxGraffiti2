
import pytest
from txgraffiti2025.forms import SubstructurePredicate
from txgraffiti2025.forms.predicates import Predicate

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

def test_substructure_negation_and_alignment(df_objects):
    # has_even defined in your df_objects fixture’s objects
    pred = SubstructurePredicate(lambda obj: getattr(obj, "has_even", lambda: False)(), object_col="object")
    m = pred.mask(df_objects)
    # Negation flips exactly
    m_not = (~pred).mask(df_objects)
    assert m.index.equals(df_objects.index)
    assert (m ^ m_not).all()  # XOR should be all True
    assert ((~m) == m_not).all()

def test_substructure_on_error_raise(df_objects):
    bad = SubstructurePredicate(lambda obj: 1/0, object_col="object", on_error="raise")
    with pytest.raises(ZeroDivisionError):
        bad.mask(df_objects)

def test_substructure_repr_smoke():
    p = SubstructurePredicate(lambda obj: True, object_col="graph", on_error="false")
    s = repr(p)
    assert "SubstructurePredicate" in s and "object_col='graph'" in s and "on_error='false'" in s
