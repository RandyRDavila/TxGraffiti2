import pytest
import pandas as pd
import numpy as np

from txgraffiti2025.forms import Conjecture, Le, Ge, Eq, to_expr
from txgraffiti2025.processing.post.hazel import hazel_filter

@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.1, 3.05, 4.5],
        "c": [0.9, 2.0, 3.0, 3.9],
    })

def test_hazel_drops_bottom_quartile(df):
    # create inequalities with varying tightness
    c1 = Conjecture(Le("a", to_expr("b")), None, name="tight_many")   # a==b once, close others
    c2 = Conjecture(Le("a", to_expr("b") + 1.0), None, name="loose")  # never tight
    c3 = Conjecture(Le("a", to_expr("b") - 0.1), None, name="tight_few") # maybe tight couple times
    c4 = Conjecture(Le("a", to_expr("b") - 0.5), None, name="very_loose")

    kept = hazel_filter([c1, c2, c3, c4], df, eps=0.11)
    names = [c.name for c in kept]

    # should drop the bottom 25% (1 of 4), keeping 3
    assert len(kept) == 3
    assert "loose" not in names

def test_hazel_sorts_by_touch(df):
    c1 = Conjecture(Le("a", to_expr("b")), None, name="almost_equal")
    c2 = Conjecture(Le("a", to_expr("b") + 2.0), None, name="always_true")  # 0 touches
    out = hazel_filter([c1, c2], df)
    # ensure ordering descending by touch count
    assert out[0].name == "almost_equal"

def test_hazel_ignores_equalities(df):
    c1 = Conjecture(Eq("a","b"), None, name="equality")
    out = hazel_filter([c1], df)
    # Equalities have zero touch; filter may discard or keep if all zeros
    # For now, should return empty list
    assert out == []

def test_hazel_handles_empty_and_exceptions(df):
    assert hazel_filter([], df) == []
    # Broken conjecture (no eval)
    class Bad:
        pass
    bad_conj = Conjecture(Le("a","b"), None, name="bad")
    bad_conj.relation.left.eval = lambda df: 1/0
    out = hazel_filter([bad_conj], df)
    assert out == []
