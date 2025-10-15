import pytest
import pandas as pd
import numpy as np

from txgraffiti2025.generators.ratios import ratios
from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.forms.generic_conjecture import Ge, Le

@pytest.fixture
def df_example():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 8, 16],
        "c": [1, 1, 2, 2],
        "connected": [True, True, False, True],
    })

def connected_pred():
    return Where(lambda df: df["connected"].astype(bool))

def test_ratios_basic_bounds(df_example):
    hy = connected_pred()
    conjs = list(ratios(df_example, features=["b"], target="a", hypothesis=hy))
    assert len(conjs) == 2
    for conj in conjs:
        assert conj.condition is hy
        assert isinstance(conj.relation, (Ge, Le))

def test_ratios_skips_zero_divisions():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 0, 0], "connected": [True, True, True]})
    hy = Where(lambda df: df["connected"])
    result = list(ratios(df, features=["y"], target="x", hypothesis=hy))
    assert result == []  # division by zero everywhere => skip

def test_ratios_directionality(df_example):
    hy = connected_pred()
    both = list(ratios(df_example, features=["b"], target="a", hypothesis=hy, direction="both"))
    lower = list(ratios(df_example, features=["b"], target="a", hypothesis=hy, direction="lower"))
    upper = list(ratios(df_example, features=["b"], target="a", hypothesis=hy, direction="upper"))
    assert len(both) == 2
    assert len(lower) == 1 and isinstance(lower[0].relation, Ge)
    assert len(upper) == 1 and isinstance(upper[0].relation, Le)

def test_ratios_handles_missing_column(df_example):
    hy = connected_pred()
    res = list(ratios(df_example, features=["nonexistent"], target="a", hypothesis=hy))
    assert res == []  # gracefully skip missing feature

def test_ratios_empty_hypothesis(df_example):
    df = df_example.copy()
    df["connected"] = False
    hy = connected_pred()
    res = list(ratios(df, features=["b"], target="a", hypothesis=hy))
    assert res == []

def test_ratios_value_shape(df_example):
    hy = connected_pred()
    res = list(ratios(df_example, features=["b"], target="a", hypothesis=hy))
    assert any("ratio_lower" in c.name for c in res)
    assert any("ratio_upper" in c.name for c in res)
    # relation.left is the target expression; ensure it evaluates properly
    for conj in res:
        out = conj.relation.evaluate(df_example)
        assert out.index.equals(df_example.index)
