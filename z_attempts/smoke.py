import pandas as pd
import numpy as np
from txgraffiti2025.forms.utils import to_expr, abs_, log, floor, ColumnTerm, Const
from txgraffiti2025.forms.predicates import EQ0, GE, LE, InSet, Between, Predicate

def test__as_series_and_columnterm_numeric():
    df = pd.DataFrame({"x":[1,2,3], "b":[True, False, True]})
    # arithmetic: should coerce to float Series
    e = to_expr("x") + 1
    s = e.eval(df)
    assert s.dtype.kind == "f"
    # boolean column preserved when used directly (predicates)
    assert df["b"].dtype == bool

test__as_series_and_columnterm_numeric()

def test_EQ0_exact_and_tolerant():
    df = pd.DataFrame({"x":[0.0, 1e-10, 1e-6, 1.0]})
    m0 = EQ0("x").mask(df)                # exact zero
    assert m0.tolist() == [True, False, False, False]
    m1 = EQ0("x", tol=1e-9).mask(df)      # tolerant
    assert m1.tolist() == [True, True, False, False]

test_EQ0_exact_and_tolerant()

def test_between_and_inset_and_compare():
    df = pd.DataFrame({"k":[1,2,3,4], "x":[0,1,2,3]})
    assert Between("x", 1, 2).mask(df).tolist() == [False, True, True, False]
    assert InSet("k", {2,4}).mask(df).tolist() == [False, True, False, True]
    assert GE("x", 1).mask(df).tolist() == [False, True, True, True]
    assert LE("x", 2).mask(df).tolist() == [True, True, True, False]

test_between_and_inset_and_compare()

def test_logop_base_validation():
    df = pd.DataFrame({"y":[1, 10, 100]})
    assert np.allclose(log("y", base=10).eval(df).values, [0,1,2])

test_logop_base_validation()

import pandas as pd
from txgraffiti2025.processing.pre.hypotheses import list_boolean_columns, detect_base_hypothesis, enumerate_boolean_hypotheses
from txgraffiti2025.forms.predicates import Predicate

def test_nullable_boolean_and_ints():
    df = pd.DataFrame({
        "connected": pd.Series([True, True, None, True], dtype="boolean"),
        "bipartite": pd.Series([1, 0, 1, None], dtype="Int64"),
        "x": [1,2,3,4]
    })
    cols = list_boolean_columns(df)
    assert set(cols) == {"connected", "bipartite"}
    H0 = detect_base_hypothesis(df)  # not all True, so TRUE
    assert isinstance(H0, Predicate)
    Hs = enumerate_boolean_hypotheses(df)
    for H in Hs:
        m = H.mask(df)
        assert m.index.equals(df.index) and m.dtype == bool

test_nullable_boolean_and_ints()

def test_pairs_strict_subset_and_names():
    df = pd.DataFrame({
        "A": [1,1,0,0],         # int binary
        "B": [1,0,1,0],
        "C": [1,1,1,1],
    })
    Hs = enumerate_boolean_hypotheses(df, treat_binary_ints=True)
    # Expect singles for A,B (C is base if include_base True, else absorbed)
    names = [getattr(h, "name", repr(h)) for h in Hs]
    assert any("((A) ∧ (B))" in n or "((B) ∧ (A))" in n for n in names)

test_pairs_strict_subset_and_names()

import pandas as pd
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses
from txgraffiti2025.processing.pre.simplify_hypotheses import simplify_and_dedup_hypotheses

def test_simplify_handles_nullable_and_ints():
    df = pd.DataFrame({
        "A": pd.Series([1,1,0,None], dtype="Int64"),
        "B": pd.Series([True, None, True, False], dtype="boolean"),
        "x": [0,1,2,3],
    })
    Hs = enumerate_boolean_hypotheses(df, treat_binary_ints=True)
    kept, eqs = simplify_and_dedup_hypotheses(df, Hs, min_support=1, treat_binary_ints=True)
    for H in kept:
        m = H.mask(df)
        assert m.index.equals(df.index) and m.dtype == bool
    # equivalences list should not duplicate symmetric pairs
    seen = set()
    for e in eqs:
        k = tuple(sorted([e.A.name, e.B.name]))
        assert k not in seen
        seen.add(k)

test_simplify_handles_nullable_and_ints()
