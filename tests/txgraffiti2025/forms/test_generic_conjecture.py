import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import GEQ, LEQ
from txgraffiti2025.forms.generic_conjecture import (
    Eq, Le, Ge, AllOf, AnyOf, Conjecture,
)

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def df_basic():
    # Matches the style used in your predicates tests
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 1.0, 4.0],
        "c": [1.5, 2.5, 3.5, 4.5],
    })


# -----------------------
# Your tests (lightly adjusted)
# -----------------------

def test_eq_le_ge(df_basic):
    assert Eq("a", "a").evaluate(df_basic).all()
    assert Le("a", "c").evaluate(df_basic).all()
    assert Ge("c", "a").evaluate(df_basic).all()

def test_slacks_sign_convention(df_basic):
    # Le: slack = right - left (>=0 when satisfied)
    le = Le("a", "c")
    s = le.slack(df_basic)
    assert (s >= 0).all()

    # Ge: slack = left - right (>=0 when satisfied)
    ge = Ge("c", "a")
    s2 = ge.slack(df_basic)
    assert (s2 >= 0).all()

    # Eq: slack = -abs(left - right)
    eq = Eq("a", "a")
    s3 = eq.slack(df_basic)
    assert (s3 == 0).all()

def test_allof_anyof(df_basic):
    r1 = Le("a", "c")
    r2 = Ge("b", 1.0)
    assert AllOf([r1, r2]).evaluate(df_basic).all()

    # AnyOf: at least one holds — with this data, Le(b, 2.0) is False on last row
    r3 = Ge("a", 10.0)      # always False
    r4 = Le("b", 2.0)       # [True, True, True, False]
    anymask = AnyOf([r3, r4]).evaluate(df_basic)
    assert anymask.tolist() == [True, True, True, False]  # not all True
    assert anymask.any()


# -----------------------
# Added coverage
# -----------------------

def test_le_ge_slack_values(df_basic):
    # Explicit values to pin down conventions
    # a vs c rows: (1,1.5), (2,2.5), (3,3.5), (4,4.5)
    le = Le("a", "c")
    ge = Ge("c", "a")
    assert le.slack(df_basic).tolist() == [0.5, 0.5, 0.5, 0.5]  # c - a
    assert ge.slack(df_basic).tolist() == [0.5, 0.5, 0.5, 0.5]  # c - a

def test_allof_anyof_slack_semantics(df_basic):
    # r1 slack: c - a = [0.5, 0.5, 0.5, 0.5]
    # r2 slack: b - 1  = [1.0, 1.0, 0.0, 3.0]
    r1 = Le("a", "c")
    r2 = Ge("b", 1.0)
    allof = AllOf([r1, r2])  # min of slacks
    anyof = AnyOf([r1, r2])  # max of slacks
    assert allof.slack(df_basic).tolist() == [0.5, 0.5, 0.0, 0.5]
    assert anyof.slack(df_basic).tolist() == [1.0, 1.0, 0.5, 3.0]

def test_allof_anyof_empty_parts_behavior(df_basic):
    # Your code returns zeros for empty lists
    assert AllOf([]).evaluate(df_basic).all()
    assert AnyOf([]).evaluate(df_basic).sum() == 0
    assert AllOf([]).slack(df_basic).tolist() == [0.0] * len(df_basic)
    assert AnyOf([]).slack(df_basic).tolist() == [0.0] * len(df_basic)

def test_conjecture_R_given_C(df_basic):
    C = GEQ("a", 2.0) & LEQ("a", 3.0)  # rows where a in {2,3}
    R = Le("a", "c")
    conj = Conjecture(R, C)
    applicable, holds, failures = conj.check(df_basic)
    assert applicable.sum() == 2
    assert holds.all()        # vacuous True for non-applicable rows
    assert failures.empty

def test_conjecture_vacuous_truth_on_non_applicable(df_basic):
    # Force a relation that fails on some rows, then restrict applicability
    R = Ge("a", "c")  # False everywhere in df_basic
    C = GEQ("a", 4.0)  # only last row applicable
    conj = Conjecture(R, C)
    applicable, holds, failures = conj.check(df_basic)
    # Only last row is applicable & it fails
    assert applicable.tolist() == [False, False, False, True]
    assert holds.tolist() == [True, True, True, False]  # vacuously True elsewhere
    assert list(failures.index) == [3]
    # __slack__ present and negative for failure (Ge slack = a - c)
    assert "__slack__" in failures.columns
    assert failures["__slack__"].iloc[0] < 0

def test_conjecture_auto_base_detection_true_column():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1, 2, 2],
        "connected": [True, True, True],
    })
    # No condition provided; auto_base picks (connected)
    conj = Conjecture(Le("a", "b"))
    applicable, holds, failures = conj.check(df, auto_base=True)
    assert applicable.tolist() == [True, True, True]
    assert holds.tolist() == [True, True, False]  # last row a<=b is False
    assert list(failures.index) == [2]
    assert "__slack__" in failures.columns
    assert failures["__slack__"].iloc[0] == (df.loc[2, "b"] - df.loc[2, "a"])  # Le slack = rhs - lhs

def test_conjecture_auto_base_detection_none_true_columns():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [2, 1, 3],
        "connected": [True, False, True],   # not all True
        "simple":    [True, True, False],   # not all True
    })
    # With no always-True boolean columns, auto_base should use TRUE (all rows applicable)
    conj = Conjecture(Le("a", "b"))
    applicable, holds, failures = conj.check(df, auto_base=True)
    assert applicable.tolist() == [True, True, True]
    assert holds.tolist() == [True, False, True]
    assert list(failures.index) == [1]
