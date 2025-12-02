import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import (
    Predicate, LE, GE, LT, GT, EQ, NE,
    InSet, Between, IsInteger, IsNaN, IsFinite,
    Where, RowWhere, GEQ, LEQ, GT0, LT0, EQ0, BETWEEN, IN, IS_INT, IS_NAN, IS_FINITE,
    to_expr,
)

# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def df_basic():
    # a,b,c: numeric; cat: categorical; n: float-y integers
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 1.0, 4.0],
        "c": [1.5, 2.5, 3.5, 4.5],
        "cat": ["x", "y", "x", "z"],
        "n": [1.0, 2.0, 3.0, 4.0],   # integer-valued floats
    })

@pytest.fixture
def df_special():
    # For NaN / +/-inf / alignment checks
    return pd.DataFrame({
        "x": [0.0, 1.0, np.nan, np.inf, -np.inf],
        "y": [0.0, 1.0, 1.0, 1.0, 1.0],
    })


# -----------------------
# Your (adjusted) tests
# -----------------------

def test_compare_ops(df_basic):
    assert GE("a", "b").mask(df_basic).equals(df_basic["a"] >= df_basic["b"])
    assert LE("a", 3.0).mask(df_basic).equals(df_basic["a"] <= 3.0)
    assert LT("a", "c").mask(df_basic).equals(df_basic["a"] < df_basic["c"])
    assert GT("b", 1).mask(df_basic).equals(df_basic["b"] > 1)
    assert EQ("a", to_expr("a")).mask(df_basic).all()
    assert NE("a", "b").mask(df_basic).equals(df_basic["a"] != df_basic["b"])

def test_eq_with_tolerance(df_basic):
    # Slight perturbation within tolerance should evaluate True
    eq = EQ(to_expr("a") + 1e-10, "a", tol=1e-9)
    assert eq.mask(df_basic).all()

def test_inset_and_between(df_basic):
    mask_in = InSet("cat", {"x", "z"}).mask(df_basic)
    assert set(df_basic.loc[mask_in, "cat"]) <= {"x", "z"}

    mask_b = Between("a", 2.0, 4.0).mask(df_basic)
    assert mask_b.to_numpy().tolist() == [False, True, True, True]

def test_numeric_checks(df_basic):
    assert IsFinite("a").mask(df_basic).all()
    assert IsNaN("a").mask(df_basic).sum() == 0
    # integer check against float column
    assert IsInteger(to_expr("n")).mask(df_basic).all()

def test_where_and_rowwhere(df_basic):
    m1 = Where(lambda df: (df["a"] + df["b"]) > 3).mask(df_basic)
    m2 = RowWhere(lambda row: (row["a"] + row["b"]) > 3).mask(df_basic)
    assert m1.equals(m2)

def test_shorthands(df_basic):
    assert GEQ("a", 1).mask(df_basic).all()
    # 'b' <= 2 is NOT true for the last row (b == 4), so don't assert .all()
    assert LEQ("b", 2).mask(df_basic).equals(df_basic["b"] <= 2)
    assert GT0(to_expr("a") - 0.5).mask(df_basic).sum() == 4
    assert LT0(0 - to_expr("a")).mask(df_basic).sum() == 4
    assert EQ0(to_expr("a") - "a").mask(df_basic).all()
    assert BETWEEN("a", 2, 4).mask(df_basic).sum() == 3
    assert IN("cat", ["x", "y"]).mask(df_basic).sum() == 3
    assert IS_INT("n").mask(df_basic).all()
    assert IS_NAN("cat").mask(df_basic).sum() == 0
    assert IS_FINITE("a").mask(df_basic).all()


# -----------------------
# Added edge-case tests
# -----------------------

def test_between_exclusive_and_inclusive(df_basic):
    # (2, 4) exclusive should only include strictly between
    ex_lo = Between("a", 2, 4, inclusive_low=False).mask(df_basic).tolist()
    ex_hi = Between("a", 2, 4, inclusive_high=False).mask(df_basic).tolist()
    assert ex_lo == [False, False, True, True]  # 2 excluded
    assert ex_hi == [False, True, True, False]  # 4 excluded

def test_inset_with_expression(df_basic):
    # IN over an expression (e.g., a+1)
    expr = to_expr("a") + 1
    mask = IN(expr, {2.0, 3.0, 5.0}).mask(df_basic).tolist()
    # a+1 = [2,3,4,5] -> membership at positions 0,1,3
    assert mask == [True, True, False, True]

def test_isfinite_and_isnan_on_special(df_special):
    m_fin = IsFinite("x").mask(df_special).tolist()
    m_nan = IsNaN("x").mask(df_special).tolist()
    # x = [0.0, 1.0, nan, inf, -inf]
    assert m_fin == [True, True, False, False, False]
    assert m_nan == [False, False, True, False, False]

def test_where_alignment(df_special):
    # Return ndarray of bools (not Series) to ensure alignment is enforced
    P = Where(lambda df: np.asarray(df["y"] == 1.0))
    m = P.mask(df_special)
    assert m.index.equals(df_special.index)
    # y == 1.0 is [False, True, True, True, True]
    assert m.tolist() == [False, True, True, True, True]

def test_rowwhere_alignment(df_special):
    # RowWhere always aligns to df.index
    R = RowWhere(lambda row: bool(row["y"] == 1.0))
    m = R.mask(df_special)
    assert m.index.equals(df_special.index)
    assert m.tolist() == [False, True, True, True, True]

def test_predicate_callable_alias_if_present(df_basic):
    # If Predicate implements __call__, ensure it mirrors .mask; otherwise, skip.
    P = Predicate.from_column("a", truthy_only=True)
    try:
        m_call = P(df_basic)
    except TypeError:
        pytest.skip("__call__ not implemented; mask-only predicates")
    else:
        assert isinstance(m_call, pd.Series)
        assert m_call.equals(P.mask(df_basic))

def test_compare_repr_smoke(df_basic):
    # Ensure Compare/derived preds have a stable, informative repr
    r1 = repr(GE("a", "b"))          # expect something like "('a' >= 'b')"
    r2 = repr(EQ("a", "b", tol=1e-9))  # your code shows "_approx" for approx eq
    r3 = repr(IN("cat", {"x", "z"}))
    assert isinstance(r1, str) and isinstance(r2, str) and isinstance(r3, str)
    assert (">=" in r1 or "=" in r1)
    # accept either explicit "==" or an approximation marker
    assert ("==" in r2) or ("approx" in r2.lower()) or ("≈" in r2)
