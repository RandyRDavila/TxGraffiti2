import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms.utils import (
    Expr, Const, ColumnTerm, LinearForm, BinOp, UnaryOp, LogOp,
    to_expr, floor, ceil, abs_, log, exp, sqrt,
)

def test_const_and_column_eval(df_basic):
    c = Const(3.14)
    s = c.eval(df_basic)
    assert (s == 3.14).all() and (s.index == df_basic.index).all()

    col = ColumnTerm("a")
    s2 = col.eval(df_basic)
    assert (s2.values == df_basic["a"].values).all()

def test_linearform_alignment_and_math(df_basic):
    lf = LinearForm(1.0, [(2.0, "a"), (-3.0, "b")])  # 1 + 2a - 3b
    out = lf.eval(df_basic)
    expected = 1.0 + 2.0*df_basic["a"] - 3.0*df_basic["b"]
    assert np.allclose(out.values, expected.values)
    assert (out.index == df_basic.index).all()

def test_expr_ops_plus_minus_mul_div_pow_mod_neg(df_basic):
    a = to_expr("a"); b = to_expr("b")
    s = ((a + 1) - (2*b)) * (a / b) ** 2 % 5
    out = s.eval(df_basic)
    assert len(out) == len(df_basic)
    assert (out.index == df_basic.index).all()

def test_unary_helpers_floor_ceil_abs_log_exp_sqrt(df_basic):
    a = to_expr("a") + 0.6
    f = floor(a).eval(df_basic)
    c = ceil(a).eval(df_basic)
    ab = abs_(-a).eval(df_basic)
    lg = log(to_expr("a")).eval(df_basic)
    ex = exp(to_expr("a")*0.0).eval(df_basic)  # exp(0)=1
    rt = sqrt(to_expr("c")).eval(df_basic)
    assert (f <= a.eval(df_basic)).all()
    assert (c >= a.eval(df_basic)).all()
    assert np.allclose(ab.values, (a.eval(df_basic) * 1.0).values)
    assert np.isfinite(lg).all()
    assert np.allclose(ex, 1.0)
    assert np.all(rt >= 0.0)


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def df_small():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 1.0, 4.0],
        "c": [1.5, 2.5, 3.5, 4.5],
    })


# -----------------------
# _as_series behavior is covered implicitly via eval() paths,
# but we verify a couple failure/success scenarios explicitly
# through public classes
# -----------------------

def test_column_term_eval_and_missing(df_small):
    x = ColumnTerm("a")
    y = x.eval(df_small)
    assert y.index.equals(df_small.index)
    assert np.allclose(y.to_numpy(), df_small["a"].to_numpy())

    with pytest.raises(KeyError):
        ColumnTerm("missing").eval(df_small)


def test_const_eval_fraction_and_repr():
    from fractions import Fraction
    c1 = Const(3)
    c2 = Const(3.0)
    c3 = Const(Fraction(3, 2))
    s = c1.eval(pd.DataFrame(index=[0,1]))
    assert np.allclose(s.to_numpy(), [3.0, 3.0])
    # repr should be stable-ish but we avoid exact string; just ensure rational shows a slash
    r = repr(c3)
    assert "/" in r


def test_to_expr_roundtrip_and_errors():
    assert isinstance(to_expr("x"), ColumnTerm)
    assert isinstance(to_expr(5), Const)
    e = to_expr(Const(1.5))
    assert isinstance(e, Const)
    with pytest.raises(TypeError):
        to_expr({"not": "supported"})


def test_linear_form_eval_and_repr(df_small):
    lf = LinearForm(1.0, [(2.0, "a"), (-1.0, "b")])  # 1 + 2a - b
    out = lf.eval(df_small)
    expected = 1.0 + 2.0 * df_small["a"] - df_small["b"]
    assert np.allclose(out.to_numpy(), expected.to_numpy())
    # Missing column should raise
    with pytest.raises(KeyError):
        LinearForm(0.0, [(1.0, "z")]).eval(df_small)
    # repr smoke
    s = repr(lf)
    assert "+" in s or "-" in s


def test_binop_eval_and_operator_overloads(df_small):
    # (a + 2) / (b - 1)
    expr = (to_expr("a") + 2) / (to_expr("b") - 1)
    out = expr.eval(df_small)
    expected = (df_small["a"] + 2) / (df_small["b"] - 1)
    assert np.allclose(out.to_numpy(), expected.to_numpy())

    # Ensure overloads produce Exprs (BinOp/UnaryOp)
    assert isinstance(to_expr("a") + 1, BinOp)
    assert isinstance(1 + to_expr("a"), BinOp)
    assert isinstance(-to_expr("a"), UnaryOp)


def test_unary_helpers_eval(df_small):
    fa = floor(to_expr("c")).eval(df_small)
    ce = ceil(to_expr("c")).eval(df_small)
    ab = abs_(to_expr("a") - 2.5).eval(df_small)
    ex = exp(0 * to_expr("a")).eval(df_small)
    rt = sqrt("a").eval(df_small)

    assert (fa <= df_small["c"]).all()
    assert (ce >= df_small["c"]).all()
    assert (ab >= 0).all()
    assert np.allclose(ex.to_numpy(), 1.0)
    assert (rt >= 0).all()


def test_log_eval_bases_and_repr(df_small):
    # natural log
    ln = log("a").eval(df_small)
    assert np.all(np.isfinite(ln.to_numpy()))

    # base 10
    lg10 = log("a", base=10).eval(df_small)
    ratio = (ln / lg10).replace([np.inf, -np.inf], np.nan)
    finite = ratio.notna()
    assert np.allclose(ratio[finite].to_numpy(), np.log(10.0), rtol=1e-6, atol=1e-6)

    # repr smoke for LogOp base formatting
    s_e = repr(log("a"))
    s_10 = repr(log("a", base=10))
    s_2 = repr(log("a", base=2))
    assert "log(" in s_e
    assert ("log10(" in s_10) or ("log_10(" in s_10)
    assert ("log2(" in s_2) or ("log_2(" in s_2)


def test_precedence_in_repr_parentheses_smoke():
    # We don't assert exact strings, just that parentheses appear
    # where needed when multiplying a sum.
    a, b, c = to_expr("a"), to_expr("b"), to_expr("c")
    expr = (a + b) * c
    s = repr(expr)
    assert "*" in s
    # parenthesized left term is expected because (a+b)*c
    assert "(" in s and ")" in s


def test_alignment_and_broadcasting_with_arrays():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
    # Broadcast scalar
    expr1 = to_expr("x") + 5
    out1 = expr1.eval(df)
    assert np.allclose(out1.to_numpy(), [15.0, 25.0, 35.0])

    # Provide a matching-length numpy array via Const + BinOp
    arr = np.array([1.0, 2.0, 3.0])
    expr2 = to_expr("x") - Const(arr)  # Const accepts ndarray indirectly through BinOp -> _as_series
    out2 = expr2.eval(df)
    assert np.allclose(out2.to_numpy(), [9.0, 18.0, 27.0])

    # Mismatched array length should error when it reaches _as_series via BinOp
    bad = BinOp(np.add, to_expr("x"), Const(np.array([1.0, 2.0])))
    with pytest.raises(ValueError):
        bad.eval(df)
