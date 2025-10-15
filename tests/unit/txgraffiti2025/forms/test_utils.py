import numpy as np
import pandas as pd
from txgraffiti2025.forms import (
    Expr, Const, ColumnTerm, LinearForm, to_expr,
    floor, ceil, abs_, log, exp, sqrt
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
