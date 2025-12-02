import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import (
    product, product_n, power, ratio, reciprocal, square, cube, to_expr,
)
from txgraffiti2025.forms.generic_conjecture import Le, Ge, Eq

def test_product_and_ratio(df_basic):
    expr = ratio(product("a","b"), "b")  # (a*b)/b = a
    out = expr.eval(df_basic)
    assert np.allclose(out, df_basic["a"])

def test_product_n_and_powers(df_basic):
    expr = product_n(["a","b","a"])  # a*b*a = a^2 b
    out = expr.eval(df_basic)
    expected = (df_basic["a"]**2) * df_basic["b"]
    assert np.allclose(out, expected)

    p = power("a", 0.5).eval(df_basic)
    assert np.all(p >= 0)

def test_recip_square_cube(df_basic):
    r = reciprocal("a").eval(df_basic)
    assert np.allclose(r, 1.0/df_basic["a"])
    s2 = square("a").eval(df_basic)
    s3 = cube("a").eval(df_basic)
    assert np.allclose(s2, df_basic["a"]**2)
    assert np.allclose(s3, df_basic["a"]**3)

def test_product_n_requires_terms():
    with pytest.raises(ValueError):
        _ = product_n([])

def test_ratio_zero_denominator_behaviour_via_relation():
    # We don’t assert exact Inf/NaN values; instead, we test behavior via a relation.
    df = pd.DataFrame({"a": [2.0, 2.0], "b": [1.0, 0.0]})
    r = ratio("a", "b")                 # [2.0, inf] under NumPy/pandas semantics
    # Check a *bounded* inequality: r <= 1e6 should fail on the div-by-zero row
    rel = Le(r, 1e6)
    m = rel.evaluate(df).tolist()
    assert m == [True, False]

def test_relation_integration_with_nonlinear_exprs():
    # α·ω ≤ n and α / ω ≥ 1 (second only when α≥ω)
    df = pd.DataFrame({"alpha": [2, 3], "omega": [2, 1], "n": [5, 10]})
    rel_prod = Le(product("alpha", "omega"), "n")  # [4<=5, 3<=10] -> [T, T]
    rel_ratio_ge1 = Ge(ratio("alpha", "omega"), 1.0)  # [1>=1, 3>=1] -> [T, T]
    assert rel_prod.evaluate(df).all()
    assert rel_ratio_ge1.evaluate(df).all()

def test_power_fractional_and_integer_exponents(df_basic):
    # fractional power should be >= 0 for nonnegative bases
    p = power("a", 0.5).eval(df_basic)
    assert np.all(p >= 0)
    # integer powers equal square/cube helpers
    assert np.allclose(power("a", 2).eval(df_basic), square("a").eval(df_basic))
    assert np.allclose(power("a", 3).eval(df_basic), cube("a").eval(df_basic))

def test_expr_inputs_accept_mixed_types(df_basic):
    # Use to_expr composition inside nonlinear
    e = product(to_expr("a") + 1, to_expr("b") - 1)  # (a+1)*(b-1)
    out = e.eval(df_basic)
    expected = (df_basic["a"] + 1) * (df_basic["b"] - 1)
    assert np.allclose(out, expected)

def test_eq_on_nonlinear_combination(df_basic):
    # Check Eq slack sign convention on a nonlinear expression (should be <= 0)
    e = product("a", "b")
    rel = Eq(e, e, tol=0.0)
    m = rel.evaluate(df_basic)
    s = rel.slack(df_basic)
    assert m.all()
    assert np.all(s.to_numpy() == 0.0)
