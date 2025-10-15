import numpy as np
from txgraffiti2025.forms import product, product_n, power, ratio, reciprocal, square, cube, to_expr

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
