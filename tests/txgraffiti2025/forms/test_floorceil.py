import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import with_floor, with_ceil, to_expr
from txgraffiti2025.forms.generic_conjecture import Le, Ge, Eq


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def df_basic():
    # Keep consistent with earlier tests: a,b,c numeric
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 1.0, 4.0],
        "c": [1.5, 2.5, 3.5, 4.5],
    })


# -----------------------
# Your original test, kept
# -----------------------

def test_floor_and_ceil(df_basic):
    expr = to_expr("a") + 0.6
    f = with_floor(expr).eval(df_basic)
    c = with_ceil(expr).eval(df_basic)
    assert (f <= expr.eval(df_basic)).all()
    assert (c >= expr.eval(df_basic)).all()
    # integrality squeeze
    assert (c - f >= 0).all()


# -----------------------
# Additional coverage
# -----------------------

def test_exact_behaviour_on_edge_values():
    # Explicit negatives / zeros / integers / fractions
    df = pd.DataFrame({"x": [-1.1, -1.0, -0.9, 0.0, 0.9, 1.0, 1.1]})
    f = with_floor("x").eval(df).tolist()
    c = with_ceil("x").eval(df).tolist()
    assert f == [-2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0]
    assert c == [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 2.0]

def test_idempotence(df_basic):
    # floor(floor(x)) == floor(x);  ceil(ceil(x)) == ceil(x)
    expr = to_expr("c")
    f = with_floor(expr)
    c = with_ceil(expr)
    ff = with_floor(f).eval(df_basic)
    cc = with_ceil(c).eval(df_basic)
    assert np.allclose(ff, f.eval(df_basic))
    assert np.allclose(cc, c.eval(df_basic))

def test_integer_invariance(df_basic):
    # For integer-valued floats, floor/ceil return the same number
    a = to_expr("a")
    f = with_floor(a).eval(df_basic)
    c = with_ceil(a).eval(df_basic)
    assert np.allclose(f, df_basic["a"])
    assert np.allclose(c, df_basic["a"])

def test_monotonicity(df_basic):
    # If x <= y elementwise, then floor(x) <= floor(y) and ceil(x) <= ceil(y)
    x = to_expr("c") - 0.25     # [1.25, 2.25, 3.25, 4.25]
    y = to_expr("c") + 0.25     # [1.75, 2.75, 3.75, 4.75]
    fx = with_floor(x).eval(df_basic)
    fy = with_floor(y).eval(df_basic)
    cx = with_ceil(x).eval(df_basic)
    cy = with_ceil(y).eval(df_basic)
    assert (fx <= fy).all()
    assert (cx <= cy).all()

def test_relations_with_slack_signs(df_basic):
    # Le: floor(expr) <= expr -> slack = expr - floor(expr) >= 0
    expr = to_expr("c") + 0.1
    r_le = Le(with_floor(expr), expr)
    s_le = r_le.slack(df_basic)
    assert (r_le.evaluate(df_basic)).all()
    assert (s_le >= 0).all()

    # Ge: ceil(expr) >= expr -> slack = ceil(expr) - expr >= 0
    r_ge = Ge(with_ceil(expr), expr)
    s_ge = r_ge.slack(df_basic)
    assert (r_ge.evaluate(df_basic)).all()
    assert (s_ge >= 0).all()

def test_equality_only_when_already_integer(df_basic):
    # Eq(floor(x), x) holds exactly when x is already integer-valued
    x = to_expr("c")  # .5 values: not integers
    r = Eq(with_floor(x), x, tol=0.0)
    m = r.evaluate(df_basic)
    assert m.sum() == 0  # none are equal

    # But for integer-valued floats (column "a"), equality holds
    r2 = Eq(with_floor("a"), "a", tol=0.0)
    assert r2.evaluate(df_basic).all()

def test_composition_inside_linear_like_forms(df_basic):
    # floor(a + 0.6) + ceil(b - 0.1) is evaluable and stable
    e = with_floor(to_expr("a") + 0.6) + with_ceil(to_expr("b") - 0.1)
    out = e.eval(df_basic).to_numpy()
    # sanity: all outputs are integers (stored as floats)
    assert np.allclose(out, np.round(out))
