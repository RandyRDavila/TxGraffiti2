import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import linear_expr, linear_le, linear_ge, linear_eq

@pytest.fixture
def df_basic():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 1.0, 4.0],
        "c": [1.5, 2.5, 3.5, 4.5],
        "d": [0.0, 0.5, 1.0, 1.5],
    })


def test_linear_expr_and_builders(df_basic):
    # 1 + 2a - 3b <= c
    rel = linear_le(1.0, [(2.0, "a"), (-3.0, "b")], "c")
    mask = rel.evaluate(df_basic)
    # 1 + 2a - 3b <= c -> [True, True, False, True] on this df
    assert mask.tolist() == [True, True, False, True]

    # mapping form (no duplicates in dicts at runtime) -> choose a trivially true bound
    # 2a >= a  holds for nonnegative a
    rel2 = linear_ge(0.0, {"a": 2.0}, "a")
    assert rel2.evaluate(df_basic).all()

    # sequence form with duplicate columns merged: (a + a) >= a  -> always true for any a
    rel3 = linear_ge(0.0, [(1.0, "a"), (1.0, "a")], "a")
    assert rel3.evaluate(df_basic).all()

    # equality within tolerance
    rel4 = linear_eq(0.0, [(1.0, "a")], "a", tol=1e-9)
    assert rel4.evaluate(df_basic).all()


def test_le_ge_slack_values(df_basic):
    # Build LHS: L = 0.5 + 1*a - 0.5*b
    L = linear_expr(0.5, [(1.0, "a"), (-0.5, "b")])
    # Le: L <= c → slack = c - L
    r_le = linear_le(0.5, [(1.0, "a"), (-0.5, "b")], "c")
    # Ge: L >= d → slack = L - d
    r_ge = linear_ge(0.5, [(1.0, "a"), (-0.5, "b")], "d")

    # Explicit slacks to pin down convention
    L_vals = L.eval(df_basic).to_numpy()
    c_vals = df_basic["c"].to_numpy()
    d_vals = df_basic["d"].to_numpy()

    np.testing.assert_allclose(r_le.slack(df_basic).to_numpy(), c_vals - L_vals, rtol=0, atol=1e-12)
    np.testing.assert_allclose(r_ge.slack(df_basic).to_numpy(), L_vals - d_vals, rtol=0, atol=1e-12)

    # Both relations should hold on this data
    assert r_le.evaluate(df_basic).all()
    assert r_ge.evaluate(df_basic).all()


def test_right_as_scalar_and_column(df_basic):
    # L = a
    # a <= 10 (scalar RHS) -> always True
    r1 = linear_le(0.0, [(1.0, "a")], 10.0)
    assert r1.evaluate(df_basic).all()

    # a >= c - 2 → put RHS as expression by passing column name and intercept in LHS
    # Equivalent to: a + 2 >= c
    r2 = linear_ge(2.0, [(1.0, "a")], "c")
    assert r2.evaluate(df_basic).all()


def test_zero_coefficient_terms_and_merging(df_basic):
    # Terms include zeros and duplicates; zeros should drop, duplicates should merge
    r = linear_ge(0.0, [(0.0, "a"), (1.0, "a"), (-1.0, "a"), (2.0, "b"), (-2.0, "b")], 0.0)
    # After normalization, LHS collapses to 0.0; so 0 >= 0 holds
    assert r.evaluate(df_basic).all()


def test_linear_eq_slack_sign(df_basic):
    # Eq slack = -abs(LHS - RHS) (0 best, negative otherwise)
    r = linear_eq(0.0, [(1.0, "a")], "c", tol=0.0)  # equality without tolerance
    slack = r.slack(df_basic).to_numpy()
    # For df_basic, a != c, so slack should be strictly negative
    assert np.all(slack <= 0)
    assert np.any(slack < 0)
