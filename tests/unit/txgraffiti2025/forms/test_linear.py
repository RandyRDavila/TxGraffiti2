import numpy as np
from txgraffiti2025.forms import linear_expr, linear_le, linear_ge, linear_eq

def test_linear_expr_and_builders(df_basic):
    # 1 + 2a - 3b <= c
    rel = linear_le(1.0, [(2.0, "a"), (-3.0, "b")], "c")
    mask = rel.evaluate(df_basic)
    assert mask.all()

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
