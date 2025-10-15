import numpy as np
from txgraffiti2025.forms import with_floor, with_ceil, to_expr

def test_floor_and_ceil(df_basic):
    expr = to_expr("a") + 0.6
    f = with_floor(expr).eval(df_basic)
    c = with_ceil(expr).eval(df_basic)
    assert (f <= expr.eval(df_basic)).all()
    assert (c >= expr.eval(df_basic)).all()
    # integrality squeeze
    assert (c - f >= 0).all()
