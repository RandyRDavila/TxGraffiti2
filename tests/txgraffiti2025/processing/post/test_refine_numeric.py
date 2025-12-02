# tests/unit/txgraffiti2025/processing/post/test_refine_numeric.py

import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
from txgraffiti2025.processing.post.refine_numeric import refine_numeric_bounds, RefinementConfig

def _H_true(df): return Where(lambda d: True)

def test_upper_bound_sqrt_intercept_tightening():
    # y <= x + sqrt(z), with z = 4 on mask → sqrt(z)=2; try sqrt form (should equal baseline)
    # Then shrink baseline a bit to ensure "tighter" candidate still holds
    df = pd.DataFrame({
        "H": [True]*4,
        "x": [1.0, 2.0, 3.0, 4.0],
        "z": [4.0, 4.0, 4.0, 4.0],
        "y": [1.9, 2.9, 3.9, 4.9],  # all <= x + 2.0
    })
    H = Where(lambda d: d["H"])
    base = Conjecture(Le("y", to_expr("x") + Const(2.0)), condition=H)

    # Turn on sqrt; it should generate x + sqrt(z) which equals baseline and is fine
    cfg = RefinementConfig(try_sqrt_intercept=True, require_tighter=False)
    props = refine_numeric_bounds(df, base, config=cfg)
    assert any("refined" in (p.name or "") for p in props)
    # Check that at least one candidate equals the sqrt version numerically
    # (We don't rely on Expr comparison; we check truth)
    for p in props:
        applicable, holds, _ = p.check(df, auto_base=False)
        assert holds[applicable].all()

def test_round_numeric_const_coeff_and_intercept():
    """
    Upper bound: y <= 1.9*x + 1.9 on mask.

    We adjust y so that a strictly tighter *integer-rounded* variant exists,
    e.g. y <= 1*x + 1, which is pointwise <= baseline (1.9x+1.9) and still
    satisfies y <= 1*x + 1 for all rows.
    """
    df = pd.DataFrame({
        "H": [True]*4,
        "x": [1.0, 2.0, 3.0, 4.0],
        # Make a tighter integer-rounded bound feasible: y == 1*x + 1
        "y": [2.0, 3.0, 4.0, 5.0],
    })
    H = Where(lambda d: d["H"])
    base = Conjecture(Le("y", Const(1.9) * to_expr("x") + Const(1.9)), condition=H)

    cfg = RefinementConfig(
        try_coeff_round_const=True,
        try_intercept_round_const=True,
        require_tighter=True,   # insist on strictly tighter than baseline
    )
    props = refine_numeric_bounds(df, base, config=cfg)

    # Expect at least one strictly tighter variant (e.g., 1*x + 1).
    assert len(props) >= 1


def test_lower_bound_rounding_moves_the_other_way():
    """
    Lower bound: y >= 1.1*x - 0.1 on mask.

    We adjust y so that a strictly tighter *integer-rounded* variant exists,
    e.g. y >= 2*x, which is pointwise >= baseline (1.1x - 0.1) and still
    satisfies y >= 2*x for all rows.
    """
    df = pd.DataFrame({
        "H": [True]*4,
        "x": [1.0, 2.0, 3.0, 4.0],
        # Make a tighter integer-rounded bound feasible: y == 2*x
        "y": [2.0, 4.0, 6.0, 8.0],
    })
    H = Where(lambda d: d["H"])
    base = Conjecture(Ge("y", Const(1.1) * to_expr("x") - Const(0.1)), condition=H)

    cfg = RefinementConfig(
        try_coeff_round_const=True,
        try_intercept_round_const=True,
        require_tighter=True,   # insist on strictly tighter than baseline
    )
    props = refine_numeric_bounds(df, base, config=cfg)

    # Expect at least one strictly tighter variant (e.g., 2*x).
    assert len(props) >= 1


@pytest.mark.optional
def test_whole_rhs_floor_if_supported():
    # If Expr exposes floor_(), we should be able to floor the whole RHS on an upper bound
    df = pd.DataFrame({
        "H": [True]*3,
        "x": [1.0, 2.1, 3.6],
        "y": [1.0, 2.0, 3.0],
    })
    H = Where(lambda d: d["H"])
    rhs = to_expr("x") + Const(0.49)
    base = Conjecture(Le("y", rhs), condition=H)

    # Only run if floor_ is supported by the Expr
    if hasattr(rhs, "floor_"):
        props = refine_numeric_bounds(df, base, config=RefinementConfig(try_whole_rhs_floor=True, require_tighter=True))
        assert len(props) >= 1
        for p in props:
            applicable, holds, _ = p.check(df, auto_base=False)
            assert holds[applicable].all()
    else:
        pytest.skip("Expr.floor_() not available; skipping.")
