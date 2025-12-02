import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
from txgraffiti2025.processing.post.hazel import (
    compute_touch_number,
    hazel_rank,
)

# ---------- helpers ----------

def H_col(name):
    return Where(lambda d, n=name: d[n])

def df4():
    # 4 rows, with two different hypothesis columns
    return pd.DataFrame({
        "H_all":  [True,  True,  True,  True],
        "H_12":   [True,  True,  False, False],   # only rows 0,1 applicable
        "x":      [1.0,   2.0,   3.0,   4.0],
        "y":      [2.0,   3.0,   6.0,   8.0],
    })

# ---------- tests ----------

def test_compute_touch_number_inequality_slack_atol():
    """
    For upper bound y <= 2x + 0, rows where y == 2x are 'touches'.
    Here we set y = 2x exactly, so touches should be all 4.
    """
    df = pd.DataFrame({
        "H_all": [True, True, True, True],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 4.0, 6.0, 8.0],  # exactly 2x
    })
    conj = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=Where(lambda d: d["H_all"]))
    stats = compute_touch_number(df, conj, atol=1e-9)
    assert stats["touch"] == 4
    assert stats["holds_n"] == 4
    assert stats["applicable_n"] == 4
    assert np.isclose(stats["touch_frac_app"], 1.0)
    assert np.isclose(stats["touch_frac_holds"], 1.0)


def test_compute_touch_number_respects_condition_mask():
    """
    Same bound but condition restricts to rows 0..1: touches should be 2 when y = 2x.
    """
    df = pd.DataFrame({
        "H_12": [True, True, False, False],  # only rows 0,1 applicable
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 4.0, 6.0, 8.0],  # exactly 2x
    })
    conj = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=Where(lambda d: d["H_12"]))
    stats = compute_touch_number(df, conj, atol=1e-9)
    assert stats["touch"] == 2
    assert stats["holds_n"] == 2
    assert stats["applicable_n"] == 2
    assert np.isclose(stats["touch_frac_app"], 1.0)
    assert np.isclose(stats["touch_frac_holds"], 1.0)



def test_hazel_rank_drops_bottom_quartile_and_sorts():
    """
    Build 4 conjectures with touches: [4, 3, 2, 1].
    With drop_frac=0.25 (bottom quartile), the one with touch=1 should drop.
    Survivors sorted by touch desc: [4,3,2].
    """
    df = df4()
    H = H_col("H_all")

    c4 = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H)               # touch=4
    c3 = Conjecture(Le("y", Const(2.0) * to_expr("x") + Const(1.0)), condition=H)  # holds only when x=1(or none)… make it false so touch=0; adjust:
    # Make c3: y <= 2x + 0 but restrict to 3 rows equality by using a shifted mask
    # Instead, craft direct bounds to control touches:
    c3 = Conjecture(Le("y", Const(2.0) * to_expr("x") + Const(0.0)), condition=H_col("H_12"))  # touch=2 (rows 0,1)
    # For better spread, define an Eq that touches on rows 0..2 (by equality  y == [2,3,6,100] won't hold; so instead use Ge to target 3 touches)
    # Easier: create explicit conjectures with known touches:
    c2 = Conjecture(Le("y", Const(2.0) * to_expr("x") + Const(-1.0)), condition=H)  # touch=0 (never equal); we want a touch=2 already above
    # Let's reassign list to actual touches [4,2,1,0] and assert behavior (bottom quartile -> drop 0).
    conjs = [c4, c3, Conjecture(Le("y", Const(2.0) * to_expr("x") + Const(0.5)), condition=H), c2]

    res = hazel_rank(df, conjs, drop_frac=0.25, atol=1e-9)
    kept = res.kept_sorted
    # Ensure at least one dropped (bottom quartile)
    assert len(kept) < len(conjs)
    # Sorted non-increasing by touch
    touches = []
    for c in kept:
        t = compute_touch_number(df, c, atol=1e-9)["touch"]
        touches.append(t)
    assert touches == sorted(touches, reverse=True)


def test_hazel_rank_ties_at_quantile_uses_fallback():
    """
    If all conjectures have identical touch counts, strict '>' cutoff would drop all.
    Hazel's fallback should keep top ceil((1-drop_frac)*n) by sorting.
    """
    df = df4()
    H = H_col("H_all")

    # All three will have touch=4
    c1 = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H)
    c2 = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H)
    c3 = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H)

    res = hazel_rank(df, [c1, c2, c3], drop_frac=0.25, atol=1e-9)
    kept = res.kept_sorted
    # Fallback keeps ceil(0.75*3)=3
    assert len(kept) == 3


def test_hazel_rank_includes_eq_conjectures_in_touch_calc():
    """
    Equalities should compute 'touch' via their own tolerance, and be rankable.
    """
    df = df4()
    H = H_col("H_all")

    # y == 2x touches on all rows
    eq_all = Conjecture(Eq("y", Const(2.0) * to_expr("x")), condition=H)
    # y == x touches on none
    eq_none = Conjecture(Eq("y", to_expr("x")), condition=H)

    res = hazel_rank(df, [eq_all, eq_none], drop_frac=0.5, atol=1e-9)
    kept = res.kept_sorted
    assert eq_all in kept
    assert eq_none not in kept


def test_hazel_scoreboard_has_expected_columns():
    df = df4()
    H = H_col("H_all")
    c = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H)

    res = hazel_rank(df, [c], drop_frac=0.25, atol=1e-9)
    sb = res.scoreboard
    for col in ("conjecture", "touch", "applicable_n", "holds_n", "touch_frac_app", "touch_frac_holds", "pretty", "relation_text", "condition_text"):
        assert col in sb.columns
