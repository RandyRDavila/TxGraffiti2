import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.predicates import Where
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
from txgraffiti2025.processing.post.dalmatian import (
    dalmatian_filter,
    _target_and_direction,
    _is_significant,
)

from txgraffiti2025.forms.pretty import format_conjecture


# ---------- helpers ----------
def H_col(name):
    return Where(lambda d, n=name: d[n])

def df4():
    # 4 rows, with two different hypothesis columns
    return pd.DataFrame({
        "H_all":  [True,  True,  True,  True],
        "H_12":   [True,  True,  False, False],   # only rows 0,1 applicable
        "x":      [1.0,   2.0,   3.0,   4.0],
        "y":      [2.0,   3.0,   6.0,   8.0],     # chosen so truth tests can pass with reasonable bounds
    })

# ---------- tests ----------

def test_target_and_direction_from_string_left():
    c1 = Conjecture(Le("y", to_expr("x") + Const(1.0)), condition=H_col("H_all"))
    c2 = Conjecture(Ge("y", Const(2.0) * to_expr("x")), condition=H_col("H_all"))

    assert _target_and_direction(c1) == ("y", "le")
    assert _target_and_direction(c2) == ("y", "ge")


def test_significance_respects_applicable_mask_upper_bound():
    """
    New conjecture applies only to rows 0..1 (H_12).
    It 'improves' previous bounds only on rows 2..3 (outside mask).
    => Should NOT be significant.
    """
    df = df4()

    # previous accepted upper bound: y <= 2*x (true on all rows)
    prev = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H_col("H_all"))

    # new candidate: y <= 1.5*x is smaller than 2*x on rows 2..3,
    # but we make it *not* smaller on rows 0..1 so that there is no improvement on its own mask.
    # Make candidate apply only to rows 0..1
    cand = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H_col("H_12"))
    # Note: RHS equal to previous on rows 0..1 → no strict improvement on applicable rows

    assert _is_significant(cand, df, [prev]) is False


def test_significance_strict_somewhere_on_mask_upper_bound():
    """
    Candidate is equal to previous everywhere on its mask except one row, where it's strictly smaller.
    => Should be significant.
    """
    df = df4()

    # prev: y <= 2*x (applies everywhere)
    prev = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H_col("H_all"))

    # cand applies to rows 0..1 and is tighter there by making RHS = 2*x - 0.1
    # (strictly smaller on mask)
    rhs_cand = Const(2.0) * to_expr("x") - Const(0.1)
    cand = Conjecture(Le("y", rhs_cand), condition=H_col("H_12"))

    assert _is_significant(cand, df, [prev]) is True


def test_significance_nan_safe_when_peers_nan():
    """
    If peers have no finite RHS on the applicable mask, the code treats the new conjecture as significant
    to avoid blocking progress.
    """
    df = df4()

    # prev: craft an RHS that becomes NaN on applicable rows (divide by zero via 0*x / 0)
    nan_rhs = (Const(0.0) * to_expr("x")) / Const(0.0)
    prev = Conjecture(Le("y", nan_rhs), condition=H_col("H_12"))

    # cand: a normal finite RHS on same mask
    cand = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H_col("H_12"))

    assert _is_significant(cand, df, [prev]) is True


def test_dalmatian_filter_rejects_untrue_conjecture():
    """
    A conjecture that fails its truth test on any applicable row should be rejected.
    """
    df = df4()

    # This bound is *too small* to be true: y <= x  (fails on several rows)
    bad = Conjecture(Le("y", to_expr("x")), condition=H_col("H_all"))

    # A good one to keep: y <= 2*x + 1
    good = Conjecture(Le("y", Const(2.0) * to_expr("x") + Const(1.0)), condition=H_col("H_all"))

    kept = dalmatian_filter([bad, good], df)
    assert good in kept
    assert bad not in kept


def test_dalmatian_filter_dedups_by_hash():
    df = df4()

    c1 = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H_col("H_all"))
    c2 = Conjecture(Le("y", Const(2.0) * to_expr("x")), condition=H_col("H_all"))  # identical

    kept = dalmatian_filter([c1, c2], df)

    assert len(kept) == 1
    # Compare canonicalized text (relation + condition), not Expr.pretty()
    assert format_conjecture(kept[0], show_condition=True) == format_conjecture(c1, show_condition=True)
