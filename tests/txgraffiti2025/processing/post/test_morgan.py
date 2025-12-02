# tests/unit/txgraffiti2025/processing/post/test_morgan.py

import pandas as pd
import numpy as np
import pytest

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Where, Predicate
from txgraffiti2025.processing.post.morgan import morgan_filter


@pytest.fixture
def df_masked():
    # Boolean columns to craft subset/incomparable masks
    return pd.DataFrame({
        "connected": [True, True, True, True, True, True],
        "A": [True, True, False, False, True, False],   # ~ 3 trues
        "B": [True, False, True, False, False, True],   # overlaps with A, neither subset
        "alltrue": [True]*6,
        "x": [1, 2, 3, 4, 5, 6],
        "y": [1, 2, 3, 4, 5, 6],
    })


def _pred_col(col: str, name=None) -> Predicate:
    # Vectorized Where over a boolean column; pretty name optional
    return Where(lambda d, c=col: d[c], name=f"({col})" if name is None else name)


def test_grouping_by_canonical_relation_and_subset_drop(df_masked):
    # Two conclusions that should canonicalize to the same key:
    # y >= x  and  y >= 1*x
    rel1 = Ge("y", to_expr("x"))
    rel2 = Ge("y", Const(1) * to_expr("x"))

    # Hypotheses: H_all (most general) and H_A (subset)
    H_all = _pred_col("alltrue", name="(all)")
    H_A   = _pred_col("A", name="(A)")

    c1 = Conjecture(relation=rel1, condition=H_A, name="A_subset")
    c2 = Conjecture(relation=rel2, condition=H_all, name="ALL_general")

    res = morgan_filter(df_masked, [c1, c2])

    # Kept should include only the most general (ALL_general), drop A_subset
    kept_names = {c.name for c in res.kept}
    dropped = {c.name for (c, _reason) in res.dropped}
    assert "ALL_general" in kept_names
    assert "A_subset" in dropped
    # Exactly one kept for this conclusion
    assert len(res.kept) == 1


def test_identical_masks_keep_simplest_name(df_masked):
    # Same conclusion; identical masks (use the same column)
    rel = Ge("y", to_expr("x"))

    # Two conditions with identical masks but different names/complexity
    H_same_simple = Where(lambda d: d["A"], name="(A)")
    H_same_verbose = Where(lambda d: d["A"], name="((A) ∧ (true))")  # deliberately more complex

    c_simple = Conjecture(relation=rel, condition=H_same_simple, name="simple")
    c_verbose = Conjecture(relation=rel, condition=H_same_verbose, name="verbose")

    res = morgan_filter(df_masked, [c_simple, c_verbose])

    kept = {c.name for c in res.kept}
    dropped = {c.name for (c, reason) in res.dropped}

    # Should keep only one representative; prefer simpler condition name
    assert kept == {"simple"} or kept == {"verbose"}  # name of kept conj
    # But due to the tie-breaker (shorter pretty name), we expect "simple" to win
    assert "simple" in kept
    assert "verbose" in dropped


def test_incomparable_masks_both_survive(df_masked):
    # Same conclusion; hypotheses A and B are not subsets of each other
    rel = Ge("y", to_expr("x"))
    H_A = _pred_col("A")
    H_B = _pred_col("B")

    cA = Conjecture(relation=rel, condition=H_A, name="on_A")
    cB = Conjecture(relation=rel, condition=H_B, name="on_B")

    res = morgan_filter(df_masked, [cA, cB])
    kept = {c.name for c in res.kept}
    assert {"on_A", "on_B"} == kept
    assert len(res.dropped) == 0


def test_condition_none_is_most_general(df_masked):
    # None → mask = all True; should dominate others for same conclusion
    rel = Ge("y", to_expr("x"))
    c_general = Conjecture(relation=rel, condition=None, name="none_general")
    c_A = Conjecture(relation=rel, condition=_pred_col("A"), name="on_A")
    c_B = Conjecture(relation=rel, condition=_pred_col("B"), name="on_B")

    res = morgan_filter(df_masked, [c_general, c_A, c_B])
    kept_names = {c.name for c in res.kept}
    assert kept_names == {"none_general"}
    assert {c.name for c, _ in res.dropped} == {"on_A", "on_B"}


def test_different_conclusions_are_grouped_separately(df_masked):
    # Different relations => separate groups => no cross-dominance
    rel_lower = Ge("y", to_expr("x"))
    rel_upper = Le("y", 2 * to_expr("x"))

    H_A = _pred_col("A")
    H_B = _pred_col("B")

    c1 = Conjecture(relation=rel_lower, condition=H_A, name="lower_A")
    c2 = Conjecture(relation=rel_upper, condition=H_B, name="upper_B")

    res = morgan_filter(df_masked, [c1, c2])
    kept = {c.name for c in res.kept}
    assert kept == {"lower_A", "upper_B"}
    assert len(res.dropped) == 0
