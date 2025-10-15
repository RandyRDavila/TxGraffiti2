import pandas as pd
import pytest

from txgraffiti2025.forms import Conjecture, Le, Ge, Where, to_expr
from txgraffiti2025.processing.post.morgan import morgan_generalize

@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.1, 3.9, 5.2],
        "flag1": [True, True, True, False],   # coverage 3
        "flag2": [True, False, True, False],  # coverage 2
        "flag_all": [True, True, True, True], # coverage 4
    })

def P(col):
    return Where(lambda df: df[col].astype(bool))

def test_morgan_keeps_most_general_same_conclusion(df):
    # Same conclusion: a <= b
    c1 = Conjecture(Le("a","b"), P("flag1"), name="le_flag1")  # cov=3
    c2 = Conjecture(Le("a","b"), P("flag2"), name="le_flag2")  # cov=2
    c3 = Conjecture(Le("a","b"), None,       name="le_global") # cov=4 (most general)
    out = morgan_generalize([c1,c2,c3], df)
    names = {c.name for c in out}
    assert names == {"le_global"}

def test_morgan_groups_by_conclusion_not_mixing(df):
    # Different conclusions shouldn't compete
    g1 = Conjecture(Le("a","b"), P("flag1"), name="le_flag1")
    g2 = Conjecture(Ge("a", to_expr("b") - 1.0), P("flag2"), name="ge_flag2")
    out = morgan_generalize([g1,g2], df)
    names = {c.name for c in out}
    assert names == {"le_flag1", "ge_flag2"}

def test_morgan_tie_coverage_prefers_simpler_predicate(df):
    # Build a tie in coverage: both cover all rows; prefer simpler predicate (None) over composed
    base = Conjecture(Le("a","b"), None, name="le_global")
    # Construct predicate that is logically identical to True (flag1 or not not flag1) but more complex;
    # for test simplicity, just reuse flag_all which is True everywhere.
    complex_pred = P("flag_all")  # coverage 4, but counted larger complexity than None
    tied = Conjecture(Le("a","b"), complex_pred, name="le_all_complex")
    out = morgan_generalize([base,tied], df)
    names = {c.name for c in out}
    assert names == {"le_global"}

def test_morgan_handles_multiple_groups_and_picks_best_in_each(df):
    # Group 1: a <= b  (keep global)
    g1a = Conjecture(Le("a","b"), P("flag1"), name="le_f1")
    g1b = Conjecture(Le("a","b"), None,       name="le_all")

    # Group 2: a >= b-2  vs a >= b-1  (same target/direction but different rhs => different conclusions)
    g2a = Conjecture(Ge("a", to_expr("b") - 2.0), P("flag1"), name="ge_loose")
    g2b = Conjecture(Ge("a", to_expr("b") - 1.0), P("flag2"), name="ge_tighter")
    out = morgan_generalize([g1a,g1b,g2a,g2b], df)
    names = {c.name for c in out}
    # We expect one from group 1 (global) and both from group 2 (different conclusions)
    assert "le_all" in names and "ge_loose" in names and "ge_tighter" in names
    assert "le_f1" not in names
