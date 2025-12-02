import pandas as pd
import numpy as np
import pytest

from txgraffiti2025.forms import Conjecture, Le, Ge, Eq, to_expr
from txgraffiti2025.processing.post.dalmatian import (
    dalmatian_filter,
    dalmatian_score,
)

# ---------- Fixtures ----------

@pytest.fixture
def df():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.1, 3.9, 5.2],
        "c": [2.0, 2.0, 2.0, 2.0],
    })


@pytest.fixture
def true_conj(df):
    return Conjecture(Le("a", "b"), None, name="a_le_b_true")


@pytest.fixture
def false_conj(df):
    return Conjecture(Ge("a", to_expr("b") + 1.0), None, name="a_ge_b_plus1_false")


@pytest.fixture
def tight_conj(df):
    # slightly tighter than true_conj (b - 0.5)
    return Conjecture(Le("a", to_expr("b") - 0.5), None, name="a_le_b_minus_half")


# ---------- Tests: truth test ----------

def test_truth_test_true(df, true_conj):
    s = dalmatian_score(true_conj, df)
    assert s["truth_ok"] is True
    assert s["touch_count"] >= 0


def test_truth_test_false(df, false_conj):
    s = dalmatian_score(false_conj, df)
    assert s["truth_ok"] is False


# ---------- Tests: significance ----------

def test_significance_tighter_bound(df, true_conj, tight_conj):
    # Use a local df with margin: min(b-a) = 0.8
    df2 = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.8, 2.8, 3.9, 4.8],  # b-a: [0.8, 0.8, 0.9, 0.8]
    })
    base = Conjecture(Le("a","b"), None, name="a_le_b_true")
    tighter = Conjecture(Le("a", to_expr("b") - 0.5), None, name="a_le_b_minus_half")
    kept = dalmatian_filter([base, tighter], df2)
    names = [c.name for c in kept]
    assert "a_le_b_true" in names
    assert "a_le_b_minus_half" in names  # tighter and still true given the margin
    assert len(kept) == 2


def test_significance_redundant_bound(df):
    c1 = Conjecture(Le("a", "b"), None, name="a_le_b_1")
    c2 = Conjecture(Le("a", "b"), None, name="a_le_b_2")  # identical
    kept = dalmatian_filter([c1, c2], df)
    assert len(kept) == 1
    assert kept[0].name in {"a_le_b_1", "a_le_b_2"}


def test_significance_lower_bounds(df):
    # Construct two true lower bounds: a >= b - 1.2 (tight) and a >= b - 2.0 (weaker)
    tight = Conjecture(Ge("a", to_expr("b") - 1.2), None, name="a_ge_b_minus_1_2")
    weak  = Conjecture(Ge("a", to_expr("b") - 2.0), None, name="a_ge_b_minus_2_0")
    kept = dalmatian_filter([weak, tight], df)
    names = [c.name for c in kept]
    # Pure Dalmatian keeps both: both are true; significance is non-retroactive
    assert "a_ge_b_minus_1_2" in names
    assert "a_ge_b_minus_2_0" in names  # both survive under pure Dalmatian (no retroactive pruning)


# ---------- Tests: duplicate detection ----------

def test_duplicate_hash_removal(df):
    c1 = Conjecture(Le("a", "b"), None, name="dup1")
    c2 = Conjecture(Le("a", "b"), None, name="dup2")
    kept = dalmatian_filter([c1, c2], df)
    assert len(kept) == 1


# ---------- Tests: robustness ----------

def test_error_handling(df):
    class Bad:
        pass
    c = Bad()
    kept = dalmatian_filter([c], df)
    assert kept == []

def test_dalmatian_score_values(df, true_conj):
    s = dalmatian_score(true_conj, df)
    assert set(["truth_ok","touch_count","min_slack","mean_slack"]).issubset(s.keys())
