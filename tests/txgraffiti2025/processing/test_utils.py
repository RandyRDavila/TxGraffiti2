import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.processing.utils import (
    truth_mask,
    touch_count,
    slack_summary,
    hash_conjecture,
    is_trivial_conjecture,
    compare_conjectures,
    safe_apply,
    describe_conjectures,
)

from txgraffiti2025.forms import (
    Conjecture,
    Eq, Le, Ge,
    Where,
    GEQ, LEQ,  # shorthands from predicates
    to_expr,
)


# ---------- Fixtures ----------

@pytest.fixture
def df_basic():
    # c = 1.5 * a; b is constant 2
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 2.0, 2.0],
        "c": [1.5, 3.0, 4.5, 6.0],
        "cat": ["x", "y", "x", "z"],
    })


@pytest.fixture
def conj_le(df_basic):
    # a <= c  (true for all rows)
    return Conjecture(Le("a", "c"), None, name="a_le_c")


@pytest.fixture
def conj_eq_all(df_basic):
    # a == a  (tautology; slack == 0 everywhere)
    return Conjecture(Eq("a", "a"), None, name="a_eq_a")


@pytest.fixture
def conj_false(df_basic):
    # a >= a + 1 (always false)
    return Conjecture(Ge("a", to_expr("a") + 1.0), None, name="a_ge_a_plus_1")


@pytest.fixture
def conj_conditional(df_basic):
    # a <= c restricted to 2 <= a <= 3  (middle two rows)
    C = GEQ("a", 2.0) & LEQ("a", 3.0)
    return Conjecture(Le("a", "c"), C, name="a_le_c_given_2_to_3")


# ---------- Tests: truth_mask & alignment ----------

def test_truth_mask_alignment(df_basic, conj_le):
    holds = truth_mask(conj_le, df_basic)
    # Should match relation.evaluate
    assert holds.equals(Le("a", "c").evaluate(df_basic))
    assert holds.index.equals(df_basic.index)


def test_truth_mask_with_condition(df_basic, conj_conditional):
    holds = truth_mask(conj_conditional, df_basic)
    # Outside condition, holds is vacuously True
    C = (df_basic["a"] >= 2.0) & (df_basic["a"] <= 3.0)
    rel = Le("a", "c").evaluate(df_basic)
    expected = (~C) | (C & rel)
    assert holds.equals(expected)


# ---------- Tests: touch_count ----------

def test_touch_count_eq_everywhere(df_basic, conj_eq_all):
    assert touch_count(conj_eq_all, df_basic) == len(df_basic)


def test_touch_count_inequality_tight_some():
    # Build a tiny df so tightness is predictable
    df = pd.DataFrame({"a": [1.0, 2.0], "c": [1.0, 3.0]})  # tight at first row
    conj = Conjecture(Le("a", "c"), None)
    assert touch_count(conj, df) == 1


# ---------- Tests: slack_summary ----------

def test_slack_summary_values(df_basic, conj_le):
    mn, avg = slack_summary(conj_le, df_basic)
    # slack = c - a
    slack = (df_basic["c"] - df_basic["a"]).to_numpy()
    assert np.isclose(mn, slack.min())
    assert np.isclose(avg, slack.mean())


def test_slack_summary_empty(df_basic):
    # Condition never true -> (nan, nan)
    C = Where(lambda df: pd.Series(False, index=df.index))
    conj = Conjecture(Le("a", "c"), C)
    mn, avg = slack_summary(conj, df_basic)
    assert np.isnan(mn) and np.isnan(avg)


# ---------- Tests: hashing & comparing ----------

def test_hash_stability_and_difference(df_basic, conj_le, conj_eq_all):
    h1 = hash_conjecture(conj_le)
    h2 = hash_conjecture(conj_le)
    h3 = hash_conjecture(conj_eq_all)
    assert h1 == h2
    assert h1 != h3


def test_compare_conjectures(df_basic, conj_le):
    clone = Conjecture(Le("a","c"), None, name="a_le_c")  # structurally identical
    other = Conjecture(Ge("a","c"), None, name="a_ge_c")
    assert compare_conjectures(conj_le, clone)
    assert not compare_conjectures(conj_le, other)


# ---------- Tests: triviality detection ----------

def test_is_trivial_when_condition_empty(df_basic):
    C = Where(lambda df: pd.Series(False, index=df.index))
    conj = Conjecture(Le("a","c"), C)
    assert is_trivial_conjecture(conj, df_basic) is True


def test_is_trivial_tautology(df_basic, conj_eq_all):
    assert is_trivial_conjecture(conj_eq_all, df_basic) is True


def test_nontrivial_false_conjecture(df_basic, conj_false):
    # Always false should not be marked as "trivial"
    assert is_trivial_conjecture(conj_false, df_basic) is False


# ---------- Tests: safe_apply & describe ----------

def test_safe_apply_skips_errors(df_basic, conj_le, conj_false):
    data = [conj_le, "not_a_conjecture", conj_false]
    def fn(x):  # raises for the middle element
        return hash_conjecture(x)
    out = safe_apply(data, fn)
    # Should only have two hashes (skipped the string)
    assert len(out) == 2
    assert isinstance(out[0], str) and isinstance(out[1], str)

def test_describe_conjectures_returns_dataframe(df_basic, conj_le, conj_eq_all, conj_false):
    df = describe_conjectures([conj_le, conj_eq_all, conj_false], df_basic)
    assert set(["name","hash","touch","min_slack","mean_slack"]).issubset(set(df.columns))
    assert len(df) == 3
    # Types: touch is int-ish, slacks are floats or nan
    assert df["touch"].dtype.kind in "iu"
    assert df["min_slack"].dtype.kind in "f"
