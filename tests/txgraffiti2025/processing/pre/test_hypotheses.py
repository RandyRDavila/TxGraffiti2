import pandas as pd
import numpy as np
import pytest

from txgraffiti2025.processing.pre.hypotheses import (
    list_boolean_columns,
    detect_base_hypothesis,
    enumerate_boolean_hypotheses,
)
from txgraffiti2025.forms.predicates import Where, AndPred
from txgraffiti2025.forms.generic_conjecture import TRUE


# ---------------------------
# list_boolean_columns
# ---------------------------

def test_list_boolean_columns_detects_bool_and_binary_ints():
    df = pd.DataFrame({
        "b_true":  [True, True, True],
        "b_mix":   [True, False, True],
        "i01":     [0, 1, 1],    # binary ints
        "i02":     [0, 2, 0],    # not binary
        "f":       [0.0, 1.0, 0.0],
        "text":    ["a", "b", "c"],
    })

    cols_default = set(list_boolean_columns(df))
    assert {"b_true", "b_mix", "i01"} <= cols_default
    assert "i02" not in cols_default
    assert "f" not in cols_default and "text" not in cols_default

    # When treat_binary_ints=False, integer 0/1 should not be treated as bool
    cols_no_ints = set(list_boolean_columns(df, treat_binary_ints=False))
    assert {"b_true", "b_mix"} <= cols_no_ints
    assert "i01" not in cols_no_ints


# ---------------------------
# detect_base_hypothesis
# ---------------------------

def test_detect_base_hypothesis_uses_all_true_booleans():
    df = pd.DataFrame({
        "connected": [True, True, True],
        "simple":    [True, True, True],
        "bip":       [True, False, True],
        "x":         [1, 2, 3],
    })
    base = detect_base_hypothesis(df)

    # Base must be conjunction of (connected) and (simple)
    assert isinstance(base, (Where, AndPred))
    m = base.mask(df)
    assert m.index.equals(df.index)
    assert m.all()

    # Pretty names propagated
    s = getattr(base, "name", repr(base))
    assert "(connected)" in s and "(simple)" in s and "∧" in s

def test_detect_base_returns_TRUE_when_no_all_true_columns():
    df = pd.DataFrame({
        "flag": [True, False, True],
        "x": [1, 2, 3],
    })
    base = detect_base_hypothesis(df)
    assert base is TRUE
    m = base.mask(df)
    assert m.all()  # TRUE over all rows


# ---------------------------
# enumerate_boolean_hypotheses
# ---------------------------

def test_enumerate_includes_base_singles_and_pairs_with_constraints():
    # Base column all True; two boolean columns with overlapping but not subset masks
    df = pd.DataFrame({
        "base":       [True, True, True, True],
        "c1":         [True, False, True, False],  # has rows outside c2
        "c2":         [False, True, True, False],  # has rows outside c1
        "irrelevant": [0, 1, 2, 3],
    })
    # rename base column to something realistic like "connected"
    df = df.rename(columns={"base": "connected"})
    preds = enumerate_boolean_hypotheses(df)

    # 1) base present
    assert len(preds) >= 3
    base = preds[0]
    assert base is not None
    assert base.mask(df).all()

    # 2) singles exist and are base ∧ (c)
    singles = preds[1:3]
    for p in singles:
        m = p.mask(df)
        assert m.index.equals(df.index)
        # Should be subset of base
        assert (m <= base.mask(df)).all()
        # Non-empty under base
        assert m.any()

    # 3) at least one pair is present (connected ∧ c1 ∧ c2)
    any_pair = False
    for p in preds[3:]:
        m = p.mask(df)
        if m.equals((df["connected"] & df["c1"] & df["c2"])):
            any_pair = True
            break
    assert any_pair

def test_enumerate_skips_always_false_under_base():
    df = pd.DataFrame({
        "connected": [True, True, True, True],
        "c_ok":      [True, False, False, True],   # has support
        "c_empty":   [False, False, False, False], # no support
    })
    preds = enumerate_boolean_hypotheses(df, include_pairs=False)
    # Expect base + single for c_ok, but not for c_empty
    assert len(preds) == 2
    base, single = preds
    m_single = single.mask(df)
    assert m_single.equals(df["connected"] & df["c_ok"])

def test_enumerate_respects_treat_binary_ints_flag():
    df = pd.DataFrame({
        "connected": [True, True, True],
        "i01":       [0, 1, 1],  # should be ignored if treat_binary_ints=False
    })
    preds_yes = enumerate_boolean_hypotheses(df, treat_binary_ints=True, include_pairs=False)
    preds_no  = enumerate_boolean_hypotheses(df, treat_binary_ints=False, include_pairs=False)

    # With ints as bools: base + single(i01)
    assert len(preds_yes) == 2
    # Without ints as bools: only base
    assert len(preds_no) == 1

def test_masks_are_aligned_and_boolean_dtype():
    df = pd.DataFrame({
        "connected": [True, True, False, True],
        "c1": [True, False, False, True],
        "c2": [False, True, False, True],
    }, index=[10, 11, 12, 13])
    preds = enumerate_boolean_hypotheses(df)
    for p in preds:
        m = p.mask(df)
        assert m.index.equals(df.index)
        assert m.dtype == bool

def test_pretty_names_of_singles_and_pairs():
    df = pd.DataFrame({
        "connected": [True, True, True],
        "bip": [True, False, True],
        "simple": [True, True, True],
    })
    preds = enumerate_boolean_hypotheses(df)

    # Expect base first (connected & simple), and a single containing "(bip)"
    names = [getattr(p, "name", repr(p)) for p in preds]
    assert any("(bip)" in s for s in names)
    # Base pretty name contains ∧ when multiple all-true columns exist
    assert any("∧" in s for s in names)
