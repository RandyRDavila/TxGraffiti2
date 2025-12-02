# tests/unit/txgraffiti2025/processing/post/test_reciprocal_generalizer.py

import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import Ge, Le, Conjecture, Predicate, Where, to_expr
from txgraffiti2025.processing.post.reciprocal_generalizer import (
    find_reciprocal_matches_for_conjecture,
    propose_generalizations_from_reciprocals,
)


def _H_true(df):
    return Predicate.from_column("H")


def test_find_matches_lower_uses_min_and_detects_coeff():
    # y >= c * x with c = min(1/(b+s)) under H
    # choose s=0 so min(1/b) = 1/3 => c ≈ 0.333...
    df = pd.DataFrame({
        "H": [True]*6,
        "x": [1, 2, 3, 4, 5, 6],
        "b": [3, 4, 5, 6, 7, 8],          # 1/b ranges [1/8, 1/3]
    })
    H = _H_true(df)
    c = 1/3
    conj = Conjecture(Ge("y", c * to_expr("x")), condition=H)  # pattern: Ge, coeff ~ 1/3
    matches = find_reciprocal_matches_for_conjecture(
        df, conj, candidate_cols=["b"], shifts=(0,), min_support=3
    )
    assert len(matches) >= 1
    # best match should be with shift 0 and column 'b'
    top = matches[0]
    assert top.column == "b" and top.shift == 0
    assert np.isclose(top.extremum_value, c, rtol=1e-9, atol=1e-9)
    # expr is 1 / b
    vals = top.expr.eval(df)
    assert np.isfinite(vals).all()


def test_find_matches_upper_uses_max_and_detects_coeff_with_shift():
    # y <= c * x with c = max(1/(b+1)) under H
    # b+1 in {2,3,4,5} => max(1/(b+1)) = 1/2
    df = pd.DataFrame({
        "H": [True]*4,
        "x": [1, 2, 3, 4],
        "b": [1, 2, 3, 4],
    })
    H = _H_true(df)
    c = 0.5
    conj = Conjecture(Le("y", c * to_expr("x")), condition=H)
    matches = find_reciprocal_matches_for_conjecture(
        df, conj, candidate_cols=["b"], shifts=(1,), min_support=2
    )
    assert len(matches) >= 1
    top = matches[0]
    assert top.column == "b" and top.shift == 1
    assert np.isclose(top.extremum_value, c, rtol=1e-9, atol=1e-9)
    # expr = 1/(b+1)
    vals = top.expr.eval(df)
    assert np.isfinite(vals).all()


def test_no_matches_for_non_ratio_style_conjecture():
    # Construct a conjecture that _extract_ratio_pattern will not recognize
    df = pd.DataFrame({"H": [True, True], "x": [1.0, 2.0], "y": [2.0, 3.0]})
    H = _H_true(df)
    # Relation not of the form target (<=/>=) c * feature (e.g., add intercept)
    conj = Conjecture(Ge("y", 2.0 * to_expr("x") + 1.0), condition=H)
    matches = find_reciprocal_matches_for_conjecture(df, conj)
    assert matches == []


def test_candidate_cols_filter_and_min_support():
    # Only 'b' is allowed by candidate_cols; 'c' should be ignored
    df = pd.DataFrame({
        "H": [True]*6,
        "x": [1, 2, 3, 4, 5, 6],
        "b": [3, 4, 5, 6, 7, 8],
        "c": [10, 10, 10, 10, 10, 10],
    })
    H = _H_true(df)
    conj = Conjecture(Ge("y", (1/3) * to_expr("x")), condition=H)
    matches = find_reciprocal_matches_for_conjecture(
        df, conj, candidate_cols=["b"], shifts=(0,), min_support=5
    )
    assert len(matches) >= 1
    assert all(m.column == "b" for m in matches)


def test_propose_generalizations_subset_filter_and_truth_check_lower():
    # Base conjecture: y >= (1/3) * x under H1
    # We will propose 1/b where min(1/b) = 1/3, and verify it holds on a superset H2.
    df = pd.DataFrame({
        "H1": [True, True, True, False, False, False],
        "H2": [True, True, True, True, False, False],   # H1 ⊆ H2
        "x":  [3,   6,   9,   3,    6,    9],
        "b":  [3,   4,   5,   6,    7,    8],           # min(1/b on H1) = 1/3
        # Make y large enough so y >= (1/b)*x holds for H2 as well
        "y":  [1.0, 3.0, 6.0, 2.0, 4.0, 6.0],
    })
    H1 = Where(lambda d: d["H1"])
    H2 = Where(lambda d: d["H2"])

    base = Conjecture(Ge("y", (1/3) * to_expr("x")), condition=H1)
    props = propose_generalizations_from_reciprocals(
        df, base, candidate_hypotheses=[H1, H2], candidate_cols=["b"], shifts=(0,), min_support=3
    )
    # Should produce a conjecture on H2 (superset) or H1; at least one overall
    assert len(props) >= 1
    for c in props:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert holds[applicable].all()
        assert failures.empty


def test_propose_generalizations_upper_case_with_shift_and_superset_filter():
    # Base: y <= (1/2) * x on H1, with c = max(1/(b+1)) on H1.
    # Only allow proposals that respect the subset relation to H2.
    df = pd.DataFrame({
        "H1": [True, True, True, False, False, False],
        "H2": [True, True, True, True, False, False],   # H1 ⊆ H2
        "x":  [2,   4,   6,   2,    4,    6],
        "b":  [1,   2,   3,   4,    5,    6],           # b+1 in {2,3,4} on H1 -> max 1/2
        "y":  [1,  1.5,  2.5,  1,   3,    10],
    })
    H1 = Where(lambda d: d["H1"])
    H2 = Where(lambda d: d["H2"])
    base = Conjecture(Le("y", 0.5 * to_expr("x")), condition=H1)

    props = propose_generalizations_from_reciprocals(
        df, base, candidate_hypotheses=[H1, H2], candidate_cols=["b"], shifts=(1,), min_support=3
    )
    assert len(props) >= 1
    for c in props:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert holds[applicable].all()
        assert failures.empty


def test_returns_empty_when_no_support_or_no_match():
    df = pd.DataFrame({
        "H": [True]*4,
        "x": [1, 2, 3, 4],
        "b": [10, 10, 10, 10],  # 1/b constant = 0.1; try coeff that doesn't match
        "y": [0, 0, 0, 0],
    })
    H = _H_true(df)
    conj = Conjecture(Ge("y", 0.3333333333 * to_expr("x")), condition=H)
    props = propose_generalizations_from_reciprocals(
        df, conj, candidate_hypotheses=[H], candidate_cols=["b"], shifts=(0,), min_support=4
    )
    assert props == []
