# tests/test_generate_lp_conjecture.py

import pandas as pd
import numpy as np
import pytest
from fractions import Fraction

from txgraffiti2 import (
    Property,
    Predicate,
    Inequality,
    Conjecture,
    generate_lp_conjecture,
)

# ─ Helpers ─────────────────────────────────────────────────────────────────

def make_df():
    # numeric x, y columns
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 9, 10],
        })

def prop(name="x"):
    return Property(name, lambda df, c=name: df[c])

TRUE = Predicate("True", lambda df: pd.Series(True, index=df.index))


# ─ Tests ────────────────────────────────────────────────────────────────────

def test_empty_subdf_raises():
    df = make_df()
    x = prop("x")
    # hypothesis never true → empty subdf
    false_hyp = Predicate("False", lambda df: df["x"] > 100)
    with pytest.raises(ValueError):
        generate_lp_conjecture(
            df=df,
            features=[x],
            target=x,
            hypothesis=false_hyp
        )

def test_clear_bound():
    df = make_df()
    x = prop("x")
    y = prop("y")

    # hypothesis is always true
    conj = generate_lp_conjecture(df, [x], y, TRUE)

    # it must hold everywhere
    assert conj.is_true(df)

    # check the learned statement
    assert conj.conclusion.rhs.name == "y"
    assert conj.conclusion.lhs.name == "(2*x)"
    assert repr(conj.conclusion) == "<Ineq (2*x) <= y>"
    assert repr(conj) == "(True) → ((2*x) <= y)"

    # check touch number
    assert conj.conclusion.touch_count(df) == 4



# @pytest.mark.parametrize("coefs,bias", [
#     ([2.0], 1.0),     # y = 2x + 1
#     ([3.0], 0.0),     # y = 3x
#     ([0.0], 5.0),     # y = constant
# ])
# def test_perfect_fit_linear(coefs, bias):
#     df = make_df()
#     x = prop("x")
#     # build y = coefs[0]*x + bias
#     df["y"] = coefs[0] * df["x"] + bias

#     y = prop("y")
#     conj = generate_lp_conjecture(df, [x], y, TRUE)

#     # 1) it must hold everywhere
#     assert conj.is_true(df)

#     # 2) check the learned coefficients
#     #    string name should reflect only the nonzero terms
#     term_strs = []
#     if abs(coefs[0]) > 1e-8:
#         frac = Fraction(str(coefs[0])).limit_denominator()
#         term_strs.append(f"{frac}*x" if frac != 1 else "x")
#     if abs(bias) > 1e-8:
#         term_strs.append(str(Fraction(str(bias)).limit_denominator()))

#     if term_strs:
#         expected = "(" + " + ".join(term_strs) + ")"
#     else:
#         expected = "(0)"

#     assert conj.conclusion.rhs.name == expected

# def test_hypothesis_filtering():
#     df = make_df()
#     x = prop("x")
#     # y = 2x + 1 again
#     df["y"] = 2*df["x"] + 1

#     # only rows with x>=3
#     hyp = Predicate("x>=3", lambda df: df["x"] >= 3)
#     y = prop("y")
#     conj = generate_lp_conjecture(df, [x], y, hyp)

#     # it must hold everywhere (including hyp=False rows via implication)
#     assert conj.is_true(df)

#     # check that coefficients still fit the sub‐subset exactly
#     # i.e. the rhs must still be "2*x + 1"
#     assert conj.conclusion.rhs.name == "(2*x + 1)"

# def test_zero_term_production():
#     df = make_df()
#     x = prop("x")
#     # y=0 so best fit is a=0, b=0
#     df["y"] = 0
#     y = prop("y")

#     conj = generate_lp_conjecture(df, [x], y, TRUE)
#     # everything zero → name should be "(0)"
#     assert conj.conclusion.rhs.name == "(0)"
#     assert conj.is_true(df)

# def test_tol_filters_small_coeffs(monkeypatch):
#     df = make_df()
#     x = prop("x")
#     # y = 0.000000001 * x + 0
#     df["y"] = 1e-9 * df["x"]
#     y = prop("y")

#     # use a tight tol so that coefficient is dropped
#     conj = generate_lp_conjecture(df, [x], y, TRUE, tol=1e-8)
#     # |a| < tol so we should get only "(0)" as rhs
#     assert conj.conclusion.rhs.name == "(0)"

# def test_nontrivial_constant_only():
#     df = make_df()
#     x = prop("x")
#     # make y constant = 5
#     df["y"] = 5
#     y = prop("y")

#     conj = generate_lp_conjecture(df, [x], y, TRUE)
#     # a=0 dropped, b=5 kept → "(5)"
#     assert conj.conclusion.rhs.name == "(5)"
#     assert conj.is_true(df)
