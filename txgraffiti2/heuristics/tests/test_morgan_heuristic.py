# tests/test_morgan_accept.py

import pandas as pd
import pytest

from txgraffiti2 import (
    Property,
    Predicate,
    Inequality,
    Conjecture,
    morgan_accept,
    Constant
)

# -- helpers to build simple Conjectures ------------------

def make_df():
    # 3 rows, two boolean columns A,B and one numeric x
    return pd.DataFrame({
        "A": [True,  True,  False],
        "B": [True,  False, False],
        "x": [1,     2,     3]
    })

def prop_x():
    return Property("x", lambda df: df["x"])

# -- the tests --------------------------------------------

def test_accept_if_no_existing():
    df = make_df()
    H = Predicate("A", lambda df: df["A"])
    x = Property("x", lambda df: df["x"])
    inq = Inequality(x, "<=", Constant(2))         # A → x ≤ 2 holds everywhere on A’s rows
    conj = H.implies(inq, as_conjecture=True)
    df = make_df()
    assert conj.is_true(df)    # no existing → accept

    conj2 = H >> inq
    assert conj2.is_true(df)    # same thing, different syntax → accept

def test_accept_different_conclusion():
    df = make_df()
    A = Predicate("A", lambda df: df["A"])
    B = Predicate("B", lambda df: df["B"])
    x = Property("x", lambda df: df["x"])
    new_conj  = A.implies(Inequality(x, ">=", Constant(2)), as_conjecture=True)  # A → x ≥ 2
    old_conj  = B.implies(Inequality(x, ">=", Constant(2)), as_conjecture=True)  # B → x ≥ 2
    assert morgan_accept(new_conj, [old_conj], df)  # different conclusion → accept

def test_accept_negation():
    df = make_df()
    B = Predicate("B", lambda df: df["B"])
    notB = ~B
    x = Property("x", lambda df: df["x"])
    new_conj  = B.implies(Inequality(x, "<=", Constant(2)), as_conjecture=True)  # A → x ≥ 2
    old_conj  = notB.implies(Inequality(x, "<=", Constant(2)), as_conjecture=True)  # B → x ≤ 2
    assert morgan_accept(new_conj, [old_conj], df)  # different conclusion → accept

# def test_new_more_general_than_existing():
#     df = make_df()
#     # hypothesis A covers rows [0,1]
#     H_A = Predicate("A", lambda df: df["A"])
#     # hypothesis B covers only row [0]
#     H_B = Predicate("B", lambda df: df["B"])
#     # both propose the same conclusion x ≤ 2
#     old = make_conj(H_B, "<=", 2)
#     new = make_conj(H_A, "<=", 2)
#     assert morgan_accept(new, [old], df)        # new_mask ⊃ old_mask → accept

# def test_reject_if_strict_subset():
#     df = make_df()
#     H_A = Predicate("A", lambda df: df["A"])  # mask [True,True,False]
#     H_B = Predicate("B", lambda df: df["B"])  # mask [True,False,False]
#     old = make_conj(H_A, "<=", 2)
#     new = make_conj(H_B, "<=", 2)
#     assert not morgan_accept(new, [old], df)    # new_mask ⊂ old_mask → reject

# def test_equal_mask_is_not_subset():
#     df = make_df()
#     H_A = Predicate("A", lambda df: df["A"])
#     old = make_conj(H_A, "<=", 2)
#     new = make_conj(H_A, "<=", 2)
#     assert morgan_accept(new, [old], df)        # equal masks → accept

# def test_flipped_conclusion_seen_as_same():
#     df = make_df()
#     H = Predicate("A", lambda df: df["A"])
#     # old: x <= 2 ; new: 2 >= x  (same logical conclusion)
#     old = make_conj(H, "<=", 2)
#     new = make_conj(H, ">=", 2)
#     # masks are equal so new is not a strict subset => accept
#     assert morgan_accept(new, [old], df)

# def test_only_same_hypothesis_compared():
#     df = make_df()
#     H1 = Predicate("A", lambda df: df["A"])
#     H2 = Predicate("B", lambda df: df["B"])
#     # old under H2, new under H1, same conclusion
#     old = make_conj(H2, "<=", 2)
#     new = make_conj(H1, "<=", 2)
#     # although B‐mask⊃?A‐mask? irrelevant because hypotheses differ → accept
#     assert morgan_accept(new, [old], df)
