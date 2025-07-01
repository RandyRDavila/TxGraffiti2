import pandas as pd
import pytest

from txgraffiti2.conjecture_logic import Property, Predicate, Conjecture, Inequality, Constant
from txgraffiti2.dalmatian_heuristic import dalmatian_accept

# A helper predicate that's always true
TRUE = Predicate("True", lambda df: pd.Series(True, index=df.index))

def make_simple_df():
    # simple DataFrame with a single numeric column 'x'
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 1, 2, 2, 3]})

def make_prop(name="x"):
    return Property(name, lambda df, c=name: df[c])

def make_ineq(lhs, op, rhs):
    return Inequality(lhs, op, rhs)

def test_reject_if_not_true_everywhere():
    df = make_simple_df()
    x = make_prop("x")
    # conjecture x <= 3  fails on rows 4,5
    new = Conjecture(TRUE, make_ineq(x, "<=", Constant(3)))
    assert not dalmatian_accept(new, [], df)

def test_accept_with_no_existing():
    df = make_simple_df()
    x = make_prop("x")
    # x <= 5 holds everywhere
    new = Conjecture(TRUE, make_ineq(x, "<=", Constant(5)))
    assert dalmatian_accept(new, [], df)

def test_accept_if_strictly_tighter_somewhere_constant():
    df = make_simple_df()
    x = make_prop("x")
    # existing bound x <= 5
    old = Conjecture(TRUE, make_ineq(x, "<=", Constant(6)))
    # new bound x <= 4 is strictly tighter at rows where x in {5}
    new = Conjecture(TRUE, make_ineq(x, "<=", Constant(5)))
    assert dalmatian_accept(new, [old], df)

def test_accept_if_strictly_tighter_somewhere_nonconstant():
    df = make_simple_df()
    x = make_prop("x")
    y = make_prop("y")
    # existing bound x <= 3*y
    old = Conjecture(TRUE, make_ineq(x, "<=", Property("3*y", lambda df: 3 * df["y"])))
    # new bound x <= 2*y is strictly tighter at rows where y in {1, 2}
    new = Conjecture(TRUE, make_ineq(x, "<=", Property("2*y", lambda df: 2 * df["y"])))
    assert dalmatian_accept(new, [old], df)
    
def test_reject_if_never_tighter():
    df = make_simple_df()
    x = make_prop("x")
    # existing bound x <= 6
    old = Conjecture(TRUE, make_ineq(x, "<=", Constant(6)))
    # new bound x <= 6 is never strictly tighter
    new_same = Conjecture(TRUE, make_ineq(x, "<=", Constant(6)))
    assert not dalmatian_accept(new_same, [old], df)
    # new bound x <= 7 is looser, also reject
    new_looser = Conjecture(TRUE, make_ineq(x, "<=", Constant(7)))
    assert not dalmatian_accept(new_looser, [old], df)

def test_only_compare_same_hypothesis():
    df = make_simple_df()
    x = make_prop("x")
    # two different hypotheses
    H1 = TRUE
    H2 = Predicate("x>0", lambda df: df["x"] > 0)

    # existing bound under H2: x <= 3
    old_diff = Conjecture(H2, make_ineq(x, "<=", Constant(3)))

    # new bound under H1: x <= 5
    new = Conjecture(H1, make_ineq(x, "<=", Constant(5)))

    # although `new` is not tighter than old_diff under H2, they have different hypotheses,
    # so new should be accepted (since no existing under H1)
    assert dalmatian_accept(new, [old_diff], df)
    
    new2 = Conjecture(H1, make_ineq(x, "<=", Constant(6)))
    assert not dalmatian_accept(new2, [new], df)
 