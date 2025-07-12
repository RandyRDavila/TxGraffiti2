import pandas as pd
import pytest

import txgraffiti as rd
from txgraffiti.heuristics.delavina import sophie_accept
from txgraffiti.logic import Property, Predicate, KnowledgeTable, Conjecture, Inequality

# ————— Fixtures —————

@pytest.fixture
def df():
    # 4 rows, two overlapping boolean predicates p and q
    return pd.DataFrame({
        'alpha': [1, 2, 3, 4],
        'p':     [True,  True,  False, False],
        'q':     [False, True,  True,  False],
    })

@pytest.fixture
def kt(df):
    return KnowledgeTable(df)


# Helper to build a “dummy” conjecture whose conclusion always holds
def make_conj(pred):
    # use alpha <= alpha as a vacuous conclusion
    A = Property('alpha', lambda df: df['alpha'])
    return pred >> (A <= A)


# ————— Tests on plain DataFrame —————

def test_accept_when_no_prior(df):
    P = Predicate('p', lambda df: df['p'])
    new = make_conj(P)
    # no accepted ⇒ any nonempty cover should be accepted
    assert sophie_accept(new, [], df)

def test_reject_empty_cover(df):
    # create a predicate that is always False
    F = Predicate('never', lambda df: pd.Series(False, index=df.index))
    new = make_conj(F)
    # empty cover ⇒ no new rows ⇒ reject
    assert not sophie_accept(new, [], df)

def test_reject_if_subset_of_existing(df):
    P = Predicate('p', lambda df: df['p'])         # covers rows [0,1]
    P_union = Predicate('p_or_q', lambda df: df['p'] | df['q'])  # covers [0,1,2]
    c_small = make_conj(P)
    c_big   = make_conj(P_union)
    # c_small’s cover ⊂ c_big’s cover ⇒ reject c_small
    assert not sophie_accept(c_small, [c_big], df)
    # but c_big adds row2 beyond c_small ⇒ accept
    assert sophie_accept(c_big, [c_small], df)

def test_accept_if_disjoint_cover(df):
    P1 = Predicate('p', lambda df: df['p'])   # covers [0,1]
    P2 = Predicate('q', lambda df: df['q'])   # covers [1,2]
    c1 = make_conj(P1)
    c2 = make_conj(P2)
    # union of accepted ([0,1]) vs new cover [1,2] ⇒ row2 is new ⇒ accept
    assert sophie_accept(c2, [c1], df)

def test_reject_if_no_new_row(df):
    P1 = Predicate('p', lambda df: df['p'])   # covers [0,1]
    # define P1_subset that only covers row1
    P1_sub = Predicate('only1', lambda df: pd.Series([False,True,False,False], index=df.index))
    c1 = make_conj(P1)
    c1_sub = make_conj(P1_sub)
    # c1_sub covers only row1, which was already in c1’s cover ⇒ reject
    assert not sophie_accept(c1_sub, [c1], df)


# ————— Parametrized test for both df and KnowledgeTable —————

@pytest.mark.parametrize("table", ["df", "kt"])
def test_sophie_accept_on_both(table, df, kt):
    T = df if table == "df" else kt
    # pick predicate accordingly
    P = Predicate('p', lambda df: df['p'])   if table=="df" else T.p
    new = make_conj(P)
    # empty accepted list
    assert sophie_accept(new, [], T)
