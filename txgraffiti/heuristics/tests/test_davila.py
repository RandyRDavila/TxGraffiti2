import pandas as pd
import pytest

import txgraffiti as rd
from txgraffiti.heuristics.davila import morgan_accept

# ——————— Fixtures ———————
@pytest.fixture
def df():
    # simple DataFrame with two rows
    return pd.DataFrame({
        'alpha': [1, 1, 1],
        'beta': [1, 1, 1],
        'connected': [True, True, True],
        'tree': [False, True, True],
    })

@pytest.fixture
def kt():
    return rd.KnowledgeTable({
        'alpha': [1, 1, 1],
        'beta': [1, 1, 1],
        'connected': [True, True, True],
        'tree': [False, True, True],
    })

def test_accept_if_no_existing(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    hyp   = rd.Predicate('connected', lambda df: df['connected'])
    conj  = hyp >> (alpha <= 100)
    # with no “old” conjectures, any valid one should be accepted
    assert morgan_accept(conj, [], df) is True

def test_accept_strictly_more_general(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    hyp1  = rd.Predicate('tree',      lambda df: df['tree'])
    hyp2  = rd.Predicate('connected', lambda df: df['connected'])
    # old bound only applies where tree==True
    old   = hyp1 >> (alpha <= 10)
    # new bound applies on all connected rows (a superset!)
    new   = hyp2 >> (alpha <= 10)
    # new covers strictly more rows, and conclusion holds on the extra ones
    assert morgan_accept(new, [old], df) is True

def test_dalmatian_accept_two_conjectures_on_dataframe(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    beta = rd.Property('beta', lambda df: df['beta'])

    general_hyp = rd.Predicate('connected', lambda df: df['connected'])
    less_general_hyp = rd.Predicate('tree', lambda df: df['tree'])

    most_general = general_hyp >> (alpha <= beta)
    less_general = less_general_hyp >> (alpha <= beta)

    # decline the weaker conjecture
    assert morgan_accept(less_general, [most_general], df) == False

    # accept the better conjecture
    assert morgan_accept(most_general, [less_general], df) == True

def test_ignore_unrelated_existing(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    beta  = rd.Property('beta',  lambda df: df['beta'])
    # existing conjecture has different conclusion LHS
    hyp      = rd.Predicate('connected', lambda df: df['connected'])
    other    = hyp >> (beta  <= alpha + 1)
    candidate = hyp >> (alpha <= beta + 1)
    # because `other` has a different LHS we ignore it entirely
    assert morgan_accept(candidate, [other], df) is True

def test_dalmatian_accept_two_conjectures_on_knowledgetable(kt):
    alpha = kt.alpha
    beta = kt.beta

    general_hyp = kt.connected
    less_general_hyp = kt.tree

    most_general = general_hyp >> (alpha <= beta)
    less_general = less_general_hyp >> (alpha <= beta)

    # decline the weaker conjecture
    assert morgan_accept(less_general, [most_general], kt) == False

    # accept the better conjecture
    assert morgan_accept(most_general, [less_general], kt) == True
