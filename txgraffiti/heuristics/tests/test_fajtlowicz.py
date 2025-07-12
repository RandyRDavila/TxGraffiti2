import pandas as pd
import pytest

import txgraffiti as rd
from txgraffiti.heuristics.fajtlowicz import dalmatian_accept

# ——————— Fixtures ———————
@pytest.fixture
def df():
    # simple DataFrame with two rows
    return pd.DataFrame({
        'alpha': [1, 1, 1],
        'beta': [1, 1, 1],
        'connected': [True, True, True],
    })

@pytest.fixture
def kt():
    return rd.KnowledgeTable({
        'alpha': [1, 1, 1],
        'beta': [1, 1, 1],
        'connected': [True, True, True],
    })

def test_accept_if_no_existing_bounds(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    hyp   = rd.Predicate('connected', lambda df: df['connected'])
    conj  = hyp >> (alpha <= 100)
    # no “old bounds” list ⇒ should accept any valid conjecture
    assert dalmatian_accept(conj, [], df) is True

def test_reject_if_not_globally_valid(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    hyp   = rd.Predicate('connected', lambda df: df['connected'])
    # α <= β is false on your df fixture (α==1, β==1 ⇒ valid, but let's break it)
    bad_conj = hyp >> (alpha <= 0)
    assert dalmatian_accept(bad_conj, [], df) is False
    # even if there are no old bounds, invalid ones must be rejected
    assert dalmatian_accept(bad_conj, [bad_conj], df) is False

def test_reject_if_equal_everywhere(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    hyp   = rd.Predicate('connected', lambda df: df['connected'])
    old   = hyp >> (alpha <= 1)
    new   = hyp >> (alpha <= 1)
    # new is valid, but offers no strictly tighter value anywhere
    assert dalmatian_accept(new, [old], df) is False

def test_dalmatian_accept_two_conjectures_dataframe(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    beta = rd.Property('beta', lambda df: df['beta'])

    hyp = rd.Predicate('connected', lambda df: df['connected'])

    weak = hyp >> (alpha <= beta + 2)
    strong = hyp >> (alpha <= beta + 1)
    best = hyp >> (alpha <= beta)

    # decline the weaker conjecture
    assert dalmatian_accept(weak, [strong], df) == False

    # accept the better conjecture
    assert dalmatian_accept(best, [strong], df) == True

def test_multiple_old_bounds(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    hyp   = rd.Predicate('connected', lambda df: df['connected'])
    # two old bounds: β+2 and β+3 ⇒ their min is β+2
    old1  = hyp >> (alpha <= alpha + 2)
    old2  = hyp >> (alpha <= alpha + 3)
    # new bound α <= α+2 is not strictly tighter than min(old1,old2)=α+2
    assert dalmatian_accept(old1, [old1, old2], df) is False
    # but α <= α+1 is strictly tighter on every row
    better = hyp >> (alpha <= alpha + 1)
    assert dalmatian_accept(better, [old1, old2], df) is True

def test_unrelated_existing_bounds_are_ignored(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    beta  = rd.Property('beta',  lambda df: df['beta'])
    hyp1  = rd.Predicate('connected', lambda df: df['connected'])
    hyp2  = rd.Predicate('tree',      lambda df: df['tree'])
    # existing bound has a different hypothesis or different LHS
    old1 = hyp2 >> (alpha <= beta + 1)
    old2 = hyp1 >> (beta  <= alpha + 1)
    new1 = hyp1 >> (alpha <= beta)
    # neither old1 nor old2 shares both hyp AND same LHS, so new1 should be accepted
    assert dalmatian_accept(new1, [old1, old2], df) is True

def test_dalmatian_accept_two_conjectures_knowledgetable(kt):
    alpha = kt.alpha
    beta = kt.beta

    hyp = kt.connected

    weak = hyp >> (alpha <= beta + 2)
    strong = hyp >> (alpha <= beta + 1)
    best = hyp >> (alpha <= beta)

    # decline the weaker conjecture
    assert dalmatian_accept(weak, [strong], kt) == False

    # accept the better conjecture
    assert dalmatian_accept(best, [strong], kt) == True
