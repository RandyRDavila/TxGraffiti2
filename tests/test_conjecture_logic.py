import pandas as pd
import pytest

from txgraffiti2.conjecture_logic import Property, Predicate, Inequality, Conjecture

# ——————— Fixtures ———————
@pytest.fixture
def df():
    # simple DataFrame with two rows
    return pd.DataFrame({
        'alpha': [1, 2, 3],
        'beta': [3, 1, 1],
        'connected': [True, True, True],
        'K_n': [True, False, False],
        'tree': [False, False, True]
    })

# ——————— Property tests ———————
def test_property_basic(df):
    p = Property('alpha', lambda df: df['alpha'])
    assert p(df).tolist() == [1, 2, 3]
    assert repr(p) == "<Property alpha>"

def test_property_arithmetic(df):
    p = Property('alpha', lambda df: df['alpha'])
    q = Property('beta', lambda df: df['beta'])
    # addition
    r = p + q
    assert isinstance(r, Property)
    assert r.name == "(alpha + beta)"
    assert r(df).tolist() == [4, 3, 4]
    # scalar lift & mul
    s = p * 3
    assert s.name == "(alpha * 3)"
    assert s(df).tolist() == [3, 6, 9]

def test_property_identities(df):
    p = Property('alpha', lambda df: df['alpha'])
    zero = Property('0', lambda df: pd.Series(0, index=df.index))
    one = Property('1', lambda df: pd.Series(1, index=df.index))
    #  p + 0 → p
    assert (p + zero) is p
    # p * 1 → p
    assert (p * one) is p
    # p * 0 → zero
    got = p * zero
    assert got.name == "0"
    assert got(df).tolist() == [0, 0, 0]


def test_boolean_property(df):
    # boolean property based on a column
    p = Property('connected', lambda df: df['connected'])
    assert p(df).tolist() == [True, True, True]
    assert repr(p) == "<Property connected>"

    # boolean property with a constant
    q = Property('K_n', lambda df: df['K_n'])
    assert q(df).tolist() == [True, False, False]
    assert repr(q) == "<Property K_n>"

# ——————— Predicate tests ———————
def test_boolean_arithmetic(df):
    p = Predicate('connected', lambda df: df['connected'])
    q = Predicate('K_n', lambda df: df['K_n'])
    # AND
    r = p & q
    assert r.name == "(connected) ∧ (K_n)"
    assert r(df).tolist() == [True, False, False]
    # OR
    s = p | q
    assert s.name == "(connected) ∨ (K_n)"
    assert s(df).tolist() == [True, True, True]
    # NOT
    t = ~p
    assert t.name == "¬(connected)"
    assert t(df).tolist() == [False, False, False]


# ——————— Inequality tests ———————
def test_inequality_basic(df):
    a = Property('alpha', lambda df: df['alpha'])
    three = Property('3', lambda df: pd.Series(3, index=df.index))
    ineq = a <= three
    # name and repr
    assert repr(ineq) == "<Ineq alpha <= 3>"
    # slack: 3 - a
    slack = ineq.slack(df)
    assert slack.tolist() == [2, 1, 0]
    # touch_count: none equal
    assert ineq.touch_count(df) == 1

def test_inequality_counterexample(df):
    # test evaluation of the underlying predicate
    a = Property('alpha', lambda df: df['alpha'])
    b = Property('beta', lambda df: df['beta'])
    ineq = a * 11 == b
    # only second row (2*11 == 22) is true
    mask = ineq(df).tolist()
    assert mask == [False, False, False]

# ——————— Conjecture tests ———————
def test_conjecture_true_and_false(df):
    # true conjecture: hypothesis implies conclusion
    hyp = Predicate('connected', lambda df: df['connected'])
    con = Predicate('beta>alpha', lambda df: df['beta'] > df['alpha'])
    conj = Conjecture(hypothesis=hyp, conclusion=con)
    # evaluate on df
    result = conj.evaluate(df)
    assert result.tolist() == [True, False, False]
