import pandas as pd
import pytest

from txgraffiti.logic import Property, Predicate

# ——————— Fixtures ———————
@pytest.fixture
def df():
    # simple DataFrame with two rows
    return pd.DataFrame({
        'alpha': [1, 2, 3],
        'beta': [3, 1, 1],
        'connected': [True, True, True],
        'K_n': [True, False, False],
        'tree': [False, False, True],
    })

@pytest.fixture
def polytope_df():
    return pd.DataFrame({
        'is_simple': [True, True, False],
        'p3': [0, 4, 4],
        'p4': [6, 0, 1],
        'f0': [8, 4, 5],
        'f1': [12, 6, 8],
    })

def test_polytope_property_basic(polytope_df):
    # basic property on polytope DataFrame
    p = Property('is_simple', lambda df: df['is_simple'])
    assert p(polytope_df).tolist() == [True, True, False]
    assert repr(p) == "<Property is_simple>"

    q = Property('p3 > 0', lambda df: df['p3'] > 0)
    assert q(polytope_df).tolist() == [False, True, True]
    assert repr(q) == "<Property p3 > 0>"

def test_polytope_property_arithmetic(polytope_df):
    # arithmetic operations on properties
    p = Property('p3', lambda df: df['p3'])
    q = Property('p4', lambda df: df['p4'])
    # addition
    r = p + q
    assert isinstance(r, Property)
    assert r.name == "(p3 + p4)"
    assert r(polytope_df).tolist() == [6, 4, 5]
    # scalar multiplication
    s = p * 2
    assert s.name == "(2 * p3)"
    assert s(polytope_df).tolist() == [0, 8, 8]
    # TODO: need division checks

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
    assert s.name == "(3 * alpha)"
    assert s(df).tolist() == [3, 6, 9]

def test_property_identities(df):
    p = Property('alpha', lambda df: df['alpha'])
    zero = Property('0', lambda df: pd.Series(0, index=df.index))
    one = Property('1', lambda df: pd.Series(1, index=df.index))
    #  p + 0 → p
    assert (p + zero) is p
    # p - 0 → p
    assert (p - zero) is p
    # p * 1 → p
    assert (p * one) is p
    # p * 0 → zero
    got = p * zero
    assert got.name == "0"
    assert got(df).tolist() == [0, 0, 0]
    # TODO: add division by zero

def test_boolean_property(df):
    # boolean property based on a column
    p = Property('connected', lambda df: df['connected'])
    assert p(df).tolist() == [True, True, True]
    assert repr(p) == "<Property connected>"

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
    # TODO: needs a test for demorgan's law

# ——————— Inequality tests ———————
def test_inequality_basic(df):
    a = Property('alpha', lambda df: df['alpha'])
    three = Property('3', lambda df: pd.Series(3, index=df.index))
    ineq = a <= three
    # name and repr
    assert repr(ineq) == "<Predicate alpha <= 3>"
    # slack: 3 - a
    slack = ineq.slack(df)
    assert slack.tolist() == [2, 1, 0]
    # touch_count: one equals
    assert ineq.touch_count(df) == 1

def test_inequality_counterexample(df):
    # test evaluation of the underlying predicate
    a = Property('alpha', lambda df: df['alpha'])
    b = Property('beta', lambda df: df['beta'])
    ineq = a * 11 == b
    mask = ineq(df).tolist()
    assert mask == [False, False, False]

# ——————— Conjecture tests ———————
def test_conjecture_true_and_false(df):
    # true conjecture on one row: hypothesis implies conclusion
    hypothesis = Predicate('connected', lambda df: df['connected'])
    conclusion = Predicate('beta>alpha', lambda df: df['beta'] > df['alpha'])
    conj1 = hypothesis.implies(conclusion, as_conjecture=True)
    # evaluate on df
    result = conj1(df)
    assert result.tolist() == [True, False, False]

    conj2 = hypothesis >> conclusion
    # should be equivalent to conj1
    assert conj1 == conj2

def test_conjecture_true(df):
    # true conjecture on one row: hypothesis implies conclusion
    hypothesis = Predicate('K_n', lambda df: df['K_n'])
    conclusion = Predicate('beta=3*alpha', lambda df: df['beta'] == 3*df['alpha'])
    conj = hypothesis.implies(conclusion, as_conjecture=True)
    # evaluate on df
    result = conj(df)
    assert result.tolist() == [True, True, True]

# ——————— DeMorgan's Laws test ———————
def test_demorgans_law(df):
    # DeMorgan's Law: ~(p & q) == ~p | ~q
    p = Predicate('connected', lambda df: df['connected'])
    q = Predicate('K_n', lambda df: df['K_n'])
    left = ~(p & q)
    right = ~p | ~q
    assert left.name == "¬((connected) ∧ (K_n))"
    assert right.name == "(¬(connected)) ∨ (¬(K_n))"
    assert left(df).tolist() == right(df).tolist()
