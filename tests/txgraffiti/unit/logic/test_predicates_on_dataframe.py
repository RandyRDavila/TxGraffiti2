import pandas as pd
import pytest
import functools

from txgraffiti.logic import Predicate, TRUE, FALSE

@pytest.fixture
def df():
    # three rows so we can test various Boolean combinations
    return pd.DataFrame({
        "A": [True, True, False],
        "B": [True, False, True],
        "C": [False, True, True],
    })

@pytest.fixture
def preds(df):
    # wrap each column as a Predicate
    return (
        Predicate("A", lambda df: df["A"]),
        Predicate("B", lambda df: df["B"]),
        Predicate("C", lambda df: df["C"]),
    )

def test_call_repr_eq(df):
    p = Predicate("A", lambda df: df["A"])
    # __call__
    assert p(df).tolist() == [True, True, False]
    # __repr__
    assert repr(p) == "<Predicate A>"
    # __eq__ & __hash__ by name only
    p2 = Predicate("A", lambda df: df["A"])
    assert p == p2
    assert hash(p) == hash(p2)
    assert p != Predicate("B", lambda df: df["B"])

def test_invert_constants_and_double_neg(df):
    # ~True → False, ~False → True
    assert (~TRUE) is FALSE
    assert (~FALSE) is TRUE
    # double-negation cancels
    A = Predicate("A", lambda df: df["A"])
    notA = ~A
    assert notA.name == "¬(A)"
    assert (~notA) is A

def test_and_complement(df):
    A = Predicate("A", lambda df: df["A"])
    notA = ~A
    # A ∧ ¬A → False
    assert (A & notA) is FALSE
    assert (notA & A) is FALSE

def test_and_absorption(df, preds):
    A, B, _ = preds
    orAB = A | B
    # A ∧ (A ∨ B) → A
    assert (A & orAB) is A
    assert (orAB & A) is A

def test_and_identity_constants(df):
    A = Predicate("A", lambda df: df["A"])
    # A ∧ True → A ; True ∧ A → A
    assert (A & TRUE) is A
    assert (TRUE & A) is A
    # A ∧ False → False ; False ∧ A → False
    assert (A & FALSE) is FALSE
    assert (FALSE & A) is FALSE

def test_and_idempotence(df):
    A = Predicate("A", lambda df: df["A"])
    assert (A & A) is A

def test_and_flattening(df, preds):
    A, B, C = preds
    combined = (A & B) & C
    # terms stored in _and_terms
    terms = getattr(combined, "_and_terms")
    assert terms == [A, B, C]
    # name flattened
    assert combined.name == "(A) ∧ (B) ∧ (C)"
    # evaluation matches bitwise-and
    expected = [a and b and c for a,b,c in zip(df["A"], df["B"], df["C"])]
    assert combined(df).tolist() == expected

def test_or_complement(df):
    A = Predicate("A", lambda df: df["A"])
    notA = ~A
    # A ∨ ¬A → True
    assert (A | notA) is TRUE
    assert (notA | A) is TRUE

def test_or_absorption(df, preds):
    A, B, _ = preds
    andAB = A & B
    # A ∨ (A ∧ B) → A
    assert (A | andAB) is A
    assert (andAB | A) is A

def test_or_identity_constants(df):
    A = Predicate("A", lambda df: df["A"])
    # A ∨ False → A ; False ∨ A → A
    assert (A | FALSE) is A
    assert (FALSE | A) is A
    # A ∨ True → True ; True ∨ A → True
    assert (A | TRUE) is TRUE
    assert (TRUE | A) is TRUE

def test_or_idempotence(df):
    A = Predicate("A", lambda df: df["A"])
    assert (A | A) is A

def test_or_flattening(df, preds):
    A, B, C = preds
    combined = (A | B) | C
    terms = getattr(combined, "_or_terms")
    assert terms == [A, B, C]
    assert combined.name == "(A) ∨ (B) ∨ (C)"
    expected = [a or b or c for a,b,c in zip(df["A"], df["B"], df["C"])]
    assert combined(df).tolist() == expected

def test_xor_rules_and_default(df):
    A = Predicate("A", lambda df: df["A"])
    B = Predicate("B", lambda df: df["B"])
    notA = ~A

    # A ⊕ ¬A → True
    assert (A ^ notA) is TRUE
    assert (notA ^ A) is TRUE

    # A ⊕ A → False
    assert (A ^ A) is FALSE

    # A ⊕ False → A ; False ⊕ A → A
    assert (A ^ FALSE) is A
    assert (FALSE ^ A) is A

    # A ⊕ True → ¬A ; True ⊕ A → ¬A
    assert (A ^ TRUE) == ~A
    assert (TRUE ^ A) == ~A

    # default XOR builds new predicate
    xorAB = A ^ B
    assert isinstance(xorAB, Predicate)
    assert xorAB.name == "(A) ⊕ (B)"
    # correct truth‐table
    expected = [a ^ b for a,b in zip(df["A"], df["B"])]
    assert xorAB(df).tolist() == expected

def test_implies(df):
    A = Predicate("A", lambda df: df["A"])
    B = Predicate("B", lambda df: df["B"])
    impl = A.implies(B)
    # name
    assert impl.name == "(A → B)"
    # evaluation: (~A) | B
    expected = [ (not a) or b for a,b in zip(df["A"], df["B"]) ]
    assert impl(df).tolist() == expected
