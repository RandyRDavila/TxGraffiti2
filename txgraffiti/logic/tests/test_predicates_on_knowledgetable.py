import pandas as pd
import pytest
import functools

from txgraffiti.logic import KnowledgeTable, Predicate, TRUE, FALSE

@pytest.fixture
def kt():
    # three rows so we can test various Boolean combinations
    return KnowledgeTable({
        "A": [True, True, False],
        "B": [True, False, True],
        "C": [False, True, True],
    })

@pytest.fixture
def preds(kt):
    # wrap each column as a Predicate
    return (
        kt.A,
        kt.B,
        kt.C,
    )

def test_call_repr_eq(kt):
    p = kt.A
    # __call__
    assert p(kt).tolist() == [True, True, False]
    # __repr__
    assert repr(p) == "<Predicate A>"
    # __eq__ & __hash__ by name only
    p2 = kt.A
    assert p == p2
    assert hash(p) == hash(p2)
    assert p != kt.B

def test_invert_constants_and_double_neg(kt):
    # ~True → False, ~False → True
    assert (~TRUE) is FALSE
    assert (~FALSE) is TRUE
    # double-negation cancels
    A = kt.A
    notA = ~A
    assert notA.name == "¬(A)"
    assert (~notA) is A

def test_and_complement(kt):
    A = kt.A
    notA = ~A
    # A ∧ ¬A → False
    assert (A & notA) is FALSE
    assert (notA & A) is FALSE

def test_and_absorption(kt, preds):
    A, B, _ = preds
    orAB = A | B
    # A ∧ (A ∨ B) → A
    assert (A & orAB) is A
    assert (orAB & A) is A

def test_and_identity_constants(kt):
    A = kt.A
    # A ∧ True → A ; True ∧ A → A
    assert (A & TRUE) is A
    assert (TRUE & A) is A
    # A ∧ False → False ; False ∧ A → False
    assert (A & FALSE) is FALSE
    assert (FALSE & A) is FALSE

def test_and_idempotence(kt):
    A = kt.A
    assert (A & A) is A

def test_and_flattening(kt, preds):
    A, B, C = preds
    combined = (A & B) & C
    # terms stored in _and_terms
    terms = getattr(combined, "_and_terms")
    assert terms == [A, B, C]
    # name flattened
    assert combined.name == "(A) ∧ (B) ∧ (C)"
    # evaluation matches bitwise-and
    expected = [a and b and c for a,b,c in zip(kt["A"], kt["B"], kt["C"])]
    assert combined(kt).tolist() == expected

def test_or_complement(kt):
    A = kt.A
    notA = ~A
    # A ∨ ¬A → True
    assert (A | notA) is TRUE
    assert (notA | A) is TRUE

def test_or_absorption(kt, preds):
    A, B, _ = preds
    andAB = A & B
    # A ∨ (A ∧ B) → A
    assert (A | andAB) is A
    assert (andAB | A) is A

def test_or_identity_constants(kt):
    A = kt.A
    # A ∨ False → A ; False ∨ A → A
    assert (A | FALSE) is A
    assert (FALSE | A) is A
    # A ∨ True → True ; True ∨ A → True
    assert (A | TRUE) is TRUE
    assert (TRUE | A) is TRUE

def test_or_idempotence(kt):
    A = kt.A
    assert (A | A) is A

def test_or_flattening(kt, preds):
    A, B, C = preds
    combined = (A | B) | C
    terms = getattr(combined, "_or_terms")
    assert terms == [A, B, C]
    assert combined.name == "(A) ∨ (B) ∨ (C)"
    expected = [a or b or c for a,b,c in zip(kt["A"], kt["B"], kt["C"])]
    assert combined(kt).tolist() == expected

def test_xor_rules_and_default(kt):
    A = kt.A
    B = kt.B
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
    expected = [a ^ b for a,b in zip(kt["A"], kt["B"])]
    assert xorAB(kt).tolist() == expected

def test_implies(kt):
    A = kt.A
    B = kt.B
    impl = A.implies(B)
    # name
    assert impl.name == "(A → B)"
    # evaluation: (~A) | B
    expected = [ (not a) or b for a,b in zip(kt["A"], kt["B"]) ]
    assert impl(kt).tolist() == expected
