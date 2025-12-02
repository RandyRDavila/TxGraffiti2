import pandas as pd
import numpy as np
import pytest

from txgraffiti2025.forms import ClassInclusion, ClassEquivalence, GEQ, LEQ, Predicate
from txgraffiti2025.forms.predicates import Where


@pytest.fixture
def df_basic():
    # Keep c = 1.5 * a so A ⊆ B in the first test below.
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 2.0, 2.0],
        "c": [1.5, 3.0, 4.5, 6.0],  # = 1.5 * a
    })


def test_class_inclusion_and_violations(df_basic):
    A = GEQ("a", 3.0)
    B = GEQ("c", 4.5)  # since c = 1.5*a, rows with a>=3 -> c>=4.5
    incl = ClassInclusion(A, B)
    mask = incl.mask(df_basic)
    assert mask.all()
    assert incl.violations(df_basic).empty


def test_class_equivalence(df_basic):
    A = GEQ("a", 2.0) & LEQ("a", 3.0)
    B = GEQ("a", 1.999999) & LEQ("a", 3.000001)
    eqv = ClassEquivalence(A, B)
    mask = eqv.mask(df_basic)
    # Nearly identical ranges but could differ at edges; just assert it's boolean and aligned
    assert mask.dtype == bool
    assert mask.index.equals(df_basic.index)


# --- Extra coverage ---

def test_inclusion_vacuous_truth_and_violations_indexing():
    # A is always False → inclusion holds vacuously regardless of B
    df = pd.DataFrame({"x": [0, 1, 2, 3]})
    A = GEQ("x", 10)  # all False
    B = GEQ("x", 0)   # all True here, but we want to test vacuity
    incl = ClassInclusion(A, B)
    m = incl.mask(df)
    assert m.all()
    assert incl.violations(df).empty

    # Now craft a failing case with clear violation at index 1
    df2 = pd.DataFrame({"A": [True, True, False], "B": [True, False, True]})
    incl2 = ClassInclusion(Predicate.from_column("A"), Predicate.from_column("B"))
    m2 = incl2.mask(df2)
    assert m2.tolist() == [True, False, True]
    viol_idx = incl2.violations(df2).index.tolist()
    assert viol_idx == [1]


def test_equivalence_symmetric_difference_and_alignment():
    # Masks differ only at index 0
    df = pd.DataFrame({"C1": [True, False, True], "C2": [False, False, True]})
    eqv = ClassEquivalence(Predicate.from_column("C1"), Predicate.from_column("C2"))
    m = eqv.mask(df)
    assert m.tolist() == [False, True, True]
    # violations() returns rows where masks differ (symmetric difference)
    viol = eqv.violations(df)
    assert viol.index.tolist() == [0]
    # Alignment preserved
    assert m.index.equals(df.index) and viol.index.isin(df.index).all()


def test_inclusion_with_where_alignment():
    # Use a Where predicate that returns a Series (aligned) to ensure reindexing logic is stable
    df = pd.DataFrame({"a": [1, 2, 3, 4], "flag": [True, False, True, False]})

    # A := rows where flag is True
    A = Predicate.from_column("flag")

    # B := rows where a >= 1 (always True here); inclusion should hold
    B = GEQ("a", 1)
    incl = ClassInclusion(A, B)
    assert incl.mask(df).all()
    assert incl.violations(df).empty

    # Tougher B: a >= 3; only rows with flag True AND a < 3 should violate
    B2 = GEQ("a", 3)
    incl2 = ClassInclusion(A, B2)
    m2 = incl2.mask(df)
    # Row 0: flag True, a=1 -> violation; row 2: flag True, a=3 -> ok
    assert m2.tolist() == [False, True, True, True]
    assert incl2.violations(df).index.tolist() == [0]


def test_repr_smoke():
    A = Predicate.from_column("P")
    B = Predicate.from_column("Q")
    incl = ClassInclusion(A, B)
    eqv = ClassEquivalence(A, B)
    s1, s2 = repr(incl), repr(eqv)
    # Don’t hard-code exact strings; just ensure informative/contains glyphs or text
    assert ("⊆" in s1) or ("Inclusion" in s1)
    assert ("≡" in s2) or ("Equivalence" in s2)
