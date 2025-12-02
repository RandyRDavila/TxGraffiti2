
import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms import Le, Ge, Eq, Implication, Equivalence, GEQ
from txgraffiti2025.forms.predicates import Predicate

def test_implication_basic(df_basic):
    # premise: a <= c  (true), conclusion: b >= 2  (true)
    imp = Implication(Le("a","c"), Ge("b", 2.0))
    applicable, holds, failures = imp.check(df_basic)
    assert applicable.all()
    assert holds.all()
    assert failures.empty

def test_implication_with_condition_and_failure(df_basic):
    C = GEQ("a", 3.0)  # last two rows
    imp = Implication(Ge("a", 3.0), Ge("b", 3.0), condition=C)  # b>=3 is false
    applicable, holds, failures = imp.check(df_basic)
    assert applicable.sum() == 2
    assert (~holds).sum() == 2
    assert len(failures) == 2
    assert "__slack__" in failures.columns

def test_equivalence(df_basic):
    eqv = Equivalence(Le("a","c"), Ge("c","a"))
    applicable, holds, failures = eqv.check(df_basic)
    assert applicable.all()
    assert holds.all()
    assert failures.empty


def test_implication_vacuous_truth_outside_condition():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 1, 3, 5], "flag": [False, False, True, False]})
    R1 = Le("a", "b")    # [True, False, True, True]
    R2 = Ge("b", "a")    # [True, True,  True, True]
    C  = Predicate.from_column("flag")  # only row 2 applicable
    impl = Implication(R1, R2, condition=C)
    applicable, holds, failures = impl.check(df)
    # Only row 2 is applicable; implication holds everywhere (vacuous outside C)
    assert applicable.tolist() == [False, False, True, False]
    assert holds.all()
    assert failures.empty

def test_implication_failures_slack_is_from_conclusion(df_basic):
    # Make a failing case under condition: premise True, conclusion False
    # Assume df_basic['b'] < 10 everywhere to force failure of Ge(b, 10)
    C = GEQ("a", 1.0)  # all rows applicable
    imp = Implication(Ge("a", 1.0), Ge("b", 10.0), condition=C)
    applicable, holds, failures = imp.check(df_basic)
    # All applicable, some failures expected if b<10
    assert applicable.all()
    assert (~holds).any()
    assert not failures.empty
    # Slack column exists and matches conclusion slack (b - 10)
    assert "__slack__" in failures.columns
    slack_expected = (df_basic["b"] - 10.0).loc[failures.index]
    np.testing.assert_allclose(failures["__slack__"].to_numpy(), slack_expected.to_numpy())

def test_equivalence_with_condition_subset():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 3, 2, 4], "use": [True, True, False, False]})
    # Within the applicable subset (rows 0-1), Le(x,y) == Ge(y,x)
    eqv = Equivalence(Le("x","y"), Ge("y","x"), condition=Predicate.from_column("use"))
    applicable, holds, failures = eqv.check(df)
    assert applicable.tolist() == [True, True, False, False]
    assert holds.all()
    assert failures.empty

def test_implication_and_equivalence_repr_smoke(df_basic):
    imp = Implication(Le("a","c"), Ge("c","a"))
    eqv = Equivalence(Le("a","c"), Ge("c","a"))
    s1, s2 = repr(imp), repr(eqv)
    # Avoid brittle exact-match; just ensure informative content
    assert ("⇒" in s1) or ("Implication" in s1)
    assert ("⇔" in s2) or ("Equivalence" in s2)
