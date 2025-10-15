import pandas as pd
from txgraffiti2025.forms import Le, Ge, Eq, Implication, Equivalence, GEQ

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
