import numpy as np
import pandas as pd
from txgraffiti2025.forms import Eq, Le, Ge, AllOf, AnyOf, Conjecture, GEQ, LEQ

def test_eq_le_ge(df_basic):
    assert Eq("a", "a").evaluate(df_basic).all()
    assert Le("a", "c").evaluate(df_basic).all()
    assert Ge("c", "a").evaluate(df_basic).all()

def test_slacks_sign_convention(df_basic):
    # Le: slack = right - left (>=0 when satisfied)
    le = Le("a", "c")
    s = le.slack(df_basic)
    assert (s >= 0).all()
    # Ge: slack = left - right (>=0 when satisfied)
    ge = Ge("c", "a")
    s2 = ge.slack(df_basic)
    assert (s2 >= 0).all()
    # Eq: negative absolute error
    eq = Eq("a", "a")
    s3 = eq.slack(df_basic)
    assert (s3 == 0).all()

def test_allof_anyof(df_basic):
    r1 = Le("a", "c")
    r2 = Ge("b", 1.0)
    assert AllOf([r1, r2]).evaluate(df_basic).all()
    # AnyOf: at least one holds
    r3 = Ge("a", 10.0)
    r4 = Le("b", 2.0)
    anymask = AnyOf([r3, r4]).evaluate(df_basic)
    assert anymask.all()

def test_conjecture_R_given_C(df_basic):
    C = GEQ("a", 2.0) & LEQ("a", 3.0)  # rows 2..3
    R = Le("a", "c")
    conj = Conjecture(R, C)
    applicable, holds, failures = conj.check(df_basic)
    assert applicable.sum() == 2
    assert holds.all()
    assert failures.empty
