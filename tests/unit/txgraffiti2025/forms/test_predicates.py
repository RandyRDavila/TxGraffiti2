import numpy as np
import pandas as pd
import pytest
from txgraffiti2025.forms import (
    Predicate, LE, GE, LT, GT, EQ, NE,
    InSet, Between, IsInteger, IsNaN, IsFinite,
    Where, RowWhere, GEQ, LEQ, GT0, LT0, EQ0, BETWEEN, IN, IS_INT, IS_NAN, IS_FINITE,
    to_expr
)

def test_compare_ops(df_basic):
    assert GE("a", "b").mask(df_basic).equals((df_basic["a"] >= df_basic["b"]))
    assert LE("a", 3.0).mask(df_basic).equals((df_basic["a"] <= 3.0))
    assert LT("a", "c").mask(df_basic).equals((df_basic["a"] < df_basic["c"]))
    assert GT("b", 1).mask(df_basic).equals((df_basic["b"] > 1))
    assert EQ("a", to_expr("a")).mask(df_basic).all()
    assert NE("a", "b").mask(df_basic).equals((df_basic["a"] != df_basic["b"]))

def test_eq_with_tolerance(df_basic):
    eq = EQ(to_expr("a") + 1e-10, "a", tol=1e-9)
    assert eq.mask(df_basic).all()

def test_inset_and_between(df_basic):
    mask_in = InSet("cat", {"x", "z"}).mask(df_basic)
    assert set(df_basic["cat"][mask_in]) <= {"x","z"}
    mask_b = Between("a", 2.0, 4.0).mask(df_basic)
    assert mask_b.to_numpy().tolist() == [False, True, True, True]

def test_numeric_checks(df_basic):
    assert IsFinite("a").mask(df_basic).all()
    assert IsNaN("a").mask(df_basic).sum() == 0
    # integer check against float column
    assert IsInteger(to_expr("n")).mask(df_basic).all()

def test_where_and_rowwhere(df_basic):
    m1 = Where(lambda df: (df["a"] + df["b"]) > 3).mask(df_basic)
    m2 = RowWhere(lambda row: (row["a"] + row["b"]) > 3).mask(df_basic)
    assert m1.equals(m2)

def test_shorthands(df_basic):
    assert GEQ("a", 1).mask(df_basic).all()
    assert LEQ("b", 2).mask(df_basic).all()
    assert GT0(to_expr("a") - 0.5).mask(df_basic).sum() == 4
    assert LT0(0 - to_expr("a")).mask(df_basic).sum() == 4
    assert EQ0(to_expr("a") - "a").mask(df_basic).all()
    assert BETWEEN("a", 2, 4).mask(df_basic).sum() == 3
    assert IN("cat", ["x", "y"]).mask(df_basic).sum() == 3
    assert IS_INT("n").mask(df_basic).all()
    assert IS_NAN("cat").mask(df_basic).sum() == 0
    assert IS_FINITE("a").mask(df_basic).all()
