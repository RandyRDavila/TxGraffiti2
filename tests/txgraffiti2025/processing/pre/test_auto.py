import pandas as pd
import numpy as np
import pytest

from txgraffiti2025.forms import Le, Ge, Predicate
from txgraffiti2025.processing.pre.auto import AutoConjecture, AutoImplication, AutoEquivalence


def test_autoconjecture_uses_detected_base_when_none():
    # Base columns are fully True → auto-base should be (connected) ∧ (simple)
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 3.0, 5.0],
        "connected": [True, True, True, True],
        "simple": [True, True, True, True],
    })
    C = AutoConjecture(Le("a", "b"), condition=None)

    # condition starts None and must remain None after check (no mutation)
    assert C.condition is None
    applicable, holds, failures = C.check(df)
    assert C.condition is None

    # auto-base => all rows applicable here
    assert applicable.all()
    assert holds.all()
    assert failures.empty


def test_autoconjecture_respects_explicit_condition():
    df = pd.DataFrame({
        "a": [1.0, 3.0, 2.0, 4.0],
        "b": [2.0, 1.0, 2.0, 3.0],
        "flag": [True, False, True, False],
        "connected": [True, True, True, True],  # even though this is True, explicit flag wins
    })
    explicit = Predicate.from_column("flag")
    C = AutoConjecture(Le("a", "b"), condition=explicit)

    applicable, holds, failures = C.check(df)
    # Only rows with flag True are applicable
    assert applicable.tolist() == [True, False, True, False]
    # On applicable rows, a <= b in rows 0 and 2 → holds True
    assert holds[applicable].all()
    assert failures.empty


def test_autoimplication_with_autobase_and_no_mutation():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, 1, 3, 5],
        "connected": [True, True, True, True],
    })
    # Premise: x <= y ; Conclusion: y >= x (logically equivalent)
    I = AutoImplication(Le("x", "y"), Ge("y", "x"), condition=None)

    assert I.condition is None
    applicable, holds, failures = I.check(df)
    assert I.condition is None  # still None after check
    assert applicable.all()
    assert holds.all()
    assert failures.empty


def test_autoequivalence_with_autobase_and_explicit_condition():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [1, 3, 2, 5],
        "mask": [True, True, False, False],
        "connected": [True, True, True, True],
    })
    # On rows where mask True, Le(a,b) equals Ge(b,a) for rows 0..1
    cond = Predicate.from_column("mask")
    E = AutoEquivalence(Le("a", "b"), Ge("b", "a"), condition=cond)

    applicable, holds, failures = E.check(df)
    # Applicability follows explicit mask
    assert applicable.tolist() == [True, True, False, False]
    # For those applicable rows, a<=b equals b>=a → equivalence holds
    assert holds[applicable].all()
    assert failures.empty


def test_autobase_detects_TRUE_when_no_boolean_columns():
    df = pd.DataFrame({
        "alpha": [1.0, 2.0, 3.0],
        "beta": [2.0, 3.0, 4.0],
    })
    C = AutoConjecture(Le("alpha", "beta"), condition=None)
    applicable, holds, failures = C.check(df)
    # With no boolean columns, auto-base → TRUE (all applicable)
    assert applicable.all()
    assert holds.all()
    assert failures.empty
