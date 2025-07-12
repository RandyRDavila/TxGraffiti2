import re

import pandas as pd
import pytest

import txgraffiti as rd
from txgraffiti.export_utils.lean4 import (
    auto_var_map,
    conjecture_to_lean4,
    LEAN_SYMBOLS,
)

# ————— Fixtures —————

@pytest.fixture
def df():
    return pd.DataFrame({
        'alpha':     [1, 2],
        'beta':      [2, 1],
        'connected': [True, False],
        'tree':      [False, True],
        'name':      ['x', 'y'],
    })

# ————— auto_var_map tests —————

def test_auto_var_map_default_skip(df):
    vm = auto_var_map(df)
    # 'name' must be skipped by default
    assert 'name' not in vm
    # all other columns map to "<col> G"
    for col in ('alpha','beta','connected','tree'):
        assert vm[col] == f"{col} G"

def test_auto_var_map_custom_skip(df):
    vm = auto_var_map(df, skip=('alpha','connected'))
    assert 'alpha' not in vm
    assert 'connected' not in vm
    # 'beta' and 'tree' remain
    assert vm['beta'] == "beta G"
    assert vm['tree'] == "tree G"


# ————— conjecture_to_lean4 basic tests —————

def test_conjecture_to_lean4_single_hypothesis(df):
    alpha = rd.Property('alpha',   lambda df: df['alpha'])
    beta  = rd.Property('beta',    lambda df: df['beta'])
    hyp   = rd.Predicate('connected', lambda df: df['connected'])

    conj  = hyp >> (alpha <= beta)
    out   = conjecture_to_lean4(conj, name="C1")

    expected = (
        "theorem C1 (G : SimpleGraph V)\n"
        "    (h1 : connected G) : alpha G ≤ beta G :=\n"
        "sorry \n"
    )
    assert out == expected

def test_conjecture_to_lean4_multiple_hypotheses_and_custom(df):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    beta  = rd.Property('beta',  lambda df: df['beta'])
    hyp1  = rd.Predicate('connected', lambda df: df['connected'])
    hyp2  = rd.Predicate('tree',      lambda df: df['tree'])

    conj = (hyp1 & hyp2) >> (alpha == beta)
    out  = conjecture_to_lean4(
        conj,
        name="C2",
        object_symbol="X",
        object_decl="MyGraphType"
    )

    # theorem header
    assert out.startswith("theorem C2 (X : MyGraphType)\n")
    # both hypotheses appear in order
    assert "(h1 : connected X)" in out
    assert "(h2 : tree X)"      in out
    # conclusion uses '=' for '=='
    assert re.search(r"alpha X = beta X", out)
    # ends with a `sorry`
    assert out.rstrip().endswith("sorry")

# ————— operator‐mapping tests —————

@pytest.mark.parametrize("op,lean_sym", [
    ("<=", "≤"),
    (">=", "≥"),
    ("<",  "<"),
    (">",  ">"),
    ("==", "="),
    ("!=", "≠"),
])
def test_all_relational_ops_map_correctly(df, op, lean_sym):
    alpha = rd.Property('alpha',   lambda df: df['alpha'])
    beta  = rd.Property('beta',    lambda df: df['beta'])
    hyp   = rd.Predicate('connected', lambda df: df['connected'])

    # build the inequality dynamically
    ineq = eval(f"alpha {op} beta")   # e.g. alpha <= beta
    conj = hyp >> ineq
    out  = conjecture_to_lean4(conj, name="OpTest")

    # check that the lean_sym appears exactly once between alpha G and beta G
    assert re.search(rf"alpha G {re.escape(lean_sym)} beta G", out)
