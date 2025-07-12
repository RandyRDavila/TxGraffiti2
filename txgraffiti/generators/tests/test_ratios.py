import pandas as pd
import pytest

import txgraffiti as rd
from txgraffiti.generators import ratios

# ——————— Fixtures ———————
@pytest.fixture
def df_simple():
    return pd.DataFrame({
        'alpha':     [1, 2, 3],
        'beta':      [3, 1, 1],
        'gamma':     [2, 4, 2],
        'connected': [True, True, True],
        'tree':      [False, False, True],
    })

@pytest.fixture
def kt_simple(df_simple):
    return rd.KnowledgeTable(df_simple)

# ——————— Helper to extract the two constants  ———————
def extract_constants(conj, df, feature):
    # evaluate T(x)/F(x) on all rows where H holds
    H = conj.hypothesis(df)
    T = conj.conclusion.lhs(df)[H]
    F = conj.conclusion.rhs(df)[H]  # because rhs is c*F or C*F
    # the constant is ratio of T/F (they should all agree)
    ratios = (T / (F / df[feature][H]))
    # return the unique scalar
    val = ratios.unique()
    assert len(val) == 1
    return float(val[0])

# ——————— Basic functionality ———————
def test_ratios_basic(df_simple):
    alpha = rd.Property('alpha', lambda df: df['alpha'])
    beta  = rd.Property('beta',  lambda df: df['beta'])
    H     = rd.Predicate('connected', lambda df: df['connected'])

    gens = list(ratios(df_simple,
                       features=[beta],
                       target=alpha,
                       hypothesis=H))
    # exactly two conjectures: lower-bound then upper-bound
    assert len(gens) == 2
    low, high = gens

    # check ops
    assert low.conclusion.op  in (">=", "≥")
    assert high.conclusion.op in ("<=", "≤")

    # check they hold on all rows
    assert low.is_true(df_simple)
    assert high.is_true(df_simple)
