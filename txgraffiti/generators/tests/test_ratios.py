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

    # # check constants
    # c = extract_constants(low,  df_simple, 'beta')
    # C = extract_constants(high, df_simple, 'beta')
    # # manual ratios: alpha/beta over connected rows: [1/3, 2/1, 3/1] → c=1/3, C=3
    # assert pytest.approx(c, rel=1e-6) == 1/3
    # assert pytest.approx(C, rel=1e-6) == 3.0

# # ——————— Multiple features ———————
# def test_ratios_multi_feature(df_simple):
#     alpha = rd.Property('alpha', lambda df: df['alpha'])
#     beta  = rd.Property('beta',  lambda df: df['beta'])
#     gamma = rd.Property('gamma', lambda df: df['gamma'])
#     H     = rd.Predicate('connected', lambda df: df['connected'])

#     gens = list(ratios(df_simple,
#                        features=[beta, gamma],
#                        target=alpha,
#                        hypothesis=H))
#     # 2 conjectures per feature
#     assert len(gens) == 4

#     # group by feature name
#     by_feat = {}
#     for conj in gens:
#         fname = conj.conclusion.rhs.name
#         by_feat.setdefault(fname, []).append(conj)

#     # each feature got exactly 2
#     assert set(by_feat) == {'beta', 'gamma'}
#     assert all(len(lst) == 2 for lst in by_feat.values())

# # ——————— Division by zero: skip or error? ———————
# def test_ratios_skip_zero_feature(df_simple):
#     # add a row where beta==0 under H
#     df = df_simple.copy()
#     df.loc[0, 'beta'] = 0
#     alpha = rd.Property('alpha', lambda df: df['alpha'])
#     beta  = rd.Property('beta',  lambda df: df['beta'])
#     H     = rd.Predicate('connected', lambda df: df['connected'])

#     # if your implementation skips rows with F==0, we still get 2 conjectures
#     gens = list(ratios(df, features=[beta], target=alpha, hypothesis=H))
#     assert len(gens) == 2
#     # and they must hold on the remaining rows
#     assert all(c.is_true(df) for c in gens)

# # ——————— Works on KnowledgeTable subclass ———————
# @pytest.mark.parametrize("table", ("df_simple", "kt_simple"))
# def test_ratios_on_both(table, df_simple, kt_simple):
#     T = df_simple if table == "df_simple" else kt_simple
#     alpha = rd.Property('alpha', lambda df: df['alpha']) if table == "df_simple" else T.alpha
#     beta  = rd.Property('beta',  lambda df: df['beta'])  if table == "df_simple" else T.beta
#     H     = rd.Predicate('connected', lambda df: df['connected']) if table == "df_simple" else T.connected

#     gens = list(ratios(T, features=[beta], target=alpha, hypothesis=H))
#     assert len(gens) == 2
#     # sanity check of validity
#     assert all(c.is_true(T) for c in gens)
