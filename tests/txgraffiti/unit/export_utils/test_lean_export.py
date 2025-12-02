# # tests/unit/export_utils/test_lean_export.py
# import re
# from fractions import Fraction

# import pandas as pd
# import pytest

# import txgraffiti as rd
# from txgraffiti.export_utils import (
#     auto_var_map,
#     conjecture_to_lean4,
#     necessary_conjecture_to_lean,
# )

# # ————— Fixtures —————

# @pytest.fixture
# def df():
#     return pd.DataFrame({
#         'alpha':     [1, 2],
#         'beta':      [2, 1],
#         'connected': [True, False],
#         'tree':      [False, True],
#         'name':      ['x', 'y'],
#     })

# # ————— auto_var_map tests —————

# def test_auto_var_map_default_skip(df):
#     vm = auto_var_map(df)
#     assert 'name' not in vm
#     for col in ('alpha','beta','connected','tree'):
#         assert vm[col] == f"{col} G"

# def test_auto_var_map_custom_skip(df):
#     vm = auto_var_map(df, skip=('alpha','connected'))
#     assert 'alpha' not in vm
#     assert 'connected' not in vm
#     assert vm['beta'] == "beta G"
#     assert vm['tree'] == "tree G"

# # ————— conjecture_to_lean4 basic tests —————

# def test_conjecture_to_lean4_single_hypothesis(df):
#     alpha = rd.Property('alpha',   lambda df: df['alpha'])
#     beta  = rd.Property('beta',    lambda df: df['beta'])
#     hyp   = rd.Predicate('connected', lambda df: df['connected'])

#     conj  = hyp >> (alpha <= beta)
#     out   = conjecture_to_lean4(conj, name="C1")

#     expected = (
#         "theorem C1 (G : SimpleGraph V)\n"
#         "    (h1 : connected G) : alpha G ≤ beta G :=\n"
#         "sorry\n"
#     )
#     assert out == expected

# def test_conjecture_to_lean4_multiple_hypotheses_and_custom(df):
#     alpha = rd.Property('alpha', lambda df: df['alpha'])
#     beta  = rd.Property('beta',  lambda df: df['beta'])
#     hyp1  = rd.Predicate('connected', lambda df: df['connected'])
#     hyp2  = rd.Predicate('tree',      lambda df: df['tree'])

#     conj = (hyp1 & hyp2) >> (alpha == beta)
#     out  = conjecture_to_lean4(
#         conj,
#         name="C2",
#         object_symbol="X",
#         object_decl="MyGraphType"
#     )

#     assert out.startswith("theorem C2 (X : MyGraphType)\n")
#     assert "(h1 : connected X)" in out
#     assert "(h2 : tree X)"      in out
#     assert re.search(r"alpha X = beta X", out)
#     assert out.endswith("sorry\n")

# # ————— operator‐mapping tests —————

# @pytest.mark.parametrize("op,lean_sym", [
#     ("<=", "≤"),
#     (">=", "≥"),
#     ("<",  "<"),
#     (">",  ">"),
#     ("==", "="),
#     ("!=", "≠"),
# ])
# def test_all_relational_ops_map_correctly(df, op, lean_sym):
#     alpha = rd.Property('alpha',   lambda df: df['alpha'])
#     beta  = rd.Property('beta',    lambda df: df['beta'])
#     hyp   = rd.Predicate('connected', lambda df: df['connected'])

#     ineq = eval(f"alpha {op} beta")
#     conj = hyp >> ineq
#     out  = conjecture_to_lean4(conj, name="OpTest")
#     assert re.search(rf"alpha G {re.escape(lean_sym)} beta G", out)

# # ————— helper to normalize exporter’s harmless formatting quirks —————

# def canonicalize(s: str) -> str:
#     # strip trailing spaces at line ends
#     s = re.sub(r"[ \t]+(?=\n)", "", s)
#     # drop a stray trailing ' G' just before ':='
#     s = re.sub(r"\)\s*G(?=\s*:=)", ")", s)
#     # ensure exactly one space before ':='
#     s = re.sub(r"\s*:=\n", " :=\n", s)
#     return s

# # ————— pipeline tests for necessary conjectures —————

# def test_necessary_casts_fraction_to_q_and_coerces_invariants():
#     indep = rd.Property('independence_number', lambda df: df['alpha'])
#     zf    = rd.Property('zero_forcing_number', lambda df: df['beta'])
#     hyp   = rd.Predicate('connected', lambda df: df['connected'])

#     q12_11 = Fraction(12, 11)
#     qm1_11 = Fraction(-1, 11)
#     conj   = hyp >> (indep >= (q12_11 * zf + qm1_11))

#     out = necessary_conjecture_to_lean(
#         [conj],
#         func_names=['independence_number','zero_forcing_number'],
#         name_prefix="Tx"
#     )[0]

#     assert "(h1 : connected G)" in out
#     assert "order G ≥ (2 : ℕ)" in out
#     assert "(independence_number G : ℚ)" in out
#     assert "(zero_forcing_number G : ℚ)" in out
#     assert "(12/11 : ℚ)" in out
#     assert "(-1/11 : ℚ)" in out
#     assert "1.0909" not in out

# def test_necessary_negative_int_triggers_int_ambient_type():
#     indep  = rd.Property('independence_number', lambda df: df['alpha'])
#     clique = rd.Property('clique_number',       lambda df: df['beta'])
#     hyp    = rd.Predicate('connected', lambda df: df['connected'])
#     conj   = hyp >> (indep >= (clique + (-6)))

#     out = necessary_conjecture_to_lean(
#         [conj],
#         func_names=['independence_number','clique_number'],
#         name_prefix="Tx"
#     )[0]

#     assert "(independence_number G : ℤ)" in out
#     assert "(clique_number G : ℤ)" in out
#     assert "(-6 : ℤ)" in out

# # ————— full-string (exact, after canonicalization) —————

# def test_necessary_full_string_q_context():
#     indep = rd.Property('independence_number', lambda df: df['alpha'])
#     zf    = rd.Property('zero_forcing_number', lambda df: df['beta'])
#     hyp   = rd.Predicate('connected', lambda df: df['connected'])

#     q12_11 = Fraction(12, 11)
#     qm1_11 = Fraction(-1, 11)
#     conj   = hyp >> (indep >= (q12_11 * zf + qm1_11))

#     out = necessary_conjecture_to_lean(
#         [conj],
#         func_names=['independence_number','zero_forcing_number'],
#         name_prefix="TxFull"
#     )[0]

#     expected = (
#         "theorem TxFull_1 (G : SimpleGraph V)\n"
#         "    (h1 : connected G)\n"
#         "    (h2 : order G ≥ (2 : ℕ)) : (independence_number G : ℚ) ≥ "
#         "(((12/11 : ℚ) * (zero_forcing_number G : ℚ)) + (-1/11 : ℚ)) :=\n"
#         "sorry\n"
#     )
#     assert canonicalize(out) == expected

# def test_necessary_full_string_z_context():
#     indep  = rd.Property('independence_number', lambda df: df['alpha'])
#     clique = rd.Property('clique_number',       lambda df: df['beta'])
#     hyp    = rd.Predicate('connected', lambda df: df['connected'])
#     conj   = hyp >> (indep >= (clique + (-6)))

#     out = necessary_conjecture_to_lean(
#         [conj],
#         func_names=['independence_number','clique_number'],
#         name_prefix="TxFullZ"
#     )[0]

#     expected = (
#         "theorem TxFullZ_1 (G : SimpleGraph V)\n"
#         "    (h1 : connected G)\n"
#         "    (h2 : order G ≥ (2 : ℕ)) : (independence_number G : ℤ) ≥ "
#         "((clique_number G : ℤ) + (-6 : ℤ)) :=\n"
#         "sorry\n"
#     )
#     assert canonicalize(out) == expected
