import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms.pretty import (
    format_expr, format_pred, format_relation, format_conjecture, to_latex
)
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge
from txgraffiti2025.forms.implication import Implication, Equivalence
from txgraffiti2025.forms.predicates import Predicate, GEQ, LEQ, IN, Where
from txgraffiti2025.forms.linear import linear_expr, linear_le
from txgraffiti2025.forms.nonlinear import product, ratio
from txgraffiti2025.forms.utils import to_expr, sqrt, log, exp, floor, ceil, abs_


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def df_basic():
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 2.0, 1.0, 4.0],
        "c": [1.5, 2.5, 3.5, 4.5],
        "connected": [True, True, True, True],
    })


# -----------------------
# format_expr
# -----------------------

def test_format_expr_constants_and_columns():
    assert format_expr(to_expr(2)) == "2"
    assert format_expr(to_expr(2.0)) == "2"
    assert format_expr(to_expr("alpha")) == "alpha"

def test_format_expr_linear_form_strip_ones_and_signs():
    lin = linear_expr(0.0, [(1.0, "x"), (-1.0, "y"), (2.0, "z")])
    s = format_expr(lin)  # default strip_ones=True
    # Expect "x - y + 2·z" (with unicode dot by default)
    assert "x" in s and "- y" in s and ("2·z" in s or "2*z" in s)

def test_format_expr_mul_div_pow_and_unary_ops():
    e_mul = product("x", "y")                     # x * y
    e_div = ratio(to_expr("x") + 1, "y")          # (x+1)/y
    e_pow = to_expr("x") ** 2
    e_sqrt = sqrt("x")
    e_log = log("x")
    e_exp = exp("x")
    e_floor = floor("x")
    e_ceil = ceil("x")
    e_abs = abs_("x")

    s_mul = format_expr(e_mul)
    s_div = format_expr(e_div)
    s_pow = format_expr(e_pow)
    s_sqrt = format_expr(e_sqrt)
    s_log = format_expr(e_log)
    s_exp = format_expr(e_exp)
    s_floor = format_expr(e_floor)
    s_ceil = format_expr(e_ceil)
    s_abs = format_expr(e_abs)

    assert "x" in s_mul and "y" in s_mul and ("·" in s_mul or "*" in s_mul)
    assert "/" in s_div and "x" in s_div and "y" in s_div
    assert "^" in s_pow or "**" in s_pow
    # Unary ops rely on repr from your Expr nodes; we just sanity-check substrings
    assert any(tok in s_sqrt.lower() for tok in ("sqrt", "√"))
    assert "log" in s_log.lower()
    assert "exp" in s_exp.lower()
    assert "floor" in s_floor.lower()
    assert "ceil" in s_ceil.lower()
    assert "abs" in s_abs.lower()


# -----------------------
# format_pred
# -----------------------

def test_format_pred_and_or_not(df_basic):
    P = Predicate.from_column("connected")  # name="(connected)"
    Q = GEQ("a", 2.0) & LEQ("a", 3.0)
    R = IN("a", {1.0, 4.0})
    s_and = format_pred(Q, unicode_ops=True)
    s_or = format_pred(Q | R, unicode_ops=True)
    s_not = format_pred(~R, unicode_ops=True)

    # Your Compare predicates carry short names ("GE", "LE"), not column names
    assert "∧" in s_and and ("GE" in s_and and "LE" in s_and)
    assert "∨" in s_or
    assert "¬" in s_not or "~" in s_not

def test_format_pred_flattening(df_basic):
    P = GEQ("a", 1.0) & LEQ("a", 4.0) & IN("b", {2.0, 4.0})
    s = format_pred(P, unicode_ops=True)
    # Expect a single pair of parentheses with three parts joined by ∧
    assert s.count("(") == 1 and s.count(")") == 1
    assert s.count("∧") == 2


# -----------------------
# format_relation
# -----------------------

def test_format_relation_unicode_and_ascii():
    r_eq = Eq("alpha", "beta")
    r_le = Le("alpha", "beta")
    r_ge = Ge("beta", "alpha")

    s_eq_u = format_relation(r_eq, unicode_ops=True)
    s_le_u = format_relation(r_le, unicode_ops=True)
    s_ge_u = format_relation(r_ge, unicode_ops=True)

    s_eq_a = format_relation(r_eq, unicode_ops=False)
    s_le_a = format_relation(r_le, unicode_ops=False)
    s_ge_a = format_relation(r_ge, unicode_ops=False)

    assert "alpha" in s_eq_u and "=" in s_eq_u and "beta" in s_eq_u
    assert "≤" in s_le_u
    assert "≥" in s_ge_u

    assert "<=" in s_le_a and ">=" in s_ge_a and "=" in s_eq_a


# -----------------------
# format_conjecture & to_latex
# -----------------------

def test_format_conjecture_suppresses_true_condition(df_basic):
    C = Conjecture(Le("a", "c"))  # no condition → may be TRUE at print time
    # show_condition=True but condition is None → renders as just relation
    txt = format_conjecture(C, unicode_ops=True, show_condition=True)
    assert "≤" in txt and "a" in txt and "c" in txt
    assert "->" not in txt and "⇒" not in txt

def test_format_conjecture_with_explicit_condition(df_basic):
    P = GEQ("a", 2.0) & LEQ("a", 3.0)
    C = Conjecture(Le("a", "c"), P)
    txt_u = format_conjecture(C, unicode_ops=True, arrow="⇒")
    txt_a = format_conjecture(C, unicode_ops=False, arrow="->")
    assert "∧" in txt_u and "⇒" in txt_u
    assert "&" in txt_a and "->" in txt_a

def test_to_latex_mapping(df_basic):
    P = GEQ("a", 2.0)
    C = Conjecture(Ge("a", "c"), P)
    tex = to_latex(C)
    # \Rightarrow, \geq, variable names; no unicode glyphs
    assert "\\Rightarrow" in tex and "\\geq" in tex and "a" in tex and "c" in tex
    assert "≥" not in tex and "≤" not in tex and "·" not in tex


# -----------------------
# Monkey-patched .pretty()
# -----------------------

def test_conjecture_pretty_method_unicode_ascii(df_basic):
    P = GEQ("a", 2.0) & LEQ("a", 3.0)
    C = Conjecture(Le("a", "c"), P)
    s_u = C.pretty(unicode_ops=True, arrow="⇒")
    s_a = C.pretty(unicode_ops=False, arrow="->")
    assert "∧" in s_u and "⇒" in s_u
    assert "&" in s_a and "->" in s_a

def test_implication_and_equivalence_pretty():
    # Build simple relations
    r1 = Le("alpha", "beta")
    r2 = Ge("gamma", "delta")
    # No extra condition (defaults to TRUE)
    impl = Implication(r1, r2)
    eqv = Equivalence(r1, r2)

    si = impl.pretty(unicode_ops=True, arrow="⇒")
    se = eqv.pretty(unicode_ops=True)
    assert "≤" in si and "≥" in si and "⇒" in si
    assert "⇔" in se  # equivalence glyph in unicode mode

def test_expr_dot_symbol_variants():
    # When dot="" we should not see a separator for multiplication
    e = product("x", "y")
    s_default = format_expr(e, dot="·")
    s_ascii = format_expr(e, dot="*")
    s_concat = format_expr(e, dot="")
    assert "·" in s_default or "*" in s_default  # depending on platform
    assert "*" in s_ascii
    # With dot="" and simple tokens, pretty may produce "xy" (no parens) or "(x)(y)"
    assert (s_concat == "xy") or ("(" in s_concat and ")" in s_concat and "x" in s_concat and "y" in s_concat)


# -----------------------
# Integration smoke
# -----------------------

def test_pretty_plays_nice_with_linear_and_nonlinear(df_basic):
    lin = linear_expr(1.0, [(2.0, "a"), (-3.0, "b")])   # 1 + 2a - 3b
    rel = Le(lin, "c")
    txt = format_relation(rel, unicode_ops=True)
    assert "a" in txt and "b" in txt and "c" in txt and ("·" in txt or "*" in txt)
