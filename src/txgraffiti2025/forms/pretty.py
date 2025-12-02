# src/txgraffiti2025/forms/pretty.py

"""
Formatting helpers (no monkey-patching).

Use:
    from txgraffiti2025.forms.pretty import format_conjecture
    print(format_conjecture(conj))  # math-style
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from fractions import Fraction

from .utils import Expr, Const, ColumnTerm, LinearForm, BinOp, UnaryOp
from .generic_conjecture import Conjecture, Relation, Eq, Le, Ge, TRUE
from .predicates import Predicate, AndPred, OrPred, NotPred
import numpy as np

__all__ = [
    "format_expr",
    "format_pred",
    "format_relation",
    "format_conjecture",
    "to_latex",
]

# -------- Expr pretty --------
def _const_to_text(c: Const) -> str:
    v = c.value
    if isinstance(v, Fraction):
        n, d = v.numerator, v.denominator
        return f"{n}" if d == 1 else f"{n}/{d}"
    f = float(v);  return str(int(f)) if f.is_integer() else f"{f}"

def _maybe_paren(s: str) -> str:
    return s if s.replace("_","").isalnum() else f"({s})"

def format_expr(e: Expr, *, dot: str = "·", strip_ones: bool = True) -> str:
    if isinstance(e, ColumnTerm):
        return e.col
    if isinstance(e, Const):
        return _const_to_text(e)
    if isinstance(e, LinearForm):
        a0 = float(e.intercept)
        parts: List[str] = []
        if not np.isclose(a0, 0.0): parts.append(_const_to_text(Const(a0)))
        for coef, col in e.terms:
            c = float(coef)
            if strip_ones and np.isclose(c,  1.0): parts.append(f"{col}");  continue
            if strip_ones and np.isclose(c, -1.0): parts.append(f"-{col}"); continue
            k = _const_to_text(Const(c))
            parts.append(f"{k}{dot}{col}" if dot else f"{k}{col}")
        if not parts: return "0"
        s = " + ".join(parts)
        return s.replace("+ -", "- ")
    if isinstance(e, BinOp):
        import numpy as _np
        sym_map = {_np.add:"+", _np.subtract:"-", _np.multiply:"*", _np.divide:"/", _np.mod:"%", _np.power:"**"}
        sym = sym_map.get(e.fn, "?")
        L = format_expr(e.left, dot=dot, strip_ones=strip_ones)
        R = format_expr(e.right, dot=dot, strip_ones=strip_ones)
        if sym == "*":
            if isinstance(e.left, Const):
                k = _const_to_text(e.left)
                if strip_ones and k in ("1","1.0"):  return R
                if strip_ones and k in ("-1","-1.0"): return f"-{_maybe_paren(R)}"
                return f"{k}{dot}{_maybe_paren(R)}" if dot else f"{k}{_maybe_paren(R)}"
            if isinstance(e.right, Const):
                k = _const_to_text(e.right)
                if strip_ones and k in ("1","1.0"):  return L
                if strip_ones and k in ("-1","-1.0"): return f"-{_maybe_paren(L)}"
                return f"{_maybe_paren(L)}{dot}{k}" if dot else f"{_maybe_paren(L)}{k}"
            return f"{_maybe_paren(L)}{dot}{_maybe_paren(R)}" if dot else f"{_maybe_paren(L)}{_maybe_paren(R)}"
        if sym == "/":  return f"{_maybe_paren(L)}/{_maybe_paren(R)}"
        if sym == "+":  return (f"{L} + {R}").replace("+ -", "- ")
        if sym == "-":  return f"{L} - {R}"
        if sym == "**": return f"{_maybe_paren(L)}^{_maybe_paren(R)}"
        return f"{L} {sym} {R}"
    if isinstance(e, UnaryOp):
        return repr(e).replace("'", "")
    return repr(e).replace("'", "")

# -------- Predicate pretty --------
def _p_name(p: Predicate) -> str:
    name = getattr(p, "name", None)
    return name.replace("'", "") if name else repr(p).replace("'", "")

def format_pred(pred: Optional[Predicate], *, unicode_ops: bool = True) -> str:
    if pred is None: return "TRUE"
    if isinstance(pred, AndPred):
        parts = [];  stack = [pred]
        while stack:
            q = stack.pop()
            if isinstance(q, AndPred): stack.extend([q.b, q.a])
            else: parts.append(_p_name(q))
        glue = " ∧ " if unicode_ops else " & "
        return "(" + glue.join(reversed(parts)) + ")"
    if isinstance(pred, OrPred):
        parts = [];  stack = [pred]
        while stack:
            q = stack.pop()
            if isinstance(q, OrPred): stack.extend([q.b, q.a])
            else: parts.append(_p_name(q))
        glue = " ∨ " if unicode_ops else " | "
        return "(" + glue.join(reversed(parts)) + ")"
    if isinstance(pred, NotPred):
        return f"¬{_p_name(pred.a)}" if unicode_ops else f"~{_p_name(pred.a)}"
    return _p_name(pred)

# -------- Relation & Conjecture pretty --------
def format_relation(rel: Relation, *, unicode_ops: bool = True, strip_ones: bool = True) -> str:
    L = R = None
    if isinstance(rel, (Eq, Le, Ge)):
        L = format_expr(rel.left, strip_ones=strip_ones)
        R = format_expr(rel.right, strip_ones=strip_ones)
    if isinstance(rel, Eq):
        return f"({L} = {R})"
    if isinstance(rel, Le):
        op = "≤" if unicode_ops else "<="
        return f"({L} {op} {R})"
    if isinstance(rel, Ge):
        op = "≥" if unicode_ops else ">="
        return f"({L} {op} {R})"
    return repr(rel).replace("'", "")

def format_conjecture(
    conj: Conjecture,
    *,
    unicode_ops: bool = True,
    arrow: str = "⇒",
    strip_ones: bool = True,
    show_condition: bool = True,
) -> str:
    rel = format_relation(conj.relation, unicode_ops=unicode_ops, strip_ones=strip_ones)
    if not show_condition:
        return rel
    cond = format_pred(conj.condition, unicode_ops=unicode_ops)
    if cond == "TRUE":
        return rel
    return f"{cond} {arrow} {rel}"

def to_latex(conj: Conjecture) -> str:
    rel = format_relation(conj.relation, unicode_ops=False, strip_ones=True)
    rel = rel.replace(">=", r"\geq").replace("<=", r"\leq").replace("=", r"=")
    rel = rel.replace("*", r"\cdot ")
    cond = format_pred(conj.condition, unicode_ops=False)
    cond = cond.replace("&", r"\wedge").replace("|", r"\vee").replace("~", r"\neg ")
    return f"${rel}$" if cond == "TRUE" else f"${cond} \\Rightarrow {rel}$"
