"""
Human-friendly, math-like pretty printing for forms.

Adds:
- format_expr(expr, ...):   expression -> "c * x" (with 1* suppressed, no quotes)
- format_pred(pred, ...):   predicate -> "(connected) ∧ (bipartite) ∧ (cubic)"
- format_relation(rel, ...): relation -> "alpha ≥ beta"
- format_conjecture(conj, ...): "((connected) ∧ (bipartite)) -> alpha ≥ beta"

Also monkey-patches:
- Conjecture.pretty(...)
- Implication.pretty(...)
- Equivalence.pretty(...)

This DOES NOT change __repr__ anywhere; it’s an opt-in view for readability.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from .utils import Expr, Const, ColumnTerm, LinearForm, BinOp, UnaryOp  # type: ignore
from .generic_conjecture import Conjecture, Relation, Eq, Le, Ge
from .implication import Implication, Equivalence
from .predicates import Predicate, AndPred, OrPred, NotPred  # type: ignore
from fractions import Fraction

# -----------------------------
# Expr pretty
# -----------------------------

def _const_to_text(c: Const) -> str:
    v = c.value
    if isinstance(v, Fraction):
        # minimal rational form (no "Fraction(...)")
        n, d = v.numerator, v.denominator
        if d == 1:
            return f"{n}"
        return f"{n}/{d}"
    # float/int
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f}"

def _maybe_paren(s: str) -> str:
    # Add parens only if the token likely contains spaces or operators
    return s if s.replace("_","").isalnum() else f"({s})"

def format_expr(e: Expr, *, unicode_ops: bool = True, dot: str = "·", strip_ones: bool = True) -> str:
    """
    Convert an Expr to a compact infix string.

    Parameters
    ----------
    unicode_ops : bool
        Use unicode symbols (e.g. '·') for multiplication.
    dot : str
        Multiplication symbol when needed ('·', '*', or '').
    strip_ones : bool
        Suppress coefficients 1 and -1: 1*x -> x, -1*x -> -x.
    """
    # Column
    if isinstance(e, ColumnTerm):
        return e.col  # no quotes

    # Constant
    if isinstance(e, Const):
        return _const_to_text(e)

    # LinearForm
    if isinstance(e, LinearForm):
        a0 = float(e.intercept)
        parts: List[str] = []
        if not np.isclose(a0, 0.0):
            parts.append(_const_to_text(Const(a0)))
        for coef, col in e.terms:
            c = coef
            name = col
            # show ± and suppress 1* when requested
            if strip_ones and np.isclose(c, 1.0):
                parts.append(f"{name}")
            elif strip_ones and np.isclose(c, -1.0):
                parts.append(f"-{name}")
            else:
                k = _const_to_text(Const(c))
                parts.append(f"{k}{dot}{name}" if dot else f"{k}{name}")
        if not parts:
            return "0"
        # join with " + " while fixing "+ -" to "-"
        s = " + ".join(parts)
        s = s.replace("+ -", "- ")
        return s

    # BinOp (handle the common ones we care about)
    if isinstance(e, BinOp):
        sym_map = {np.add:"+", np.subtract:"-", np.multiply:"*", np.divide:"/", np.mod:"%", np.power:"**"}
        sym = sym_map.get(e.fn, "?")
        L = format_expr(e.left, unicode_ops=unicode_ops, dot=dot, strip_ones=strip_ones)
        R = format_expr(e.right, unicode_ops=unicode_ops, dot=dot, strip_ones=strip_ones)

        # Special-case multiplication by constants
        if sym == "*":
            # Detect Const on either side
            if isinstance(e.left, Const):
                k = _const_to_text(e.left)
                if strip_ones and k in ("1","1.0"):
                    return R
                if strip_ones and k in ("-1","-1.0"):
                    return f"-{_maybe_paren(R)}"
                return f"{k}{dot}{_maybe_paren(R)}" if dot else f"{k}{_maybe_paren(R)}"
            if isinstance(e.right, Const):
                k = _const_to_text(e.right)
                if strip_ones and k in ("1","1.0"):
                    return L
                if strip_ones and k in ("-1","-1.0"):
                    return f"-{_maybe_paren(L)}"
                return f"{_maybe_paren(L)}{dot}{k}" if dot else f"{_maybe_paren(L)}{k}"
            # generic multiply
            return f"{_maybe_paren(L)}{dot}{_maybe_paren(R)}" if dot else f"{_maybe_paren(L)}{_maybe_paren(R)}"

        if sym == "/":
            return f"{_maybe_paren(L)}/{_maybe_paren(R)}"

        if sym == "+":
            s = f"{L} + {R}"
            return s.replace("+ -", "- ")

        if sym == "-":
            return f"{L} - {R}"

        if sym == "**":
            return f"{_maybe_paren(L)}^{_maybe_paren(R)}"

        # Fallback
        return f"{L} {sym} {R}"

    # UnaryOp (log, exp, sqrt, floor, ceil, abs, neg)
    if isinstance(e, UnaryOp):
        name = getattr(e, "fn", None)
        # We'll inspect repr(e) to get the function name we installed in utils
        r = repr(e)  # e.g., "sqrt(x)" or "log(x)" from your LogOp/UnaryOp
        # strip quotes if any leaked
        return r.replace("'", "")

    # Fallback: repr without quotes
    return repr(e).replace("'", "")

# -----------------------------
# Predicate pretty
# -----------------------------

def _flatten_and(pred: Predicate) -> List[Predicate]:
    out: List[Predicate] = []
    def rec(p: Predicate):
        if isinstance(p, AndPred):
            rec(p.a)
            rec(p.b)
        else:
            out.append(p)
    rec(pred)
    return out

def _flatten_or(pred: Predicate) -> List[Predicate]:
    out: List[Predicate] = []
    def rec(p: Predicate):
        if isinstance(p, OrPred):
            rec(p.a)
            rec(p.b)
        else:
            out.append(p)
    rec(pred)
    return out

def format_pred(pred: Optional[Predicate], *, unicode_ops: bool = True) -> str:
    if pred is None:
        return "TRUE"
    # And-chains
    if isinstance(pred, AndPred):
        parts = [_p_name(p) for p in _flatten_and(pred)]
        glue = " ∧ " if unicode_ops else " & "
        return "(" + glue.join(parts) + ")"
    # Or-chains
    if isinstance(pred, OrPred):
        parts = [_p_name(p) for p in _flatten_or(pred)]
        glue = " ∨ " if unicode_ops else " | "
        return "(" + glue.join(parts) + ")"
    # Not
    if isinstance(pred, NotPred):
        return f"¬{_p_name(pred.a)}" if unicode_ops else f"~{_p_name(pred.a)}"
    # Fallback: use its name or repr without quotes
    return _p_name(pred)

def _p_name(p: Predicate) -> str:
    name = getattr(p, "name", None)
    if name:
        return name.replace("'", "")
    return repr(p).replace("'", "")

# -----------------------------
# Relation & Conjecture pretty
# -----------------------------

def format_relation(rel: Relation, *, unicode_ops: bool = True, strip_ones: bool = True) -> str:
    if isinstance(rel, Eq):
        op = "="
        L = format_expr(rel.left, unicode_ops=unicode_ops, strip_ones=strip_ones)
        R = format_expr(rel.right, unicode_ops=unicode_ops, strip_ones=strip_ones)
        return f"{L} {op} {R}"
    if isinstance(rel, Le):
        op = "≤" if unicode_ops else "<="
        L = format_expr(rel.left, unicode_ops=unicode_ops, strip_ones=strip_ones)
        R = format_expr(rel.right, unicode_ops=unicode_ops, strip_ones=strip_ones)
        return f"{L} {op} {R}"
    if isinstance(rel, Ge):
        op = "≥" if unicode_ops else ">="
        L = format_expr(rel.left, unicode_ops=unicode_ops, strip_ones=strip_ones)
        R = format_expr(rel.right, unicode_ops=unicode_ops, strip_ones=strip_ones)
        return f"{L} {op} {R}"
    # Fallback
    return repr(rel).replace("'", "")

def format_conjecture(
    conj: Conjecture,
    *,
    unicode_ops: bool = True,
    arrow: str = "->",
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

# -----------------------------
# LaTeX helpers
# -----------------------------

def to_latex(conj: Conjecture) -> str:
    """Render a conjecture as LaTeX (math mode fragment)."""
    rel = format_relation(conj.relation, unicode_ops=False, strip_ones=True)
    # map ops
    rel = rel.replace(">=", r"\geq").replace("<=", r"\leq").replace("=", r"=")
    # swap * to \cdot when needed (we kept · out by stripping ones)
    rel = rel.replace("*", r"\cdot ")
    cond = format_pred(conj.condition, unicode_ops=False)
    cond = cond.replace("&", r"\wedge").replace("|", r"\vee").replace("~", r"\neg ")
    if cond == "TRUE":
        return f"${rel}$"
    return f"${cond} \\Rightarrow {rel}$"

# -----------------------------
# Attach .pretty() methods
# -----------------------------

def _conj_pretty(self: Conjecture, *, unicode_ops: bool = True, arrow: str = "->", strip_ones: bool = True, show_condition: bool = True) -> str:
    return format_conjecture(self, unicode_ops=unicode_ops, arrow=arrow, strip_ones=strip_ones, show_condition=show_condition)

def _impl_pretty(self: Implication, *, unicode_ops: bool = True, arrow: str = "->", strip_ones: bool = True, show_condition: bool = True) -> str:
    # R1 -> R2 [under C]
    lhs = format_relation(self.premise, unicode_ops=unicode_ops, strip_ones=strip_ones)
    rhs = format_relation(self.conclusion, unicode_ops=unicode_ops, strip_ones=strip_ones)
    body = f"{lhs} {arrow} {rhs}"
    if not show_condition:
        return body
    cond = format_pred(self.condition, unicode_ops=unicode_ops)
    if cond == "TRUE":
        return body
    return f"{cond} {arrow} {body}"

def _eqv_pretty(self: Equivalence, *, unicode_ops: bool = True, strip_ones: bool = True, show_condition: bool = True) -> str:
    a = format_relation(self.a, unicode_ops=unicode_ops, strip_ones=strip_ones)
    b = format_relation(self.b, unicode_ops=unicode_ops, strip_ones=strip_ones)
    body = f"{a} ⇔ {b}" if unicode_ops else f"{a} <=> {b}"
    if not show_condition:
        return body
    cond = format_pred(self.condition, unicode_ops=unicode_ops)
    if cond == "TRUE":
        return body
    arrow = "->" if not unicode_ops else "⇒"
    return f"{cond} {arrow} {body}"

# Monkey patch
Conjecture.pretty = _conj_pretty             # type: ignore[attr-defined]
Implication.pretty = _impl_pretty            # type: ignore[attr-defined]
Equivalence.pretty = _eqv_pretty             # type: ignore[attr-defined]
