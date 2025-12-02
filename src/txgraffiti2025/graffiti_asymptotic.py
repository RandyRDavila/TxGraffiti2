# src/txgraffiti2025/forms/asymptotic.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Optional

import numpy as np

from txgraffiti2025.forms.utils import Expr, Const, to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, TRUE, Relation
from txgraffiti2025.forms.predicates import Predicate


# ────────────────────────────── Helpers ────────────────────────────── #

def _nice_const(val: float | int | Fraction, max_denom: int = 50) -> Const:
    """
    Rationalize to small denominator for pretty printing, but still keep numeric semantics.
    """
    if isinstance(val, Fraction):
        return Const(val.limit_denominator(max_denom))
    try:
        fr = Fraction(float(val)).limit_denominator(max_denom)
        # accept if it really matches
        if abs(float(fr) - float(val)) <= 1e-12:
            return Const(fr)
        return Const(float(val))
    except Exception:
        return Const(float(val))


# ─────────────────────── Limit relation primitives ─────────────────────── #

class LimitKind(Enum):
    TO_CONST  = "to_const"     # lim f(t) = c
    TO_ZERO   = "to_zero"      # lim f(t) = 0
    TO_INF    = "to_infty"     # lim f(t) = +∞
    RATIO_CONST = "ratio_const"  # lim f(t)/g(t) = c


@dataclass(frozen=True)
class LimitRelation(Relation):
    """
    Symbolic relation to express asymptotic statements.

    Modes
    -----
    • TO_CONST:   lim_{t→∞} f(t) = c                 (const provided)
    • TO_ZERO:    lim_{t→∞} f(t) = 0
    • TO_INF:     lim_{t→∞} f(t) = ∞
    • RATIO_CONST:lim_{t→∞} f(t)/g(t) = c            (const provided)

    Notes
    -----
    - This is **symbolic**; .evaluate(df) is intentionally NotImplemented to avoid
      accidental row-wise truth evaluation of a global/asymptotic claim.
    - Use it inside a Conjecture that carries a hypothesis Predicate.
    """
    kind: LimitKind
    f: Expr                 # target expression (e.g., independence_number)
    t: Expr                 # diverging parameter (e.g., order)
    g: Optional[Expr] = None  # ratio denominator, when kind == RATIO_CONST
    const: Optional[Const] = None  # the c in the limit, when applicable

    # ---- interface expected from Relation ----
    def evaluate(self, df):
        raise NotImplementedError("Asymptotic LimitRelation is not row-wise evaluable.")

    # Stable identity used by dedup/sorting
    def signature(self) -> str:
        k = self.kind.value
        f = repr(self.f)
        t = repr(self.t)
        if self.kind == LimitKind.RATIO_CONST:
            g = repr(self.g) if self.g is not None else "None"
            c = repr(self.const) if self.const is not None else "None"
            return f"LIM[{k}]:f={f};g={g};t={t};c={c}"
        c = repr(self.const) if self.const is not None else "None"
        return f"LIM[{k}]:f={f};t={t};c={c}"

    # Human-readable
    def pretty(self) -> str:
        tname = getattr(self.t, "name", None) or repr(self.t)
        if self.kind == LimitKind.TO_CONST:
            return f"lim_{{{tname}→∞}} {self.f.pretty()} = {self.const.pretty()}"
        if self.kind == LimitKind.TO_ZERO:
            return f"lim_{{{tname}→∞}} {self.f.pretty()} = 0"
        if self.kind == LimitKind.TO_INF:
            return f"lim_{{{tname}→∞}} {self.f.pretty()} = ∞"
        if self.kind == LimitKind.RATIO_CONST:
            return f"lim_{{{tname}→∞}} ({self.f.pretty()} / {self.g.pretty()}) = {self.const.pretty()}"
        return self.signature()

    # LaTeX-friendly
    def pretty_latex(self) -> str:
        def _p(e: Expr) -> str:
            return getattr(e, "to_latex", None)() if hasattr(e, "to_latex") else e.pretty()
        tname = getattr(self.t, "name", None) or _p(self.t)
        if self.kind == LimitKind.TO_CONST:
            return rf"\lim_{{{tname}\to\infty}} {_p(self.f)} \;=\; {self.const.pretty()}"
        if self.kind == LimitKind.TO_ZERO:
            return rf"\lim_{{{tname}\to\infty}} {_p(self.f)} \;=\; 0"
        if self.kind == LimitKind.TO_INF:
            return rf"\lim_{{{tname}\to\infty}} {_p(self.f)} \;=\; \infty"
        if self.kind == LimitKind.RATIO_CONST:
            return rf"\lim_{{{tname}\to\infty}} \frac{{{_p(self.f)}}}{{{_p(self.g)}}} \;=\; {self.const.pretty()}"
        return self.signature()


# ───────────────────── Specialized Conjecture subclass ───────────────────── #

class AsymptoticConjecture(Conjecture):
    """
    Conjecture specialized for asymptotic statements (LimitRelation).

    Examples
    --------
    • H ⇒ lim_{n→∞} independence_number = ∞
    • H ⇒ lim_{n→∞} independence_number / n = 1/4
    """

    def __init__(self, relation: LimitRelation, condition: Optional[Predicate] = None, name: Optional[str] = None):
        if condition is None:
            condition = TRUE
        super().__init__(relation=relation, condition=condition, name=name)

    # Stable identity for dedup
    def signature(self) -> str:
        cond = repr(self.condition) if (self.condition is not None and self.condition is not TRUE) else "TRUE"
        return f"{cond} :: {self.relation.signature()}"

    # Textual pretty
    def pretty(self, show_tol: bool = False) -> str:
        cond = "TRUE" if (self.condition is None or self.condition is TRUE) \
               else (getattr(self.condition, "pretty", None) and self.condition.pretty()) or repr(self.condition)
        return f"({cond}) ⇒ {self.relation.pretty()}"

    # LaTeX block
    def pretty_latex(self) -> str:
        cond = r"\mathrm{TRUE}" if (self.condition is None or self.condition is TRUE) \
               else (getattr(self.condition, "to_latex", None) and self.condition.to_latex()) \
               or getattr(self.condition, "pretty", lambda: repr(self.condition))()
        return rf"\big({cond}\big)\;\Rightarrow\; {self.relation.pretty_latex()}"


# ───────────────────── Convenience constructors (API) ───────────────────── #

def lim_to_infinity(*, f: Expr | str, t: Expr | str, condition: Optional[Predicate] = None) -> AsymptoticConjecture:
    f_expr = to_expr(f) if isinstance(f, str) else f
    t_expr = to_expr(t) if isinstance(t, str) else t
    rel = LimitRelation(kind=LimitKind.TO_INF, f=f_expr, t=t_expr)
    return AsymptoticConjecture(relation=rel, condition=condition, name="lim→∞")

def lim_to_zero(*, f: Expr | str, t: Expr | str, condition: Optional[Predicate] = None) -> AsymptoticConjecture:
    f_expr = to_expr(f) if isinstance(f, str) else f
    t_expr = to_expr(t) if isinstance(t, str) else t
    rel = LimitRelation(kind=LimitKind.TO_ZERO, f=f_expr, t=t_expr)
    return AsymptoticConjecture(relation=rel, condition=condition, name="lim→0")

def lim_to_const(*, f: Expr | str, t: Expr | str, c: float | int | Fraction,
                 condition: Optional[Predicate] = None, max_denom: int = 50) -> AsymptoticConjecture:
    f_expr = to_expr(f) if isinstance(f, str) else f
    t_expr = to_expr(t) if isinstance(t, str) else t
    rel = LimitRelation(kind=LimitKind.TO_CONST, f=f_expr, t=t_expr, const=_nice_const(c, max_denom))
    return AsymptoticConjecture(relation=rel, condition=condition, name="lim→const")

def lim_ratio_const(*, f: Expr | str, g: Expr | str, t: Expr | str, c: float | int | Fraction,
                    condition: Optional[Predicate] = None, max_denom: int = 50) -> AsymptoticConjecture:
    f_expr = to_expr(f) if isinstance(f, str) else f
    g_expr = to_expr(g) if isinstance(g, str) else g
    t_expr = to_expr(t) if isinstance(t, str) else t
    rel = LimitRelation(kind=LimitKind.RATIO_CONST, f=f_expr, g=g_expr, t=t_expr, const=_nice_const(c, max_denom))
    return AsymptoticConjecture(relation=rel, condition=condition, name="lim ratio→const")
