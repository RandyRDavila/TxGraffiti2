from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import operator
import numpy as np
import pandas as pd

from .graffiti_utils import Expr, to_expr
from .graffiti_predicates import Predicate, TRUE as _TRUE

__all__ = [
    # core boolean DSL
    "BoolFormula", "coerce_formula",
    # relations over Expr
    "Relation", "Ge", "Le", "Eq", "Lt", "Gt",
    # logical combinators
    "AndF", "OrF", "XorF", "NotF", "Implies", "Iff", "AllOf",
    # conjectures + helpers
    "Conjecture",
    # piecewise numeric
    "ite",
    # re-export TRUE for convenience
    "TRUE",
]

TRUE = _TRUE


# ─────────────────────── Boolean Formula base ─────────────────────── #

class BoolFormula:
    """Boolean-valued formula over rows. evaluate(df) -> bool Series aligned to df.index."""

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def pretty(self) -> str:
        return repr(self)

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.evaluate(df)

    # boolean operators (return BoolFormula)
    def __and__(self, other) -> "BoolFormula":
        return AndF(self, coerce_formula(other))

    def __or__(self, other) -> "BoolFormula":
        return OrF(self, coerce_formula(other))

    def __xor__(self, other) -> "BoolFormula":
        return XorF(self, coerce_formula(other))

    def __invert__(self) -> "BoolFormula":
        return NotF(self)

    # implication via >>
    def __rshift__(self, other) -> "BoolFormula":
        return Implies(self, coerce_formula(other))

    # biconditional helper
    def iff(self, other) -> "BoolFormula":
        return Iff(self, coerce_formula(other))


@dataclass(frozen=True)
class _PredFormula(BoolFormula):
    """Adapter that treats a Predicate as a BoolFormula."""
    pred: Predicate

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        # Ensure aligned, boolean Series, NA->False
        return self.pred.mask(df).reindex(df.index).fillna(False).astype(bool, copy=False)

    def __repr__(self) -> str:
        p = getattr(self.pred, "pretty", None)
        return p() if callable(p) else repr(self.pred)


def coerce_formula(x) -> BoolFormula:
    if isinstance(x, BoolFormula):
        return x
    if isinstance(x, Predicate):
        return _PredFormula(x)
    raise TypeError(f"Cannot coerce {type(x).__name__} to BoolFormula")


# ───────────────────── Relations from numeric Expr comparisons ───────────────────── #

@dataclass(frozen=True)
class Relation(BoolFormula):
    left: Expr
    right: Expr
    symbol: str
    fn: Callable[[pd.Series, pd.Series], pd.Series]

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = pd.to_numeric(self.left.evaluate(df), errors="coerce")
        r = pd.to_numeric(self.right.evaluate(df), errors="coerce")
        # elementwise compare; keep alignment and NA->False
        out = self.fn(l, r)
        s = (out if isinstance(out, pd.Series) else pd.Series(out, index=df.index)).reindex(df.index)
        return s.fillna(False).astype(bool, copy=False)

    def __repr__(self) -> str:
        return f"{self.left!r} {self.symbol} {self.right!r}"


def Ge(x: Expr, y: Expr) -> Relation: return Relation(to_expr(x), to_expr(y), "≥", operator.ge)
def Le(x: Expr, y: Expr) -> Relation: return Relation(to_expr(x), to_expr(y), "≤", operator.le)
def Eq(x: Expr, y: Expr) -> Relation: return Relation(to_expr(x), to_expr(y), "=", operator.eq)
def Lt(x: Expr, y: Expr) -> Relation: return Relation(to_expr(x), to_expr(y), "<", operator.lt)
def Gt(x: Expr, y: Expr) -> Relation: return Relation(to_expr(x), to_expr(y), ">", operator.gt)


# ───────────────────── Logical combinators ───────────────────── #

@dataclass(frozen=True)
class AndF(BoolFormula):
    a: BoolFormula
    b: BoolFormula
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        av = self.a.evaluate(df).to_numpy(dtype=bool, copy=False)
        bv = self.b.evaluate(df).to_numpy(dtype=bool, copy=False)
        return pd.Series(np.logical_and(av, bv), index=df.index)
    def __repr__(self) -> str:
        return f"({self.a!r} ∧ {self.b!r})"

@dataclass(frozen=True)
class OrF(BoolFormula):
    a: BoolFormula
    b: BoolFormula
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        av = self.a.evaluate(df).to_numpy(dtype=bool, copy=False)
        bv = self.b.evaluate(df).to_numpy(dtype=bool, copy=False)
        return pd.Series(np.logical_or(av, bv), index=df.index)
    def __repr__(self) -> str:
        return f"({self.a!r} ∨ {self.b!r})"

@dataclass(frozen=True)
class XorF(BoolFormula):
    a: BoolFormula
    b: BoolFormula
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        av = self.a.evaluate(df).to_numpy(dtype=bool, copy=False)
        bv = self.b.evaluate(df).to_numpy(dtype=bool, copy=False)
        return pd.Series(np.logical_xor(av, bv), index=df.index)
    def __repr__(self) -> str:
        return f"({self.a!r} ⊕ {self.b!r})"

@dataclass(frozen=True)
class NotF(BoolFormula):
    a: BoolFormula
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        av = self.a.evaluate(df).to_numpy(dtype=bool, copy=False)
        return pd.Series(np.logical_not(av), index=df.index)
    def __repr__(self) -> str:
        return f"¬({self.a!r})"

@dataclass(frozen=True)
class Implies(BoolFormula):
    a: BoolFormula
    b: BoolFormula
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        av = self.a.evaluate(df).to_numpy(dtype=bool, copy=False)
        bv = self.b.evaluate(df).to_numpy(dtype=bool, copy=False)
        # (~A) | B
        return pd.Series(np.logical_or(np.logical_not(av), bv), index=df.index)
    def __repr__(self) -> str:
        return f"({self.a!r} ⇒ {self.b!r})"

@dataclass(frozen=True)
class Iff(BoolFormula):
    a: BoolFormula
    b: BoolFormula
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        av = self.a.evaluate(df).to_numpy(dtype=bool, copy=False)
        bv = self.b.evaluate(df).to_numpy(dtype=bool, copy=False)
        # (A & B) | (~A & ~B)
        return pd.Series(
            np.logical_or(
                np.logical_and(av, bv),
                np.logical_and(np.logical_not(av), np.logical_not(bv)),
            ),
            index=df.index,
        )
    def __repr__(self) -> str:
        return f"({self.a!r} ⇔ {self.b!r})"

@dataclass(frozen=True)
class AllOf(BoolFormula):
    # accept Predicates *or* BoolFormula; coerce at runtime
    parts: list[Union["BoolFormula", Predicate]]

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        coerced = [coerce_formula(p) for p in self.parts]
        if not coerced:
            return pd.Series(True, index=df.index)
        m = coerced[0].evaluate(df).to_numpy(dtype=bool, copy=False)
        for p in coerced[1:]:
            m = np.logical_and(m, p.evaluate(df).to_numpy(dtype=bool, copy=False))
        return pd.Series(m, index=df.index)

    def __repr__(self) -> str:
        return " ∧ ".join(repr(coerce_formula(p)) for p in self.parts)

# ───────────────────── Piecewise numeric: ite(cond, then, else) ───────────────────── #

@dataclass(frozen=True)
class _IteExpr(Expr):
    cond: BoolFormula
    t: Expr
    e: Expr
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        m = self.cond.evaluate(df)  # BoolFormula evaluation
        tv = pd.to_numeric(self.t.evaluate(df), errors="coerce").astype(float)
        ev = pd.to_numeric(self.e.evaluate(df), errors="coerce").astype(float)
        return pd.Series(
            np.where(m.to_numpy(dtype=bool, copy=False), tv, ev),
            index=df.index
        )
    def __repr__(self) -> str:
        return f"if({self.cond!r}, {self.t!r}, {self.e!r})"

def ite(condition: Union[BoolFormula, Predicate], then_expr: Expr, else_expr: Expr) -> Expr:
    """Piecewise numeric expression: if condition then then_expr else else_expr."""
    return _IteExpr(cond=coerce_formula(condition), t=then_expr, e=else_expr)


# ───────────────────── Conjecture wrapper ───────────────────── #

@dataclass(frozen=True)
class Conjecture:
    """
    A conjecture is a BoolFormula, optionally conditioned on another BoolFormula.
    When condition is None/TRUE: it's a global formula.
    """
    relation: BoolFormula
    condition: Optional[Union[BoolFormula, Predicate]] = None
    name: Optional[str] = None

    def formula(self) -> BoolFormula:
        if self.condition is None:
            return self.relation
        return Implies(coerce_formula(self.condition), coerce_formula(self.relation))

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        # Evaluate as “condition ⇒ relation” row-wise
        return self.formula().evaluate(df)

    def pretty(self) -> str:
        if self.condition is None:
            return repr(self.relation)
        return f"{coerce_formula(self.condition)!r} ⇒ {self.relation!r}"
