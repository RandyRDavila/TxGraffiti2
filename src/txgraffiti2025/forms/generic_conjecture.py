# src/txgraffiti2025/forms/generic_conjecture.py
from __future__ import annotations

"""
Generic, DataFrame-agnostic conjecture primitives.

- Relation types (row-wise on a DataFrame): Eq, Le, Ge, AllOf, AnyOf
- Conjecture: (R | C) meaning “for all rows in class C, relation R holds”

Conventions
-----------
- Relation.evaluate(df) -> boolean Series aligned to df.index.
- Relation.slack(df) -> float Series aligned to df.index where >= 0 means satisfied.
  Le:  slack = rhs - lhs
  Ge:  slack = lhs - rhs
  Eq:  slack = tol - |lhs - rhs|
  AllOf: min(child slacks)
  AnyOf: max(child slacks)

- Conjecture.check(df) returns:
    applicable : mask where the condition holds,
    holds      : mask indicating satisfaction of (R | C),
    failures   : df rows with applicable & ~evaluate, plus "__slack__".

User-facing display
-------------------
Conjecture.pretty() yields math-style strings like:
    (planar ∧ regular) ⇒ (alpha ≤ mu) ∧ (alpha ≥ ⌊order/3⌋)
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from .utils import Expr, to_expr
from .predicates import Predicate, Where, AndPred

__all__ = [
    "Relation",
    "Eq",
    "Le",
    "Ge",
    "AllOf",
    "AnyOf",
    "Conjecture",
    "TRUE",
]

# =========================================================
# TRUE predicate (universal class)
# =========================================================

class TRUE_Predicate(Predicate):
    name: str = "TRUE"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index, dtype=bool)
    def __repr__(self) -> str:
        return "TRUE"

TRUE = TRUE_Predicate()

# =========================================================
# Small pretty helpers
# =========================================================

def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        return s[1:-1].strip()
    return s

def _pretty_predicate(cond: Predicate, *, unicode_ops: bool = True) -> str:
    """
    Render predicates compactly: strip one outer () if present, then wrap in ().
    """
    s = repr(cond)
    s = _strip_outer_parens(s)
    return f"({s})"

# =========================================================
# Relations
# =========================================================

class Relation:
    """Abstract base: row-wise boolean relation + slack margin."""
    name: str = "Relation"

    # --- core API ---
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def slack(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    # --- helpers and sugar ---
    def is_tight(self, df: pd.DataFrame, *, atol: float = 1e-12) -> pd.Series:
        """
        Rows where the relation is satisfied at equality (boundary), robust to FP error.
        For Le/Ge/Eq this corresponds to slack ≈ 0.
        """
        s = self.slack(df).reindex(df.index)
        return pd.Series(np.isclose(s.values, 0.0, atol=atol), index=s.index, dtype=bool)

    # composition: R1 & R2, R1 | R2
    def __and__(self, other: "Relation") -> "AllOf":
        return AllOf([self, other])

    def __or__(self, other: "Relation") -> "AnyOf":
        return AnyOf([self, other])

    # unified pretty signature (subclasses may override)
    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"{self.name}()"


@dataclass
class Eq(Relation):
    """Equality with absolute tolerance: left == right (within tol)."""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    tol: float = 1e-9
    name: str = "Equality"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        m = np.isclose(l.values, r.values, atol=self.tol)
        return pd.Series(m, index=df.index, dtype=bool)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series(self.tol - np.abs(l - r), index=df.index, dtype=float)

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        eq = "=" if unicode_ops else "=="
        lhs = repr(self.left)
        rhs = repr(self.right)
        if show_tol:
            pm = "±" if unicode_ops else "+/-"
            return f"{lhs} {eq} {rhs} {pm} {self.tol:g}"
        return f"{lhs} {eq} {rhs}"

    def __repr__(self) -> str:
        # Pretty by default (unicode), include tol only when nonzero and helpful
        pm = f" ± {self.tol:g}" if self.tol else ""
        return f"{repr(self.left)} = {repr(self.right)}{pm}"

class Lt(Relation):
    """
    Strict less-than: lhs < rhs - tol.
    If tol==0, this is a pointwise strict <.
    """
    def __init__(self, lhs: Expr, rhs: Union[Expr, float, int], *, tol: float = 0.0, name: Optional[str] = None):
        self.lhs = to_expr(lhs)
        self.rhs = to_expr(rhs)
        self.tol = float(tol)
        self._name = name

    def pretty(self) -> str:
        L = self.lhs.pretty() if hasattr(self.lhs, "pretty") else repr(self.lhs)
        R = self.rhs.pretty() if hasattr(self.rhs, "pretty") else repr(self.rhs)
        if self.tol > 0.0:
            return self._name or f"{L} < {R} - {self.tol:g}"
        return self._name or f"{L} < {R}"

    def evaluate(self, df: pd.DataFrame, condition: Optional["Predicate"] = None) -> pd.Series:
        a = self.lhs.eval(df).astype(float, copy=False)
        b = self.rhs.eval(df).astype(float, copy=False)
        mask = a < (b - self.tol)
        # Ensure boolean series aligned to df; drop NaNs as False
        if hasattr(mask, "fillna"):
            mask = mask.fillna(False)
        mask = mask.astype(bool, copy=False)
        if condition is not None:
            C = condition.mask(df).astype(bool, copy=False)
            mask = mask & C
        return mask

@dataclass
class Le(Relation):
    """Inequality: left <= right ; slack = (right - left)."""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    name: str = "Inequality(<=)"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series((l <= r).values, index=df.index, dtype=bool)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series((r - l).values, index=df.index, dtype=float)

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        sym = "≤" if unicode_ops else "<="
        return f"{repr(self.left)} {sym} {repr(self.right)}"

    def __repr__(self) -> str:
        # Pretty by default (unicode)
        return f"{repr(self.left)} ≤ {repr(self.right)}"

class Gt(Relation):
    """
    Strict greater-than: lhs > rhs + tol.
    If tol==0, this is a pointwise strict >.
    """
    def __init__(self, lhs: Expr, rhs: Union[Expr, float, int], *, tol: float = 0.0, name: Optional[str] = None):
        self.lhs = to_expr(lhs)
        self.rhs = to_expr(rhs)
        self.tol = float(tol)
        self._name = name

    def pretty(self) -> str:
        L = self.lhs.pretty() if hasattr(self.lhs, "pretty") else repr(self.lhs)
        R = self.rhs.pretty() if hasattr(self.rhs, "pretty") else repr(self.rhs)
        if self.tol > 0.0:
            return self._name or f"{L} > {R} + {self.tol:g}"
        return self._name or f"{L} > {R}"

    def evaluate(self, df: pd.DataFrame, condition: Optional["Predicate"] = None) -> pd.Series:
        a = self.lhs.eval(df).astype(float, copy=False)
        b = self.rhs.eval(df).astype(float, copy=False)
        mask = a > (b + self.tol)
        if hasattr(mask, "fillna"):
            mask = mask.fillna(False)
        mask = mask.astype(bool, copy=False)
        if condition is not None:
            C = condition.mask(df).astype(bool, copy=False)
            mask = mask & C
        return mask

@dataclass
class Ge(Relation):
    """Inequality: left >= right ; slack = (left - right)."""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    name: str = "Inequality(>=)"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series((l >= r).values, index=df.index, dtype=bool)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series((l - r).values, index=df.index, dtype=float)

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        sym = "≥" if unicode_ops else ">="
        return f"{repr(self.left)} {sym} {repr(self.right)}"

    def __repr__(self) -> str:
        # Pretty by default (unicode)
        return f"{repr(self.left)} ≥ {repr(self.right)}"


@dataclass
class AllOf(Relation):
    """Conjunction of relations: R1 ∧ R2 ∧ ... ; slack = min(child slacks)."""
    parts: Iterable[Relation]
    name: str = "AllOf"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(True, index=df.index, dtype=bool)
        for r in self.parts:
            out &= r.evaluate(df).reindex(df.index).astype(bool)
        return out

    def slack(self, df: pd.DataFrame) -> pd.Series:
        slacks: List[pd.Series] = [r.slack(df).reindex(df.index) for r in self.parts]
        if not slacks:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.concat(slacks, axis=1).min(axis=1)

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        glue = " ∧ " if unicode_ops else " & "
        items: List[str] = []
        for p in self.parts:
            if hasattr(p, "pretty"):
                items.append(p.pretty(unicode_ops=unicode_ops, show_tol=show_tol))  # type: ignore[call-arg]
            else:
                items.append(repr(p))
        return glue.join(items)

    def __repr__(self) -> str:
        # Use each part's __repr__ (already pretty) and join with ∧
        return " ∧ ".join(repr(p) for p in self.parts)


@dataclass
class AnyOf(Relation):
    """Disjunction of relations: R1 ∨ R2 ∨ ... ; slack = max(child slacks)."""
    parts: Iterable[Relation]
    name: str = "AnyOf"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(False, index=df.index, dtype=bool)
        for r in self.parts:
            out |= r.evaluate(df).reindex(df.index).astype(bool)
        return out

    def slack(self, df: pd.DataFrame) -> pd.Series:
        slacks: List[pd.Series] = [r.slack(df).reindex(df.index) for r in self.parts]
        if not slacks:
            return pd.Series(0.0, index=df.index, dtype=float)
        return pd.concat(slacks, axis=1).max(axis=1)

    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        glue = " ∨ " if unicode_ops else " | "
        items: List[str] = []
        for p in self.parts:
            if hasattr(p, "pretty"):
                items.append(p.pretty(unicode_ops=unicode_ops, show_tol=show_tol))  # type: ignore[call-arg]
            else:
                items.append(repr(p))
        return glue.join(items)

    def __repr__(self) -> str:
        # Use each part's __repr__ (already pretty) and join with ∨
        return " ∨ ".join(repr(p) for p in self.parts)


# =========================================================
# Conjecture: (R | C)
# =========================================================

@dataclass
class Conjecture:
    """
    General form: For any object in class C, relation R holds.  (R | C)

    .check(df, auto_base=True) returns:
      - applicable: mask where C holds,
      - holds:      mask for (R | C),
      - failures:   rows of df with applicable & ~evaluate + "__slack__".
    """
    relation: Relation
    condition: Optional[Predicate] = None
    name: str = "Conjecture"
    coefficient_pairs = None
    intercept = None

    # cached for nicer repr/pretty if condition is None and auto_base=True was used
    _resolved_condition: Optional[Predicate] = field(default=None, init=False, repr=False)

    # --------------------------- internals ---------------------------

    def _auto_base(self, df: pd.DataFrame) -> Predicate:
        """
        Detect a base predicate from boolean always-True columns.
        Supports both bool and pandas' nullable BooleanDtype.
        """
        if df is None or df.empty:
            return TRUE

        always_true_cols: List[str] = []
        for col in df.columns:
            s = df[col]
            if is_bool_dtype(s):
                # treat NaN as False for this test
                if bool(pd.Series(s).fillna(False).all()):
                    always_true_cols.append(col)

        if not always_true_cols:
            return TRUE

        preds = [Where(lambda d, c=col: d[c], name=f"{col}") for col in always_true_cols]
        base: Predicate = preds[0]
        for p in preds[1:]:
            base = AndPred(base, p)
        base.name = " ∧ ".join(f"{c}" for c in always_true_cols)
        return base

    # --------------------------- public API ---------------------------

    def check(
        self,
        df: pd.DataFrame,
        *,
        auto_base: bool = True,
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Evaluate the conjecture on a DataFrame.

        Returns:
            applicable, holds, failures
        """
        # resolve condition
        if self.condition is not None:
            cond = self.condition
        else:
            cond = self._auto_base(df) if auto_base else TRUE

        applicable = cond.mask(df).reindex(df.index, fill_value=False).astype(bool)
        eval_mask = self.relation.evaluate(df).reindex(df.index).astype(bool)
        holds = (~applicable) | (applicable & eval_mask)

        failing = (applicable & ~eval_mask)
        failures = df.loc[failing].copy()
        if failing.any():
            s = self.relation.slack(df).reindex(df.index)
            failures["__slack__"] = s.loc[failing]

        # cache for nicer __repr__/pretty
        self._resolved_condition = cond
        return applicable, holds, failures

    def is_true(self, df: pd.DataFrame, *, auto_base: bool = True) -> bool:
        applicable, holds, _ = self.check(df, auto_base=auto_base)
        return bool(holds[applicable].all())

    # def touch_count(self, df: pd.DataFrame, *, auto_base: bool = True, atol: float = 1e-12) -> int:
    #     applicable, _, _ = self.check(df, auto_base=auto_base)
    #     tight = self.relation.is_tight(df, atol=atol).reindex(df.index)
    #     self.touch = int((applicable & tight).sum())
    #     return self.touch
    def touch_count(self, df: pd.DataFrame, *, auto_base: bool = True, atol: float = 1e-12) -> int:
        applicable, _, _ = self.check(df, auto_base=auto_base)
        tight = self.relation.is_tight(df, atol=atol).reindex(df.index)
        val = int((applicable & tight).sum())
        # keep both for backward compatibility
        setattr(self, "touch", val)
        setattr(self, "touch_count", val)
        return val

    def violation_count(self, df: pd.DataFrame, *, auto_base: bool = True) -> int:
        applicable, holds, _ = self.check(df, auto_base=auto_base)
        return int((applicable & ~holds).sum())

    def pretty(
        self,
        arrow: Optional[str] = None,
        *,
        unicode_ops: bool = True,
        show_tol: bool = False,
    ) -> str:
        """
        Human-facing rendering:
            (cond) ⇒ (lhs ≤ rhs) ∧ (lhs ≥ rhs)  ...
        If the condition is TRUE, returns just the relation string.

        Parameters
        ----------
        arrow : Optional[str]
            Force the arrow symbol. Defaults to '⇒' if unicode_ops else '->'.
        unicode_ops : bool
            Use unicode math symbols.
        show_tol : bool
            If True, show ±tol for Eq relations.
        """
        cond = self.condition or self._resolved_condition or TRUE

        if hasattr(self.relation, "pretty"):
            rel_str = self.relation.pretty(unicode_ops=unicode_ops, show_tol=show_tol)  # type: ignore[call-arg]
        else:
            rel_str = repr(self.relation)

        # If TRUE, omit condition.
        if isinstance(cond, TRUE_Predicate):
            return rel_str

        cond_str = _pretty_predicate(cond, unicode_ops=unicode_ops)
        arr = ("⇒" if unicode_ops else "->") if arrow is None else arrow
        return f"{cond_str} {arr} {rel_str}"

    def signature(self) -> str:
        """Canonical-ish string signature for deduplication (mirrors pretty())."""
        return self.pretty(unicode_ops=True, show_tol=False)

    def __repr__(self) -> str:
        # Keep a compact debug form that is still pretty—same as pretty() w/o arrow
        cond = self.condition or self._resolved_condition or TRUE
        if isinstance(cond, TRUE_Predicate):
            return f"{repr(self.relation)}"
        return f"Conjecture({repr(self.relation)} | {repr(cond)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Conjecture):
            return False
        # Use the canonical signature (pretty + normalized condition)
        return self.signature() == other.signature()

    def __hash__(self) -> int:
        # Avoid recursion; hash a stable, human-readable signature
        return hash(self.signature())
