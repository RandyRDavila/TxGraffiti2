# src/txgraffiti2025/forms/implication.py
"""
Implications and equivalences between relations (logical R₄ forms).

This module expresses logical relationships among **relations** (not classes),
optionally restricted by a class predicate `C`:

- Implication:  R₁ ⇒ R₂     (within C if provided)
- Equivalence:  R₁ ⇔ R₂     (i.e., (R₁ ⇒ R₂) ∧ (R₂ ⇒ R₁))

Semantics are row-wise over a DataFrame. If a condition `C` is supplied, rows
outside `C` **vacuously satisfy** the statement (counted as holds=True).

Returns from .check(df):
    applicable : boolean mask where C holds (or all True if C is None)
    holds      : boolean mask for the statement’s truth under C
    failures   : df rows that violate the statement under C, with "__slack__"
                 attached to help rank counterexamples (see notes for each).

Notes
-----
Implication R₁ ⇒ R₂ (under C)
    holds := (~C) OR (~R₁) OR (R₁ AND R₂)
    failures := C & R₁ & ~R₂, with "__slack__" from R₂.slack(df)

Equivalence R₁ ⇔ R₂ (under C)
    holds := (~C) OR (R₁ == R₂)
    failures := C & (R₁ != R₂); includes "__lhs__"=R₁, "__rhs__"=R₂ masks for clarity
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from .generic_conjecture import Relation
from .predicates import Predicate

__all__ = ["Implication", "Equivalence"]


# =====================================================================
# Implication
# =====================================================================

@dataclass
class Implication:
    """
    Row-wise implication between relations: premise ⇒ conclusion  (optionally under `condition`).

    - failures carry "__slack__" from `conclusion.slack(df)` on failing rows.
    - `touch_count(df)`: counts applicable rows where premise holds and
      `conclusion.slack(df) == 0` (tight conclusion).
    """
    premise: Relation
    conclusion: Relation
    condition: Optional[Predicate] = None
    name: str = "Implication"

    # --------------------------- core API ---------------------------

    def check(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Evaluate the implication on a DataFrame.

        Returns:
            applicable, holds, failures
        """
        if self.condition is None:
            applicable = pd.Series(True, index=df.index, dtype=bool)
        else:
            applicable = self.condition.mask(df).reindex(df.index, fill_value=False).astype(bool)

        p = self.premise.evaluate(df).reindex(df.index).astype(bool)
        q = self.conclusion.evaluate(df).reindex(df.index).astype(bool)

        # (~C) OR (~P) OR (P & Q)
        holds = (~applicable) | (~p) | (p & q)

        failing = (applicable & p & ~q)
        failures = df.loc[failing].copy()
        if failing.any():
            s = self.conclusion.slack(df).reindex(df.index)
            failures["__slack__"] = s.loc[failing]

        return applicable, holds, failures

    def violation_count(self, df: pd.DataFrame) -> int:
        applicable, holds, _ = self.check(df)
        return int((applicable & ~holds).sum())

    def touch_count(self, df: pd.DataFrame) -> int:
        """Count applicable rows where premise holds and the conclusion is tight."""
        if self.condition is None:
            applicable = pd.Series(True, index=df.index, dtype=bool)
        else:
            applicable = self.condition.mask(df).reindex(df.index, fill_value=False).astype(bool)

        p = self.premise.evaluate(df).reindex(df.index).astype(bool)
        s = self.conclusion.slack(df).reindex(df.index)
        return int((applicable & p & (s == 0)).sum())

    # --------------------------- pretty / repr ---------------------------

    def pretty(self, *, unicode_ops: bool = True, arrow: Optional[str] = None, show_condition: bool = True) -> str:
        """Human-friendly text, e.g. '(planar ∧ regular) ⇒ alpha ≤ mu'."""
        # prefer relation-specific pretty() if available
        def fmt_rel(r: Relation) -> str:
            if hasattr(r, "pretty"):
                return r.pretty(unicode_ops=unicode_ops)  # type: ignore[call-arg]
            return repr(r)

        body = f"{fmt_rel(self.premise)} {'⇒' if (unicode_ops and arrow is None) else (arrow or '->')} {fmt_rel(self.conclusion)}"
        if not show_condition or self.condition is None:
            return body
        cond = repr(self.condition)
        # omit explicit TRUE
        if cond.strip().upper() == "TRUE":
            return body
        return f"{cond} {'⇒' if (unicode_ops and arrow is None) else (arrow or '->')} {body}"

    def signature(self) -> str:
        """Stable textual signature (good for logs)."""
        return self.pretty(unicode_ops=True, arrow="⇒", show_condition=True)

    def __repr__(self) -> str:
        c = "" if self.condition is None else f" | {self.condition!r}"
        return f"({self.premise!r} ⇒ {self.conclusion!r}{c})"


# =====================================================================
# Equivalence
# =====================================================================

@dataclass
class Equivalence:
    """
    Row-wise equivalence between relations: a ⇔ b  (optionally under `condition`).

    - failures are the symmetric difference where masks differ under C.
    - For debugging, failures include boolean columns "__lhs__" and "__rhs__".
    """
    a: Relation
    b: Relation
    condition: Optional[Predicate] = None
    name: str = "Equivalence"

    # --------------------------- core API ---------------------------

    def check(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Evaluate the equivalence on a DataFrame.

        Returns:
            applicable, holds, failures
        """
        if self.condition is None:
            applicable = pd.Series(True, index=df.index, dtype=bool)
        else:
            applicable = self.condition.mask(df).reindex(df.index, fill_value=False).astype(bool)

        pa = self.a.evaluate(df).reindex(df.index).astype(bool)
        pb = self.b.evaluate(df).reindex(df.index).astype(bool)

        holds = (~applicable) | (pa == pb)
        failing = (applicable & (pa != pb))

        failures = df.loc[failing].copy()
        if failing.any():
            failures["__lhs__"] = pa.loc[failing]
            failures["__rhs__"] = pb.loc[failing]

        return applicable, holds, failures

    def violation_count(self, df: pd.DataFrame) -> int:
        applicable, holds, _ = self.check(df)
        return int((applicable & ~holds).sum())

    def touch_count(self, df: pd.DataFrame) -> int:
        """
        Count applicable rows that are 'tight' in an equivalence sense:
        We interpret 'tight' as both relations holding and at least one is tight
        (i.e., min(slack_a, slack_b) == 0 where both evaluate True), if slacks exist.
        If either relation lacks .slack, we return 0 conservatively.
        """
        try:
            if self.condition is None:
                applicable = pd.Series(True, index=df.index, dtype=bool)
            else:
                applicable = self.condition.mask(df).reindex(df.index, fill_value=False).astype(bool)

            pa = self.a.evaluate(df).reindex(df.index).astype(bool)
            pb = self.b.evaluate(df).reindex(df.index).astype(bool)

            sa = self.a.slack(df).reindex(df.index)
            sb = self.b.slack(df).reindex(df.index)

            both = (applicable & pa & pb)
            if not both.any():
                return 0
            tight = (sa.loc[both].combine(sb.loc[both], min) == 0)
            return int(tight.sum())
        except Exception:
            return 0

    # --------------------------- pretty / repr ---------------------------

    def pretty(self, *, unicode_ops: bool = True, show_condition: bool = True) -> str:
        """Human-friendly text, e.g. '(planar) ⇒ (alpha ≤ mu ⇔ size = order−1)'."""
        def fmt_rel(r: Relation) -> str:
            if hasattr(r, "pretty"):
                return r.pretty(unicode_ops=unicode_ops)  # type: ignore[call-arg]
            return repr(r)

        body = f"{fmt_rel(self.a)} {'⇔' if unicode_ops else '<=>'} {fmt_rel(self.b)}"
        if not show_condition or self.condition is None:
            return body
        cond = repr(self.condition)
        if cond.strip().upper() == "TRUE":
            return body
        return f"{cond} {'⇒' if unicode_ops else '->'} {body}"

    def signature(self) -> str:
        """Stable textual signature (good for logs)."""
        return self.pretty(unicode_ops=True, show_condition=True)

    def __repr__(self) -> str:
        c = "" if self.condition is None else f" | {self.condition!r}"
        return f"({self.a!r} ⇔ {self.b!r}{c})"
