# src/txgraffiti2025/forms/class_relations.py
"""
R₁-type conjectures: inclusion and equivalence between classes.

Express logical statements purely at the level of **classes** (predicates),
independent of numeric invariants:

- Inclusion:    C₁ ⊆ C₂   (“all objects of class C₁ are in class C₂”)
- Equivalence:  C₁ ≡ C₂   (“classes C₁ and C₂ coincide”)

Classes are represented by :class:`~txgraffiti2025.forms.predicates.Predicate`
objects that produce boolean masks over a DataFrame.

API (summary)
-------------
Both classes expose:
    .mask(df)              -> boolean Series for row-wise truth
    .violations(df)        -> DataFrame subset of counterexamples
    .violation_count(df)   -> int
    .holds_all(df)         -> bool (i.e., .mask(df).all())
    .pretty(...)           -> human-friendly string (unicode or ASCII)
    .signature()           -> stable string (good for logs/tests)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .predicates import Predicate

__all__ = [
    "ClassInclusion",
    "ClassEquivalence",
]


# -----------------------------
# Helpers
# -----------------------------

def _bool_mask(p: Predicate, df: pd.DataFrame) -> pd.Series:
    """Mask from a predicate, aligned, dtype=bool, NA->False."""
    m = p.mask(df).reindex(df.index, fill_value=False)
    if m.dtype != bool:
        m = m.fillna(False).astype(bool, copy=False)
    return m


def _pred_name(p: Predicate) -> str:
    """
    Compact display name for a predicate.
    Uses repr(p) but strips redundant outer parens like '((planar))' -> '(planar)'.
    """
    s = repr(p).strip()
    # normalize any accidental double-wrapping
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        # stop if removing would change grouping (look for bare ' ∧ ', ' ∨ ', etc.)
        # but since our Predicate.__repr__ already adds its own parens sensibly,
        # one layer strip is enough for tidiness.
        s = inner
        break
    return f"({s})"


# -----------------------------
# Class inclusion:  A ⊆ B
# -----------------------------

@dataclass
class ClassInclusion:
    """
    Logical class inclusion: ``A ⊆ B`` (i.e., implication ``A → B`` holds row-wise).

    Methods
    -------
    mask(df)            : (~A) | B
    violations(df)      : rows with A & ~B
    violation_count(df) : count of violations
    holds_all(df)       : mask(df).all()
    pretty(...)         : "(A) ⊆ (B)" (or ASCII: "(A) <= (B)")
    signature()         : stable pretty string
    """
    A: Predicate
    B: Predicate
    name: str = "ClassInclusion"

    # --- core ---

    def mask(self, df: pd.DataFrame) -> pd.Series:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        return (~a) | b

    def violations(self, df: pd.DataFrame) -> pd.DataFrame:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        bad = a & ~b
        return df.loc[bad]

    # --- handy helpers ---

    def violation_count(self, df: pd.DataFrame) -> int:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        return int((a & ~b).sum())

    def holds_all(self, df: pd.DataFrame) -> bool:
        """True iff inclusion holds for every row (vacuous where A is False)."""
        return bool(self.mask(df).all())

    # --- formatting ---

    def pretty(self, *, unicode_ops: bool = True) -> str:
        op = "⊆" if unicode_ops else "<="
        return f"{_pred_name(self.A)} {op} {_pred_name(self.B)}"

    def signature(self) -> str:
        return self.pretty(unicode_ops=True)

    def __repr__(self) -> str:
        return f"({self.A!r} ⊆ {self.B!r})"


# -----------------------------
# Class equivalence:  A ≡ B
# -----------------------------

@dataclass
class ClassEquivalence:
    """
    Logical class equivalence: ``A ≡ B`` (row-wise equality of masks).

    Methods
    -------
    mask(df)            : A == B
    violations(df)      : rows where A ^ B
    violation_count(df) : count of violations
    holds_all(df)       : mask(df).all()
    pretty(...)         : "(A) ≡ (B)" (or ASCII: "(A) == (B)")
    signature()         : stable pretty string
    """
    A: Predicate
    B: Predicate
    name: str = "ClassEquivalence"

    # --- core ---

    def mask(self, df: pd.DataFrame) -> pd.Series:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        return a == b

    def violations(self, df: pd.DataFrame) -> pd.DataFrame:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        bad = a ^ b
        return df.loc[bad]

    # --- handy helpers ---

    def violation_count(self, df: pd.DataFrame) -> int:
        a = _bool_mask(self.A, df)
        b = _bool_mask(self.B, df)
        return int((a ^ b).sum())

    def holds_all(self, df: pd.DataFrame) -> bool:
        return bool(self.mask(df).all())

    # --- formatting ---

    def pretty(self, *, unicode_ops: bool = True) -> str:
        op = "≡" if unicode_ops else "=="
        return f"{_pred_name(self.A)} {op} {_pred_name(self.B)}"

    def signature(self) -> str:
        return self.pretty(unicode_ops=True)

    def __repr__(self) -> str:
        return f"({self.A!r} ≡ {self.B!r})"
