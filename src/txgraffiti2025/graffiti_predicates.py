from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import pandas as pd

__all__ = ["Predicate", "Where", "TRUE"]


# ────────────────────────── boolean predicates ────────────────────────── #

class Predicate:
    """
    Boolean predicate over rows. `mask(df) -> bool Series` aligned to df.index.
    Supports &, |, ^, ~. Also supports implication (>>) and biconditional (.iff()).
    """

    def mask(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.mask(df)

    def pretty(self) -> str:
        return repr(self)

    # boolean ops (on predicates)
    def __and__(self, other: "Predicate") -> "Predicate":
        return _And(self, other)

    def __or__(self, other: "Predicate") -> "Predicate":
        return _Or(self, other)

    def __xor__(self, other: "Predicate") -> "Predicate":
        return _Xor(self, other)

    def __invert__(self) -> "Predicate":
        return _Not(self)

    # implication / biconditional as BoolFormula
    def __rshift__(self, other):
        from .graffiti_generic_conjecture import coerce_formula, Implies
        return Implies(coerce_formula(self), coerce_formula(other))

    def iff(self, other):
        from .graffiti_generic_conjecture import coerce_formula, Iff
        return Iff(coerce_formula(self), coerce_formula(other))

    @classmethod
    def from_column(cls, name: str, truthy_only: bool = True) -> "Predicate":
        return _ColumnPredicate(name, truthy_only=truthy_only)


@dataclass(frozen=True)
class _ColumnPredicate(Predicate):
    name: str
    truthy_only: bool = True
    def mask(self, df: pd.DataFrame) -> pd.Series:
        if self.name not in df.columns:
            raise KeyError(f"Column '{self.name}' not in DataFrame.")
        s = df[self.name]
        if s.dtype == bool or str(s.dtype).lower().startswith("boolean"):
            out = s
        else:
            # Treat nonzero as True; NA -> False
            out = (s.fillna(0) != 0)
        return out.reindex(df.index, fill_value=False).astype(bool, copy=False)
    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Where(Predicate):
    fn: Callable[[pd.DataFrame], pd.Series]
    name: Optional[str] = None
    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = self.fn(df)
        if not isinstance(m, pd.Series):
            m = pd.Series(m, index=df.index)
        return m.reindex(df.index, fill_value=False).fillna(False).astype(bool, copy=False)
    def __repr__(self) -> str:
        return self.name or "where(...)"

@dataclass(frozen=True)
class _And(Predicate):
    a: Predicate
    b: Predicate
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (self.a.mask(df) & self.b.mask(df)).astype(bool, copy=False)
    def __repr__(self) -> str:
        return f"({self.a!r} ∧ {self.b!r})"

@dataclass(frozen=True)
class _Or(Predicate):
    a: Predicate
    b: Predicate
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (self.a.mask(df) | self.b.mask(df)).astype(bool, copy=False)
    def __repr__(self) -> str:
        return f"({self.a!r} ∨ {self.b!r})"

@dataclass(frozen=True)
class _Xor(Predicate):
    a: Predicate
    b: Predicate
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (self.a.mask(df) ^ self.b.mask(df)).astype(bool, copy=False)
    def __repr__(self) -> str:
        return f"({self.a!r} ⊕ {self.b!r})"

@dataclass(frozen=True)
class _Not(Predicate):
    a: Predicate
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return (~self.a.mask(df)).astype(bool, copy=False)
    def __repr__(self) -> str:
        return f"¬({self.a!r})"


# Canonical TRUE predicate
class _TruePred(Predicate):
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)
    def __repr__(self) -> str:
        return "TRUE"

TRUE = _TruePred()
