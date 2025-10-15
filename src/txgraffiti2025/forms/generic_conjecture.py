"""
Generic, dataframe-agnostic forms:

- Relation types R: Eq, Le, Ge, AllOf, AnyOf
- Conjecture: (R | C) meaning "for all rows in class C, relation R holds"

Other form-specific helpers live in:
- linear.py, nonlinear.py, floorceil.py, logexp.py (algebraic)
- qualitative.py (R6)
- implication.py (R4 between relations)
- predicates.py (class conditions C)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .utils import Expr, to_expr
from .predicates import Predicate

__all__ = [
    "Relation",
    "Eq",
    "Le",
    "Ge",
    "AllOf",
    "AnyOf",
    "Conjecture",
]


# =========================================================
# Relations R (evaluate to a boolean Series on a DataFrame)
# =========================================================

class Relation:
    """A relation R that can be checked row-wise on a DataFrame."""
    name: str = "Relation"
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series indexed like df: True where the relation holds."""
        raise NotImplementedError
    def slack(self, df: pd.DataFrame) -> pd.Series:
        """Optional real-valued 'margin' (positive good) for inequalities; zeros for equalities."""
        raise NotImplementedError

@dataclass
class Eq(Relation):
    """left == right (within tolerance)"""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    tol: float = 1e-9
    name: str = "Equality"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series(np.isclose(l, r, atol=self.tol), index=df.index)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        # negative absolute error so "larger is better" convention remains (0 best)
        return pd.Series(-np.abs(l - r), index=df.index)

@dataclass
class Le(Relation):
    """left <= right"""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    name: str = "Inequality(<=)"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series(l <= r, index=df.index)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series(r - l, index=df.index)

@dataclass
class Ge(Relation):
    """left >= right"""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    name: str = "Inequality(>=)"

    def __post_init__(self):
        self.left = to_expr(self.left)
        self.right = to_expr(self.right)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series(l >= r, index=df.index)

    def slack(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.eval(df); r = self.right.eval(df)
        return pd.Series(l - r, index=df.index)

@dataclass
class AllOf(Relation):
    """Conjunction of relations: R1 ∧ R2 ∧ ..."""
    parts: Iterable[Relation]
    name: str = "AllOf"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(True, index=df.index)
        for r in self.parts:
            out &= r.evaluate(df)
        return out

    def slack(self, df: pd.DataFrame) -> pd.Series:
        slacks = [r.slack(df) for r in self.parts]
        if not slacks:
            return pd.Series(0.0, index=df.index)
        return pd.concat(slacks, axis=1).min(axis=1)

@dataclass
class AnyOf(Relation):
    """Disjunction of relations: R1 ∨ R2 ∨ ..."""
    parts: Iterable[Relation]
    name: str = "AnyOf"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(False, index=df.index)
        for r in self.parts:
            out |= r.evaluate(df)
        return out

    def slack(self, df: pd.DataFrame) -> pd.Series:
        slacks = [r.slack(df) for r in self.parts]
        if not slacks:
            return pd.Series(0.0, index=df.index)
        return pd.concat(slacks, axis=1).max(axis=1)

# =========================================================
# Conjecture: (R | C)
# =========================================================

@dataclass
class Conjecture:
    """General form: For any object in class C, relation R holds.  (R | C)"""
    relation: Relation
    condition: Optional[Predicate] = None
    name: str = "Conjecture"

    def check(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Returns:
            applicable: mask where C holds (or all True if no C)
            holds: mask where R holds among applicable rows
            failures: df slice for applicable rows where R fails, with '__slack__'
        """
        applicable = (self.condition.mask(df) if self.condition
                      else pd.Series(True, index=df.index)).reindex(df.index, fill_value=False)
        eval_mask = self.relation.evaluate(df)
        holds = (~applicable) | (applicable & eval_mask)

        # failures (within applicable)
        failing = applicable & ~eval_mask
        failures = df.loc[failing].copy()
        if len(failures):
            failures["__slack__"] = self.relation.slack(df).loc[failing]
        return applicable, holds, failures
