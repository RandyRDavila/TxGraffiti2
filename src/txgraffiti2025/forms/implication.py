"""
Implications between relations (R₄):
    R1 ⇒ R2   (optionally under class C)

Also provides an Equivalence helper:
    R1 ⇔ R2   (i.e., (R1 ⇒ R2) ∧ (R2 ⇒ R1))
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

from .generic_conjecture import Relation
from .predicates import Predicate

__all__ = [
    "Implication",
    "Equivalence",
]


@dataclass
class Implication:
    """Row-wise implication: where premise holds, conclusion must hold (within C if provided)."""
    premise: Relation
    conclusion: Relation
    condition: Optional[Predicate] = None
    name: str = "Implication"

    def check(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        applicable = (self.condition.mask(df) if self.condition
                      else pd.Series(True, index=df.index)).reindex(df.index, fill_value=False)
        p = self.premise.evaluate(df).reindex(df.index)
        q = self.conclusion.evaluate(df).reindex(df.index)

        # Holds if: not applicable OR not premise OR (premise and conclusion)
        holds = (~applicable) | (~p) | (p & q)

        failing = applicable & p & ~q
        failures = df.loc[failing].copy()
        if len(failures):
            failures["__slack__"] = self.conclusion.slack(df).loc[failing]
        return applicable.astype(bool), holds.astype(bool), failures


@dataclass
class Equivalence:
    """Row-wise equivalence: R1 ⇔ R2 (optionally under C)."""
    a: Relation
    b: Relation
    condition: Optional[Predicate] = None
    name: str = "Equivalence"

    def check(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        applicable = (self.condition.mask(df) if self.condition
                      else pd.Series(True, index=df.index)).reindex(df.index, fill_value=False)
        pa = self.a.evaluate(df).reindex(df.index)
        pb = self.b.evaluate(df).reindex(df.index)

        holds = (~applicable) | (pa == pb)
        failing = applicable & (pa != pb)
        failures = df.loc[failing].copy()
        return applicable.astype(bool), holds.astype(bool), failures
