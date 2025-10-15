"""
R₁-type conjectures: inclusion and equivalence between classes.

Expresses logical statements like:
    C1 ⊆ C2      ("all objects of class C1 are in class C2")
    C1 ≡ C2      ("classes C1 and C2 coincide")

Purely logical relations between predicates (boolean masks on a DataFrame),
independent of numeric invariants.
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .predicates import Predicate

__all__ = [
    "ClassInclusion",
    "ClassEquivalence",
]

@dataclass
class ClassInclusion:
    """Represents C1 ⊆ C2."""
    A: Predicate
    B: Predicate
    name: str = "ClassInclusion"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series where the implication A→B holds row-wise."""
        a_mask = self.A.mask(df).reindex(df.index, fill_value=False)
        b_mask = self.B.mask(df).reindex(df.index, fill_value=False)
        return (~a_mask) | b_mask

    def violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return subset of df where A holds but B does not."""
        bad = self.A.mask(df).reindex(df.index, fill_value=False) & ~self.B.mask(df).reindex(df.index, fill_value=False)
        return df.loc[bad]


@dataclass
class ClassEquivalence:
    """Represents C1 ≡ C2."""
    A: Predicate
    B: Predicate
    name: str = "ClassEquivalence"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        a_mask = self.A.mask(df).reindex(df.index, fill_value=False)
        b_mask = self.B.mask(df).reindex(df.index, fill_value=False)
        return a_mask == b_mask

    def violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return subset of df where class membership differs."""
        bad = ~self.mask(df)
        return df.loc[bad]
