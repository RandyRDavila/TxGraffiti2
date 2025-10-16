# src/txgraffiti2025/forms/class_relations.py

"""
R₁-type conjectures: inclusion and equivalence between classes.

Express logical statements purely at the level of **classes** (predicates),
independent of numeric invariants:

- Inclusion:    ``C1 ⊆ C2``  (“all objects of class C1 are in class C2”)
- Equivalence:  ``C1 ≡ C2``  (“classes C1 and C2 coincide”)

Classes are represented by :class:`~txgraffiti2025.forms.predicates.Predicate`
objects that produce boolean masks over a DataFrame.
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
    """
    Logical class inclusion: ``A ⊆ B`` (i.e., implication ``A → B`` holds row-wise).

    Parameters
    ----------
    A, B : Predicate
        Class predicates. The inclusion asserts: for every row with ``A`` True,
        ``B`` is also True.

    Attributes
    ----------
    A : Predicate
        Left-hand (subset) class.
    B : Predicate
        Right-hand (superset) class.
    name : str
        Display name (default: ``"ClassInclusion"``).

    Returns
    -------
    ClassInclusion
        Object providing `mask(df)` for satisfaction and `violations(df)`.

    Methods
    -------
    mask(df) : pd.Series
        Boolean Series aligned to ``df.index`` indicating where the implication
        ``(~A) | B`` is True (i.e., the inclusion holds for that row).
    violations(df) : pd.DataFrame
        Subset of ``df`` where ``A`` holds but ``B`` does not (counterexamples).

    Notes
    -----
    - Rows with ``A`` False vacuously satisfy the implication.
    - Useful for checking containment relationships among graph classes
      (e.g., ``trees ⊆ bipartite``, ``planar ∧ triangle_free ⊆ 3-colorable``).

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Predicate, GEQ
    >>> from txgraffiti2025.forms.class_relations import ClassInclusion
    >>> df = pd.DataFrame({"connected": [True, False, True], "deg_min": [1, 0, 3]})
    >>> C_conn = Predicate.from_column("connected")
    >>> C_min_deg_ge1 = GEQ("deg_min", 1)
    >>> incl = ClassInclusion(C_min_deg_ge1, C_conn)   # "min degree ≥1 ⊆ connected"
    >>> incl.mask(df).tolist()                         # (~A) | B
    [True, True, True]
    >>> incl.violations(df).empty
    True

    A failing example:

    >>> df2 = pd.DataFrame({"A": [True, True], "B": [True, False]})
    >>> incl2 = ClassInclusion(Predicate.from_column("A"), Predicate.from_column("B"))
    >>> incl2.mask(df2).tolist()
    [True, False]
    >>> incl2.violations(df2).index.tolist()
    [1]
    """
    A: Predicate
    B: Predicate
    name: str = "ClassInclusion"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate the implication ``A → B`` per row.

        Parameters
        ----------
        df : pd.DataFrame
            Table of objects; columns used by the predicates must be present.

        Returns
        -------
        pd.Series
            Boolean Series where ``(~A) | B`` holds.
        """
        a_mask = self.A.mask(df).reindex(df.index, fill_value=False)
        b_mask = self.B.mask(df).reindex(df.index, fill_value=False)
        return (~a_mask) | b_mask

    def violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the subset of rows that violate the inclusion (``A ∧ ¬B``).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            ``df.loc[ A & ~B ]`` — the counterexamples to ``A ⊆ B``.

        Examples
        --------
        >>> import pandas as pd
        >>> from txgraffiti2025.forms.class_relations import ClassInclusion
        >>> from txgraffiti2025.forms.predicates import Predicate
        >>> df = pd.DataFrame({"A": [True, False], "B": [False, True]})
        >>> incl = ClassInclusion(Predicate.from_column("A"), Predicate.from_column("B"))
        >>> incl.violations(df).index.tolist()
        [0]
        """
        bad = self.A.mask(df).reindex(df.index, fill_value=False) & ~self.B.mask(df).reindex(df.index, fill_value=False)
        return df.loc[bad]

    def __repr__(self) -> str:
        return f"({self.A!r} ⊆ {self.B!r})"


@dataclass
class ClassEquivalence:
    """
    Logical class equivalence: ``A ≡ B`` (row-wise equality of masks).

    Parameters
    ----------
    A, B : Predicate
        Class predicates. The equivalence asserts the masks coincide: ``A == B``.

    Attributes
    ----------
    A : Predicate
        First class.
    B : Predicate
        Second class.
    name : str
        Display name (default: ``"ClassEquivalence"``).

    Returns
    -------
    ClassEquivalence
        Object providing `mask(df)` and `violations(df)`.

    Methods
    -------
    mask(df) : pd.Series
        Boolean Series where ``A.mask(df) == B.mask(df)``.
    violations(df) : pd.DataFrame
        Subset of ``df`` where class membership differs (symmetric difference).

    Notes
    -----
    - Equivalences often arise from characterizations (e.g., “`G` is a tree
      iff it is connected and acyclic” represented as equality of masks).

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Predicate
    >>> from txgraffiti2025.forms.class_relations import ClassEquivalence
    >>> df = pd.DataFrame({"C1": [True, False, True], "C2": [True, False, True]})
    >>> eqv = ClassEquivalence(Predicate.from_column("C1"), Predicate.from_column("C2"))
    >>> eqv.mask(df).tolist()
    [True, True, True]
    >>> eqv.violations(df).empty
    True

    Mismatch example:

    >>> df2 = pd.DataFrame({"C1": [True, False], "C2": [False, False]})
    >>> eqv2 = ClassEquivalence(Predicate.from_column("C1"), Predicate.from_column("C2"))
    >>> eqv2.mask(df2).tolist()
    [False, True]
    >>> eqv2.violations(df2).index.tolist()
    [0]
    """
    A: Predicate
    B: Predicate
    name: str = "ClassEquivalence"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate the equivalence per row.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.Series
            Boolean Series where ``A.mask(df) == B.mask(df)``.
        """
        a_mask = self.A.mask(df).reindex(df.index, fill_value=False)
        b_mask = self.B.mask(df).reindex(df.index, fill_value=False)
        return a_mask == b_mask

    def violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the subset of rows that violate the equivalence.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            ``df.loc[ A ^ B ]`` — rows where membership differs (symmetric diff).

        Examples
        --------
        >>> import pandas as pd
        >>> from txgraffiti2025.forms.class_relations import ClassEquivalence
        >>> from txgraffiti2025.forms.predicates import Predicate
        >>> df = pd.DataFrame({"A": [True, False], "B": [False, False]})
        >>> ClassEquivalence(Predicate.from_column("A"), Predicate.from_column("B")).violations(df).index.tolist()
        [0]
        """
        bad = ~self.mask(df)
        return df.loc[bad]

    def __repr__(self) -> str:
        return f"({self.A!r} ≡ {self.B!r})"
