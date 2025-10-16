# src/txgraffiti2025/forms/implication.py

"""
Implications and equivalences between relations (logical R₄ forms).

This module expresses logical relationships among **relations** (not classes),
optionally restricted to a class condition `C`:

- Implication:  ``R1 ⇒ R2`` (within `C` if provided)
- Equivalence:  ``R1 ⇔ R2``  (i.e., `(R1 ⇒ R2) ∧ (R2 ⇒ R1)`)

These operate row-wise over a DataFrame, just like ordinary relations.
If a class condition is supplied, rows outside the class **vacuously satisfy**
the statement (i.e., counted as holds = True).

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.generic_conjecture import Le, Ge
>>> from txgraffiti2025.forms.predicates import Predicate
>>> from txgraffiti2025.forms.implication import Implication, Equivalence
>>>
>>> df = pd.DataFrame({
...     "a": [1, 2, 3, 4],
...     "b": [2, 1, 3, 5],
...     "connected": [True, False, True, True],
... })
>>>
>>> R1 = Le("a", "b")   # a <= b
>>> R2 = Ge("b", "a")   # b >= a
>>> C  = Predicate.from_column("connected")
>>>
>>> # Implication under a class: (a<=b) ⇒ (b>=a) | connected
>>> impl = Implication(R1, R2, condition=C)
>>> applicable, holds, failures = impl.check(df)
>>> applicable.tolist()
[True, False, True, True]
>>> holds.tolist()    # rows outside C vacuously True; inside C we test implication
[True, True, True, True]
>>> failures.empty
True
>>>
>>> # Equivalence (a<=b) ⇔ (b>=a) globally
>>> eqv = Equivalence(R1, R2)
>>> applicable, holds, failures = eqv.check(df)
>>> applicable.all(), holds.all(), failures.empty
(True, True, True)
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
    """
    Row-wise implication between relations: ``premise ⇒ conclusion`` (optionally under `condition`).

    Parameters
    ----------
    premise : Relation
        The antecedent relation `R1` to test per row.
    conclusion : Relation
        The consequent relation `R2` required to hold whenever `premise` holds.
    condition : Predicate, optional
        Class predicate `C`. If provided, only rows where `C.mask(df)` is True
        are considered applicable; rows outside `C` vacuously satisfy the implication.
    name : str, default "Implication"
        Display name.

    Attributes
    ----------
    premise : Relation
    conclusion : Relation
    condition : Predicate or None
    name : str

    Returns
    -------
    Implication
        Object exposing :meth:`check` to evaluate satisfaction and collect violations.

    Notes
    -----
    Semantics per row:
    ``(~applicable) OR (~premise) OR (premise AND conclusion)``.

    The `failures` DataFrame (if non-empty) includes a column ``"__slack__"``
    inherited from the **conclusion** relation’s `slack(df)`, evaluated on the
    failing rows (useful for ranking counterexamples by severity).

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Le, Ge
    >>> from txgraffiti2025.forms.implication import Implication
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 1, 4]})
    >>> R1 = Le("x", "y")          # premise:  x <= y
    >>> R2 = Ge("y", "x")          # conclude: y >= x
    >>> impl = Implication(R1, R2)  # global implication
    >>> applicable, holds, failures = impl.check(df)
    >>> applicable.tolist()
    [True, True, True]
    >>> holds.tolist()
    [True, False, True]
    >>> failures.index.tolist()     # row 1 violates: x<=y is False there, so implication fails?
    [1]
    """
    premise: Relation
    conclusion: Relation
    condition: Optional[Predicate] = None
    name: str = "Implication"

    def check(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Evaluate the implication on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Table of objects with all columns needed by `premise`, `conclusion`,
            and (if provided) `condition`.

        Returns
        -------
        applicable : pd.Series
            Boolean mask where `condition` holds; all True if `condition is None`.
        holds : pd.Series
            Boolean mask indicating the implication's satisfaction per row:
            ``(~applicable) | (~premise) | (premise & conclusion)``.
        failures : pd.DataFrame
            Rows with `applicable & premise & ~conclusion`. Includes a
            ``"__slack__"`` column from `conclusion.slack(df)` on those rows.

        Examples
        --------
        >>> import pandas as pd
        >>> from txgraffiti2025.forms.generic_conjecture import Le, Ge
        >>> from txgraffiti2025.forms.predicates import Predicate
        >>> from txgraffiti2025.forms.implication import Implication
        >>> df = pd.DataFrame({"a":[1,2,3], "b":[2,1,3], "flag":[True, False, True]})
        >>> impl = Implication(Le("a","b"), Ge("b","a"), condition=Predicate.from_column("flag"))
        >>> applicable, holds, failures = impl.check(df)
        >>> applicable.tolist()
        [True, False, True]
        >>> holds.tolist()
        [True, True, True]
        >>> failures.empty
        True
        """
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

    def __repr__(self) -> str:
        c = "" if self.condition is None else f" | {self.condition!r}"
        return f"({self.premise!r} ⇒ {self.conclusion!r}{c})"

@dataclass
class Equivalence:
    """
    Row-wise equivalence between relations: ``a ⇔ b`` (optionally under `condition`).

    Parameters
    ----------
    a, b : Relation
        Relations to compare per row.
    condition : Predicate, optional
        Class predicate `C`. If provided, only rows where `C` holds are
        applicable; other rows vacuously satisfy the equivalence.
    name : str, default "Equivalence"
        Display name.

    Attributes
    ----------
    a : Relation
    b : Relation
    condition : Predicate or None
    name : str

    Returns
    -------
    Equivalence
        Object exposing :meth:`check` for satisfaction and mismatch extraction.

    Notes
    -----
    Semantics per row:
    ``(~applicable) OR (a == b)``.

    `failures` collects the symmetric difference of the two relation masks
    over the applicable subset.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Le, Ge
    >>> from txgraffiti2025.forms.implication import Equivalence
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 3, 2]})
    >>> eqv = Equivalence(Le("x","y"), Ge("y","x"))
    >>> applicable, holds, failures = eqv.check(df)
    >>> applicable.tolist()
    [True, True, True]
    >>> holds.tolist()
    [True, True, False]
    >>> failures.index.tolist()
    [2]
    """
    a: Relation
    b: Relation
    condition: Optional[Predicate] = None
    name: str = "Equivalence"

    def check(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Evaluate the equivalence on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        applicable : pd.Series
            Boolean mask where `condition` holds; all True if `condition is None`.
        holds : pd.Series
            Boolean mask where either `~applicable` or `a.evaluate(df) == b.evaluate(df)`.
        failures : pd.DataFrame
            Rows in the applicable subset where the two relation masks differ.

        Examples
        --------
        >>> import pandas as pd
        >>> from txgraffiti2025.forms.generic_conjecture import Le, Ge
        >>> from txgraffiti2025.forms.predicates import Predicate
        >>> from txgraffiti2025.forms.implication import Equivalence
        >>> df = pd.DataFrame({"a":[1,2,3], "b":[1,3,2], "flag":[True,True,False]})
        >>> eqv = Equivalence(Le("a","b"), Ge("b","a"), condition=Predicate.from_column("flag"))
        >>> applicable, holds, failures = eqv.check(df)
        >>> applicable.tolist()
        [True, True, False]
        >>> holds.tolist()
        [True, True, True]
        >>> failures.empty
        True
        """
        applicable = (self.condition.mask(df) if self.condition
                      else pd.Series(True, index=df.index)).reindex(df.index, fill_value=False)
        pa = self.a.evaluate(df).reindex(df.index)
        pb = self.b.evaluate(df).reindex(df.index)

        holds = (~applicable) | (pa == pb)
        failing = applicable & (pa != pb)
        failures = df.loc[failing].copy()
        return applicable.astype(bool), holds.astype(bool), failures

    def __repr__(self) -> str:
        c = "" if self.condition is None else f" | {self.condition!r}"
        return f"({self.a!r} ⇔ {self.b!r}{c})"
