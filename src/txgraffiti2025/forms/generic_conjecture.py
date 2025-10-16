# src/txgraffiti2025/forms/generic_conjecture.py

"""
Generic, dataframe-agnostic forms:

- Relation types R: Eq, Le, Ge, AllOf, AnyOf
- Conjecture: (R | C) meaning "for all rows in class C, relation R holds"

Other form-specific helpers live in:
- linear.py, nonlinear.py, floorceil.py, logexp.py (algebraic)
- qualitative.py (R6)
- implication.py (R between relations)
- predicates.py (class conditions C)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd

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


class TRUE_Predicate(Predicate):
    """A predicate that always returns True for all rows of any DataFrame."""
    name: str = "TRUE"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)

    def __repr__(self):
        return "TRUE"

# export constant
TRUE = TRUE_Predicate()


# =========================================================
# Relations R (evaluate to a boolean Series on a DataFrame)
# =========================================================

class Relation:
    """
    Abstract base class for row-wise relations over a DataFrame of invariants.

    A `Relation` evaluates to a boolean mask aligned to the input DataFrame
    and exposes an optional real-valued `slack` that quantifies margin/tightness.

    Methods
    -------
    evaluate(df) : pd.Series
        Boolean Series (index-aligned) where the relation holds.
    slack(df) : pd.Series
        Real-valued margin used by post-processing (e.g., touch counting).
        Convention depends on the concrete subclass; see Notes.

    Notes
    -----
    Slack conventions in this module:
      - `Le`: slack = (rhs - lhs)          (>= 0 iff satisfied)
      - `Ge`: slack = (lhs - rhs)          (>= 0 iff satisfied)
      - `Eq`: slack = -abs(lhs - rhs)      (0 best; negative if violated)
      - `AllOf`: min of child slacks       (tightest term dominates)
      - `AnyOf`: max of child slacks       (any satisfied term can dominate)

    Examples
    --------
    These classes are abstract; see concrete subclasses like :class:`Le`
    and :class:`Eq` for runnable examples.
    """
    name: str = "Relation"

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series indexed like `df`: True where the relation holds."""
        raise NotImplementedError

    def slack(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a real-valued margin (positive is “more satisfied”).

        The exact convention is relation-specific; see class docstring Notes.
        """
        raise NotImplementedError


@dataclass
class Eq(Relation):
    """
    Equality relation: ``left == right`` within an absolute tolerance.

    Parameters
    ----------
    left, right : Union[Expr, float, int, str]
        Expressions or literals. Strings are parsed via :func:`to_expr`.
    tol : float, default=1e-9
        Absolute tolerance used by `numpy.isclose`.
    name : str, default="Equality"
        Display name.

    Returns
    -------
    Eq
        A relation object usable in :class:`Conjecture` or directly.

    Notes
    -----
    The slack is ``-abs(left - right)`` so that equality has slack 0 and
    deviations are negative (keeps the “larger is better” convention aligned
    with inequalities when used in `AllOf`/`AnyOf`).

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Eq
    >>> from txgraffiti2025.forms.utils import to_expr
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2.000000001, 4]})
    >>> r = Eq("a", "b", tol=1e-6)
    >>> r.evaluate(df).tolist()
    [True, True, False]
    >>> list(round(x, 6) for x in r.slack(df).tolist())
    [-0.0, -0.0, -1.0]
    """
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

    def __repr__(self) -> str:
        return f"Eq({self.left!r} == {self.right!r}, tol={self.tol})"


@dataclass
class Le(Relation):
    """
    Inequality relation: ``left <= right``.

    Parameters
    ----------
    left, right : Union[Expr, float, int, str]
        Expressions or literals. Strings are parsed via :func:`to_expr`.
    name : str, default="Inequality(<=)"
        Display name.

    Returns
    -------
    Le
        A relation object usable in :class:`Conjecture` or directly.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Le
    >>> df = pd.DataFrame({"alpha": [2, 3, 4], "mu": [2, 5, 3]})
    >>> r = Le("alpha", "mu")
    >>> r.evaluate(df).tolist()
    [True, True, False]
    >>> r.slack(df).tolist()
    [0, 2, -1]
    """
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

    def __repr__(self) -> str:
        return f"Le({self.left!r} <= {self.right!r})"

@dataclass
class Ge(Relation):
    """
    Inequality relation: ``left >= right``.

    Parameters
    ----------
    left, right : Union[Expr, float, int, str]
        Expressions or literals. Strings are parsed via :func:`to_expr`.
    name : str, default="Inequality(>=)"
        Display name.

    Returns
    -------
    Ge
        A relation object usable in :class:`Conjecture` or directly.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Ge
    >>> df = pd.DataFrame({"alpha": [2, 3, 4], "residue": [1, 5, 4]})
    >>> r = Ge("alpha", "residue")
    >>> r.evaluate(df).tolist()
    [True, False, True]
    >>> r.slack(df).tolist()
    [1, -2, 0]
    """
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

    def __repr__(self) -> str:
        return f"Ge({self.left!r} >= {self.right!r})"

@dataclass
class AllOf(Relation):
    """
    Logical conjunction (AND) of relations: ``R1 ∧ R2 ∧ ...``.

    Parameters
    ----------
    parts : Iterable[Relation]
        Child relations to be AND-ed.
    name : str, default="AllOf"

    Returns
    -------
    AllOf
        Composite relation whose slack is the elementwise minimum of parts.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Le, Ge, AllOf
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3], "c": [0, 0, 2]})
    >>> r = AllOf([Le("a", "b"), Ge("b", "c")])
    >>> r.evaluate(df).tolist()   # a<=b AND b>=c
    [True, False, True]
    >>> r.slack(df).tolist()      # min( (b-a), (b-c) )
    [1, -1, 1]
    """
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

    def __repr__(self) -> str:
        parts = " ∧ ".join(repr(p) for p in self.parts)
        return f"AllOf({parts})"

@dataclass
class AnyOf(Relation):
    """
    Logical disjunction (OR) of relations: ``R1 ∨ R2 ∨ ...``.

    Parameters
    ----------
    parts : Iterable[Relation]
        Child relations to be OR-ed.
    name : str, default="AnyOf"

    Returns
    -------
    AnyOf
        Composite relation whose slack is the elementwise maximum of parts.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Le, Ge, AnyOf
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3], "c": [0, 3, 2]})
    >>> r = AnyOf([Le("a", "b"), Ge("b", "c")])
    >>> r.evaluate(df).tolist()   # a<=b OR b>=c
    [True, True, True]
    >>> r.slack(df).tolist()      # max( (b-a), (b-c) )
    [1, 0, 1]
    """
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

    def __repr__(self) -> str:
        parts = " ∨ ".join(repr(p) for p in self.parts)
        return f"AnyOf({parts})"


# =========================================================
# Conjecture: (R | C)
# =========================================================
@dataclass
class Conjecture:
    """
    General form: For any object in class C, relation R holds.  (R | C)

    Parameters
    ----------
    relation : Relation
        The relation `R` to check row-wise.
    condition : Predicate, optional
        Class predicate `C`. If None and `auto_base=True` in :meth:`check`,
        a base predicate is auto-detected from the DataFrame; otherwise the
        universal `TRUE` is used.
    name : str, default "Conjecture"
        Display name.

    Notes
    -----
    **Auto base detection (when `condition is None` and `auto_base=True`):**
    - If one or more boolean columns are True for all rows, they define the base.
      - One column -> `(col)`
      - Multiple -> `((col1) ∧ (col2) ∧ ...)`
    - If none, uses `TRUE` (all rows applicable).

    Examples
    --------
    Auto base picks `(connected)` when that column is all True:

    >>> import pandas as pd
    >>> from txgraffiti2025.forms.generic_conjecture import Conjecture, Le
    >>> df = pd.DataFrame({"a":[1,2,3], "b":[2,2,4], "connected":[True,True,True]})
    >>> C = Conjecture(Le("a","b"))              # no condition provided
    >>> applicable, holds, _ = C.check(df)       # auto_base=True by default
    >>> C  # doctest: +ELLIPSIS
    Conjecture(Le('a' <= 'b') | (connected))

    Turn off auto base (treat as global):

    >>> C2 = Conjecture(Le("a","b"))
    >>> _, holds2, _ = C2.check(df, auto_base=False)
    >>> holds2.all()
    True
    """
    relation: Relation
    condition: Optional[Predicate] = None
    name: str = "Conjecture"

    def _auto_base(self, df: pd.DataFrame) -> Predicate:
        """
        Detect a base predicate from boolean always-True columns.
        Returns a Where/AndPred with nice names like `(connected)` or
        `((connected) ∧ (simple))`; falls back to TRUE.
        """
        # Local import to avoid cycles if TRUE lives here
        try:
            TRUE_obj = TRUE  # type: ignore[name-defined]
        except NameError:
            # Define an inline TRUE if not available for some reason
            class _TRUE(Predicate):
                name = "TRUE"
                def mask(self, df: pd.DataFrame) -> pd.Series:
                    return pd.Series(True, index=df.index)
            TRUE_obj = _TRUE()

        if df is None or df.empty:
            return TRUE_obj

        always_true_cols = [
            col for col in df.columns
            if df[col].dtype == bool and bool(df[col].all())
        ]
        if not always_true_cols:
            return TRUE_obj

        preds = [Where(lambda d, c=col: d[c], name=f"({col})") for col in always_true_cols]
        if len(preds) == 1:
            return preds[0]

        base = preds[0]
        for p in preds[1:]:
            base = AndPred(base, p)
        base.name = "(" + " ∧ ".join(f"({c})" for c in always_true_cols) + ")"
        return base

    def check(
        self,
        df: pd.DataFrame,
        *,
        auto_base: bool = True,
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Evaluate the conjecture on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset of objects.
        auto_base : bool, default True
            If `condition is None` and `auto_base=True`, a base predicate is
            auto-detected from `df` as described in the class notes. If False,
            uses the universal TRUE instead.

        Returns
        -------
        applicable : pd.Series
            Boolean mask where the class condition holds.
        holds : pd.Series
            Boolean mask indicating `(R | C)` satisfaction per row.
        failures : pd.DataFrame
            Applicable rows failing the relation; includes a ``"__slack__"`` column
            when provided by the relation.
        """
        if self.condition is not None:
            cond = self.condition
        else:
            if auto_base:
                cond = self._auto_base(df)
            else:
                try:
                    cond = TRUE  # type: ignore[name-defined]
                except NameError:
                    class _TRUE(Predicate):
                        name = "TRUE"
                        def mask(self, df: pd.DataFrame) -> pd.Series:
                            return pd.Series(True, index=df.index)
                    cond = _TRUE()

        applicable = cond.mask(df).reindex(df.index, fill_value=False)
        eval_mask = self.relation.evaluate(df).reindex(df.index)
        holds = (~applicable) | (applicable & eval_mask)

        failing = applicable & ~eval_mask
        failures = df.loc[failing].copy()
        if len(failures):
            failures["__slack__"] = self.relation.slack(df).loc[failing]
        return applicable.astype(bool), holds.astype(bool), failures

    def is_true(self, df: pd.DataFrame, *, auto_base: bool = True) -> bool:
        """Return True iff the conjecture holds on all applicable rows."""
        applicable, holds, _ = self.check(df, auto_base=auto_base)
        return bool(holds[applicable].all())

    def __repr__(self) -> str:
        c = "True" if self.condition is None else repr(self.condition)
        # If auto base is used later, __repr__ will still show "True" here.
        # For better UX, users typically print after calling .check(), or set condition explicitly.
        return f"Conjecture({self.relation!r} | {c})"

