# src/txgraffiti2025/forms/predicates.py

"""
DataFrame-agnostic predicates C(df) -> boolean masks.

- Composable boolean logic: AND (&), OR (|), NOT (~)
- Column/value comparisons via vectorized expressions (no graph assumptions)
- Helpers for common forms: comparisons, between, in-set, is-integer, approx-eq
- Also supports arbitrary vectorized functions df -> Series[bool] and row-wise fallbacks

Examples
--------
Basic composition:

>>> import pandas as pd
>>> from txgraffiti2025.forms.predicates import Predicate, GEQ, LT0, IN
>>> df = pd.DataFrame({"connected": [True, False, True], "deg": [3, -1, 2]})
>>> P_conn = Predicate.from_column("connected")
>>> P_deg_ok = GEQ("deg", 0) & ~LT0("deg")
>>> m = (P_conn & P_deg_ok & IN("deg", {2, 3})).mask(df)
>>> m.tolist()
[True, False, True]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Any, Union
import numpy as np
import pandas as pd

from .utils import Expr, to_expr  # reuse numeric Expr system (columns, constants, ops)

__all__ = [
    "Predicate",
    "AndPred",
    "OrPred",
    "NotPred",
    "Compare",
    "LE",
    "GE",
    "LT",
    "GT",
    "EQ",
    "NE",
    "InSet",
    "Between",
    "IsInteger",
    "IsNaN",
    "IsFinite",
    "Where",
    "RowWhere",
    "GEQ",
    "LEQ",
    "GT0",
    "LT0",
    "EQ0",
    "BETWEEN",
    "IN",
    "IS_INT",
    "IS_NAN",
    "IS_FINITE",
]


# =========================
# Internal helpers
# =========================

def _as_bool_series(arr: Any, index: pd.Index) -> pd.Series:
    """
    Normalize any array-like to a boolean Series aligned to a given index.

    Parameters
    ----------
    arr : Any
        Array-like input or pandas Series.
    index : pd.Index
        Target index for alignment.

    Returns
    -------
    pd.Series
        Boolean Series aligned to `index`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import _as_bool_series
    >>> _as_bool_series([1, 0, 2], pd.RangeIndex(3)).tolist()
    [True, False, True]
    """
    if isinstance(arr, pd.Series):
        if arr.dtype != bool:
            arr = arr.astype(bool, copy=False)
        return arr.reindex(index, fill_value=False)
    return pd.Series(np.asarray(arr, dtype=bool), index=index)


# =========================
# Base predicate + combinators
# =========================

class Predicate:
    """
    Base class for DataFrame-agnostic predicates producing boolean masks.

    Predicates are composable with bitwise operators:
    - `&` (AND) yields :class:`AndPred`
    - `|` (OR) yields :class:`OrPred`
    - `~` (NOT) yields :class:`NotPred`

    Methods
    -------
    mask(df) : pd.Series
        Return a boolean Series aligned to `df.index`.

    Notes
    -----
    - Rows where the mask is False are treated as outside the class when used
      as conditions in :class:`~txgraffiti2025.forms.generic_conjecture.Conjecture`.

    Examples
    --------
    From a boolean column:

    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Predicate
    >>> df = pd.DataFrame({"connected": [True, False, True]})
    >>> P_conn = Predicate.from_column("connected")
    >>> P_conn.mask(df).tolist()
    [True, False, True]

    Using composition:

    >>> from txgraffiti2025.forms.predicates import GEQ, LT0
    >>> df = pd.DataFrame({"x": [0, 1, -2]})
    >>> m = (GEQ("x", 0) & ~LT0("x")).mask(df)
    >>> m.tolist()
    [True, True, False]
    """
    name: str = "Predicate"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series aligned to `df.index`. Subclasses must implement."""
        raise NotImplementedError

    def __and__(self, other: "Predicate") -> "Predicate":
        """Logical AND of two predicates."""
        return AndPred(self, other)

    def __or__(self, other: "Predicate") -> "Predicate":
        """Logical OR of two predicates."""
        return OrPred(self, other)

    def __invert__(self) -> "Predicate":
        """Logical NOT of the predicate."""
        return NotPred(self)

    def __repr__(self) -> str:
        return getattr(self, "name", self.__class__.__name__)

    @staticmethod
    def from_column(col: Union[str, Expr], *, truthy_only: bool = False) -> "Predicate":
        """
        Build a predicate directly from a column/Expr evaluated as booleans.

        Parameters
        ----------
        col : str or Expr
            Column name or expression to evaluate.
        truthy_only : bool, default False
            If True, cast with `astype(bool)`; otherwise preserves boolean dtype if present.

        Returns
        -------
        Predicate
            A predicate whose mask is `to_expr(col).eval(df).astype(bool)` if needed.

        Examples
        --------
        >>> import pandas as pd
        >>> from txgraffiti2025.forms.predicates import Predicate
        >>> df = pd.DataFrame({"flag": [1, 0, 3]})
        >>> P = Predicate.from_column("flag", truthy_only=True)
        >>> P.mask(df).tolist()
        [True, False, True]
        """
        def _fn(df: pd.DataFrame) -> pd.Series:
            s = to_expr(col).eval(df)
            return s.astype(bool) if truthy_only or s.dtype != bool else s
        # give Where a readable label so __repr__ shows "(connected)" etc.
        label = f"({col})" if isinstance(col, str) else f"({repr(col)})"
        return Where(_fn, name=label)


@dataclass
class AndPred(Predicate):
    """
    Logical conjunction (AND) of two predicates.

    Parameters
    ----------
    a, b : Predicate
        Left and right predicates.

    Returns
    -------
    AndPred
        Composite predicate evaluating `a.mask(df) & b.mask(df)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Predicate, AndPred
    >>> df = pd.DataFrame({"p": [True, False, True], "q": [True, True, False]})
    >>> P = AndPred(Predicate.from_column("p"), Predicate.from_column("q"))
    >>> P.mask(df).tolist()
    [True, False, False]
    """
    a: Predicate
    b: Predicate
    name: str = "C_and"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(self.a.mask(df) & self.b.mask(df), df.index)

    def __repr__(self) -> str:
        return f"({self.a!r} ∧ {self.b!r})"

@dataclass
class OrPred(Predicate):
    """
    Logical disjunction (OR) of two predicates.

    Parameters
    ----------
    a, b : Predicate
        Left and right predicates.

    Returns
    -------
    OrPred
        Composite predicate evaluating `a.mask(df) | b.mask(df)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Predicate, OrPred
    >>> df = pd.DataFrame({"p": [True, False, True], "q": [True, True, False]})
    >>> P = OrPred(Predicate.from_column("p"), Predicate.from_column("q"))
    >>> P.mask(df).tolist()
    [True, True, True]
    """
    a: Predicate
    b: Predicate
    name: str = "C_or"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(self.a.mask(df) | self.b.mask(df), df.index)

    def __repr__(self) -> str:
        return f"({self.a!r} ∨ {self.b!r})"

@dataclass
class NotPred(Predicate):
    """
    Logical negation (NOT) of a predicate.

    Parameters
    ----------
    a : Predicate
        Predicate to negate.

    Returns
    -------
    NotPred
        Predicate evaluating `~a.mask(df)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Predicate, NotPred
    >>> df = pd.DataFrame({"p": [True, False, True]})
    >>> P = NotPred(Predicate.from_column("p"))
    >>> P.mask(df).tolist()
    [False, True, False]
    """
    a: Predicate
    name: str = "C_not"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(~self.a.mask(df), df.index)

    def __repr__(self) -> str:
        return f"(~{self.a!r})"

# =========================
# Vectorized comparison predicates
# =========================

@dataclass
class Compare(Predicate):
    """
    Vectorized comparison: ``left OP right`` evaluated per row.

    Parameters
    ----------
    left, right : Expr or float or int or str
        Expressions, scalars, or column names. Strings are parsed via :func:`to_expr`.
    op : Callable[[Any, Any], Any]
        Vectorized NumPy/pandas-compatible binary operator (e.g., `np.less_equal`).
    name : str, default="Compare"
        Display name.

    Returns
    -------
    Compare
        Predicate whose mask is `_as_bool_series(op(to_expr(left).eval(df), to_expr(right).eval(df)))`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from txgraffiti2025.forms.predicates import Compare
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]})
    >>> P = Compare("a", "b", np.less_equal)  # a <= b
    >>> P.mask(df).tolist()
    [True, False, True]
    """
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    op: Callable[[Any, Any], Any]  # expects vectorized numpy/pandas-compatible op
    name: str = "Compare"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        l = to_expr(self.left).eval(df)
        r = to_expr(self.right).eval(df)
        out = self.op(l, r)
        return _as_bool_series(out, df.index)

    def __repr__(self) -> str:
        import numpy as np
        op_sym = {
            np.less_equal: "<=",
            np.greater_equal: ">=",
            np.less: "<",
            np.greater: ">",
            np.equal: "==",
            np.not_equal: "!=",
        }.get(self.op, getattr(self.op, "__name__", "op"))
        return f"({self.left!r} {op_sym} {self.right!r})"

# Convenience constructors

def LE(left, right) -> Predicate:
    """
    ``left <= right``

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import LE
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 1, 4]})
    >>> LE("x", "y").mask(df).tolist()
    [True, False, True]
    """
    return Compare(left, right, np.less_equal, name="LE")


def GE(left, right) -> Predicate:
    """
    ``left >= right``

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import GE
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 1, 3]})
    >>> GE("x", "y").mask(df).tolist()
    [False, True, True]
    """
    return Compare(left, right, np.greater_equal, name="GE")


def LT(left, right) -> Predicate:
    """
    ``left < right``

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import LT
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 1, 3]})
    >>> LT("x", "y").mask(df).tolist()
    [True, False, False]
    """
    return Compare(left, right, np.less, name="LT")


def GT(left, right) -> Predicate:
    """
    ``left > right``

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import GT
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 2, 3]})
    >>> GT("x", "y").mask(df).tolist()
    [True, False, False]
    """
    return Compare(left, right, np.greater, name="GT")


def EQ(left, right, tol: float = 0.0) -> Predicate:
    """
    ``left == right`` (or approximately equal if `tol > 0`).

    Parameters
    ----------
    left, right : Expr or float or int or str
    tol : float, default 0.0
        Absolute tolerance. If positive, `np.isclose` is used.

    Examples
    --------
    Exact:

    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import EQ
    >>> df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 4.0]})
    >>> EQ("a", "b").mask(df).tolist()
    [True, True, False]

    Approximate:

    >>> EQ("a", "b", tol=1e-9).mask(pd.DataFrame({"a":[1.0, 2.0], "b":[1.0+1e-12, 2.0-1e-12]})).tolist()
    [True, True]
    """
    if tol == 0.0:
        return Compare(left, right, np.equal, name="EQ")
    # approx equal within tol
    def _approx(l, r): return np.isclose(l, r, atol=tol)
    return Compare(left, right, _approx, name=f"EQ≈(tol={tol})")


def NE(left, right) -> Predicate:
    """
    ``left != right``

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import NE
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 0, 3]})
    >>> NE("a", "b").mask(df).tolist()
    [False, True, False]
    """
    return Compare(left, right, np.not_equal, name="NE")


# =========================
# Set membership / ranges
# =========================

@dataclass
class InSet(Predicate):
    """
    Membership predicate: value of `col` is in `values`.

    Parameters
    ----------
    col : Expr or str
        Column or expression to evaluate.
    values : Iterable[Any]
        Hashable values to test against.

    Returns
    -------
    InSet
        Predicate evaluating `to_expr(col).eval(df).isin(values)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import InSet
    >>> df = pd.DataFrame({"k": [1, 2, 3, 4]})
    >>> InSet("k", {2, 4}).mask(df).tolist()
    [False, True, False, True]
    """
    col: Union[Expr, str]
    values: Iterable[Any]
    name: str = "InSet"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        s = to_expr(self.col).eval(df)
        out = pd.Series(s).isin(set(self.values))
        return _as_bool_series(out, df.index)

    def __repr__(self) -> str:
        vals = list(self.values)
        preview = vals if len(vals) <= 5 else vals[:5] + ["…"]
        return f"InSet({self.col!r} in {preview})"

@dataclass
class Between(Predicate):
    """
    Range predicate: `low <= x <= high` (bounds optionally strict).

    Parameters
    ----------
    x : Expr or str or float or int
        Value/column to test.
    low, high : Expr or str or float or int
        Range bounds.
    inclusive_low : bool, default True
        If True, test `x >= low`, else `x > low`.
    inclusive_high : bool, default True
        If True, test `x <= high`, else `x < high`.

    Returns
    -------
    Between
        Predicate for interval membership.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Between
    >>> df = pd.DataFrame({"x": [0, 1, 2, 3]})
    >>> Between("x", 1, 2).mask(df).tolist()
    [False, True, True, False]
    >>> Between("x", 1, 2, inclusive_low=False).mask(df).tolist()
    [False, False, True, False]
    """
    x: Union[Expr, str, float, int]
    low: Union[Expr, str, float, int]
    high: Union[Expr, str, float, int]
    inclusive_low: bool = True
    inclusive_high: bool = True
    name: str = "Between"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = to_expr(self.x).eval(df)
        lv = to_expr(self.low).eval(df)
        hv = to_expr(self.high).eval(df)
        left_ok = (xv >= lv) if self.inclusive_low else (xv > lv)
        right_ok = (xv <= hv) if self.inclusive_high else (xv < hv)
        return _as_bool_series(left_ok & right_ok, df.index)

    def __repr__(self) -> str:
        lo = "[" if self.inclusive_low else "("
        hi = "]" if self.inclusive_high else ")"
        return f"Between({lo}{self.low!r}, {self.high!r}{hi}) on {self.x!r}"

# =========================
# Numeric property checks
# =========================

@dataclass
class IsInteger(Predicate):
    """
    Integer-valued check within tolerance.

    Parameters
    ----------
    x : Expr or str or float or int
        Value/column to test.
    tol : float, default 1e-9
        Absolute tolerance on the fractional part.

    Returns
    -------
    IsInteger
        Predicate testing `isclose(x mod 1, 0, atol=tol)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import IsInteger
    >>> df = pd.DataFrame({"x": [1.0, 2.5, 3.0000000001]})
    >>> IsInteger("x").mask(df).tolist()
    [True, False, True]
    """
    x: Union[Expr, str, float, int]
    tol: float = 1e-9
    name: str = "IsInteger"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = to_expr(self.x).eval(df)
        frac = np.mod(np.asarray(xv, dtype=float), 1.0)
        out = np.isclose(frac, 0.0, atol=self.tol)
        return _as_bool_series(out, df.index)

    def __repr__(self) -> str:
        return f"IsInteger({self.x!r}, tol={self.tol})"


@dataclass
class IsNaN(Predicate):
    """
    NaN check for a column/expression.

    Parameters
    ----------
    x : Expr or str or float or int

    Returns
    -------
    IsNaN
        Predicate evaluating `pd.isna(x)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import IsNaN
    >>> df = pd.DataFrame({"x": [1.0, float("nan"), 2.0]})
    >>> IsNaN("x").mask(df).tolist()
    [False, True, False]
    """
    x: Union[Expr, str, float, int]
    name: str = "IsNaN"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = to_expr(self.x).eval(df)
        return _as_bool_series(pd.isna(xv), df.index)

    def __repr__(self) -> str:
        return f"IsNaN({self.x!r})"

@dataclass
class IsFinite(Predicate):
    """
    Finite (non-inf, non-NaN) check for a column/expression.

    Parameters
    ----------
    x : Expr or str or float or int

    Returns
    -------
    IsFinite
        Predicate evaluating `np.isfinite(x)`.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import IsFinite
    >>> df = pd.DataFrame({"x": [1.0, float("inf"), float("-inf"), float("nan")]})
    >>> IsFinite("x").mask(df).tolist()
    [True, False, False, False]
    """
    x: Union[Expr, str, float, int]
    name: str = "IsFinite"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = to_expr(self.x).eval(df)
        out = np.isfinite(np.asarray(xv, dtype=float))
        return _as_bool_series(out, df.index)

    def __repr__(self) -> str:
        return f"IsFinite({self.x!r})"

# =========================
# Functional predicates
# =========================

@dataclass
class Where(Predicate):
    """
    Vectorized predicate from a function `fn(df) -> Series[bool]`.

    Parameters
    ----------
    fn : Callable[[pd.DataFrame], pd.Series]
        Must return a boolean Series aligned (or alignable) to `df.index`.
    name : str, default "Where"
        Display name.

    Returns
    -------
    Where
        Predicate that defers to `fn`.

    Raises
    ------
    ValueError
        If `fn` does not return a boolean Series.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Where
    >>> df = pd.DataFrame({"x": [1, 2, 3]})
    >>> P = Where(lambda d: (d["x"] % 2 == 1))
    >>> P.mask(df).tolist()
    [True, False, True]
    """
    fn: Callable[[pd.DataFrame], pd.Series]
    name: str = "Where"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = self.fn(df)
        if not isinstance(m, pd.Series) or m.dtype != bool:
            raise ValueError("Where(fn) must return a boolean pandas Series.")
        return m.reindex(df.index, fill_value=False)

    def __repr__(self) -> str:
        # Prefer descriptive name if customized (e.g., from_column(col))
        if self.name and self.name != "Where":
            return self.name
        fn = getattr(self.fn, "__name__", "fn")
        return f"Where({fn})"

@dataclass
class RowWhere(Predicate):
    """
    Row-wise fallback predicate from `fn(row) -> bool`.

    Slower than :class:`Where` but universal when vectorization is awkward.

    Parameters
    ----------
    fn : Callable[[pd.Series], bool]
        Returns True/False for a single row.
    name : str, default "RowWhere"

    Returns
    -------
    RowWhere
        Predicate applying `fn` to each row.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import RowWhere
    >>> df = pd.DataFrame({"x": [1, 2, 3]})
    >>> P = RowWhere(lambda row: row["x"] > 1)
    >>> P.mask(df).tolist()
    [False, True, True]
    """
    fn: Callable[[pd.Series], bool]
    name: str = "RowWhere"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        out = df.apply(lambda row: bool(self.fn(row)), axis=1)
        return _as_bool_series(out, df.index)

    def __repr__(self) -> str:
        fn = getattr(self.fn, "__name__", "fn")
        return f"RowWhere({fn})"

# =========================
# Handy shorthands (readable DSL)
# =========================

def GEQ(col_or_expr, val) -> Predicate:
    """Alias for :func:`GE` — ``col_or_expr >= val``."""
    return GE(col_or_expr, val)

def LEQ(col_or_expr, val) -> Predicate:
    """Alias for :func:`LE` — ``col_or_expr <= val``."""
    return LE(col_or_expr, val)

def GT0(col_or_expr) -> Predicate:
    """``col_or_expr > 0``."""
    return GT(col_or_expr, 0)

def LT0(col_or_expr) -> Predicate:
    """``col_or_expr < 0``."""
    return LT(col_or_expr, 0)

def EQ0(col_or_expr, tol: float = 0.0) -> Predicate:
    """``col_or_expr == 0`` (approximate if `tol > 0`)."""
    return EQ(col_or_expr, 0, tol=tol)

def BETWEEN(col_or_expr, lo, hi, inc_lo=True, inc_hi=True) -> Predicate:
    """Shorthand for :class:`Between`."""
    return Between(col_or_expr, lo, hi, inc_lo, inc_hi)

def IN(col, values: Iterable[Any]) -> Predicate:
    """Shorthand for :class:`InSet`."""
    return InSet(col, values)

def IS_INT(col_or_expr, tol: float = 1e-9) -> Predicate:
    """Shorthand for :class:`IsInteger`."""
    return IsInteger(col_or_expr, tol)

def IS_NAN(col_or_expr) -> Predicate:
    """Shorthand for :class:`IsNaN`."""
    return IsNaN(col_or_expr)

def IS_FINITE(col_or_expr) -> Predicate:
    """Shorthand for :class:`IsFinite`."""
    return IsFinite(col_or_expr)
