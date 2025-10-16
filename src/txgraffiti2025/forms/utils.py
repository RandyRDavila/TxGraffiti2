# src/txgraffiti2025/forms/utils.py

"""
Expression utilities for symbolic arithmetic over pandas DataFrames.

This module defines the lightweight `Expr` system used throughout the
TxGraffiti conjecture forms (R1–R6).  Each `Expr` represents a column-wise
numeric expression that can be evaluated on a `pandas.DataFrame` to yield a
`Series` aligned to `df.index`.

Features
--------
- Symbolic expressions referencing DataFrame columns.
- Arithmetic operator overloading (`+`, `-`, `*`, `/`, `%`, `**`, unary `-`).
- Safe broadcasting and index alignment.
- Unary and binary math helpers (`floor`, `ceil`, `log`, `exp`, `sqrt`, etc.).
- Building blocks for linear, nonlinear, and composite relations.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.utils import to_expr, floor, log
>>> df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 8]})
>>> e = to_expr("y") / to_expr("x") + 1
>>> e.eval(df).tolist()
[3.0, 3.0, 3.6666666666666665]

Unary and composed expressions:

>>> floor(log("y")).eval(df).tolist()
[0.0, 1.0, 2.0]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Union, Sequence, Optional
from fractions import Fraction
import numpy as np
import pandas as pd

__all__ = [
    "Expr",
    "Const",
    "ColumnTerm",
    "LinearForm",
    "BinOp",
    "UnaryOp",
    "LogOp",
    "to_expr",
    "floor",
    "ceil",
    "abs_",
    "log",
    "exp",
    "sqrt",
]

SeriesLike = Union[pd.Series, float, int, np.ndarray]

# --- Pretty-print helpers for Expr ---
_PRECEDENCE = {
    "**": 4,
    "unary": 3,
    "*": 2, "/": 2, "%": 2,
    "+": 1, "-": 1,
}


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _need_parens(child_prec: int, parent_prec: int, is_right_assoc: bool = False, is_right_child: bool = False) -> bool:
    if child_prec < parent_prec:
        return True
    if is_right_assoc and is_right_child and child_prec == parent_prec:
        # a ** (b ** c) stays; (a ** b) ** c already groups right in python but we print clearly
        return True
    return False

def _as_series(x: SeriesLike, index: pd.Index) -> pd.Series:
    """
    Normalize scalars, arrays, or Series to a float Series aligned to `index`.

    Parameters
    ----------
    x : scalar, np.ndarray, or pd.Series
        Input value(s) to convert.
    index : pd.Index
        Target index for alignment.

    Returns
    -------
    pd.Series
        Series of dtype float, aligned to `index`.

    Raises
    ------
    ValueError
        If an array is given with a length not matching `len(index)`.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from txgraffiti2025.forms.utils import _as_series
    >>> idx = pd.Index(range(3))
    >>> _as_series(5, idx).tolist()
    [5.0, 5.0, 5.0]
    >>> _as_series(np.array([1, 2, 3]), idx).tolist()
    [1.0, 2.0, 3.0]
    """
    if isinstance(x, pd.Series):
        return x.reindex(index)
    if isinstance(x, (float, int)):
        return pd.Series(float(x), index=index, dtype=float)
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        return pd.Series(float(x_arr), index=index, dtype=float)
    if x_arr.shape[0] != len(index):
        raise ValueError("Array length does not match DataFrame length.")
    return pd.Series(x_arr, index=index)


# ---------------------------------------------------------------------
# Expression base interface
# ---------------------------------------------------------------------
class Expr:
    """
    Abstract base for expressions that evaluate to a Series on a DataFrame.

    Notes
    -----
    - Subclasses implement :meth:`eval(df)`.
    - Arithmetic operators are overloaded to build composite expressions.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.utils import to_expr
    >>> df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    >>> expr = to_expr("x") + 2 * to_expr("y")
    >>> expr.eval(df).tolist()
    [7.0, 10.0]
    """

    def eval(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate on a DataFrame. Must be implemented by subclasses."""
        raise NotImplementedError

    # Operator overloads: return new Exprs built from current one
    def __add__(self, other): return BinOp(np.add, self, to_expr(other))
    def __radd__(self, other): return BinOp(np.add, to_expr(other), self)
    def __sub__(self, other): return BinOp(np.subtract, self, to_expr(other))
    def __rsub__(self, other): return BinOp(np.subtract, to_expr(other), self)
    def __mul__(self, other): return BinOp(np.multiply, self, to_expr(other))
    def __rmul__(self, other): return BinOp(np.multiply, to_expr(other), self)
    def __truediv__(self, other): return BinOp(np.divide, self, to_expr(other))
    def __rtruediv__(self, other): return BinOp(np.divide, to_expr(other), self)
    def __mod__(self, other): return BinOp(np.mod, self, to_expr(other))
    def __rmod__(self, other): return BinOp(np.mod, to_expr(other), self)
    def __pow__(self, power): return BinOp(np.power, self, to_expr(power))
    def __neg__(self): return UnaryOp(np.negative, self)


# ---------------------------------------------------------------------
# Concrete expression classes
# ---------------------------------------------------------------------
class Const(Expr):
    """
    Constant numeric expression.

    Parameters
    ----------
    value : float or int

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.utils import Const
    >>> Const(3).eval(pd.DataFrame(index=[0, 1])).tolist()
    [3.0, 3.0]
    """
    def __init__(self, value: float | int | Fraction):
        self.value = value

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index, dtype=float)

    def __repr__(self) -> str:
        if isinstance(self.value, Fraction):
            if self.value == 1:
                return "1"
            elif self.value == -1:
                return "-1"
            elif self.value == 0:
                return "0"
            elif self.value.denominator == 1:
                return f'{self.value.numerator}'
            elif self.value.denominator == - 1:
                return f'-{self.value.numerator}'
            else:
                return f"({self.value.numerator}/{self.value.denominator})"
        v = float(self.value)
        return str(int(v)) if v.is_integer() else repr(v)


class ColumnTerm(Expr):
    """
    Expression referencing a DataFrame column by name.

    Parameters
    ----------
    col : str
        Column name to extract from the DataFrame.

    Raises
    ------
    KeyError
        If the column is not present.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.utils import ColumnTerm
    >>> df = pd.DataFrame({"a": [1, 2, 3]})
    >>> ColumnTerm("a").eval(df).tolist()
    [1, 2, 3]
    """
    def __init__(self, col: str):
        self.col = col

    def eval(self, df: pd.DataFrame) -> pd.Series:
        if self.col not in df.columns:
            raise KeyError(f"Required column '{self.col}' not found in DataFrame.")
        return df[self.col]

    def __repr__(self):  # pretty column
        return repr(self.col)


@dataclass
class LinearForm(Expr):
    """
    Linear combination expression ``a0 + Σ ai * col_i``.

    Parameters
    ----------
    intercept : float
        Constant term.
    terms : sequence of (float, str)
        Pairs of coefficient and column name.

    Raises
    ------
    KeyError
        If a referenced column is missing.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.utils import LinearForm
    >>> df = pd.DataFrame({"x":[1,2], "y":[3,4]})
    >>> LinearForm(1.0, [(2.0,"x"), (-1.0,"y")]).eval(df).tolist()
    [0.0, -1.0]
    """
    intercept: float
    terms: Sequence[tuple[float, str]]  # (coef, column)

    def eval(self, df: pd.DataFrame) -> pd.Series:
        if len(df) == 0:
            return pd.Series(float(self.intercept), index=df.index, dtype=float)
        y = pd.Series(float(self.intercept), index=df.index, dtype=float)
        for a, c in self.terms:
            if c not in df.columns:
                raise KeyError(f"Required column '{c}' not found.")
            y = y.add(float(a) * pd.to_numeric(df[c], errors="coerce"), fill_value=0.0)
        return y

    def __repr__(self) -> str:
        parts = []
        for a, c in self.terms:
            coef = int(a) if float(a).is_integer() else a
            if coef == 1:
                parts.append(f"{repr(c)}")
            elif coef == -1:
                parts.append(f"-{repr(c)}")
            else:
                parts.append(f"{coef}*{repr(c)}")
        body = " + ".join(parts) if parts else "0"
        a0 = int(self.intercept) if float(self.intercept).is_integer() else self.intercept
        return f"{a0} + {body}"

# ---------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------
class BinOp(Expr):
    """
    Binary operator expression combining two sub-expressions.

    Parameters
    ----------
    fn : callable
        Binary numpy-like function, e.g. ``np.add``, ``np.divide``.
    left, right : Expr
        Operand expressions.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from txgraffiti2025.forms.utils import BinOp, ColumnTerm
    >>> df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> BinOp(np.add, ColumnTerm("a"), ColumnTerm("b")).eval(df).tolist()
    [4.0, 6.0]
    """
    def __init__(self, fn: Callable[[SeriesLike, SeriesLike], SeriesLike], left: Expr, right: Expr):
        self.fn, self.left, self.right = fn, left, right

    def eval(self, df: pd.DataFrame) -> pd.Series:
        l = _as_series(self.left.eval(df), df.index)
        r = _as_series(self.right.eval(df), df.index)
        return _as_series(self.fn(l, r), df.index)

    def __repr__(self) -> str:
        sym = {
            np.add: "+", np.subtract: "-", np.multiply: "*",
            np.divide: "/", np.mod: "%", np.power: "**"
        }.get(self.fn, getattr(self.fn, "__name__", "op"))
        # precedence
        parent_prec = _PRECEDENCE.get(sym, 1)
        left_s = repr(self.left)
        right_s = repr(self.right)
        # child precedence (guess from outer symbol of repr):
        def _child_prec(s: str) -> int:
            if s.startswith("log"):
                return _PRECEDENCE["unary"]
            if s.startswith(("floor(", "ceil(", "abs(", "exp(", "sqrt(", "neg(")):
                return _PRECEDENCE["unary"]
            if "**" in s and s.count("**") == 1:
                return _PRECEDENCE["**"]
            for op in ("*", "/", "%"):
                if f" {op} " in s:
                    return _PRECEDENCE[op]
            for op in ("+", "-"):
                if f" {op} " in s:
                    return _PRECEDENCE[op]
            return _PRECEDENCE["unary"]
        lp = _child_prec(left_s)
        rp = _child_prec(right_s)
        if _need_parens(lp, parent_prec, is_right_assoc=(sym=="**"), is_right_child=False):
            left_s = f"({left_s})"
        if _need_parens(rp, parent_prec, is_right_assoc=(sym=="**"), is_right_child=True):
            right_s = f"({right_s})"
        return f"{left_s} {sym} {right_s}"


class UnaryOp(Expr):
    """
    Unary operator expression wrapping one sub-expression.

    Parameters
    ----------
    fn : callable
        Unary numpy-like function (e.g. ``np.exp``, ``np.floor``).
    arg : Expr
        Operand expression.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from txgraffiti2025.forms.utils import UnaryOp, ColumnTerm
    >>> df = pd.DataFrame({"x": [1, 2, 3]})
    >>> UnaryOp(np.negative, ColumnTerm("x")).eval(df).tolist()
    [-1.0, -2.0, -3.0]
    """
    def __init__(self, fn: Callable[[SeriesLike], SeriesLike], arg: Expr):
        self.fn, self.arg = fn, arg

    def eval(self, df: pd.DataFrame) -> pd.Series:
        a = _as_series(self.arg.eval(df), df.index)
        return _as_series(self.fn(a), df.index)

    def __repr__(self) -> str:
        name = {
            np.floor: "floor",
            np.ceil: "ceil",
            np.abs: "abs",
            np.log: "log",
            np.exp: "exp",
            np.sqrt: "sqrt",
            np.negative: "neg",
        }.get(self.fn, getattr(self.fn, "__name__", "unary"))
        return f"{name}({self.arg!r})"


class LogOp(Expr):
    def __init__(self, arg: Expr, base: Optional[float]):
        self.arg = arg
        self.base = base

    def eval(self, df: pd.DataFrame) -> pd.Series:
        a = _as_series(self.arg.eval(df), df.index)
        if self.base is None:
            return _as_series(np.log(a), df.index)
        # log_b(x) = ln(x) / ln(b)
        denom = np.log(float(self.base))
        return _as_series(np.log(a) / denom, df.index)

    def __repr__(self) -> str:
        if self.base is None:
            return f"log({self.arg!r})"
        if self.base == 10:
            return f"log10({self.arg!r})"
        if self.base == 2:
            return f"log2({self.arg!r})"
        return f"log_{self.base}({self.arg!r})"

# ---------------------------------------------------------------------
# Safe conversion
# ---------------------------------------------------------------------
def to_expr(x: Union[Expr, float, int, str, Fraction]) -> Expr:
    """
    Convert scalars or column names into `Expr` objects.

    Parameters
    ----------
    x : Expr or float or int or str
        Input to convert.

    Returns
    -------
    Expr
        Existing expression or a `Const` / `ColumnTerm`.

    Raises
    ------
    TypeError
        If `x` cannot be converted.

    Examples
    --------
    >>> from txgraffiti2025.forms.utils import to_expr
    >>> to_expr("x"), to_expr(5)
    (Col(x), Const(5.0))
    """
    if isinstance(x, Expr):
        return x
    if isinstance(x, Fraction):
        return Const(x)
    if isinstance(x, (float, int)):
        return Const(float(x))
    if isinstance(x, str):
        return ColumnTerm(x)
    raise TypeError(f"Cannot convert {type(x)} to Expr")

# ---------------------------------------------------------------------
# Math helpers (wrapped as Expr)
# ---------------------------------------------------------------------
def floor(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.floor, to_expr(x))

def ceil(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.ceil, to_expr(x))

def abs_(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.abs, to_expr(x))

def log(x: Union[Expr, float, int, str], base: Optional[float] = None) -> Expr:
    return LogOp(to_expr(x), base=base)

def exp(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.exp, to_expr(x))

def sqrt(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.sqrt, to_expr(x))
