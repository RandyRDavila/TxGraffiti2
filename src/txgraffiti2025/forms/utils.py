"""
Utilities for building expressions over pandas DataFrames:
- ColumnTerm, Const, LinearForm
- Unary ops: floor, ceil, abs, log, exp, sqrt, negation
- Binary ops: +, -, *, /, %, **

These are used by Relation/Predicate classes in generic_conjecture.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Union, Sequence, Optional
import numpy as np
import pandas as pd

__all__ = [
    "Expr",
    "Const",
    "ColumnTerm",
    "LinearForm",
    "BinOp",
    "UnaryOp",
    "to_expr",
    "floor",
    "ceil",
    "abs_",
    "log",
    "exp",
    "sqrt",
]

SeriesLike = Union[pd.Series, float, int, np.ndarray]

# -------------------------
# Internal helpers
# -------------------------

def _as_series(x: SeriesLike, index: pd.Index) -> pd.Series:
    """
    Normalize scalars/arrays to a Series aligned to `index`.
    If `x` is already a Series, reindex to align.
    """
    if isinstance(x, pd.Series):
        # Align to index; if missing, fill with NaN (let math ops propagate)
        return x.reindex(index)
    if isinstance(x, (float, int)):
        return pd.Series(float(x), index=index, dtype=float)
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        return pd.Series(float(x_arr), index=index, dtype=float)
    if x_arr.shape[0] != len(index):
        raise ValueError("Array length does not match DataFrame length.")
    return pd.Series(x_arr, index=index)

# -------------------------
# Expression base interface
# -------------------------

class Expr:
    """An expression that can be evaluated on a pandas DataFrame to a Series (aligned to df.index)."""

    def eval(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    # arithmetic operators
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

class Const(Expr):
    def __init__(self, value: float):
        self.value = float(value)
    def eval(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index, dtype=float)
    def __repr__(self): return f"Const({self.value})"

class ColumnTerm(Expr):
    """Refers to a DataFrame column by name."""
    def __init__(self, col: str):
        self.col = col
    def eval(self, df: pd.DataFrame) -> pd.Series:
        if self.col not in df.columns:
            raise KeyError(f"Required column '{self.col}' not found in DataFrame.")
        # Return as-is; downstream math will cast to float if needed
        return df[self.col]
    def __repr__(self): return f"Col({self.col})"

@dataclass
class LinearForm(Expr):
    """a0 + sum_i ai * col_i"""
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

# -------------
# Combinators
# -------------

class BinOp(Expr):
    def __init__(self, fn: Callable[[SeriesLike, SeriesLike], SeriesLike], left: Expr, right: Expr):
        self.fn, self.left, self.right = fn, left, right
    def eval(self, df: pd.DataFrame) -> pd.Series:
        l = _as_series(self.left.eval(df), df.index)
        r = _as_series(self.right.eval(df), df.index)
        return _as_series(self.fn(l, r), df.index)

class UnaryOp(Expr):
    def __init__(self, fn: Callable[[SeriesLike], SeriesLike], arg: Expr):
        self.fn, self.arg = fn, arg
    def eval(self, df: pd.DataFrame) -> pd.Series:
        a = _as_series(self.arg.eval(df), df.index)
        return _as_series(self.fn(a), df.index)

# -------------
# Safe conversion
# -------------

def to_expr(x: Union[Expr, float, int, str]) -> Expr:
    if isinstance(x, Expr): return x
    if isinstance(x, (float, int)): return Const(float(x))
    if isinstance(x, str): return ColumnTerm(x)
    raise TypeError(f"Cannot convert {type(x)} to Expr")

# -------------
# Math helpers
# -------------

def floor(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.floor, to_expr(x))

def ceil(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.ceil, to_expr(x))

def abs_(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.abs, to_expr(x))

def log(x: Union[Expr, float, int, str], base: Optional[float] = None) -> Expr:
    if base is None:
        return UnaryOp(np.log, to_expr(x))
    # log base change: log_b(x) = ln(x)/ln(b)
    return UnaryOp(lambda v: np.log(v) / np.log(base), to_expr(x))

def exp(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.exp, to_expr(x))

def sqrt(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.sqrt, to_expr(x))
