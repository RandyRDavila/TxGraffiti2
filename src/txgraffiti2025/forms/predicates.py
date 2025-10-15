"""
DataFrame-agnostic predicates C(df) -> boolean masks.

- Composable boolean logic: AND (&), OR (|), NOT (~)
- Column/value comparisons via vectorized expressions (no graph assumptions)
- Helpers for common forms: comparisons, between, in-set, is-integer, approx-eq
- Also supports arbitrary vectorized functions df -> Series[bool] and row-wise fallbacks
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
    """Normalize any array-like to a boolean Series aligned to index."""
    if isinstance(arr, pd.Series):
        if arr.dtype != bool:
            arr = arr.astype(bool, copy=False)
        return arr.reindex(index, fill_value=False)
    return pd.Series(np.asarray(arr, dtype=bool), index=index)

# =========================
# Base predicate + combinators
# =========================

class Predicate:
    name: str = "Predicate"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
    def __and__(self, other: "Predicate") -> "Predicate": return AndPred(self, other)
    def __or__(self, other: "Predicate") -> "Predicate": return OrPred(self, other)
    def __invert__(self) -> "Predicate": return NotPred(self)

@dataclass
class AndPred(Predicate):
    a: Predicate
    b: Predicate
    name: str = "C_and"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(self.a.mask(df) & self.b.mask(df), df.index)

@dataclass
class OrPred(Predicate):
    a: Predicate
    b: Predicate
    name: str = "C_or"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(self.a.mask(df) | self.b.mask(df), df.index)

@dataclass
class NotPred(Predicate):
    a: Predicate
    name: str = "C_not"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(~self.a.mask(df), df.index)

# =========================
# Vectorized comparison predicates
# =========================

@dataclass
class Compare(Predicate):
    """left OP right, where left/right are Expr | scalar | column name (str)."""
    left: Union[Expr, float, int, str]
    right: Union[Expr, float, int, str]
    op: Callable[[Any, Any], Any]  # expects vectorized numpy/pandas-compatible op
    name: str = "Compare"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        l = to_expr(self.left).eval(df)
        r = to_expr(self.right).eval(df)
        out = self.op(l, r)
        return _as_bool_series(out, df.index)

# Convenience constructors
def LE(left, right) -> Predicate: return Compare(left, right, np.less_equal, name="LE")
def GE(left, right) -> Predicate: return Compare(left, right, np.greater_equal, name="GE")
def LT(left, right) -> Predicate: return Compare(left, right, np.less, name="LT")
def GT(left, right) -> Predicate: return Compare(left, right, np.greater, name="GT")
def EQ(left, right, tol: float = 0.0) -> Predicate:
    if tol == 0.0:
        return Compare(left, right, np.equal, name="EQ")
    # approx equal within tol
    def _approx(l, r): return np.isclose(l, r, atol=tol)
    return Compare(left, right, _approx, name=f"EQ≈(tol={tol})")
def NE(left, right) -> Predicate: return Compare(left, right, np.not_equal, name="NE")

# =========================
# Set membership / ranges
# =========================

@dataclass
class InSet(Predicate):
    col: Union[Expr, str]
    values: Iterable[Any]
    name: str = "InSet"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        s = to_expr(self.col).eval(df)
        out = pd.Series(s).isin(set(self.values))
        return _as_bool_series(out, df.index)

@dataclass
class Between(Predicate):
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

# =========================
# Numeric property checks
# =========================

@dataclass
class IsInteger(Predicate):
    x: Union[Expr, str, float, int]
    tol: float = 1e-9
    name: str = "IsInteger"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = to_expr(self.x).eval(df)
        frac = np.mod(np.asarray(xv, dtype=float), 1.0)
        out = np.isclose(frac, 0.0, atol=self.tol)
        return _as_bool_series(out, df.index)

@dataclass
class IsNaN(Predicate):
    x: Union[Expr, str, float, int]
    name: str = "IsNaN"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = to_expr(self.x).eval(df)
        return _as_bool_series(pd.isna(xv), df.index)

@dataclass
class IsFinite(Predicate):
    x: Union[Expr, str, float, int]
    name: str = "IsFinite"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = to_expr(self.x).eval(df)
        out = np.isfinite(np.asarray(xv, dtype=float))
        return _as_bool_series(out, df.index)

# =========================
# Functional predicates
# =========================

@dataclass
class Where(Predicate):
    """Vectorized: f(df) -> Series[bool]. You promise it's vectorized."""
    fn: Callable[[pd.DataFrame], pd.Series]
    name: str = "Where"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = self.fn(df)
        if not isinstance(m, pd.Series) or m.dtype != bool:
            raise ValueError("Where(fn) must return a boolean pandas Series.")
        return m.reindex(df.index, fill_value=False)

@dataclass
class RowWhere(Predicate):
    """Row-wise fallback: f(row) -> bool; slower but universal."""
    fn: Callable[[pd.Series], bool]
    name: str = "RowWhere"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        out = df.apply(lambda row: bool(self.fn(row)), axis=1)
        return _as_bool_series(out, df.index)

# =========================
# Handy shorthands (readable DSL)
# =========================

def GEQ(col_or_expr, val) -> Predicate: return GE(col_or_expr, val)
def LEQ(col_or_expr, val) -> Predicate: return LE(col_or_expr, val)
def GT0(col_or_expr) -> Predicate: return GT(col_or_expr, 0)
def LT0(col_or_expr) -> Predicate: return LT(col_or_expr, 0)
def EQ0(col_or_expr, tol: float = 0.0) -> Predicate: return EQ(col_or_expr, 0, tol=tol)
def BETWEEN(col_or_expr, lo, hi, inc_lo=True, inc_hi=True) -> Predicate:
    return Between(col_or_expr, lo, hi, inc_lo, inc_hi)
def IN(col, values: Iterable[Any]) -> Predicate: return InSet(col, values)
def IS_INT(col_or_expr, tol: float = 1e-9) -> Predicate: return IsInteger(col_or_expr, tol)
def IS_NAN(col_or_expr) -> Predicate: return IsNaN(col_or_expr)
def IS_FINITE(col_or_expr) -> Predicate: return IsFinite(col_or_expr)
