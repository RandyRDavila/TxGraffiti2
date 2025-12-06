# src/txgraffiti/graffiti3/predicates.py
"""
DataFrame-agnostic predicates C(df) -> boolean masks.

Features
--------
- Composable boolean logic: AND (&), OR (|), NOT (~)
- Vectorized comparisons against columns/Expr/scalars/Series
- Set membership, numeric ranges, and numeric property checks
- Arbitrary vectorized and row-wise predicates (Where / RowWhere)
- Safe handling of pandas nullable booleans (NA -> False)
- Unicode-first pretty reprs (∧, ∨, ¬, ∈, ℤ, ≤, ≥, ∞, →)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Any, Union

import numpy as np
import pandas as pd

from .exprs import Expr, to_expr, abs_

__all__ = [
    # Core
    "Predicate", "AndPred", "OrPred", "NotPred",
    # Binary comparisons
    "Compare", "LE", "GE", "LT", "GT", "EQ", "NE",
    # Set/range
    "InSet", "Between",
    # Numeric properties
    "IsInteger", "IsNaN", "IsFinite",
    # Functional
    "Where", "RowWhere",
    # Quantifier-style predicates (semantic, optional symbol)
    "ForallFinite", "ExistsDivergent",
    # Shorthands / DSL
    "GEQ", "LEQ", "GT0", "LT0", "EQ0", "BETWEEN", "IN", "IS_INT", "IS_NAN", "IS_FINITE",
]


# =====================================================================
# Internal helpers
# =====================================================================

def _as_bool_series(arr: Any, index: pd.Index) -> pd.Series:
    """
    Normalize any array-like to a boolean Series aligned to `index`.
    NA values are treated as False.
    """
    if isinstance(arr, pd.Series):
        s = arr.reindex(index)
        return s.fillna(False).astype(bool, copy=False)

    if np.isscalar(arr):
        return pd.Series(bool(arr), index=index, dtype=bool)

    a = np.asarray(arr)
    if a.ndim != 1 or len(a) != len(index):
        raise ValueError("Array-like must be 1D and match index length.")
    if a.dtype != bool:
        a = a.astype(bool, copy=False)
    return pd.Series(a, index=index, dtype=bool).fillna(False)


def _fmt_values(values: Iterable[Any]) -> str:
    """
    Format a small set/list of values as a compact comma list without quotes where possible.
    Keeps input order when not sortable.
    """
    try:
        seq = list(values)
        # Attempt a safe sort that won't crash on mixed types
        try:
            seq = sorted(seq)
        except Exception:
            pass
        parts = []
        for v in seq:
            if isinstance(v, str):
                parts.append(v)
            else:
                parts.append(str(v))
        return ", ".join(parts)
    except Exception:
        return ", ".join(map(str, values))


def _operand_key_for_pred(op) -> tuple:
    # Normalize common operand types into a stable, comparable key.
    import numpy as _np
    from .utils import to_expr

    if isinstance(op, str):
        return ("col", op)
    if isinstance(op, (int, float, _np.integer, _np.floating)):
        return ("const", float(op))
    try:
        # Covers Expr and anything convertible to Expr (including strings again).
        return ("expr", repr(to_expr(op)))
    except Exception:
        # Series/arrays/functions: last-resort identity
        return ("obj", repr(op))


# =====================================================================
# Base predicate + combinators
# =====================================================================

class Predicate:
    """
    Base class for DataFrame-agnostic predicates producing boolean masks.

    Predicates are composable with `&` (AND), `|` (OR), and `~` (NOT).
    """
    name: str = "Predicate"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean Series aligned to `df.index`."""
        raise NotImplementedError

    def __and__(self, other: "Predicate") -> "Predicate":
        return AndPred(self, other)

    def __or__(self, other: "Predicate") -> "Predicate":
        return OrPred(self, other)

    def __invert__(self) -> "Predicate":
        return NotPred(self)

    def __repr__(self) -> str:
        return getattr(self, "name", self.__class__.__name__)

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self.mask(df)

    # --- structural identity ---
    def cache_key(self) -> tuple:
        return (self.__class__.__name__, repr(self))

    def __hash__(self) -> int:
        return hash(self.cache_key())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Predicate) and self.cache_key() == other.cache_key()

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self.mask(df)

    @staticmethod
    def from_column(col: Union[str, Expr], truthy_only: bool = False) -> "Predicate":
        """
        Build a predicate directly from a column or Expr evaluated as booleans.

        Accepts positional or keyword for `truthy_only` to keep backward compatibility.

        Parameters
        ----------
        col : str or Expr
            Column name or expression to read from the DataFrame.
        truthy_only : bool, default=False
            If True, any nonzero / non-null value counts as True, and NA values
            are treated as False. This avoids conversion errors for nullable
            booleans and 0/1 Int64 columns.
        """
        def _fn(df: pd.DataFrame) -> pd.Series:
            s = to_expr(col).eval(df)

            # Ensure Series
            if not isinstance(s, pd.Series):
                return pd.Series(bool(s), index=df.index, dtype=bool)

            # Nullable boolean dtype
            if pd.api.types.is_bool_dtype(s) or str(s.dtype).lower().startswith("boolean"):
                return s.fillna(False).astype(bool, copy=False)

            # Nullable integer dtype (e.g. Int64)
            if pd.api.types.is_integer_dtype(s):
                if truthy_only:
                    return s.ne(0).fillna(False).astype(bool, copy=False)
                # fallback for non-truthy_only integers: treat 0 as False
                return s.fillna(0).ne(0).astype(bool, copy=False)

            # Fallback for other numeric/object dtypes
            if truthy_only:
                return s.fillna(False).astype(bool, copy=False)

            # Already boolean-like? keep as-is
            if s.dtype == bool:
                return s

            # Last resort: conservative truthiness (NA → False)
            return s.fillna(False).astype(bool, copy=False)

        label = f"({col})" if isinstance(col, str) else f"({repr(col)})"
        return Where(_fn, name=label)

# @dataclass
# class AndPred(Predicate):
#     a: Predicate
#     b: Predicate
#     name: str = "C_and"

#     def mask(self, df: pd.DataFrame) -> pd.Series:
#         return _as_bool_series(self.a.mask(df) & self.b.mask(df), df.index)

#     def __repr__(self) -> str:
#         return f"({self.a!r} ∧ {self.b!r})"

@dataclass(eq=False)
class AndPred(Predicate):
    a: Predicate
    b: Predicate
    name: str = "C_and"

    def cache_key(self) -> tuple:
        ak, bk = self.a.cache_key(), self.b.cache_key()
        # Order-independent: sort the two keys
        pair = tuple(sorted((ak, bk)))
        return ("AND", pair)

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(self.a.mask(df) & self.b.mask(df), df.index)

    def __repr__(self) -> str:
        return f"({self.a!r} ∧ {self.b!r})"


# @dataclass
# class OrPred(Predicate):
#     a: Predicate
#     b: Predicate
#     name: str = "C_or"

#     def mask(self, df: pd.DataFrame) -> pd.Series:
#         return _as_bool_series(self.a.mask(df) | self.b.mask(df), df.index)

#     def __repr__(self) -> str:
#         return f"({self.a!r} ∨ {self.b!r})"


@dataclass(eq=False)
class OrPred(Predicate):
    a: Predicate
    b: Predicate
    name: str = "C_or"

    def cache_key(self) -> tuple:
        ak, bk = self.a.cache_key(), self.b.cache_key()
        pair = tuple(sorted((ak, bk)))
        return ("OR", pair)

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(self.a.mask(df) | self.b.mask(df), df.index)

    def __repr__(self) -> str:
        return f"({self.a!r} ∨ {self.b!r})"

# @dataclass
# class NotPred(Predicate):
#     a: Predicate
#     name: str = "C_not"

#     def mask(self, df: pd.DataFrame) -> pd.Series:
#         return _as_bool_series(~self.a.mask(df), df.index)

#     def __repr__(self) -> str:
#         return f"(¬{self.a!r})"

@dataclass(eq=False)
class NotPred(Predicate):
    a: Predicate
    name: str = "C_not"

    def cache_key(self) -> tuple:
        return ("NOT", self.a.cache_key())

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return _as_bool_series(~self.a.mask(df), df.index)

    def __repr__(self) -> str:
        return f"(¬{self.a!r})"


# =====================================================================
# Vectorized comparison predicates
# =====================================================================

class Compare(Predicate):
    """
    Binary comparison predicate: left (op) right
      - left/right: column name, Expr, numeric scalar, or pd.Series
      - robust numeric coercion with alignment to df.index
    """

    def __init__(self, left: Any, right: Any, fn: Callable[[Any, Any], Any]):
        self.left = left
        self.right = right
        self.fn = fn
        self._symbol = (
            "<"  if fn is np.less          else
            ">"  if fn is np.greater       else
            "<=" if fn is np.less_equal    else
            ">=" if fn is np.greater_equal else
            "==" if fn is np.equal         else
            "!=" if fn is np.not_equal     else
            getattr(fn, "__name__", "cmp")
        )
        self.name = f"({self._disp(self.left)} {self._symbol} {self._disp(self.right)})"

    def _eval_operand(self, op: Any, df: pd.DataFrame) -> pd.Series:
        # Prefer Expr pipeline (covers strings via to_expr)
        try:
            s = to_expr(op).eval(df)
            return pd.to_numeric(s, errors="coerce")
        except Exception:
            if isinstance(op, pd.Series):
                return pd.to_numeric(op.reindex(df.index), errors="coerce")
            if np.isscalar(op):
                return pd.Series(float(op), index=df.index, dtype=float)
            a = np.asarray(op)
            if a.ndim == 0:
                return pd.Series(float(a), index=df.index, dtype=float)
            if len(a) != len(df.index):
                raise ValueError("Array-like operand length does not match DataFrame length.")
            return pd.Series(pd.to_numeric(a, errors="coerce"), index=df.index)

    def _disp(self, op: Any) -> str:
        if isinstance(op, str):
            return op
        name = getattr(op, "name", None)
        if isinstance(name, str):
            return name
        try:
            return str(op)
        except Exception:
            return repr(op)

    def mask(self, df: pd.DataFrame) -> pd.Series:
        L = self._eval_operand(self.left, df)
        R = self._eval_operand(self.right, df)
        out = pd.Series(False, index=df.index, dtype=bool)
        good = L.notna() & R.notna()
        if good.any():
            out.loc[good] = self.fn(L[good], R[good]).astype(bool)
        return out

    def __repr__(self) -> str:
        return self.name

    def cache_key(self) -> tuple:
        return ("CMP", self._symbol,
                _operand_key_for_pred(self.left),
                _operand_key_for_pred(self.right))


# Convenience constructors
def LT(left, right) -> Predicate: return Compare(left, right, np.less)
def LE(left, right) -> Predicate: return Compare(left, right, np.less_equal)
def GT(left, right) -> Predicate: return Compare(left, right, np.greater)
def GE(left, right) -> Predicate: return Compare(left, right, np.greater_equal)
def EQ(left, right) -> Predicate: return Compare(left, right, np.equal)
def NE(left, right) -> Predicate: return Compare(left, right, np.not_equal)
def GT0(col_or_expr) -> Predicate: return GT(col_or_expr, 0)
def LT0(col_or_expr) -> Predicate: return LT(col_or_expr, 0)


# =====================================================================
# Set membership / ranges
# =====================================================================

# @dataclass
# class InSet(Predicate):
#     col: Union[Expr, str]
#     values: Iterable[Any]
#     name: str = "InSet"

#     def mask(self, df: pd.DataFrame) -> pd.Series:
#         s = to_expr(self.col).eval(df)
#         out = pd.Series(s, index=df.index).isin(set(self.values))
#         return _as_bool_series(out, df.index)

#     def __repr__(self) -> str:
#         col_txt = to_expr(self.col).__repr__()  # use Expr pretty (__repr__)
#         return f"[{col_txt} ∈ {{{_fmt_values(self.values)}}}]"

@dataclass(eq=False)
class InSet(Predicate):
    col: Union[Expr, str]
    values: Iterable[Any]
    name: str = "InSet"

    def cache_key(self) -> tuple:
        # Order-independent set of values
        try:
            # If values are hashable, use frozenset; else, tuple of reprs
            fs = frozenset(self.values)
        except TypeError:
            fs = tuple(sorted(map(repr, self.values)))
        return ("IN", _operand_key_for_pred(self.col), fs)

    # def mask(self, df: pd.DataFrame) -> pd.Series:
    #     s = to_expr(self.col).eval(df)
    #     out = pd.Series(s, index=df.index).isin(set(self.values))
    #     return _as_bool_series(out, df.index)
    def mask(self, df: pd.DataFrame) -> pd.Series:
        s = to_expr(self.col).eval(df)
        out = pd.Series(s, index=df.index).isin(list(self.values))
        return _as_bool_series(out, df.index)

# @dataclass
# class Between(Predicate):
#     x: Union[Expr, str, float, int]
#     low: Union[Expr, str, float, int]
#     high: Union[Expr, str, float, int]
#     inclusive_low: bool = True
#     inclusive_high: bool = True
#     name: str = "Between"

#     def mask(self, df: pd.DataFrame) -> pd.Series:
#         xv = pd.to_numeric(to_expr(self.x).eval(df), errors="coerce")
#         lv = pd.to_numeric(to_expr(self.low).eval(df), errors="coerce")
#         hv = pd.to_numeric(to_expr(self.high).eval(df), errors="coerce")
#         left_ok  = (xv >= lv) if self.inclusive_low  else (xv >  lv)
#         right_ok = (xv <= hv) if self.inclusive_high else (xv <  hv)
#         return _as_bool_series(left_ok & right_ok, df.index)

#     def __repr__(self) -> str:
#         x_txt = to_expr(self.x).__repr__()
#         lo_txt = to_expr(self.low).__repr__()
#         hi_txt = to_expr(self.high).__repr__()
#         # choose symbols per bound
#         lo_sym = "≤" if self.inclusive_low  else "<"
#         hi_sym = "≤" if self.inclusive_high else "<"
#         return f"[{lo_txt} {lo_sym} {x_txt} {hi_sym} {hi_txt}]"

@dataclass(eq=False)
class Between(Predicate):
    x: Union[Expr, str, float, int]
    low: Union[Expr, str, float, int]
    high: Union[Expr, str, float, int]
    inclusive_low: bool = True
    inclusive_high: bool = True
    name: str = "Between"

    def cache_key(self) -> tuple:
        return ("BETWEEN",
                _operand_key_for_pred(self.x),
                _operand_key_for_pred(self.low),
                _operand_key_for_pred(self.high),
                bool(self.inclusive_low),
                bool(self.inclusive_high))

    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = pd.to_numeric(to_expr(self.x).eval(df), errors="coerce")
        lv = pd.to_numeric(to_expr(self.low).eval(df), errors="coerce")
        hv = pd.to_numeric(to_expr(self.high).eval(df), errors="coerce")
        left_ok  = (xv >= lv) if self.inclusive_low  else (xv >  lv)
        right_ok = (xv <= hv) if self.inclusive_high else (xv <  hv)
        return _as_bool_series(left_ok & right_ok, df.index)

    def __repr__(self) -> str:
        x_txt = to_expr(self.x).__repr__()
        lo_txt = to_expr(self.low).__repr__()
        hi_txt = to_expr(self.high).__repr__()
        # choose symbols per bound
        lo_sym = "≤" if self.inclusive_low  else "<"
        hi_sym = "≤" if self.inclusive_high else "<"
        return f"[{lo_txt} {lo_sym} {x_txt} {hi_sym} {hi_txt}]"

# =====================================================================
# Numeric property checks
# =====================================================================

@dataclass(eq=False)
class IsInteger(Predicate):
    x: Union[Expr, str, float, int]
    tol: float = 1e-9
    name: str = "IsInteger"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = pd.to_numeric(to_expr(self.x).eval(df), errors="coerce")
        frac = np.mod(np.asarray(xv, dtype=float), 1.0)
        out = np.isclose(frac, 0.0, atol=self.tol)
        return _as_bool_series(out, df.index)

    def __repr__(self) -> str:
        x_txt = to_expr(self.x).__repr__()
        return f"[{x_txt} ∈ ℤ]"

    def cache_key(self) -> tuple:
        return ("IS_INT", _operand_key_for_pred(self.x), float(self.tol))


@dataclass(eq=False)
class IsNaN(Predicate):
    x: Union[Expr, str, float, int]
    name: str = "IsNaN"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = to_expr(self.x).eval(df)
        return _as_bool_series(pd.isna(xv), df.index)

    def __repr__(self) -> str:
        x_txt = to_expr(self.x).__repr__()
        return f"[isnan({x_txt})]"

    def cache_key(self) -> tuple:
        return ("IS_NAN", _operand_key_for_pred(self.x))

@dataclass(eq=False)
class IsFinite(Predicate):
    x: Union[Expr, str, float, int]
    name: str = "IsFinite"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        xv = pd.to_numeric(to_expr(self.x).eval(df), errors="coerce")
        out = np.isfinite(np.asarray(xv, dtype=float))
        return _as_bool_series(out, df.index)

    def __repr__(self) -> str:
        x_txt = to_expr(self.x).__repr__()
        return f"[|{x_txt}| < ∞]"

    def cache_key(self) -> tuple:
        return ("IS_FINITE", _operand_key_for_pred(self.x))


# =====================================================================
# Quantifier-style pretty predicates (semantic sugar)
# =====================================================================
@dataclass(eq=False)
class ForallFinite(Predicate):
    """Universal finiteness: ∀ objects (or optional symbol) have finite expr."""
    def __init__(self, expr: Expr, symbol: str | None = None):
        self.expr = expr
        self.symbol = symbol

    def mask(self, df: pd.DataFrame) -> pd.Series:
        # Semantically: "finite for all rows" is equivalent to IsFinite(expr) mask
        return IsFinite(self.expr).mask(df)

    def __repr__(self) -> str:
        sym = f" {self.symbol}" if self.symbol else ""
        return f"(∀{sym} : |{self.expr!r}| < ∞)"

@dataclass(eq=False)
class ExistsDivergent(Predicate):
    """Existential divergence: ∃ object (or optional symbol) such that expr → ∞."""
    def __init__(self, expr: Expr, symbol: str | None = None):
        self.expr = expr
        self.symbol = symbol

    def mask(self, df: pd.DataFrame) -> pd.Series:
        # There exists a divergence row-wise ≈ not isfinite somewhere.
        return _as_bool_series(~IsFinite(self.expr).mask(df), df.index)

    def __repr__(self) -> str:
        sym = f" {self.symbol}" if self.symbol else ""
        return f"(∃{sym} : {self.expr!r} → ∞)"


# =====================================================================
# Functional predicates
# =====================================================================

@dataclass(eq=False)
class Where(Predicate):
    fn: Callable[[pd.DataFrame], Any]
    name: str = "Where"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        m = self.fn(df)
        if not isinstance(m, (pd.Series, np.ndarray, list, tuple, bool, np.bool_)):
            raise ValueError("Where(fn) must return a boolean array-like/Series.")
        return _as_bool_series(m, df.index)

    def __repr__(self) -> str:
        if self.name and self.name != "Where":
            return self.name
        fn = getattr(self.fn, "__name__", "fn")
        return f"Where({fn})"

    def cache_key(self) -> tuple:
        # Prefer explicit name; else use qualname+filename+firstlineno if possible
        if self.name and self.name != "Where":
            return ("WHERE", self.name)
        f = self.fn
        qn = getattr(f, "__qualname__", getattr(f, "__name__", "fn"))
        co = getattr(f, "__code__", None)
        if co is not None:
            sig = (co.co_filename, co.co_firstlineno, co.co_argcount, co.co_varnames)
        else:
            sig = (repr(f),)
        return ("WHERE", qn, sig)

@dataclass(eq=False)
class RowWhere(Predicate):
    fn: Callable[[pd.Series], bool]
    name: str = "RowWhere"

    def mask(self, df: pd.DataFrame) -> pd.Series:
        out = df.apply(lambda row: bool(self.fn(row)), axis=1)
        return _as_bool_series(out, df.index)

    def __repr__(self) -> str:
        fn = getattr(self.fn, "__name__", "fn")
        return f"RowWhere({fn})"

    def cache_key(self) -> tuple:
        if self.name and self.name != "RowWhere":
            return ("ROWW", self.name)
        f = self.fn
        qn = getattr(f, "__qualname__", getattr(f, "__name__", "fn"))
        co = getattr(f, "__code__", None)
        if co is not None:
            sig = (co.co_filename, co.co_firstlineno, co.co_argcount, co.co_varnames)
        else:
            sig = (repr(f),)
        return ("ROWW", qn, sig)

# =====================================================================
# Handy shorthands (readable DSL)
# =====================================================================

def GEQ(col_or_expr, val) -> Predicate:  # alias for GE
    return GE(col_or_expr, val)

def LEQ(col_or_expr, val) -> Predicate:  # alias for LE
    return LE(col_or_expr, val)

def EQ0(col_or_expr, tol: float = 0.0) -> Predicate:
    """|x| <= tol if tol>0, else x == 0."""
    if tol and tol > 0:
        return LE(abs_(to_expr(col_or_expr)), tol)
    return EQ(col_or_expr, 0)

def BETWEEN(col_or_expr, lo, hi, inc_lo=True, inc_hi=True) -> Predicate:
    return Between(col_or_expr, lo, hi, inc_lo, inc_hi)

def IN(col, values: Iterable[Any]) -> Predicate:
    return InSet(col, values)

def IS_INT(col_or_expr, tol: float = 1e-9) -> Predicate:
    return IsInteger(col_or_expr, tol)

def IS_NAN(col_or_expr) -> Predicate:
    return IsNaN(col_or_expr)

def IS_FINITE(col_or_expr) -> Predicate:
    return IsFinite(col_or_expr)
