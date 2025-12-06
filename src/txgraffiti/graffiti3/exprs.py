# src/txgraffiti/graffiti3/exprs.py
"""
Expression utilities for symbolic arithmetic over pandas DataFrames.

This module defines the lightweight `Expr` system used throughout the
TxGraffiti conjecture forms (R1–R6).  Each `Expr` represents a column-wise
numeric expression that can be evaluated on a `pandas.DataFrame` to yield a
`Series` aligned to `df.index`.

What’s new in this rewrite
--------------------------
- Robust, Unicode-first pretty-printing via Expr.pretty() (and __repr__ delegates to it):
  • floor  ⟶  ⌊x⌋
  • ceil   ⟶  ⌈x⌉
  • abs    ⟶  |x|
  • sqrt   ⟶  √(x)              (with parentheses for clarity)
  • log    ⟶  ln(x) / log₂(x) / log₁₀(x) / log_b(x)
  • exp    ⟶  exp(x)
  • mult   ⟶  ·
  • power  ⟶  x², x³, or x^(y) (with precedence-aware parentheses)

- Arithmetic and evaluation semantics unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Union, Sequence, Optional
from fractions import Fraction
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

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
    "min_",        # ← add
    "max_",
]

SeriesLike = Union[pd.Series, float, int, np.ndarray]

# --- Pretty-print helpers for precedence ---
_PRECEDENCE = {
    "**": 4,
    "unary": 3,
    "*": 2, "/": 2, "%": 2,
    "+": 1, "-": 1,
}


# def safe_sqrt_array(a: np.ndarray) -> np.ndarray:
#     """
#     sqrt but never emits runtime warnings.
#     Returns NaN where input is negative or non-finite.
#     """
#     a = np.asarray(a, dtype=float)
#     out = np.full_like(a, np.nan, dtype=float)
#     m = np.isfinite(a) & (a >= 0.0)
#     if np.any(m):
#         out[m] = np.sqrt(a[m])
#     return out



# SeriesLike = Union[pd.Series, np.ndarray, float, int]

def safe_sqrt_series(x: SeriesLike) -> pd.Series:
    """
    sqrt, but never emits runtime warnings.
    Returns NaN where x is negative or non-finite.
    """
    if isinstance(x, pd.Series):
        idx = x.index
        a = x.to_numpy(dtype=float, copy=False)
    else:
        idx = None
        a = np.asarray(x, dtype=float)

    out = np.full_like(a, np.nan, dtype=float)
    m = np.isfinite(a) & (a >= 0.0)
    if np.any(m):
        out[m] = np.sqrt(a[m])
    return pd.Series(out, index=idx)


def _need_parens(child_prec: int, parent_prec: int, is_right_assoc: bool = False, is_right_child: bool = False) -> bool:
    if child_prec < parent_prec:
        return True
    # Right-assoc power: a ** (b ** c) is clearer than a ** b ** c
    if is_right_assoc and is_right_child and child_prec == parent_prec:
        return True
    return False

def _as_series(x: SeriesLike, index: pd.Index) -> pd.Series:
    """
    Normalize scalars, arrays, or Series to a float Series aligned to `index`.
    Guarantees float dtype; booleans become 0.0/1.0.
    """
    if isinstance(x, pd.Series):
        s = x.reindex(index)
        if is_bool_dtype(s):
            return s.astype(float, copy=False)
        s_num = pd.to_numeric(s, errors="coerce")
        return s_num.astype(float, copy=False)

    if isinstance(x, (float, int, np.floating, np.integer)):
        return pd.Series(float(x), index=index, dtype=float)

    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        return pd.Series(float(x_arr), index=index, dtype=float)
    if x_arr.shape[0] != len(index):
        raise ValueError("Array length does not match DataFrame length.")
    return pd.Series(x_arr.astype(float, copy=False), index=index)


# ---------------------------------------------------------------------
# Expression base
# ---------------------------------------------------------------------
class Expr:
    """
    Abstract base for expressions that evaluate to a Series on a DataFrame.

    Subclasses must implement eval(df).
    """

    # ---------- evaluation ----------
    def eval(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    # ---------- operators ----------
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

    # ---------- pretty ----------
    def pretty(self) -> str:
        """Subclasses override; must return a human-friendly Unicode string."""
        return repr(self)  # pragma: no cover

    def __repr__(self) -> str:
        # By design, __repr__ returns pretty() so everything downstream benefits.
        return self.pretty()


# ---------------------------------------------------------------------
# Concrete expressions
# ---------------------------------------------------------------------
# class Const(Expr):
#     def __init__(self, value: float | int | Fraction):
#         self.value = value

#     def eval(self, df: pd.DataFrame) -> pd.Series:
#         return pd.Series(float(self.value), index=df.index, dtype=float)

#     def pretty(self) -> str:
#         v = self.value
#         if isinstance(v, Fraction):
#             if v == 1: return "1"
#             if v == -1: return "-1"
#             if v == 0: return "0"
#             n, d = v.numerator, v.denominator
#             if d == 1: return f"{n}"
#             if d == -1: return f"-{n}"
#             return f"({n}/{d})"
#         f = float(v)
#         return str(int(f)) if f.is_integer() else f"{f}"

class Const(Expr):
    def __init__(self, value: float | int | Fraction):
        # Keep whatever we're given; we'll normalize only at pretty-time.
        self.value = value

    def eval(self, df: pd.DataFrame) -> pd.Series:
        # Numerical semantics stay purely float-based.
        return pd.Series(float(self.value), index=df.index, dtype=float)

    def pretty(self) -> str:
        """
        Pretty-print constants with nice rational forms when possible.

        - Exact Fractions are shown as n/d, 1, -1, or 0.
        - Floats/ints are:
            * printed as an integer if they are (numerically) integral;
            * otherwise converted to a small-denominator Fraction and
              printed as n/d.
        """
        from fractions import Fraction

        v = self.value

        # Normalize to a Fraction when v is not already one
        if isinstance(v, Fraction):
            fr = v
        else:
            f = float(v)
            # Clean integer case first
            if f.is_integer():
                return str(int(f))
            # Small-denominator rational approximation (tune 50 if you like)
            fr = Fraction(f).limit_denominator(50)

        # Now pretty-print the Fraction fr
        if fr == 0:
            return "0"
        if fr == 1:
            return "1"
        if fr == -1:
            return "-1"

        n, d = fr.numerator, fr.denominator
        if d == 1:
            return f"{n}"
        if d == -1:
            return f"-{n}"
        return f"({n}/{d})"


# class ColumnTerm(Expr):
#     def __init__(self, col: str):
#         self.col = col

#     # def eval(self, df: pd.DataFrame) -> pd.Series:
#     #     if self.col not in df.columns:
#     #         raise KeyError(f"Required column '{self.col}' not found in DataFrame.")
#     #     s = df[self.col]
#     #     if is_bool_dtype(s):
#     #         return s.astype(float, copy=False).reindex(df.index)
#     #     return pd.to_numeric(s, errors="coerce").reindex(df.index)
#     def eval(self, df: pd.DataFrame) -> pd.Series:
#         if self.col not in df.columns:
#             raise KeyError(f"Required column '{self.col}' not found in DataFrame.")
#         s = df[self.col].reindex(df.index)
#         if is_bool_dtype(s):
#             return s.astype(float, copy=False)
#         return pd.to_numeric(s, errors="coerce").astype(float, copy=False)


#     def pretty(self) -> str:
#         return self.col

# src/txgraffiti2025/forms/utils.py

class ColumnTerm(Expr):
    """
    Atomic expression referring to a single DataFrame column.

    Normally `col` must be an actual column name in the DataFrame. For
    backward compatibility, we also support certain *derived* naming
    patterns for abs-differences, e.g.:

        - "abs_order_minus_size"
        - "|(order - size)|"

    In those cases, the value is computed on-the-fly from the base columns.
    """

    def __init__(self, col: str):
        self.col = col

    def __repr__(self) -> str:
        return self.pretty()

    def eval(self, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate this column term on `df`.

        If `self.col` is not an actual column but matches a known derived
        pattern (e.g., an abs-difference), compute it from the underlying
        columns instead of raising a KeyError.
        """
        # Fast path: real column
        if self.col in df.columns:
            return df[self.col]

        # Try derived patterns before giving up
        series = self._eval_derived(df)
        if series is not None:
            return series

        # If we get here, we truly don't know how to interpret this name.
        raise KeyError(f"Required column '{self.col}' not found in DataFrame.")

    def _eval_derived(self, df: pd.DataFrame) -> pd.Series | None:
        """
        Handle special naming conventions like abs(x - y).

        Supported patterns:
        - "abs_<x>_minus_<y>"
        - "|(x - y)|"   or "|x - y|"
        - "min(x, y)"   or "max(x, y)"  with simple column names x, y
        """
        name = self.col

        # Pattern 1: "abs_<x>_minus_<y>"
        # e.g. "abs_order_minus_size"
        if name.startswith("abs_") and "_minus_" in name:
            core = name[len("abs_") :]            # strip "abs_"
            left, right = core.split("_minus_", 1)
            left, right = left.strip(), right.strip()
            if left in df.columns and right in df.columns:
                return (df[left] - df[right]).abs()

        # Pattern 2: "|(x - y)|" or "|x - y|"
        # Allow optional parentheses and spaces.
        if name.startswith("|") and name.endswith("|") and "-" in name:
            inner = name[1:-1].strip()  # strip outer pipes
            # Remove optional outer parentheses: "(order - size)" -> "order - size"
            if inner.startswith("(") and inner.endswith(")"):
                inner = inner[1:-1].strip()

            parts = inner.split("-")
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()
                if left in df.columns and right in df.columns:
                    return (df[left] - df[right]).abs()

        # Pattern 3: "min(x, y)" or "max(x, y)"
        if (name.startswith("min(") or name.startswith("max(")) and name.endswith(")"):
            inner = name[name.index("(") + 1 : -1]    # between the parentheses
            # Split on first comma, and allow spaces
            if "," in inner:
                left_str, right_str = inner.split(",", 1)
                left = left_str.strip()
                right = right_str.strip()
                if left in df.columns and right in df.columns:
                    if name.startswith("min("):
                        return np.minimum(df[left], df[right])
                    else:  # max(
                        return np.maximum(df[left], df[right])

        # If no pattern matched, return None so the caller can decide
        return None


    def pretty(self) -> str:
        return self.col



@dataclass
class LinearForm(Expr):
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

    def pretty(self) -> str:
        # a0 + Σ ai*xi, using · for multiply, suppress 1* and -1*.
        parts: list[str] = []
        a0 = float(self.intercept)
        if not np.isclose(a0, 0.0):
            parts.append(Const(a0).pretty())
        for coef, col in self.terms:
            c = float(coef)
            name = ColumnTerm(col).pretty()
            if np.isclose(c, 1.0):
                parts.append(f"{name}")
            elif np.isclose(c, -1.0):
                parts.append(f"-{name}")
            else:
                parts.append(f"{Const(c).pretty()}·{name}")
        if not parts:
            return "0"
        s = " + ".join(parts)
        return s.replace("+ -", "- ")

    def __repr__(self) -> str:
        return self.pretty()


# ---------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------
# Small helpers for Unicode exponents 2 and 3
_SUP2 = "²"
_SUP3 = "³"

def _expr_outer_symbol(s: str) -> str:
    # Greedy-but-fast way to detect top-level operator for precedence inference in strings.
    if s.startswith(("⌊","⌈","|","√","ln","log")) or s.startswith(("exp","neg(")):
        return "unary"
    if "^" in s:
        return "**"
    for op in ("·", "/", "%"):
        if op in s:
            return op
    for op in (" + ", " - "):
        if op in s:
            return op.strip()
    return "unary"

# class BinOp(Expr):
#     def __init__(self, fn: Callable[[SeriesLike, SeriesLike], SeriesLike], left: Expr, right: Expr):
#         self.fn, self.left, self.right = fn, left, right

#     def eval(self, df: pd.DataFrame) -> pd.Series:
#         l = _as_series(self.left.eval(df), df.index)
#         r = _as_series(self.right.eval(df), df.index)
#         out = self.fn(l, r)  # may be Series or ndarray
#         return _as_series(out, df.index)

#     def pretty(self) -> str:
#         # Map numpy ufunc to symbol
#         sym = {
#             np.add: "+", np.subtract: "-", np.multiply: "·",
#             np.divide: "/", np.mod: "%", np.power: "**",
#         }.get(self.fn, "op")

#         # ---------- Special pretty for power with small integer exponents ----------
#         if sym == "**":
#             if isinstance(self.right, Const):
#                 try:
#                     f = float(self.right.value)
#                     if f.is_integer():
#                         n = int(f)
#                         if n == 2:
#                             base = self.left.pretty()
#                             if _need_parens(
#                                 _PRECEDENCE.get(_expr_outer_symbol(base), 3),
#                                 _PRECEDENCE["**"],
#                             ):
#                                 base = f"({base})"
#                             return f"{base}{_SUP2}"
#                         if n == 3:
#                             base = self.left.pretty()
#                             if _need_parens(
#                                 _PRECEDENCE.get(_expr_outer_symbol(base), 3),
#                                 _PRECEDENCE["**"],
#                             ):
#                                 base = f"({base})"
#                             return f"{base}{_SUP3}"
#                 except Exception:
#                     pass
#             # generic power
#             left_s = self.left.pretty()
#             right_s = self.right.pretty()
#             lp = _PRECEDENCE.get(_expr_outer_symbol(left_s), 3)
#             rp = _PRECEDENCE.get(_expr_outer_symbol(right_s), 3)
#             if _need_parens(lp, _PRECEDENCE["**"]):
#                 left_s = f"({left_s})"
#             if _need_parens(rp, _PRECEDENCE["**"], is_right_assoc=True, is_right_child=True):
#                 right_s = f"({right_s})"
#             return f"{left_s}^{right_s}"

#         # ---------- Tiny algebraic simplifications with 0 ----------
#         def _is_zero_expr(e: Expr) -> bool:
#             from fractions import Fraction
#             if isinstance(e, Const):
#                 try:
#                     v = e.value
#                     if isinstance(v, Fraction):
#                         return v == 0
#                     f = float(v)
#                     return abs(f) < 1e-12
#                 except Exception:
#                     return False
#             return False

#         # x + 0  →  x ;  0 + x → x
#         if sym == "+":
#             if _is_zero_expr(self.left):
#                 return self.right.pretty()
#             if _is_zero_expr(self.right):
#                 return self.left.pretty()

#         # x - 0 → x  (but keep 0 - x as-is: that's handled downstream)
#         if sym == "-":
#             if _is_zero_expr(self.right):
#                 return self.left.pretty()

#         # ---------- Normal binary ops with precedence-aware parentheses ----------
#         parent_prec = 2 if sym in ("·", "/", "%") else 1
#         left_s = self.left.pretty()
#         right_s = self.right.pretty()
#         lp = _PRECEDENCE.get(_expr_outer_symbol(left_s), 3)
#         rp = _PRECEDENCE.get(_expr_outer_symbol(right_s), 3)
#         if _need_parens(lp, parent_prec, is_right_assoc=False, is_right_child=False):
#             left_s = f"({left_s})"
#         if _need_parens(rp, parent_prec, is_right_assoc=False, is_right_child=True):
#             right_s = f"({right_s})"
#         return f"({left_s} {sym} {right_s})"


class BinOp(Expr):
    def __init__(self, fn: Callable[[SeriesLike, SeriesLike], SeriesLike], left: Expr, right: Expr):
        self.fn, self.left, self.right = fn, left, right

    def eval(self, df: pd.DataFrame) -> pd.Series:
        l = _as_series(self.left.eval(df), df.index)
        r = _as_series(self.right.eval(df), df.index)
        out = self.fn(l, r)  # may be Series or ndarray
        return _as_series(out, df.index)

    def pretty(self) -> str:
        # Map numpy ufunc to symbol
        sym = {
            np.add: "+",
            np.subtract: "-",
            np.multiply: "·",
            np.divide: "/",
            np.mod: "%",
            np.power: "**",
        }.get(self.fn, "op")

        # ---------- Tiny helpers for constants ----------
        from fractions import Fraction

        def _is_zero_expr(e: Expr) -> bool:
            if isinstance(e, Const):
                try:
                    v = e.value
                    if isinstance(v, Fraction):
                        return v == 0
                    f = float(v)
                    return abs(f) < 1e-12
                except Exception:
                    return False
            return False

        def _is_one_expr(e: Expr) -> bool:
            if isinstance(e, Const):
                try:
                    v = e.value
                    if isinstance(v, Fraction):
                        return v == 1
                    f = float(v)
                    return abs(f - 1.0) < 1e-12
                except Exception:
                    return False
            return False

        def _is_minus_one_expr(e: Expr) -> bool:
            if isinstance(e, Const):
                try:
                    v = e.value
                    if isinstance(v, Fraction):
                        return v == -1
                    f = float(v)
                    return abs(f + 1.0) < 1e-12
                except Exception:
                    return False
            return False

        # ---------- Special pretty for power with small integer exponents ----------
        if sym == "**":
            if isinstance(self.right, Const):
                try:
                    f = float(self.right.value)
                except Exception:
                    f = None

                if f is not None:
                    # x^0 → 1
                    if abs(f) < 1e-12:
                        return "1"

                    # x^1 → x
                    if abs(f - 1.0) < 1e-12:
                        base = self.left.pretty()
                        if _need_parens(
                            _PRECEDENCE.get(_expr_outer_symbol(base), 3),
                            _PRECEDENCE["**"],
                        ):
                            base = f"({base})"
                        return base

                    # x², x³ with nice Unicode
                    if f.is_integer():
                        n = int(f)
                        if n == 2:
                            base = self.left.pretty()
                            if _need_parens(
                                _PRECEDENCE.get(_expr_outer_symbol(base), 3),
                                _PRECEDENCE["**"],
                            ):
                                base = f"({base})"
                            return f"{base}{_SUP2}"
                        if n == 3:
                            base = self.left.pretty()
                            if _need_parens(
                                _PRECEDENCE.get(_expr_outer_symbol(base), 3),
                                _PRECEDENCE["**"],
                            ):
                                base = f"({base})"
                            return f"{base}{_SUP3}"

            # generic power
            left_s = self.left.pretty()
            right_s = self.right.pretty()
            lp = _PRECEDENCE.get(_expr_outer_symbol(left_s), 3)
            rp = _PRECEDENCE.get(_expr_outer_symbol(right_s), 3)
            if _need_parens(lp, _PRECEDENCE["**"]):
                left_s = f"({left_s})"
            if _need_parens(
                rp,
                _PRECEDENCE["**"],
                is_right_assoc=True,
                is_right_child=True,
            ):
                right_s = f"({right_s})"
            return f"{left_s}^{right_s}"

        # ---------- Tiny algebraic simplifications with 0 ----------
        # x + 0  →  x ;  0 + x → x
        if sym == "+":
            if _is_zero_expr(self.left):
                return self.right.pretty()
            if _is_zero_expr(self.right):
                return self.left.pretty()

        # x - 0 → x  (but keep 0 - x as-is: that's handled downstream)
        if sym == "-":
            if _is_zero_expr(self.right):
                return self.left.pretty()

        # ---------- Simplifications for multiplication by 1 / -1 ----------
        if sym == "·":
            # 1·x → x,  x·1 → x
            if _is_one_expr(self.left):
                return self.right.pretty()
            if _is_one_expr(self.right):
                return self.left.pretty()

            # (-1)·x → -x,  x·(-1) → -x
            if _is_minus_one_expr(self.left):
                inner = self.right.pretty()
                ip = _PRECEDENCE.get(_expr_outer_symbol(inner), 3)
                return f"-({inner})" if ip < _PRECEDENCE["unary"] else f"-{inner}"
            if _is_minus_one_expr(self.right):
                inner = self.left.pretty()
                ip = _PRECEDENCE.get(_expr_outer_symbol(inner), 3)
                return f"-({inner})" if ip < _PRECEDENCE["unary"] else f"-{inner}"

        # ---------- Normal binary ops with precedence-aware parentheses ----------
        parent_prec = 2 if sym in ("·", "/", "%") else 1
        left_s = self.left.pretty()
        right_s = self.right.pretty()
        lp = _PRECEDENCE.get(_expr_outer_symbol(left_s), 3)
        rp = _PRECEDENCE.get(_expr_outer_symbol(right_s), 3)

        if _need_parens(lp, parent_prec, is_right_assoc=False, is_right_child=False):
            left_s = f"({left_s})"
        if _need_parens(rp, parent_prec, is_right_assoc=False, is_right_child=True):
            right_s = f"({right_s})"

        return f"({left_s} {sym} {right_s})"


class UnaryOp(Expr):
    def __init__(self, fn: Callable[[SeriesLike], SeriesLike], arg: Expr):
        self.fn, self.arg = fn, arg

    def eval(self, df: pd.DataFrame) -> pd.Series:
        a = _as_series(self.arg.eval(df), df.index)
        return _as_series(self.fn(a), df.index)

    def pretty(self) -> str:
        name = {
            np.floor: "floor",
            np.ceil: "ceil",
            np.abs: "abs",
            np.exp: "exp",
            np.sqrt: "sqrt",          # legacy sqrt (not used after patch, but ok)
            safe_sqrt_series: "sqrt",# recognize safe sqrt
            np.negative: "neg",
        }.get(self.fn, getattr(self.fn, "__name__", "unary"))

        inner = self.arg.pretty()

        def paren(s: str) -> str:
            return f"({s})"

        if name == "floor":
            return f"⌊{inner}⌋"
        if name == "ceil":
            return f"⌈{inner}⌉"
        if name == "abs":
            return f"|{inner}|"
        if name == "sqrt":
            return f"√{paren(inner)}"
        if name == "exp":
            return f"exp({inner})"
        if name == "neg":
            ip = _PRECEDENCE.get(_expr_outer_symbol(inner), 3)
            return f"-({inner})" if ip < _PRECEDENCE["unary"] else f"-{inner}"
        return f"{name}{paren(inner)}"


class LogOp(Expr):
    def __init__(self, arg: Expr, base: Optional[float], epsilon: float = 0.0):
        self.arg = arg
        self.base = base
        self.epsilon = float(epsilon)

    # def eval(self, df: pd.DataFrame) -> pd.Series:
    #     a = _as_series(self.arg.eval(df), df.index)
    #     # clamp away from nonpositive if epsilon > 0
    #     if self.epsilon > 0.0:
    #         a = np.maximum(a, self.epsilon)
    #     if self.base is None:
    #         return _as_series(np.log(a), df.index)
    #     denom = np.log(float(self.base))
    #     return _as_series(np.log(a) / denom, df.index)
    def eval(self, df: pd.DataFrame) -> pd.Series:
        a = _as_series(self.arg.eval(df), df.index)
        if self.epsilon > 0.0:
            a = np.maximum(a, self.epsilon)
        out = np.log(a)
        if self.base is None:
            return _as_series(out, df.index)
        b = float(self.base)
        if not np.isfinite(b) or b <= 0.0 or np.isclose(b, 1.0):
            # Return NaNs rather than raising to keep row-wise robustness
            return pd.Series(np.nan, index=df.index, dtype=float)
        return _as_series(out / np.log(b), df.index)


    def pretty(self) -> str:
        x = self.arg.pretty()
        if self.base is None:
            return f"ln({x})"
        try:
            b = float(self.base)
            if b.is_integer():
                n = int(b)
                sub = {0:"₀",1:"₁",2:"₂",3:"₃",4:"₄",5:"₅",6:"₆",7:"₇",8:"₈",9:"₉"}
                if 0 <= n <= 9:
                    return f"log{sub[n]}({x})"
        except Exception:
            pass
        if self.base == 10:
            return f"log₁₀({x})"
        if self.base == 2:
            return f"log₂({x})"
        return f"log_{self.base}({x})"


# ---------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------
def to_expr(x: Union[Expr, float, int, str, Fraction]) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, Fraction):
        return Const(x)
    if isinstance(x, (float, int, np.floating, np.integer)):
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

def log(x: Union[Expr, float, int, str], base: Optional[float] = None, *, epsilon: float = 0.0) -> Expr:
    return LogOp(to_expr(x), base=base, epsilon=epsilon)

def exp(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(np.exp, to_expr(x))

def sqrt(x: Union[Expr, float, int, str]) -> Expr:
    return UnaryOp(safe_sqrt_series, to_expr(x))

# In forms/utils.py

class Func2Op(Expr):
    """
    Two-argument functional combinator, pretty-printed as name(left, right).

    Examples:
        min_(x, y) -> Func2Op(np.minimum, "min", x, y)
        max_(x, y) -> Func2Op(np.maximum, "max", x, y)
    """
    __slots__ = ("fn", "name", "left", "right")

    def __init__(self, fn: Callable[[SeriesLike, SeriesLike], SeriesLike], name: str,
                 left: Expr, right: Expr):
        self.fn = fn
        self.name = name
        self.left = left
        self.right = right
        self.exclude = (repr(left), repr(right))

    def eval(self, df: pd.DataFrame) -> pd.Series:
        l = _as_series(self.left.eval(df), df.index)
        r = _as_series(self.right.eval(df), df.index)
        out = self.fn(l, r)  # supports Series/ndarray broadcasting
        return _as_series(out, df.index)

    def pretty(self) -> str:
        # Keep it simple and Unicode-clean; nested Exprs print themselves.
        return f"{self.name}({self.left.pretty()}, {self.right.pretty()})"

    def __repr__(self) -> str:
        return self.pretty()




# --- Helpers for min/max canonicalization and tiny simplifications ----

def _is_commutative(name: str) -> bool:
    return name in ("min", "max")

def _canon_pair_for_commutative(a: Expr, b: Expr) -> tuple[Expr, Expr]:
    """
    Stable, readable canonicalization for commutative binary ops to reduce duplicates.
    Order by the `pretty()` string; if equal objects, keep as-is.
    """
    sa, sb = a.pretty(), b.pretty()
    if sa < sb:
        return a, b
    if sb < sa:
        return b, a
    # Equal strings – prefer structural tie-break that keeps original order
    return a, b

def _same_expr(a: Expr, b: Expr) -> bool:
    # Cheap identity check first; fallback to string form (robust for our DSL)
    return (a is b) or (a.pretty() == b.pretty())

def _const_value(e: Expr) -> float | None:
    if isinstance(e, Const):
        try:
            return float(e.value)
        except Exception:
            return None
    return None


# --- Public constructors: min_ / max_ with simplifications ---------------

def min_(x: Union[Expr, float, int, str], y: Union[Expr, float, int, str]) -> Expr:
    """
    Pointwise minimum: returns an Expr representing min(x, y).
    Pretty: min(x, y)
    """
    a, b = to_expr(x), to_expr(y)

    # Canonicalize order to limit duplicates like min(a,b) vs min(b,a)
    a, b = _canon_pair_for_commutative(a, b)

    # Trivial simplifications
    if _same_expr(a, b):
        return a

    # Constant folding if both are Const
    va, vb = _const_value(a), _const_value(b)
    if (va is not None) and (vb is not None):
        return Const(min(va, vb))

    return Func2Op(np.minimum, "min", a, b)


def max_(x: Union[Expr, float, int, str], y: Union[Expr, float, int, str]) -> Expr:
    """
    Pointwise maximum: returns an Expr representing max(x, y).
    Pretty: max(x, y)
    """
    a, b = to_expr(x), to_expr(y)

    # Canonicalize order
    a, b = _canon_pair_for_commutative(a, b)

    # Trivial simplifications
    if _same_expr(a, b):
        return a

    # Constant folding if both are Const
    va, vb = _const_value(a), _const_value(b)
    if (va is not None) and (vb is not None):
        return Const(max(va, vb))

    return Func2Op(np.maximum, "max", a, b)
