# # src/txgraffiti2025/graffiti_utils.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Callable, Any, Union
# import operator
# import numpy as np
# import pandas as pd
# import math

# __all__ = [
#     "Expr", "Const", "ColumnTerm", "BinOp", "FuncOp",
#     "to_expr",
#     "sqrt", "abs_", "floor", "ceil", "log",
#     "min_", "max_", "absdiff",
# ]

# Number = Union[int, float, np.number]


# # ────────────────────────── core numeric Expr ────────────────────────── #

# class Expr:
#     """
#     Symbolic numeric expression over dataframe columns.
#     `evaluate(df) -> pd.Series` aligned to df.index.

#     Comparisons (>= <= = < >) return BoolFormula relations
#     (see graffiti_generic_conjecture).
#     """

#     # ---- evaluate ----
#     def evaluate(self, df: pd.DataFrame) -> pd.Series:
#         raise NotImplementedError

#     # ---- comparisons -> BoolFormula (new DSL) ----
#     def __ge__(self, other: Any):
#         from .graffiti_generic_conjecture import Ge
#         return Ge(self, to_expr(other))

#     def __le__(self, other: Any):
#         from .graffiti_generic_conjecture import Le
#         return Le(self, to_expr(other))

#     def __eq__(self, other: Any):  # noqa: E711
#         from .graffiti_generic_conjecture import Eq
#         return Eq(self, to_expr(other))

#     def __lt__(self, other: Any):
#         from .graffiti_generic_conjecture import Lt
#         return Lt(self, to_expr(other))

#     def __gt__(self, other: Any):
#         from .graffiti_generic_conjecture import Gt
#         return Gt(self, to_expr(other))

#     def __ne__(self, other: Any):
#         # != is defined as ¬(==)
#         return ~self.__eq__(other)

#     # ---- arithmetic ----
#     def _bin(self, op: Callable, rhs: Any, pretty_op: str) -> "BinOp":
#         return BinOp(op, self, to_expr(rhs), pretty_op)

#     def __add__(self, other: Any): return self._bin(operator.add, other, "+")
#     def __radd__(self, other: Any): return to_expr(other)._bin(operator.add, self, "+")
#     def __sub__(self, other: Any): return self._bin(operator.sub, other, "−")
#     def __rsub__(self, other: Any): return to_expr(other)._bin(operator.sub, self, "−")
#     def __mul__(self, other: Any): return self._bin(operator.mul, other, "·")
#     def __rmul__(self, other: Any): return to_expr(other)._bin(operator.mul, self, "·")
#     def __truediv__(self, other: Any): return self._bin(operator.truediv, other, "⁄")
#     def __rtruediv__(self, other: Any): return to_expr(other)._bin(operator.truediv, self, "⁄")
#     def __floordiv__(self, other: Any): return self._bin(operator.floordiv, other, "//")
#     def __rfloordiv__(self, other: Any): return to_expr(other)._bin(operator.floordiv, self, "//")
#     def __mod__(self, other: Any): return self._bin(operator.mod, other, "%")
#     def __rmod__(self, other: Any): return to_expr(other)._bin(operator.mod, self, "%")
#     def __pow__(self, other: Any): return self._bin(operator.pow, other, "^")
#     def __rpow__(self, other: Any): return to_expr(other)._bin(operator.pow, self, "^")

#     def __neg__(self): return BinOp(operator.mul, Const(-1), self, "·")

#     # Block accidental truthiness of Expr
#     def __bool__(self):
#         raise TypeError("Numeric Expr cannot be used as a boolean. Compare it to form a BoolFormula.")

#     def pretty(self) -> str:
#         return repr(self)


# # NOTE: eq=False prevents dataclasses from overriding Expr.__eq__.
# @dataclass(frozen=True, eq=False)
# class Const(Expr):
#     value: Number
#     def evaluate(self, df: pd.DataFrame) -> pd.Series:
#         return pd.Series(float(self.value), index=df.index)
#     def __repr__(self) -> str:
#         v = float(self.value)
#         return f"{int(v)}" if v.is_integer() else f"{v:g}"


# @dataclass(frozen=True, eq=False)
# class ColumnTerm(Expr):
#     name: str
#     def evaluate(self, df: pd.DataFrame) -> pd.Series:
#         if self.name not in df.columns:
#             raise KeyError(f"Column '{self.name}' not in DataFrame.")
#         s = df[self.name]
#         if s.dtype == bool or str(s.dtype).lower().startswith("boolean"):
#             return s.astype(float)
#         if pd.api.types.is_numeric_dtype(s):
#             return s.astype(float)
#         # Non-numeric passes through; comparisons will coerce numerically later.
#         return s
#     def __repr__(self) -> str:
#         return self.name


# @dataclass(frozen=True, eq=False)
# class BinOp(Expr):
#     op: Callable[[Any, Any], Any]
#     left: Expr
#     right: Expr
#     pretty_op: str = "?"
#     def evaluate(self, df: pd.DataFrame) -> pd.Series:
#         l = self.left.evaluate(df)
#         r = self.right.evaluate(df)
#         out = self.op(l, r)
#         if not isinstance(out, pd.Series):
#             out = pd.Series(out, index=df.index)
#         return out.reindex(df.index)
#     def __repr__(self) -> str:
#         return f"({self.left!r} {self.pretty_op} {self.right!r})"


# @dataclass(frozen=True, eq=False)
# class FuncOp(Expr):
#     fn: Callable[[pd.Series], pd.Series]
#     arg: Expr
#     name: str
#     def evaluate(self, df: pd.DataFrame) -> pd.Series:
#         out = self.fn(self.arg.evaluate(df))
#         if not isinstance(out, pd.Series):
#             out = pd.Series(out, index=df.index)
#         return out.reindex(df.index)
#     def __repr__(self) -> str:
#         return f"{self.name}({self.arg!r})"


# # ────────────────────────── helpers & functions ────────────────────────── #

# def to_expr(x: Any) -> Expr:
#     """Coerce python literal / column name / Expr into an Expr."""
#     if isinstance(x, Expr):
#         return x
#     if isinstance(x, (int, float, np.number)):
#         return Const(float(x))
#     if isinstance(x, str):
#         return ColumnTerm(x)
#     raise TypeError(f"Cannot coerce {type(x).__name__} to Expr")


# def _to_float_series(s: pd.Series) -> pd.Series:
#     return pd.to_numeric(s, errors="coerce").astype(float)


# def sqrt(x: Any) -> Expr:
#     x = to_expr(x)
#     return FuncOp(lambda s: np.sqrt(_to_float_series(s)), x, "√")


# def abs_(x: Any) -> Expr:
#     x = to_expr(x)
#     return FuncOp(lambda s: _to_float_series(s).abs(), x, "abs")


# def floor(x: Any) -> Expr:
#     x = to_expr(x)
#     return FuncOp(lambda s: np.floor(_to_float_series(s)), x, "⌊·⌋")


# def ceil(x: Any) -> Expr:
#     x = to_expr(x)
#     return FuncOp(lambda s: np.ceil(_to_float_series(s)), x, "⌈·⌉")


# def log(x: Any, base: float = math.e) -> Expr:
#     if base <= 0 or base == 1:
#         raise ValueError("log base must be > 0 and != 1")
#     x = to_expr(x)
#     return FuncOp(lambda s: np.log(_to_float_series(s)) / math.log(base),
#                   x, f"log_{base:g}")


# # ────────────────────────── binary function Exprs (pretty) ────────────────────────── #

# @dataclass(frozen=True, eq=False)
# class Func2(Expr):
#     fn: callable
#     a: Expr
#     b: Expr
#     name: str
#     def evaluate(self, df: pd.DataFrame) -> pd.Series:
#         return self.fn(self.a.evaluate(df), self.b.evaluate(df))
#     def __repr__(self) -> str:
#         # print like min(a, b) / max(a, b) / absdiff(a, b) as a fallback
#         return f"{self.name}({self.a!r}, {self.b!r})"

# @dataclass(frozen=True, eq=False)
# class AbsOp(Expr):
#     arg: Expr
#     def evaluate(self, df: pd.DataFrame) -> pd.Series:
#         s = self.arg.evaluate(df)
#         return s.astype(float).abs()
#     def __repr__(self) -> str:
#         # | x |
#         return f"|{self.arg!r}|"

# @dataclass(frozen=True, eq=False)
# class AbsDiff(Expr):
#     a: Expr
#     b: Expr
#     def evaluate(self, df: pd.DataFrame) -> pd.Series:
#         sa = pd.to_numeric(self.a.evaluate(df), errors="coerce").astype(float)
#         sb = pd.to_numeric(self.b.evaluate(df), errors="coerce").astype(float)
#         return (sa - sb).abs()
#     def __repr__(self) -> str:
#         # | x − y |
#         return f"|{self.a!r} − {self.b!r}|"

# def min_(a: Any, b: Any) -> Expr:
#     return Func2(np.minimum, to_expr(a), to_expr(b), "min")

# def max_(a: Any, b: Any) -> Expr:
#     return Func2(np.maximum, to_expr(a), to_expr(b), "max")

# def abs_(x: Any) -> Expr:
#     return AbsOp(to_expr(x))

# def absdiff(a: Any, b: Any) -> Expr:
#     return AbsDiff(to_expr(a), to_expr(b))

# src/txgraffiti2025/graffiti_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Union
import operator
import numpy as np
import pandas as pd
import math

__all__ = [
    "Expr", "Const", "ColumnTerm", "BinOp", "FuncOp",
    "to_expr",
    "sqrt", "abs_", "floor", "ceil", "log",
    "min_", "max_", "absdiff",
]

Number = Union[int, float, np.number]


# ────────────────────────── core numeric Expr ────────────────────────── #

class Expr:
    """
    Symbolic numeric expression over dataframe columns.
    `evaluate(df) -> pd.Series` aligned to df.index.

    Comparisons (>= <= = < >) return BoolFormula relations
    (see graffiti_generic_conjecture).
    """

    # ---- evaluate ----
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    # ---- comparisons -> BoolFormula (new DSL) ----
    def __ge__(self, other: Any):
        from .graffiti_generic_conjecture import Ge
        return Ge(self, to_expr(other))

    def __le__(self, other: Any):
        from .graffiti_generic_conjecture import Le
        return Le(self, to_expr(other))

    def __eq__(self, other: Any):  # noqa: E711
        from .graffiti_generic_conjecture import Eq
        return Eq(self, to_expr(other))

    def __lt__(self, other: Any):
        from .graffiti_generic_conjecture import Lt
        return Lt(self, to_expr(other))

    def __gt__(self, other: Any):
        from .graffiti_generic_conjecture import Gt
        return Gt(self, to_expr(other))

    def __ne__(self, other: Any):
        # != is defined as ¬(==)
        from .graffiti_generic_conjecture import NotF
        return NotF(self.__eq__(other))

    # ---- arithmetic ----
    def _bin(self, op: Callable, rhs: Any, pretty_op: str) -> "BinOp":
        return BinOp(op, self, to_expr(rhs), pretty_op)

    def __add__(self, other: Any): return self._bin(operator.add, other, "+")
    def __radd__(self, other: Any): return to_expr(other)._bin(operator.add, self, "+")
    def __sub__(self, other: Any): return self._bin(operator.sub, other, "−")
    def __rsub__(self, other: Any): return to_expr(other)._bin(operator.sub, self, "−")
    def __mul__(self, other: Any): return self._bin(operator.mul, other, "·")
    def __rmul__(self, other: Any): return to_expr(other)._bin(operator.mul, self, "·")
    def __truediv__(self, other: Any): return self._bin(operator.truediv, other, "⁄")
    def __rtruediv__(self, other: Any): return to_expr(other)._bin(operator.truediv, self, "⁄")
    def __floordiv__(self, other: Any): return self._bin(operator.floordiv, other, "//")
    def __rfloordiv__(self, other: Any): return to_expr(other)._bin(operator.floordiv, self, "//")
    def __mod__(self, other: Any): return self._bin(operator.mod, other, "%")
    def __rmod__(self, other: Any): return to_expr(other)._bin(operator.mod, self, "%")
    def __pow__(self, other: Any): return self._bin(operator.pow, other, "^")
    def __rpow__(self, other: Any): return to_expr(other)._bin(operator.pow, self, "^")

    def __neg__(self): return BinOp(operator.mul, Const(-1), self, "·")

    # Block accidental truthiness of Expr
    def __bool__(self):
        raise TypeError("Numeric Expr cannot be used as a boolean. Compare it to form a BoolFormula.")

    def pretty(self) -> str:
        return repr(self)


# NOTE: eq=False prevents dataclasses from overriding Expr.__eq__.
@dataclass(frozen=True, eq=False)
class Const(Expr):
    value: Number
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(float(self.value), index=df.index)
    def __repr__(self) -> str:
        v = float(self.value)
        return f"{int(v)}" if v.is_integer() else f"{v:g}"


@dataclass(frozen=True, eq=False)
class ColumnTerm(Expr):
    name: str
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if self.name not in df.columns:
            raise KeyError(f"Column '{self.name}' not in DataFrame.")
        s = df[self.name]
        if s.dtype == bool or str(s.dtype).lower().startswith("boolean"):
            return s.astype(float)
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        # Non-numeric passes through; comparisons will coerce numerically later.
        return s
    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class BinOp(Expr):
    op: Callable[[Any, Any], Any]
    left: Expr
    right: Expr
    pretty_op: str = "?"
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l = self.left.evaluate(df)
        r = self.right.evaluate(df)
        out = self.op(l, r)
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=df.index)
        return out.reindex(df.index)
    def __repr__(self) -> str:
        return f"({self.left!r} {self.pretty_op} {self.right!r})"


@dataclass(frozen=True, eq=False)
class FuncOp(Expr):
    fn: Callable[[pd.Series], pd.Series]
    arg: Expr
    name: str
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        out = self.fn(self.arg.evaluate(df))
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=df.index)
        return out.reindex(df.index)
    def __repr__(self) -> str:
        return f"{self.name}({self.arg!r})"


# ────────────────────────── helpers & functions ────────────────────────── #

def to_expr(x: Any) -> Expr:
    """Coerce python literal / column name / Expr into an Expr."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, float, np.number)):
        return Const(float(x))
    if isinstance(x, str):
        return ColumnTerm(x)
    raise TypeError(f"Cannot coerce {type(x).__name__} to Expr")


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def sqrt(x: Any) -> Expr:
    x = to_expr(x)
    return FuncOp(lambda s: np.sqrt(_to_float_series(s)), x, "√")


def floor(x: Any) -> Expr:
    x = to_expr(x)
    return FuncOp(lambda s: np.floor(_to_float_series(s)), x, "⌊·⌋")


def ceil(x: Any) -> Expr:
    x = to_expr(x)
    return FuncOp(lambda s: np.ceil(_to_float_series(s)), x, "⌈·⌉")


def log(x: Any, base: float = math.e) -> Expr:
    if base <= 0 or base == 1:
        raise ValueError("log base must be > 0 and != 1")
    x = to_expr(x)
    return FuncOp(lambda s: np.log(_to_float_series(s)) / math.log(base),
                  x, f"log_{base:g}")


# ─────────────── absolute value & absolute difference (pretty) ─────────────── #

@dataclass(frozen=True, eq=False)
class AbsOp(Expr):
    arg: Expr
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        s = self.arg.evaluate(df)
        return _to_float_series(s).abs()
    def __repr__(self) -> str:
        # |x|
        return f"|{self.arg!r}|"

def abs_(x: Any) -> Expr:
    return AbsOp(to_expr(x))


@dataclass(frozen=True, eq=False)
class AbsDiff(Expr):
    a: Expr
    b: Expr
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        sa = _to_float_series(self.a.evaluate(df))
        sb = _to_float_series(self.b.evaluate(df))
        return (sa - sb).abs()
    def __repr__(self) -> str:
        # |x − y|
        return f"|{self.a!r} − {self.b!r}|"

def absdiff(a: Any, b: Any) -> Expr:
    return AbsDiff(to_expr(a), to_expr(b))


# ────────────────────────── binary function Exprs (min/max) ────────────────────────── #

@dataclass(frozen=True, eq=False)
class Func2(Expr):
    fn: Callable[[pd.Series, pd.Series], pd.Series]
    a: Expr
    b: Expr
    name: str
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        sa = _to_float_series(self.a.evaluate(df))
        sb = _to_float_series(self.b.evaluate(df))
        out = self.fn(sa, sb)
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=df.index)
        return out.reindex(df.index)
    def __repr__(self) -> str:
        # print like min(a, b) / max(a, b)
        return f"{self.name}({self.a!r}, {self.b!r})"

def min_(a: Any, b: Any) -> Expr:
    return Func2(np.minimum, to_expr(a), to_expr(b), "min")

def max_(a: Any, b: Any) -> Expr:
    return Func2(np.maximum, to_expr(a), to_expr(b), "max")
