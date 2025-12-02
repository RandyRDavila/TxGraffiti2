# src/txgraffiti2025/forms/expr_utils.py
from __future__ import annotations
from typing import Iterable, Set
import numpy as np
from txgraffiti2025.forms.utils import Expr, BinOp, UnaryOp, ColumnTerm, to_expr, Const

__all__ = [
    "expr_depends_on",
    "cancel_feature_in_product",
    "simplify_coeff_times_feature",
    "structurally_equal",
    "is_one",
]


def expr_depends_on(expr: Expr, columns: Iterable[str] | str) -> bool:
    if isinstance(columns, str):
        cols: Set[str] = {columns}
    else:
        cols = set(columns)
    def _walk(e: Expr) -> bool:
        if isinstance(e, ColumnTerm): return e.col in cols
        if isinstance(e, BinOp):      return _walk(e.left) or _walk(e.right)
        if isinstance(e, UnaryOp):    return _walk(e.arg)
        return False
    return _walk(expr)

def cancel_feature_in_product(coeff: Expr, feature: str) -> Expr | None:
    # handle (num / Col(feature)) * Col(feature) -> num
    if isinstance(coeff, BinOp) and coeff.fn is np.divide:
        den = coeff.right
        if isinstance(den, ColumnTerm) and den.col == feature:
            return coeff.left
    return None

def simplify_coeff_times_feature(coeff: Expr, feature: str) -> Expr:
    num = cancel_feature_in_product(coeff, feature)
    if num is not None:
        return num
    return coeff * to_expr(feature)

def structurally_equal(a: Expr, b: Expr) -> bool:
    return repr(a) == repr(b)

def is_one(expr) -> bool:
    try:
        from txgraffiti2025.forms.utils import Const
        if isinstance(expr, Const):
            return float(expr.value) == 1.0
    except Exception:
        pass
    return False
