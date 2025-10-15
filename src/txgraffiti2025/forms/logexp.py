"""
Logarithmic / Exponential (R5) and sqrt helpers.

These express transformations commonly seen in R5 forms, e.g.:
    log(f(x), base=b)
    exp(f(x))
    sqrt(f(x))
"""

from __future__ import annotations
from typing import Optional, Union
from .utils import to_expr, log, exp, sqrt as _sqrt, Expr

__all__ = [
    "log_base",
    "exp_e",
    "sqrt",
]

NumLike = Union[Expr, str, int, float]

def log_base(x: NumLike, base: Optional[float] = None) -> Expr:
    """Return log_base(x) as an Expr.  Defaults to natural log if base=None."""
    return log(to_expr(x), base=base)

def exp_e(x: NumLike) -> Expr:
    """Return e ** x as an Expr."""
    return exp(to_expr(x))

def sqrt(x: NumLike) -> Expr:
    """Return sqrt(x) as an Expr."""
    return _sqrt(to_expr(x))
