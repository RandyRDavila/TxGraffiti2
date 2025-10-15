"""
Floor/Ceiling wrappers (R3): discrete rounding forms to compose with any expression.

Used to express relations of the form:
    floor(f(x)) <= g(x)
    or
    h(x) <= ceil(k(x))
"""

from __future__ import annotations
from typing import Union
from .utils import floor, ceil, to_expr, Expr

__all__ = [
    "with_floor",
    "with_ceil",
]

NumLike = Union[Expr, float, int, str]

def with_floor(expr: NumLike) -> Expr:
    """Return floor(expr) as an Expr."""
    return floor(to_expr(expr))

def with_ceil(expr: NumLike) -> Expr:
    """Return ceil(expr) as an Expr."""
    return ceil(to_expr(expr))
