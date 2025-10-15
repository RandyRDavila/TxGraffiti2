"""
Nonlinear forms (R4): products, powers, and simple rational combos over DataFrame columns.
All helpers return an Expr and accept Expr | str | int | float for convenience.
"""

from __future__ import annotations
from typing import Iterable, Union
from functools import reduce
from operator import mul

from .utils import Expr, to_expr

__all__ = [
    "product",
    "product_n",
    "power",
    "ratio",
    "reciprocal",
    "square",
    "cube",
]

NumLike = Union[Expr, str, int, float]

def product(a: NumLike, b: NumLike) -> Expr:
    """Binary product a * b."""
    return to_expr(a) * to_expr(b)

def product_n(xs: Iterable[NumLike]) -> Expr:
    """n-ary product over an iterable of terms. Raises on empty iterable."""
    it = iter(xs)
    try:
        first = to_expr(next(it))
    except StopIteration:
        raise ValueError("product_n requires at least one term")
    return reduce(lambda acc, x: acc * to_expr(x), it, first)

def power(x: NumLike, p: float) -> Expr:
    """x ** p (fractional/real powers allowed)."""
    return to_expr(x) ** p

def ratio(a: NumLike, b: NumLike) -> Expr:
    """a / b."""
    return to_expr(a) / to_expr(b)

def reciprocal(x: NumLike) -> Expr:
    """1 / x."""
    return 1.0 / to_expr(x)

# small shorthands
def square(x: NumLike) -> Expr:  # x^2
    return to_expr(x) ** 2.0

def cube(x: NumLike) -> Expr:    # x^3
    return to_expr(x) ** 3.0
