# src/txgraffiti2025/forms/nonlinear.py
"""
Nonlinear forms (R4): powers, products, and rational combinations of DataFrame columns.

These helpers build composite :class:`Expr` objects representing nonlinear
combinations of invariants. They correspond to **Form R4** in Hansen–Aouchiche–
Caporossi, where conjectures involve products or powers of graph invariants such as:

  χ(G)·χ(Ĝ), α(G)·ω(G), √n, 1/μ(G), n², α(G)³, a(G)/b(G)

All helpers accept :class:`Expr`, column names (str), or numeric constants,
and return an :class:`Expr` that can be used inside relations like
:class:`~txgraffiti2025.forms.generic_conjecture.Le`, :class:`Ge`, or :class:`Eq`.
"""

from __future__ import annotations
from typing import Iterable, Union
from functools import reduce

from .utils import Expr, to_expr

__all__ = [
    "product",
    "product_n",
    "power",
    "ratio",
    "safe_ratio",
    "reciprocal",
    "reciprocal_eps",
    "square",
    "cube",
    "nth_root",
    "geometric_mean",
]

NumLike = Union[Expr, str, int, float]


# -------------------------------------------------------------------
# Products
# -------------------------------------------------------------------

def product(a: NumLike, b: NumLike) -> Expr:
    """
    Binary product ``a * b``.
    """
    return to_expr(a) * to_expr(b)


def product_n(xs: Iterable[NumLike]) -> Expr:
    """
    n-ary product over an iterable of terms.

    Raises
    ------
    ValueError
        If the iterable is empty.
    """
    it = iter(xs)
    try:
        first = to_expr(next(it))
    except StopIteration:
        raise ValueError("product_n requires at least one term")
    return reduce(lambda acc, x: acc * to_expr(x), it, first)


# -------------------------------------------------------------------
# Powers & roots
# -------------------------------------------------------------------

def power(x: NumLike, p: float) -> Expr:
    """
    Raise an expression to a (possibly fractional) power: ``x ** p``.
    """
    return to_expr(x) ** p


def square(x: NumLike) -> Expr:
    """Shorthand for ``x ** 2``."""
    return to_expr(x) ** 2.0


def cube(x: NumLike) -> Expr:
    """Shorthand for ``x ** 3``."""
    return to_expr(x) ** 3.0


def nth_root(x: NumLike, n: float) -> Expr:
    """
    n-th root: ``x ** (1/n)``.
    """
    return to_expr(x) ** (1.0 / float(n))


# -------------------------------------------------------------------
# Ratios & reciprocals
# -------------------------------------------------------------------

def ratio(a: NumLike, b: NumLike) -> Expr:
    """
    Ratio ``a / b``.
    """
    return to_expr(a) / to_expr(b)


def safe_ratio(a: NumLike, b: NumLike, eps: float = 0.0) -> Expr:
    """
    Ratio with small positive clamp on denominator: ``a / (b + eps)``.

    Useful in exploratory passes or examples to avoid division-by-zero warnings.
    """
    return to_expr(a) / (to_expr(b) + float(eps))


def reciprocal(x: NumLike) -> Expr:
    """
    Reciprocal ``1 / x``.
    """
    return 1.0 / to_expr(x)


def reciprocal_eps(x: NumLike, eps: float = 0.0) -> Expr:
    """
    Guarded reciprocal: ``1 / (x + eps)``.
    """
    return 1.0 / (to_expr(x) + float(eps))


# -------------------------------------------------------------------
# Aggregates
# -------------------------------------------------------------------

def geometric_mean(xs: Iterable[NumLike]) -> Expr:
    """
    Geometric mean of terms: ``(∏ x_i) ** (1/k)`` for k = number of terms.

    Raises
    ------
    ValueError
        If the iterable is empty.
    """
    xs_list = list(xs)
    k = len(xs_list)
    if k == 0:
        raise ValueError("geometric_mean requires at least one term")
    return product_n(xs_list) ** (1.0 / k)
