# src/txgraffiti2025/forms/floorceil.py

"""
Floor/Ceiling wrappers (R3): discrete rounding forms to compose with any expression.

These helpers wrap numeric expressions in ⌊·⌋ / ⌈·⌉ so you can build sharp
integer bounds like “⌊f(x)⌋ ≤ g(x)” or “h(x) ≤ ⌈k(x)⌉”. They return Expr nodes
that evaluate to float Series (integer-valued, but dtype float for alignment).

They interoperate seamlessly with the Expr system (see utils.py) and plug into
relations (Le/Ge/Eq) and Conjecture as usual.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.floorceil import with_floor, with_ceil
>>> df = pd.DataFrame({"x": [0.4, 1.0, 1.6, 2.1]})
>>> with_floor("x").eval(df).tolist()
[0.0, 1.0, 1.0, 2.0]
>>> with_ceil("x").eval(df).tolist()
[1.0, 1.0, 2.0, 3.0]

Inside a relation:
>>> from txgraffiti2025.forms.generic_conjecture import Le
>>> Le("alpha", with_ceil("size") / 2).pretty()
'(alpha ≤ ⌈size⌉ / 2)'
"""

from __future__ import annotations
from typing import Union

from .utils import Expr, to_expr, floor, ceil, Const

__all__ = [
    "with_floor",
    "with_ceil",
    "is_one",
]

NumLike = Union[Expr, float, int, str]


def with_floor(expr: NumLike) -> Expr:
    """
    Wrap an expression with the floor operator: ⌊expr⌋.

    Parameters
    ----------
    expr : Expr | float | int | str
        Expression or column name (strings are parsed via to_expr).

    Returns
    -------
    Expr
        Node representing ⌊expr⌋ (printed as `⌊…⌋` by Expr.pretty()).
    """
    return floor(to_expr(expr))


def with_ceil(expr: NumLike) -> Expr:
    """
    Wrap an expression with the ceiling operator: ⌈expr⌉.

    Parameters
    ----------
    expr : Expr | float | int | str
        Expression or column name (strings are parsed via to_expr).

    Returns
    -------
    Expr
        Node representing ⌈expr⌉ (printed as `⌈…⌉` by Expr.pretty()).
    """
    return ceil(to_expr(expr))


def is_one(expr: object) -> bool:
    """
    Lightweight check: True iff `expr` is exactly the constant 1.

    Notes
    -----
    This is intentionally strict and only recognizes a literal Const(1).
    It does not attempt algebraic normalization (e.g., 2/2, 1*alpha/alpha).
    """
    return isinstance(expr, Const) and float(expr.value) == 1.0
