# src/txgraffiti2025/forms/nonlinear.py

"""
Nonlinear forms (R4): powers, products, and rational combinations of DataFrame columns.

These helpers build composite :class:`Expr` objects representing nonlinear
combinations of invariants. They correspond to **Form R4** in Hansen–Aouchiche–
Caporossi *“What Forms Do Interesting Conjectures Have in Graph Theory?”*,
where conjectures involve products or powers of graph invariants such as:

    χ(G)·χ(Ĝ), α(G)·ω(G), √n, 1/μ(G), n², α(G)³, and a(G)/b(G).

Such expressions extend the linear relations of R2 and R3 to nonlinear, often
multiplicative bounds—central to many Nordhaus–Gaddum and temperature-type
conjectures.

All helpers accept :class:`Expr`, column names (str), or numeric constants,
and return an :class:`Expr` that can be passed to relations like
:class:`~txgraffiti2025.forms.generic_conjecture.Le` or
:class:`~txgraffiti2025.forms.generic_conjecture.Ge`.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.nonlinear import product, power, ratio, reciprocal
>>> from txgraffiti2025.forms.utils import to_expr
>>> df = pd.DataFrame({"alpha": [2, 3], "omega": [2, 1], "order": [5, 10]})

>>> # Product: α·ω  (e.g., α(G)·ω(G) ≤ n)
>>> e = product("alpha", "omega")
>>> e.eval(df).tolist()
[4, 3]

>>> # Power: n^0.5
>>> power("order", 0.5).eval(df).round(3).tolist()
[2.236, 3.162]

>>> # Ratio: α / ω
>>> ratio("alpha", "omega").eval(df).tolist()
[1.0, 3.0]

>>> # Reciprocal: 1 / α
>>> reciprocal("alpha").eval(df).tolist()
[0.5, 0.3333333333]

These expressions can be composed:
>>> from txgraffiti2025.forms.generic_conjecture import Le
>>> r = Le(product("alpha", "omega"), "order")   # α·ω ≤ n
>>> r.evaluate(df).tolist()
[True, True]
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
    """
    Binary product ``a * b``.

    Parameters
    ----------
    a, b : Expr or str or int or float
        Expressions or columns to multiply. Strings are parsed via :func:`to_expr`.

    Returns
    -------
    Expr
        Expression representing the product.

    Notes
    -----
    Corresponds to R4 multiplicative forms, e.g.
    ``χ(G)·χ(Ĝ)``, ``α(G)·ω(G)``, or ``size·order``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.nonlinear import product
    >>> df = pd.DataFrame({"a": [2, 3], "b": [4, 5]})
    >>> product("a", "b").eval(df).tolist()
    [8, 15]
    """
    return to_expr(a) * to_expr(b)


def product_n(xs: Iterable[NumLike]) -> Expr:
    """
    n-ary product over an iterable of terms.

    Parameters
    ----------
    xs : Iterable[Expr or str or int or float]
        Sequence of expressions or columns to multiply.

    Returns
    -------
    Expr
        Expression representing the product of all terms.

    Raises
    ------
    ValueError
        If the iterable is empty.

    Notes
    -----
    Useful for compound multiplicative forms, e.g.
    ``α(G)·ω(G)·μ(G)`` in higher-order conjectures.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.nonlinear import product_n
    >>> df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    >>> product_n(["x", "y", "z"]).eval(df).tolist()
    [15, 48]
    """
    it = iter(xs)
    try:
        first = to_expr(next(it))
    except StopIteration:
        raise ValueError("product_n requires at least one term")
    return reduce(lambda acc, x: acc * to_expr(x), it, first)


def power(x: NumLike, p: float) -> Expr:
    """
    Raise an expression to a (possibly fractional) power.

    Parameters
    ----------
    x : Expr or str or int or float
        Base expression or column.
    p : float
        Exponent (real-valued).

    Returns
    -------
    Expr
        Expression representing ``x ** p``.

    Notes
    -----
    Covers R4 “power” forms—squares, cubes, roots, or fractional powers.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.nonlinear import power
    >>> df = pd.DataFrame({"n": [1, 4, 9]})
    >>> power("n", 0.5).eval(df).tolist()
    [1.0, 2.0, 3.0]
    """
    return to_expr(x) ** p


def ratio(a: NumLike, b: NumLike) -> Expr:
    """
    Construct a ratio expression ``a / b``.

    Parameters
    ----------
    a, b : Expr or str or int or float
        Numerator and denominator.

    Returns
    -------
    Expr
        Expression representing ``a / b``.

    Notes
    -----
    Used for rational R4 forms such as
    ``α(G)/μ(G)``, ``size/order``, or ``f(x)/(g(x)+1)``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.nonlinear import ratio
    >>> df = pd.DataFrame({"a": [2, 4], "b": [1, 2]})
    >>> ratio("a", "b").eval(df).tolist()
    [2.0, 2.0]
    """
    return to_expr(a) / to_expr(b)


def reciprocal(x: NumLike) -> Expr:
    """
    Construct the reciprocal ``1 / x``.

    Parameters
    ----------
    x : Expr or str or int or float
        Denominator expression.

    Returns
    -------
    Expr
        Expression representing ``1 / x``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.nonlinear import reciprocal
    >>> df = pd.DataFrame({"x": [1, 2, 4]})
    >>> reciprocal("x").eval(df).tolist()
    [1.0, 0.5, 0.25]
    """
    return 1.0 / to_expr(x)


def square(x: NumLike) -> Expr:
    """
    Convenience shorthand for ``x ** 2``.

    Parameters
    ----------
    x : Expr or str or int or float

    Returns
    -------
    Expr
        Expression representing ``x²``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.nonlinear import square
    >>> df = pd.DataFrame({"x": [1, 3, 5]})
    >>> square("x").eval(df).tolist()
    [1.0, 9.0, 25.0]
    """
    return to_expr(x) ** 2.0


def cube(x: NumLike) -> Expr:
    """
    Convenience shorthand for ``x ** 3``.

    Parameters
    ----------
    x : Expr or str or int or float

    Returns
    -------
    Expr
        Expression representing ``x³``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.nonlinear import cube
    >>> df = pd.DataFrame({"x": [1, 2, 3]})
    >>> cube("x").eval(df).tolist()
    [1.0, 8.0, 27.0]
    """
    return to_expr(x) ** 3.0
