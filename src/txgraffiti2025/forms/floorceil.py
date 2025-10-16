# src/txgraffiti2025/forms/floorceil.py

"""
Floor/Ceiling wrappers (R3): discrete rounding forms to compose with any expression.

These helpers wrap numeric expressions in integer rounding, enabling conjecture
forms that frequently appear as sharp bounds in the survey (R3):

- ``floor(f(x)) <= g(x)``
- ``h(x) <= ceil(k(x))``

They interoperate with any :class:`Expr` produced by your expression system
(e.g., arithmetic on columns via :func:`to_expr`, linear builders, etc.), and
can be passed directly to relations like :class:`~txgraffiti2025.forms.generic_conjecture.Le`
and :class:`~txgraffiti2025.forms.generic_conjecture.Ge`.

Notes
-----
R3 (floor/ceil) forms are common in “interesting” conjectures—rounding tightens
otherwise fractional linear/nonlinear bounds. See the attached survey’s
discussion of integer rounding in sharp inequalities.

Examples
--------
Wrap a column expression and evaluate:

>>> import pandas as pd
>>> from txgraffiti2025.forms.floorceil import with_floor, with_ceil
>>> from txgraffiti2025.forms.utils import to_expr
>>> df = pd.DataFrame({"n": [1, 2, 3, 4], "x": [0.4, 1.0, 1.6, 2.1]})
>>> e_floor = with_floor(to_expr("x") + 0.5)
>>> e_ceil  = with_ceil(to_expr("x") / 2)
>>> e_floor.eval(df).tolist()
[0.0, 1.0, 2.0, 2.0]
>>> e_ceil.eval(df).tolist()
[1.0, 1.0, 1.0, 2.0]

Use inside a relation / conjecture:

>>> from txgraffiti2025.forms.generic_conjecture import Le, Conjecture
>>> # Example bound: alpha <= ceil(size / 2)
>>> df = pd.DataFrame({"alpha": [1, 2, 3], "size": [1, 3, 5]})
>>> r = Le("alpha", with_ceil(to_expr("size") / 2))
>>> r.evaluate(df).tolist()
[True, True, False]
>>> r.slack(df).tolist()   # right - left
[0.0, -0.5, -0.5]
>>> conj = Conjecture(r)   # global conjecture (no class condition)
>>> _, holds, failures = conj.check(df)
>>> holds.tolist()
[True, True, False]
>>> list(failures.index)
[2]
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
    """
    Wrap an expression with the integer floor operator.

    Parameters
    ----------
    expr : Expr or float or int or str
        Any numeric expression or column name. Strings are parsed via :func:`to_expr`.

    Returns
    -------
    Expr
        An expression representing ``floor(expr)`` (as a real-valued :class:`Expr`).

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.floorceil import with_floor
    >>> from txgraffiti2025.forms.utils import to_expr
    >>> df = pd.DataFrame({"x": [0.9, 1.0, 1.1]})
    >>> with_floor(to_expr("x")).eval(df).tolist()
    [0.0, 1.0, 1.0]
    """
    return floor(to_expr(expr))


def with_ceil(expr: NumLike) -> Expr:
    """
    Wrap an expression with the integer ceiling operator.

    Parameters
    ----------
    expr : Expr or float or int or str
        Any numeric expression or column name. Strings are parsed via :func:`to_expr`.

    Returns
    -------
    Expr
        An expression representing ``ceil(expr)`` (as a real-valued :class:`Expr`).

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.floorceil import with_ceil
    >>> from txgraffiti2025.forms.utils import to_expr
    >>> df = pd.DataFrame({"x": [0.0, 0.1, 0.9, 1.0]})
    >>> with_ceil(to_expr("x")).eval(df).tolist()
    [0.0, 1.0, 1.0, 1.0]
    """
    return ceil(to_expr(expr))
