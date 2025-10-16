# src/txgraffiti2025/forms/logexp.py

"""
Logarithmic / Exponential forms (R5) and square-root helper.

R5 conjecture forms introduce transcendental or root-type transformations
of graph invariants, extending beyond the algebraic (R4) cases.  Common
examples from the literature include:

    log(f(x), base=b)
    exp(f(x))
    sqrt(f(x))  ≡  (f(x))**0.5

These are typical of growth-rate conjectures (e.g., exponential bounds on
graph counts) or information-theoretic analogues of entropy/energy bounds.
All functions return :class:`Expr` objects compatible with the rest of the
conjecture system.

References
----------
Hansen, P., Aouchiche, M., & Caporossi, G. (2013).
*What Forms Do Interesting Conjectures Have in Graph Theory?*
Electronic Notes in Discrete Mathematics, 44, 289–292.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.logexp import log_base, exp_e, sqrt
>>> from txgraffiti2025.forms.utils import to_expr
>>> df = pd.DataFrame({"x": [1.0, 2.0, 4.0]})

>>> # Natural log
>>> log_base("x").eval(df).round(3).tolist()
[0.0, 0.693, 1.386]

>>> # Base-10 log
>>> log_base("x", base=10).eval(df).round(3).tolist()
[0.0, 0.301, 0.602]

>>> # Exponential
>>> exp_e("x").eval(df).round(3).tolist()
[2.718, 7.389, 54.598]

>>> # Square root
>>> sqrt("x").eval(df).tolist()
[1.0, 1.4142135624, 2.0]

These transformations can appear inside conjectures such as
``log(alpha(G)) <= c * log(order(G))`` or ``exp(residue(G)) <= size(G)``.
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
    """
    Logarithm of an expression, optionally with a specified base.

    Parameters
    ----------
    x : Expr or str or int or float
        Expression or column whose logarithm is taken.
    base : float, optional
        Logarithmic base.  If ``None`` (default), computes the natural
        logarithm (base e).

    Returns
    -------
    Expr
        Expression representing ``log(x)`` or ``log_base(x)``.

    Notes
    -----
    Implements the R5 logarithmic transformation form.
    Used in conjectures involving logarithmic growth or entropy-type
    measures, e.g. ``log(alpha(G)) <= c·log(order(G))``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.logexp import log_base
    >>> df = pd.DataFrame({"x": [1.0, 10.0, 100.0]})
    >>> log_base("x", base=10).eval(df).tolist()
    [0.0, 1.0, 2.0]
    """
    return log(to_expr(x), base=base)


def exp_e(x: NumLike) -> Expr:
    """
    Exponential of an expression with base e.

    Parameters
    ----------
    x : Expr or str or int or float
        Exponent expression.

    Returns
    -------
    Expr
        Expression representing ``exp(x) = e ** x``.

    Notes
    -----
    Implements the R5 exponential transformation form.
    Appears in conjectures modeling exponential bounds, e.g.
    ``exp(residue(G)) <= size(G)``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.logexp import exp_e
    >>> df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    >>> exp_e("x").eval(df).round(3).tolist()
    [1.0, 2.718, 7.389]
    """
    return exp(to_expr(x))


def sqrt(x: NumLike) -> Expr:
    """
    Square-root transformation of an expression.

    Parameters
    ----------
    x : Expr or str or int or float
        Base expression.

    Returns
    -------
    Expr
        Expression representing ``sqrt(x)``.

    Notes
    -----
    Equivalent to :func:`~txgraffiti2025.forms.nonlinear.power(x, 0.5)`.
    Common in R5 conjectures involving geometric means or
    root-bounded relationships, e.g. ``α(G) <= √n``.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.logexp import sqrt
    >>> df = pd.DataFrame({"x": [1, 4, 9]})
    >>> sqrt("x").eval(df).tolist()
    [1.0, 2.0, 3.0]
    """
    return _sqrt(to_expr(x))
