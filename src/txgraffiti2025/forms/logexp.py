# src/txgraffiti2025/forms/logexp.py
"""
Logarithmic / Exponential / Square-root forms (R5).

R5 conjecture forms introduce transcendental or root-type transformations
of invariants, extending beyond algebraic (R2–R4) expressions.

Provided helpers return `Expr` nodes compatible with the rest of the system:
- log_base(x, base=None, epsilon=0.0)   # ln by default; clamp via epsilon
- ln(x, epsilon=0.0), log2(x), log10(x) # ergonomic log variants
- exp_e(x)                               # e^x
- sqrt(x)                                 # √x

`epsilon` is useful to avoid -inf on zeros: log(max(x, epsilon)).
(Requires the `log(..., base=..., epsilon=...)` support in utils.)

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.logexp import ln, log2, log10, exp_e, sqrt
>>> df = pd.DataFrame({"x": [1.0, 2.0, 4.0]})
>>> ln("x").eval(df).round(3).tolist()
[0.0, 0.693, 1.386]
>>> log2("x").eval(df).round(3).tolist()
[0.0, 1.0, 2.0]
>>> log10("x").eval(df).round(3).tolist()
[0.0, 0.301, 0.602]
>>> exp_e("x").eval(df).round(3).tolist()
[2.718, 7.389, 54.598]
>>> sqrt("x").eval(df).round(3).tolist()
[1.0, 1.414, 2.0]
"""

from __future__ import annotations
from typing import Optional, Union

from .utils import to_expr, log, exp, sqrt as _sqrt, Expr

__all__ = [
    "log_base", "ln", "log2", "log10",
    "exp_e",
    "sqrt",
]

NumLike = Union[Expr, str, int, float]


def log_base(x: NumLike, *, base: Optional[float] = None, epsilon: float = 0.0) -> Expr:
    """
    Logarithm of an expression with optional base and epsilon clamp.

    Parameters
    ----------
    x : Expr | str | int | float
        Expression or column.
    base : float | None, default None
        Logarithmic base. None ⇒ natural log.
    epsilon : float, default 0.0
        If > 0, evaluates log(max(x, epsilon)) elementwise (helps avoid -inf on zeros).

    Returns
    -------
    Expr
        Expression node for log_base(x).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [0.0, 1.0, 10.0]})
    >>> log_base("x", base=10, epsilon=1e-9).eval(df).round(6).tolist()
    [-9.0, 0.0, 1.0]
    """
    return log(to_expr(x), base=base, epsilon=epsilon)


def ln(x: NumLike, *, epsilon: float = 0.0) -> Expr:
    """Natural logarithm ln(x); supports epsilon clamp."""
    return log_base(x, base=None, epsilon=epsilon)


def log2(x: NumLike, *, epsilon: float = 0.0) -> Expr:
    """Base-2 logarithm log₂(x); supports epsilon clamp."""
    return log_base(x, base=2.0, epsilon=epsilon)


def log10(x: NumLike, *, epsilon: float = 0.0) -> Expr:
    """Base-10 logarithm log₁₀(x); supports epsilon clamp."""
    return log_base(x, base=10.0, epsilon=epsilon)


def exp_e(x: NumLike) -> Expr:
    """
    Exponential e^x.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    >>> exp_e("x").eval(df).round(3).tolist()
    [1.0, 2.718, 7.389]
    """
    return exp(to_expr(x))


def sqrt(x: NumLike) -> Expr:
    """
    Square-root transformation √x.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1, 4, 9]})
    >>> sqrt("x").eval(df).tolist()
    [1.0, 2.0, 3.0]
    """
    return _sqrt(to_expr(x))
