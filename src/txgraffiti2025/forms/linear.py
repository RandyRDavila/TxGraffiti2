# src/txgraffiti2025/forms/linear.py
"""
Linear conjecture builders (R2): build relations from affine forms

    a0 + Σ (a_i * col_i)  (<=, >=, ==)  right

This module provides small helpers to construct linear expressions and wrap them
in R2-style relations (Le/Ge/Eq). Terms may be given as a sequence of
``(coef, column)`` pairs or as a mapping ``{column: coef}``; duplicate columns
are merged and zero coefficients are dropped.

Examples
--------
>>> from txgraffiti2025.forms.linear import linear_expr, linear_le, linear_ge, linear_eq
>>> from txgraffiti2025.forms.utils import to_expr
>>> import pandas as pd
>>>
>>> df = pd.DataFrame({"x":[1,2,3], "y":[4,5,6], "z":[0,1,1]})
>>>
>>> # Expression:  1.0 + 2*x - y
>>> e = linear_expr(1.0, [(2.0, "x"), (-1.0, "y")])
>>> e.eval(df).tolist()
[-1.0, 0.0, 1.0]
>>>
>>> # Relation:  2*x + y <= 10
>>> R = linear_le(0.0, {"x": 2.0, "y": 1.0}, 10)
>>> R.evaluate(df).tolist()
[True, True, True]
>>>
>>> # Equality with tolerance
>>> EqR = linear_eq(0.0, {"x": 1.0, "y": -1.0}, 0, tol=1e-9)   # x - y ≈ 0
>>> EqR.evaluate(df).tolist()
[False, False, False]
"""

from __future__ import annotations

from typing import Iterable, Tuple, Mapping, Union, List, Dict

from .utils import LinearForm
from .generic_conjecture import Le, Ge, Eq

__all__ = [
    "linear_expr", "linear_le", "linear_ge", "linear_eq",
    # Friendly aliases
    "lin", "leq", "geq",
    # Convenience
    "from_dict",
]

TermSeq = Iterable[Tuple[float, str]]
TermMap = Mapping[str, float]
Right = Union[float, int, str]  # Relations will coerce via to_expr downstream


# ---------------------------------------------------------------------
# Internal: normalize and stabilize term lists
# ---------------------------------------------------------------------
def _normalize_terms(terms: Union[TermSeq, TermMap]) -> List[Tuple[float, str]]:
    """
    Normalize linear terms to a stable list of (coef, column).

    - Accepts sequence of pairs or mapping {col: coef}.
    - Merges duplicates by summing coefficients.
    - Drops zero coefficients.
    - Returns list sorted by column name for deterministic repr/debugging.

    Examples
    --------
    >>> _normalize_terms([ (1, "x"), (2, "x"), (-3, "y") ])
    [(3.0, 'x'), (-3.0, 'y')]
    >>> _normalize_terms({"y": -3, "x": 3})
    [(3.0, 'x'), (-3.0, 'y')]
    """
    acc: Dict[str, float] = {}
    if isinstance(terms, Mapping):
        for col, coef in terms.items():
            acc[col] = acc.get(col, 0.0) + float(coef)
    else:
        for coef, col in terms:
            acc[col] = acc.get(col, 0.0) + float(coef)
    # drop zeros and sort by column for stable output
    items = [(coef, col) for col, coef in acc.items() if coef != 0.0]
    items.sort(key=lambda t: t[1])
    return items


# ---------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------
def linear_expr(a0: float, terms: Union[TermSeq, TermMap]) -> LinearForm:
    """
    Build a linear expression ``a0 + Σ a_i * col_i``.

    Parameters
    ----------
    a0 : float
        Intercept term.
    terms : Iterable[tuple[float, str]] or Mapping[str, float]
        Linear terms (sequence or mapping); duplicates are merged and zeros dropped.

    Returns
    -------
    LinearForm
        Expression node evaluable on a DataFrame with ``.eval(df)``.

    Examples
    --------
    >>> e = linear_expr(1.0, [(2.0, "x"), (-1.0, "y")])   # 1 + 2x - y
    >>> isinstance(e, LinearForm)
    True
    """
    return LinearForm(float(a0), _normalize_terms(terms))


def linear_le(a0: float, terms: Union[TermSeq, TermMap], right: Right) -> Le:
    """
    Build relation: ``a0 + Σ a_i * col_i <= right``.

    Slack: ``right - (a0 + Σ a_i * col_i)``  (≥ 0 iff satisfied)
    """
    return Le(linear_expr(a0, terms), right)


def linear_ge(a0: float, terms: Union[TermSeq, TermMap], right: Right) -> Ge:
    """
    Build relation: ``a0 + Σ a_i * col_i >= right``.

    Slack: ``(a0 + Σ a_i * col_i) - right``  (≥ 0 iff satisfied)
    """
    return Ge(linear_expr(a0, terms), right)


def linear_eq(a0: float, terms: Union[TermSeq, TermMap], right: Right, tol: float = 1e-9) -> Eq:
    """
    Build approximate equality: ``a0 + Σ a_i * col_i == right`` within ``tol``.

    Slack: ``tol - |(a0 + Σ a_i * col_i) - right|``  (≥ 0 iff within tolerance)
    """
    return Eq(linear_expr(a0, terms), right, tol=tol)


# ---------------------------------------------------------------------
# Aliases & convenience
# ---------------------------------------------------------------------
def lin(a0: float, terms: Union[TermSeq, TermMap]) -> LinearForm:
    """Alias for :func:`linear_expr`."""
    return linear_expr(a0, terms)


def leq(a0: float, terms: Union[TermSeq, TermMap], right: Right) -> Le:
    """Alias for :func:`linear_le`."""
    return linear_le(a0, terms, right)


def geq(a0: float, terms: Union[TermSeq, TermMap], right: Right) -> Ge:
    """Alias for :func:`linear_ge`."""
    return linear_ge(a0, terms, right)


def from_dict(a0: float, **coef_by_col: float) -> LinearForm:
    """
    Convenience: build ``a0 + Σ coef_by_col[col] * col`` from keyword args.

    Examples
    --------
    >>> e = from_dict(1.0, x=2.0, y=-1.0)   # 1 + 2x - y
    >>> isinstance(e, LinearForm)
    True
    """
    return linear_expr(a0, coef_by_col)
