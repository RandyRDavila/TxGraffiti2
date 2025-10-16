# src/txgraffiti2025/forms/linear.py

"""
Linear conjecture builders (R2):
    a0 + Σ ai * col_i   (<=, >=, ==)   right
"""

from __future__ import annotations
from typing import Iterable, Tuple, Mapping, Union, List
from .utils import LinearForm
from .generic_conjecture import Le, Ge, Eq

__all__ = [
    "linear_expr",
    "linear_le",
    "linear_ge",
    "linear_eq",
]

TermSeq = Iterable[Tuple[float, str]]
TermMap = Mapping[str, float]
Right = Union[float, int, str]  # Eq/Le/Ge will auto-coerce via to_expr

def _normalize_terms(terms: Union[TermSeq, TermMap]) -> List[Tuple[float, str]]:
    """
    Accepts either:
      - sequence of (coef, column) pairs, or
      - mapping {column: coef}
    Returns a normalized list with duplicate columns merged and zeroed terms dropped.
    """
    acc: dict[str, float] = {}
    if isinstance(terms, Mapping):
        for col, coef in terms.items():
            acc[col] = acc.get(col, 0.0) + float(coef)
    else:
        for coef, col in terms:
            acc[col] = acc.get(col, 0.0) + float(coef)
    # drop zeros and produce stable order
    return [(coef, col) for col, coef in acc.items() if coef != 0.0]

def linear_expr(a0: float, terms: Union[TermSeq, TermMap]) -> LinearForm:
    """Build just the LinearForm expression a0 + Σ ai * col_i."""
    return LinearForm(float(a0), _normalize_terms(terms))

def linear_le(a0: float, terms: Union[TermSeq, TermMap], right: Right) -> Le:
    """a0 + Σ ai * col_i  <=  right"""
    return Le(linear_expr(a0, terms), right)

def linear_ge(a0: float, terms: Union[TermSeq, TermMap], right: Right) -> Ge:
    """a0 + Σ ai * col_i  >=  right"""
    return Ge(linear_expr(a0, terms), right)

def linear_eq(a0: float, terms: Union[TermSeq, TermMap], right: Right, tol: float = 1e-9) -> Eq:
    """a0 + Σ ai * col_i  ==  right  (≈ within tol)"""
    return Eq(linear_expr(a0, terms), right, tol=tol)
