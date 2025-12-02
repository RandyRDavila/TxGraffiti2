from __future__ import annotations

from fractions import Fraction
from typing import List, Tuple
import numpy as np

from txgraffiti2025.forms.utils import Expr, Const


def rconst(x: float, *, max_denom: int, tol: float) -> Const:
    """
    Rationalize x to a Const using limit_denominator(max_denom).
    Returns 0 if |x| <= tol or not finite.
    """
    if not np.isfinite(x) or abs(x) <= tol:
        return Const(0.0)
    return Const(Fraction(x).limit_denominator(max_denom))


def build_affine(a: np.ndarray, feats: List[Expr], b: float, *, max_denom: int, tol: float) -> Expr:
    """
    Build ∑ a_j * feats[j] + b with coefficient/constant rationalization.
    Zero-suppresses tiny coefficients by 'tol'. Returns Const(0) if empty.
    """
    acc: Expr | None = None
    for coef, f in zip(a, feats):
        if abs(float(coef)) <= tol:
            continue
        term = rconst(float(coef), max_denom=max_denom, tol=tol) * f
        acc = term if acc is None else (acc + term)
    if abs(float(b)) > tol:
        c = rconst(float(b), max_denom=max_denom, tol=tol)
        acc = c if acc is None else (acc + c)
    return acc if acc is not None else Const(0.0)


def finite_mask(arrs: List[np.ndarray]) -> np.ndarray:
    """
    Return boolean mask of rows that are finite across all arrays in `arrs`.
    Assumes all arrays are 1D and the same length.
    """
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def touch_stats(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> tuple[int, float, int]:
    """
    Compute (touch_count, touch_rate, n), where a 'touch' means
        |lhs - rhs| <= atol + rtol * |rhs|.
    Only counts rows where mask is True and both sides are finite.
    """
    m = mask & np.isfinite(lhs) & np.isfinite(rhs)
    n = int(m.sum())
    if n == 0:
        return 0, 0.0, 0
    d = np.abs(lhs[m] - rhs[m])
    t = atol + rtol * np.abs(rhs[m])
    tc = int((d <= t).sum())
    return tc, (tc / n), n
