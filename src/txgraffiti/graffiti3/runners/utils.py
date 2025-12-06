# src/txgraffiti/graffiti3/runners/utils.py

from __future__ import annotations

"""
Linear-programming-based affine bounds for Graffiti4.

This module provides:

  - solve_lp_min_slack(X, y, sense, coef_bound=...)
      Solve for an affine function w·x + b that bounds y from above or
      below, minimizing the total slack.

  - lp_runner(...)
      A stage-style runner that, for each hypothesis h and each small
      subset of feature Exprs, generates conjectures of the form

          h ⇒ target ≤ w·x + b
          h ⇒ target ≥ w·x + b

      with coefficients rationalized to small denominators.

You can plug lp_runner into Graffiti4.conjecture as a “Stage 2”
after constant and ratio stages.
"""

from typing import Callable, Dict, Sequence, Tuple, List

import numpy as np
from fractions import Fraction

from txgraffiti.graffiti3.exprs import Expr, Const


# ───────────────────── helper: rationalizing coeffs ───────────────────── #
def _to_const_fraction(x: float, max_denom: int) -> Const:
    """Return a Const representing a bounded-denominator rational close to x."""
    return Const(Fraction(x).limit_denominator(max_denom))


def _rationalize_coeffs(coeffs: np.ndarray, max_denom: int) -> np.ndarray:
    """Apply _rationalize_scalar elementwise."""
    coeffs = np.asarray(coeffs, dtype=float)
    out = [ _rationalize_scalar(c, max_denom) for c in coeffs ]
    return np.array(out, dtype=float)


# ───────────────────── helper: build a “nice” affine Expr ───────────────────── #

def _rationalize_scalar(x: float, max_denom: int) -> float:
    """Return a float that is a bounded-denominator rational approximation to x."""
    return float(Fraction(x).limit_denominator(max_denom))

def _build_affine_expr(
    *,
    const_val: float,
    coefs: Sequence[float],
    feats: Sequence[Expr],
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 8.0,
    max_denom: int = 20,
) -> Expr | None:
    """
    Build a *canonical* affine expression

        const_val + sum_j coefs[j] * feats[j]

    with normalization rules:

      - coefficients with |c| < zero_tol are dropped;
      - if any |c| > max_coef_abs, we drop the entire form (return None);
      - if intercept |b| > max_intercept_abs, drop entire form (return None);
      - if the same feat appears multiple times, their coefficients are summed;
      - all active coefficients are rationalized via limit_denominator(max_denom);
      - coefficients equal to ±1 *after rationalization* are rendered as
        ±feat (no explicit “1·”);
      - active (feat, coef) pairs are sorted by repr(feat) for a stable order;
      - the intercept is also rationalized.

    This ensures, e.g., that all of

        target ≤ annihilation_number
        target ≤ 1 · annihilation_number
        target ≤ (1 · annihilation_number) + (0 · residue)

    normalize to the same Expr, and likewise for multi-feature sums.
    """
    assert len(coefs) == len(feats), "coefs and feats must have same length"

    # ---------- aggregate coefficients by feature repr ----------
    # key: repr(feat)  ->  (feat, summed_coef)
    acc: Dict[str, Tuple[Expr, float]] = {}

    for c_raw, feat in zip(coefs, feats):
        c = float(c_raw)

        # Kill tiny coefficients immediately
        if abs(c) < zero_tol:
            continue

        # If one coefficient is insane, bail on the whole expression
        if abs(c) > max_coef_abs:
            return None

        key = repr(feat)
        if key in acc:
            old_feat, old_c = acc[key]
            acc[key] = (old_feat, old_c + c)
        else:
            acc[key] = (feat, c)

    # Remove any features that canceled out to ~0 after aggregation
    cleaned_terms: List[Tuple[str, Expr, float]] = []
    for key, (feat, c_sum) in acc.items():
        if abs(c_sum) < zero_tol:
            continue
        cleaned_terms.append((key, feat, float(c_sum)))

    # Guard intercept magnitude
    b = float(const_val)
    if abs(b) > max_intercept_abs:
        return None

    # Nothing left and no intercept: return 0
    if not cleaned_terms and abs(b) < zero_tol:
        return Const(0)

    # ---------- build canonical term list ----------
    # Sort by key (repr(feat)) for deterministic ordering
    cleaned_terms.sort(key=lambda tup: tup[0])

    terms: List[Expr] = []

    for _key, feat, c in cleaned_terms:
        # Rationalize coefficient FIRST
        c_rat = Fraction(c).limit_denominator(max_denom)
        if c_rat.denominator == max_denom:
            return None

        # Decide ±1 based on the rational value (this fixes 1·x vs x issues)
        if c_rat == 1:
            term = feat
        elif c_rat == -1:
            term = Const(-1) * feat
        else:
            term = Const(c_rat) * feat

        terms.append(term)

    # Intercept (if nontrivial)
    if abs(b) >= zero_tol:
        b_rat = Fraction(b).limit_denominator(max_denom)
        terms.append(Const(b_rat))

    # Still nothing? Then it's just the intercept 0, but we handled |b|<zero_tol above.
    if not terms:
        return Const(0)

    # Fold terms with + to a single Expr
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t

    return expr


# ───────────────────── helpers: pick the best bounds ───────────────────── #

def _pick_best_ge(
    t_arr: np.ndarray,
    rhs_variants: Sequence[tuple[str, np.ndarray, Callable]],
) -> Callable | None:
    """
    For lower bounds t >= rhs, pick the variant whose rhs is:
      - valid on all finite rows (t >= rhs), and
      - has the *largest mean rhs* (least conservative but still valid).
    Returns the Expr-constructor (callable) or None.
    """
    best = None
    best_score = -np.inf

    for _tag, rhs, make_expr in rhs_variants:
        ok = np.isfinite(t_arr) & np.isfinite(rhs)
        if not np.any(ok):
            continue
        if np.all(t_arr[ok] >= rhs[ok]):
            score = float(np.mean(rhs[ok]))
            if score > best_score:
                best = make_expr
                best_score = score
    return best

def _pick_best_le(
    t_arr: np.ndarray,
    rhs_variants: Sequence[tuple[str, np.ndarray, Callable]],
) -> Callable | None:
    """
    For upper bounds t <= rhs, pick the variant whose rhs is:
      - valid on all finite rows (t <= rhs), and
      - has the *smallest mean rhs* (tightest upper bound).
    Returns the Expr-constructor (callable) or None.
    """
    best = None
    best_score = np.inf

    for _tag, rhs, make_expr in rhs_variants:
        ok = np.isfinite(t_arr) & np.isfinite(rhs)
        if not np.any(ok):
            continue
        if np.all(t_arr[ok] <= rhs[ok]):
            score = float(np.mean(rhs[ok]))
            if score < best_score:
                best = make_expr
                best_score = score
    return best


def _safe_sqrt_array(y_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 'safe' sqrt array and its validity mask.

    Parameters
    ----------
    y_arr : np.ndarray
        Raw y values on a hypothesis mask.

    Returns
    -------
    sqrt_y : np.ndarray
        Same shape as y_arr; sqrt applied only where y is finite and >= 0,
        elsewhere 0.
    valid : np.ndarray[bool]
        Mask where sqrt_y is genuinely valid (finite and > 0), suitable
        for forming ratios.
    """
    sqrt_y = np.zeros_like(y_arr, dtype=float)
    domain = np.isfinite(y_arr) & (y_arr >= 0.0)
    sqrt_y[domain] = np.sqrt(y_arr[domain], dtype=float)

    valid = domain & np.isfinite(sqrt_y) & (sqrt_y != 0.0)
    return sqrt_y, valid
