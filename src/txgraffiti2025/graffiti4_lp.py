# src/txgraffiti2025/graffiti4_lp.py
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

from typing import Any, Dict, List, Sequence, Tuple
from itertools import combinations
from fractions import Fraction

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import Expr, to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge

from fractions import Fraction
from typing import Sequence, Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import Expr, to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge

# ───────────────────── helper: rationalizing coeffs ───────────────────── #

# def _rationalize_scalar(val: float, max_denom: int) -> float:
#     """Project a float to a nearby rational with bounded denominator."""
#     frac = Fraction(float(val)).limit_denominator(max_denom)
#     return float(frac)


# def _rationalize_coeffs(coeffs: np.ndarray, max_denom: int) -> np.ndarray:
#     """Apply _rationalize_scalar elementwise."""
#     coeffs = np.asarray(coeffs, dtype=float)
#     out = [_rationalize_scalar(c, max_denom) for c in coeffs]
#     return np.array(out, dtype=float)


# ───────────────────── helper: build a “nice” affine Expr ───────────────────── #

def _build_affine_expr(
    const_val: float,
    coefs: Sequence[float],
    feats: Sequence[Expr],
    *,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 4.0,
) -> Optional[Expr]:
    """
    Build a “nice” affine expression

        const_val + sum_j coefs[j] * feats[j]

    with the following rules:

      * Drop any term with |coefs[j]| < zero_tol  (avoids 0·x).
      * Reject the whole expression if |coefs[j]| > max_coef_abs
        or |const_val| > max_intercept_abs.
      * For coefs very close to ±1, use ±feat instead of 1·feat.

    Returns None if we decide the affine form is too ugly and should be skipped.
    """
    const_val = float(const_val)

    # Intercept too large → skip this affine form entirely
    if abs(const_val) > max_intercept_abs:
        return None

    pieces: List[Expr] = []

    for coef, feat in zip(coefs, feats):
        c = float(coef)
        if abs(c) < zero_tol:
            # Drop 0·feat terms
            continue
        if abs(c) > max_coef_abs:
            # Any coefficient too large → skip the whole affine combo
            return None

        # Clean up ±1 specially
        if abs(c - 1.0) < zero_tol:
            term = feat
        elif abs(c + 1.0) < zero_tol:
            term = (-1.0) * feat
        else:
            term = c * feat
        pieces.append(term)

    expr: Optional[Expr] = None

    # Only keep the constant if it’s non-negligible
    if abs(const_val) >= zero_tol:
        expr = Const(const_val)

    for term in pieces:
        expr = term if expr is None else expr + term

    if expr is None:
        # Everything vanished → treat as purely zero; usually not very
        # interesting, but it's better than returning None by surprise.
        expr = Const(0.0)

    return expr


# ───────────────────────── LP solver core ───────────────────────── #

def solve_lp_min_slack(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sense: str,
    coef_bound: float = 10.0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Solve the min-sum-slack LP for an affine bound:

      sense = "upper":
          find w, b, s >= 0 s.t.  w·x_i + b - s_i = y_i
          (⇒ w·x_i + b >= y_i)

      sense = "lower":
          find w, b, s >= 0 s.t.  w·x_i + b + s_i = y_i
          (⇒ w·x_i + b <= y_i)

      Objective: minimize sum_i s_i.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    sense : {"upper", "lower"}
        Which side to enforce.
    coef_bound : float, optional
        Box bound for coefficients: |w_j| <= coef_bound, |b| <= coef_bound.

    Returns
    -------
    w : ndarray of shape (n_features,)
    b : float
    s : ndarray of shape (n_samples,)

    Raises
    ------
    ImportError
        If SciPy is not installed.
    ValueError
        If the LP is infeasible or solver fails.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as e:
        raise ImportError(
            "solve_lp_min_slack requires scipy. "
            "Please install scipy or provide your own LP solver."
        ) from e

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    n_samples, n_features = X.shape

    if n_samples == 0:
        raise ValueError("No samples passed to solve_lp_min_slack.")

    # Variables: [w_1,...,w_m, b, s_1,...,s_n]
    n_w = n_features
    n_b = 1
    n_s = n_samples
    n_vars = n_w + n_b + n_s

    # Objective: minimize sum s_i
    c = np.zeros(n_vars, dtype=float)
    c[n_w + n_b :] = 1.0  # coefficients for s_i

    # Equality constraints: one row per sample
    A_eq = np.zeros((n_samples, n_vars), dtype=float)
    b_eq = np.zeros(n_samples, dtype=float)

    if sense == "upper":
        # w·x_i + b - s_i = y_i
        for i in range(n_samples):
            A_eq[i, :n_w] = X[i, :]
            A_eq[i, n_w] = 1.0                # b
            A_eq[i, n_w + n_b + i] = -1.0     # -s_i
            b_eq[i] = y[i]
    elif sense == "lower":
        # w·x_i + b + s_i = y_i
        for i in range(n_samples):
            A_eq[i, :n_w] = X[i, :]
            A_eq[i, n_w] = 1.0                # b
            A_eq[i, n_w + n_b + i] = 1.0      # +s_i
            b_eq[i] = y[i]
    else:
        raise ValueError("sense must be 'upper' or 'lower'.")

    # Bounds:
    #   w_j ∈ [-coef_bound, coef_bound]
    #   b   ∈ [-coef_bound, coef_bound]
    #   s_i ∈ [0, ∞)
    bounds: List[Tuple[float, float | None]] = []

    for _ in range(n_w):
        bounds.append((-coef_bound, coef_bound))
    bounds.append((-coef_bound, coef_bound))  # b
    for _ in range(n_s):
        bounds.append((0.0, None))            # s_i

    res = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise ValueError(f"LP solve failed: {res.message}")

    z = res.x
    w = z[:n_w]
    b = z[n_w]
    s = z[n_w + n_b :]

    return w, float(b), s


# ───────────────────── helper: rationalizing coeffs ───────────────────── #

def _rationalize_scalar(val: float, max_denom: int) -> float:
    """Project a float to a nearby rational with bounded denominator."""
    frac = Fraction(float(val)).limit_denominator(max_denom)
    return float(frac)


def _rationalize_coeffs(coeffs: np.ndarray, max_denom: int) -> np.ndarray:
    """Apply _rationalize_scalar elementwise."""
    coeffs = np.asarray(coeffs, dtype=float)
    out = [ _rationalize_scalar(c, max_denom) for c in coeffs ]
    return np.array(out, dtype=float)


# ───────────────────── main LP runner (stage-style) ───────────────────── #

def lp_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence[Any],   # expects .mask, .pred, .name
    df: pd.DataFrame,
    max_features: int = 2,
    max_denom: int = 20,
    coef_bound: float = 10.0,
    direction: str = "both",     # "upper", "lower", or "both"
    solve_lp_func=None,
    # “human niceness” controls
    zero_tol: float = 1e-8,
    max_coef_abs: float = 2.5,
    max_intercept_abs: float = 2.5,
) -> List[Conjecture]:
    """
    Linear-programming-based affine bounds:

        h ⇒ target ≤ w·x + b   (upper, Le)
        h ⇒ target ≥ w·x + b   (lower, Ge)

    for each hypothesis h and each small subset of feature Exprs from `others`.

    LP-level box constraint is |w_j|, |b| ≤ coef_bound.  After solving, we
    rationalize the coefficients and THEN apply a stricter “human niceness”
    filter via |w_j| ≤ max_coef_abs and |b| ≤ max_intercept_abs, dropping any
    affine form that fails this test or collapses to 0·x.
    """
    if solve_lp_func is None:
        solve_lp_func = solve_lp_min_slack

    target_vals = df[target_col].to_numpy(dtype=float)
    feature_items = list(others.items())
    m = len(feature_items)

    # All nonempty subsets up to max_features
    all_subsets: List[List[Tuple[str, Expr]]] = []
    for k in range(1, min(max_features, m) + 1):
        for combo in combinations(feature_items, k):
            all_subsets.append(list(combo))

    conjs: List[Conjecture] = []

    for hyp in hypotheses:
        H = np.asarray(hyp.mask, dtype=bool)
        idx = np.where(H)[0]
        if idx.size == 0:
            continue

        y = target_vals[idx]
        if not np.isfinite(y).any():
            continue

        for subset in all_subsets:
            feat_names = [name for name, _ in subset]
            feat_exprs = [expr for _, expr in subset]

            # Build X for this hypothesis + subset
            cols: List[np.ndarray] = []
            valid_subset = True
            for expr in feat_exprs:
                try:
                    col_full = expr.eval(df).to_numpy(dtype=float)
                except Exception:
                    valid_subset = False
                    break

                col = col_full[idx]
                if not np.isfinite(col).any():
                    valid_subset = False
                    break
                cols.append(col)

            if not valid_subset or not cols:
                continue

            X = np.vstack(cols).T  # shape (n_h, k)
            if X.shape[0] == 0:
                continue

            # ── Upper bound: y ≤ w·x + b ─────────────────────
            if direction in ("both", "upper"):
                try:
                    w_u, b_u, _ = solve_lp_func(
                        X,
                        y,
                        sense="upper",
                        coef_bound=coef_bound,
                    )
                except Exception:
                    w_u = None

                if w_u is not None:
                    # Rationalize first, then build a “nice” affine Expr
                    w_u = _rationalize_coeffs(w_u, max_denom=max_denom)
                    b_u = _rationalize_scalar(b_u, max_denom=max_denom)

                    rhs = _build_affine_expr(
                        const_val=b_u,
                        coefs=w_u,
                        feats=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                    )
                    if rhs is not None:
                        rel = Le(left=target_expr, right=rhs)
                        conj = Conjecture(
                            relation=rel,
                            condition=hyp.pred,
                            name=f"[LP-upper] {target_col} vs {', '.join(feat_names)} under {hyp.name}",
                        )
                        conj.target_name = target_col
                        conjs.append(conj)

            # ── Lower bound: y ≥ w·x + b ─────────────────────
            if direction in ("both", "lower"):
                try:
                    w_l, b_l, _ = solve_lp_func(
                        X,
                        y,
                        sense="lower",
                        coef_bound=coef_bound,
                    )
                except Exception:
                    w_l = None

                if w_l is not None:
                    w_l = _rationalize_coeffs(w_l, max_denom=max_denom)
                    b_l = _rationalize_scalar(b_l, max_denom=max_denom)

                    rhs = _build_affine_expr(
                        const_val=b_l,
                        coefs=w_l,
                        feats=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                    )
                    if rhs is not None:
                        rel = Ge(left=target_expr, right=rhs)
                        conj = Conjecture(
                            relation=rel,
                            condition=hyp.pred,
                            name=f"[LP-lower] {target_col} vs {', '.join(feat_names)} under {hyp.name}",
                        )
                        conj.target_name = target_col

                        conjs.append(conj)

    return conjs
