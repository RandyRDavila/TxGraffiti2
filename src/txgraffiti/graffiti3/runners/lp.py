# src/txgraffiti/graffiti3/runners/lp.py

from __future__ import annotations

from typing import Any, Dict, Sequence, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from itertools import combinations

from txgraffiti.graffiti3.exprs import Expr
from txgraffiti.graffiti3.relations import Conjecture, Le, Ge

# Only needed for type checking, not at runtime (avoids circular import)
if TYPE_CHECKING:
    from txgraffiti.graffiti3.types import HypothesisInfo

# If you already have this helper in another LP module, adjust the import
from txgraffiti.graffiti3.runners.utils import _build_affine_expr, _rationalize_coeffs, _rationalize_scalar


def lp_single_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence["HypothesisInfo"],
    df: pd.DataFrame,
    direction: str = "both",           # "upper", "lower", or "both"
    min_support: int = 8,
    max_denom: int = 20,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 8.0,
) -> List[Conjecture]:
    """
    LP stage: for each hypothesis h and each single 'other' invariant x, solve

        upper:  target <= a x + b
        lower:  target >= a x + b

    with (a, b) chosen by a small LP, and then normalized via _build_affine_expr.

    This gives conjectures of the form

        h ⇒ target ≤ a · x + b
        h ⇒ target ≥ a · x + b,

    which the pure ratio stage (no intercept) cannot see.
    """
    conjs: List[Conjecture] = []

    y_all = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        H = np.asarray(hyp.mask, dtype=bool)

        for other_name, other_expr in others.items():
            # Evaluate x on the whole df
            try:
                x_all = other_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            valid = (
                H
                & np.isfinite(y_all)
                & np.isfinite(x_all)
            )
            if valid.sum() < min_support:
                continue

            x = x_all[valid]
            y = y_all[valid]

            # Center x a bit to reduce correlation with intercept
            x_mean = float(x.mean())
            x_centered = x - x_mean

            n = len(x)
            sum_x = float(x_centered.sum())

            # ------------------ upper bound: y <= a x + b ------------------
            if direction in ("both", "upper"):
                # minimize sum_i (a x_i + b) = a * sum_x + b * n
                c = np.array([sum_x, n], dtype=float)

                # constraints: a x_i + b >= y_i  ->  -a x_i - b <= -y_i
                A_ub = np.column_stack([-x_centered, -np.ones_like(x_centered)])
                b_ub = -y

                res = linprog(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    method="highs",
                )

                if res.success:
                    a_hat, b_hat = res.x
                    # un-center intercept: x = x_centered + x_mean
                    # y <= a(x - x_mean) + b_hat  ⇒  y <= a x + (b_hat - a x_mean)
                    b_hat = b_hat - a_hat * x_mean

                    a_rat = _rationalize_scalar(a_hat, max_denom)
                    b_rat = _rationalize_scalar(b_hat, max_denom)

                    rhs = _build_affine_expr(
                        const_val=b_rat,
                        coefs=[a_rat],
                        feats=[other_expr],
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs is not None:
                        rel_le = Le(left=target_expr, right=rhs)
                        c_le = Conjecture(
                            relation=rel_le,
                            condition=hyp.pred,
                            name=f"[lp1-upper] {target_col} vs {other_name} under {hyp.name}",
                        )
                        c_le.target_name = target_col
                        conjs.append(c_le)

            # ------------------ lower bound: y >= a x + b ------------------
            if direction in ("both", "lower"):
                # maximize sum_i (a x_i + b) subject to a x_i + b <= y_i
                # ⇔ minimize -(a sum_x + b n)
                c = np.array([-sum_x, -n], dtype=float)

                # constraints: a x_i + b <= y_i -> a x_i + b - y_i <= 0
                A_ub = np.column_stack([x_centered, np.ones_like(x_centered)])
                b_ub = y

                res = linprog(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    method="highs",
                )

                if res.success:
                    a_hat, b_hat = res.x
                    b_hat = b_hat - a_hat * x_mean

                    a_rat = _rationalize_scalar(a_hat, max_denom)
                    b_rat = _rationalize_scalar(b_hat, max_denom)

                    rhs = _build_affine_expr(
                        const_val=b_rat,
                        coefs=[a_rat],
                        feats=[other_expr],
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                    )
                    if rhs is not None:
                        rel_ge = Ge(left=target_expr, right=rhs)
                        c_ge = Conjecture(
                            relation=rel_ge,
                            condition=hyp.pred,
                            name=f"[lp1-lower] {target_col} vs {other_name} under {hyp.name}",
                        )
                        c_ge.target_name = target_col
                        conjs.append(c_ge)

    return conjs

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
                        max_denom=max_denom,
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
                        max_denom=max_denom,
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
