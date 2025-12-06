from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.exprs import (
    Expr,
    sqrt as sqrt_expr,
    log as log_expr,
)
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le
from txgraffiti.graffiti3.runners.utils import (
    _build_affine_expr,
    _rationalize_coeffs,
    _rationalize_scalar,
)
from txgraffiti.graffiti3.runners.poly import _solve_lp_lower, _solve_lp_upper

if TYPE_CHECKING:
    from txgraffiti.graffiti3.types import HypothesisInfo


# ───────────────────────── helpers ───────────────────────── #

def _prepare_valid_rows_multi(
    t_arr_full: np.ndarray,
    feature_arrays_full: Sequence[np.ndarray],
    *,
    min_support: int,
) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Restrict to rows where t and *all* feature arrays are finite,
    and enforce a minimum support.

    Parameters
    ----------
    t_arr_full : np.ndarray
        Target values for rows under a fixed hypothesis H.
    feature_arrays_full : sequence of np.ndarray
        Feature values for the same rows and hypothesis.
    min_support : int
        Minimum number of valid rows required.

    Returns
    -------
    (t_arr, [f1, f2, ...]) or None
    """
    if not feature_arrays_full:
        return None

    n = t_arr_full.shape[0]
    mask = np.isfinite(t_arr_full)

    for arr in feature_arrays_full:
        if arr.shape[0] != n:
            # shape mismatch -> bail out
            return None
        mask &= np.isfinite(arr)

    if mask.sum() < min_support:
        return None

    t_arr = t_arr_full[mask]
    feats = [arr[mask] for arr in feature_arrays_full]
    return t_arr, feats


def _fit_and_build_bounds(
    *,
    t_arr: np.ndarray,
    F: np.ndarray,
    feat_exprs: List[Expr],
    target_expr: Expr,
    hyp_pred,
    target_col: str,
    hyp_name: str,
    runner_tag: str,
    min_support: int,
    max_denom: int,
    coef_bound: float,
    zero_tol: float,
    max_coef_abs: float,
    max_intercept_abs: float,
) -> List[Conjecture]:
    """
    Shared LP + canonicalization logic for nonlinear feature sets.

    Given design matrix F and corresponding Exprs, solve both
    lower (Ge) and upper (Le) bounds and return the resulting
    Conjecture objects (if any).
    """
    conjs: List[Conjecture] = []
    n = F.shape[0]
    if n < min_support:
        return conjs

    # ── Lower bound: t >= beta·F + c0 ───────────────────────────────
    lo_res = _solve_lp_lower(
        t_arr,
        F,
        coef_bound=coef_bound,
    )
    if lo_res is not None:
        beta_lo, c0_lo = lo_res
        # Rationalize before applying “human niceness”
        beta_lo = _rationalize_coeffs(beta_lo, max_denom=max_denom)
        c0_lo = _rationalize_scalar(c0_lo, max_denom=max_denom)

        rhs_lo = _build_affine_expr(
            const_val=c0_lo,
            coefs=beta_lo,
            feats=feat_exprs,
            zero_tol=zero_tol,
            max_coef_abs=max_coef_abs,
            max_intercept_abs=max_intercept_abs,
            max_denom=max_denom,
        )
        if rhs_lo is not None:
            conjs.append(
                Conjecture(
                    relation=Ge(target_expr, rhs_lo),
                    condition=hyp_pred,
                    name=f"[{runner_tag}-lower] {target_col} under {hyp_name}",
                )
            )

    # ── Upper bound: t <= beta·F + c0 ───────────────────────────────
    up_res = _solve_lp_upper(
        t_arr,
        F,
        coef_bound=coef_bound,
    )
    if up_res is not None:
        beta_up, c0_up = up_res
        beta_up = _rationalize_coeffs(beta_up, max_denom=max_denom)
        c0_up = _rationalize_scalar(c0_up, max_denom=max_denom)

        rhs_up = _build_affine_expr(
            const_val=c0_up,
            coefs=beta_up,
            feats=feat_exprs,
            zero_tol=zero_tol,
            max_coef_abs=max_coef_abs,
            max_intercept_abs=max_intercept_abs,
            max_denom=max_denom,
        )
        if rhs_up is not None:
            conjs.append(
                Conjecture(
                    relation=Le(target_expr, rhs_up),
                    condition=hyp_pred,
                    name=f"[{runner_tag}-upper] {target_col} under {hyp_name}",
                )
            )

    return conjs


# ───────────────────────── 1) x, sqrt(x), log(x) ───────────────────────── #

def x_sqrt_log_single_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence["HypothesisInfo"],
    df: pd.DataFrame,
    min_support: int = 8,
    max_denom: int = 20,
    coef_bound: float = 4.0,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 8.0,
    log_base: float | None = None,
    log_epsilon: float = 0.0,
) -> List[Conjecture]:
    """
    Bounds of the form

        H ⇒ t ≥ a x + b √x + c log(x) + d
        H ⇒ t ≤ a x + b √x + c log(x) + d

    for each hypothesis H and each 'other' invariant x.
    """
    conjs: List[Conjecture] = []
    t_all = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        for name, x_expr in others.items():
            if name == target_col:
                continue

            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
                sqrt_all = sqrt_expr(x_expr).eval(df).to_numpy(dtype=float)
                log_all = log_expr(
                    x_expr, base=log_base, epsilon=log_epsilon
                ).eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr_full = x_all[mask]
            sqrt_arr_full = sqrt_all[mask]
            log_arr_full = log_all[mask]

            prepared = _prepare_valid_rows_multi(
                t_arr_full,
                [x_arr_full, sqrt_arr_full, log_arr_full],
                min_support=min_support,
            )
            if prepared is None:
                continue

            t_arr, feat_arrays = prepared
            x_arr, sqrt_arr, log_arr = feat_arrays

            F = np.column_stack([x_arr, sqrt_arr, log_arr])
            feat_exprs: List[Expr] = [
                x_expr,
                sqrt_expr(x_expr),
                log_expr(x_expr, base=log_base, epsilon=log_epsilon),
            ]

            conjs.extend(
                _fit_and_build_bounds(
                    t_arr=t_arr,
                    F=F,
                    feat_exprs=feat_exprs,
                    target_expr=target_expr,
                    hyp_pred=hyp.pred,
                    target_col=target_col,
                    hyp_name=f"{name} under {hyp.name}",
                    runner_tag="x-sqrt-log-single",
                    min_support=min_support,
                    max_denom=max_denom,
                    coef_bound=coef_bound,
                    zero_tol=zero_tol,
                    max_coef_abs=max_coef_abs,
                    max_intercept_abs=max_intercept_abs,
                )
            )

    return conjs


# ───────────────────────── 2) sqrt(x), sqrt(y) ───────────────────────── #

def sqrt_pair_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence["HypothesisInfo"],
    df: pd.DataFrame,
    min_support: int = 8,
    max_denom: int = 20,
    coef_bound: float = 4.0,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 8.0,
) -> List[Conjecture]:
    """
    Bounds of the form

        H ⇒ t ≥ a √x + b √y + c
        H ⇒ t ≤ a √x + b √y + c

    for each hypothesis H and each unordered pair (x, y) of invariants.
    """
    from itertools import combinations

    conjs: List[Conjecture] = []
    t_all = df[target_col].to_numpy(dtype=float)
    items = list(others.items())

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        for (name_x, x_expr), (name_y, y_expr) in combinations(items, 2):
            # Avoid target in any role
            if name_x == target_col or name_y == target_col:
                continue

            try:
                sx_all = sqrt_expr(x_expr).eval(df).to_numpy(dtype=float)
                sy_all = sqrt_expr(y_expr).eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            sx_arr_full = sx_all[mask]
            sy_arr_full = sy_all[mask]

            prepared = _prepare_valid_rows_multi(
                t_arr_full,
                [sx_arr_full, sy_arr_full],
                min_support=min_support,
            )
            if prepared is None:
                continue

            t_arr, feat_arrays = prepared
            sx_arr, sy_arr = feat_arrays

            F = np.column_stack([sx_arr, sy_arr])
            feat_exprs: List[Expr] = [
                sqrt_expr(x_expr),
                sqrt_expr(y_expr),
            ]

            conjs.extend(
                _fit_and_build_bounds(
                    t_arr=t_arr,
                    F=F,
                    feat_exprs=feat_exprs,
                    target_expr=target_expr,
                    hyp_pred=hyp.pred,
                    target_col=target_col,
                    hyp_name=f"{name_x}, {name_y} under {hyp.name}",
                    runner_tag="sqrt-pair",
                    min_support=min_support,
                    max_denom=max_denom,
                    coef_bound=coef_bound,
                    zero_tol=zero_tol,
                    max_coef_abs=max_coef_abs,
                    max_intercept_abs=max_intercept_abs,
                )
            )

    return conjs


# ───────────────────────── 3) x, sqrt(x*y) ───────────────────────── #

def geom_mean_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence["HypothesisInfo"],
    df: pd.DataFrame,
    min_support: int = 8,
    max_denom: int = 20,
    coef_bound: float = 4.0,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 8.0,
) -> List[Conjecture]:
    """
    Bounds of the form

        H ⇒ t ≥ a x + b √(x y) + c
        H ⇒ t ≤ a x + b √(x y) + c

    for each hypothesis H and each unordered pair (x, y) of invariants.
    """
    from itertools import combinations

    conjs: List[Conjecture] = []
    t_all = df[target_col].to_numpy(dtype=float)
    items = list(others.items())

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        for (name_x, x_expr), (name_y, y_expr) in combinations(items, 2):
            if name_x == target_col or name_y == target_col:
                continue

            # Use x in linear term, √(x y) as nonlinear feature
            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
                gm_expr = sqrt_expr(x_expr * y_expr)
                gm_all = gm_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr_full = x_all[mask]
            gm_arr_full = gm_all[mask]

            prepared = _prepare_valid_rows_multi(
                t_arr_full,
                [x_arr_full, gm_arr_full],
                min_support=min_support,
            )
            if prepared is None:
                continue

            t_arr, feat_arrays = prepared
            x_arr, gm_arr = feat_arrays

            F = np.column_stack([x_arr, gm_arr])
            feat_exprs: List[Expr] = [
                x_expr,
                gm_expr,
            ]

            conjs.extend(
                _fit_and_build_bounds(
                    t_arr=t_arr,
                    F=F,
                    feat_exprs=feat_exprs,
                    target_expr=target_expr,
                    hyp_pred=hyp.pred,
                    target_col=target_col,
                    hyp_name=f"{name_x}, √({name_x}·{name_y}) under {hyp.name}",
                    runner_tag="geom-mean",
                    min_support=min_support,
                    max_denom=max_denom,
                    coef_bound=coef_bound,
                    zero_tol=zero_tol,
                    max_coef_abs=max_coef_abs,
                    max_intercept_abs=max_intercept_abs,
                )
            )

    return conjs


# ───────────────────────── 4) x, sqrt(x + y) ───────────────────────── #

def sqrt_sum_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence["HypothesisInfo"],
    df: pd.DataFrame,
    min_support: int = 8,
    max_denom: int = 20,
    coef_bound: float = 4.0,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 8.0,
) -> List[Conjecture]:
    """
    Bounds of the form

        H ⇒ t ≥ a x + b √(x + y) + c
        H ⇒ t ≤ a x + b √(x + y) + c

    for each hypothesis H and each unordered pair (x, y) of invariants.
    """
    from itertools import combinations

    conjs: List[Conjecture] = []
    t_all = df[target_col].to_numpy(dtype=float)
    items = list(others.items())

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        for (name_x, x_expr), (name_y, y_expr) in combinations(items, 2):
            if name_x == target_col or name_y == target_col:
                continue

            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
                sum_expr = x_expr + y_expr
                sqrt_sum_expr = sqrt_expr(sum_expr)
                sqrt_sum_all = sqrt_sum_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr_full = x_all[mask]
            sqrt_sum_arr_full = sqrt_sum_all[mask]

            prepared = _prepare_valid_rows_multi(
                t_arr_full,
                [x_arr_full, sqrt_sum_arr_full],
                min_support=min_support,
            )
            if prepared is None:
                continue

            t_arr, feat_arrays = prepared
            x_arr, sqrt_sum_arr = feat_arrays

            F = np.column_stack([x_arr, sqrt_sum_arr])
            feat_exprs: List[Expr] = [
                x_expr,
                sqrt_sum_expr,
            ]

            conjs.extend(
                _fit_and_build_bounds(
                    t_arr=t_arr,
                    F=F,
                    feat_exprs=feat_exprs,
                    target_expr=target_expr,
                    hyp_pred=hyp.pred,
                    target_col=target_col,
                    hyp_name=f"{name_x}, √({name_x}+{name_y}) under {hyp.name}",
                    runner_tag="sqrt-sum",
                    min_support=min_support,
                    max_denom=max_denom,
                    coef_bound=coef_bound,
                    zero_tol=zero_tol,
                    max_coef_abs=max_coef_abs,
                    max_intercept_abs=max_intercept_abs,
                )
            )

    return conjs


# ───────────────────────── 5) x, log(x + y) ───────────────────────── #

def log_sum_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence["HypothesisInfo"],
    df: pd.DataFrame,
    min_support: int = 8,
    max_denom: int = 20,
    coef_bound: float = 4.0,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 8.0,
    log_base: float | None = None,
    log_epsilon: float = 0.0,
) -> List[Conjecture]:
    """
    Bounds of the form

        H ⇒ t ≥ a x + b log(x + y) + c
        H ⇒ t ≤ a x + b log(x + y) + c

    for each hypothesis H and each unordered pair (x, y) of invariants.
    """
    from itertools import combinations

    conjs: List[Conjecture] = []
    t_all = df[target_col].to_numpy(dtype=float)
    items = list(others.items())

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        for (name_x, x_expr), (name_y, y_expr) in combinations(items, 2):
            if name_x == target_col or name_y == target_col:
                continue

            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
                sum_expr = x_expr + y_expr
                log_sum_expr = log_expr(
                    sum_expr, base=log_base, epsilon=log_epsilon
                )
                log_sum_all = log_sum_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr_full = x_all[mask]
            log_sum_arr_full = log_sum_all[mask]

            prepared = _prepare_valid_rows_multi(
                t_arr_full,
                [x_arr_full, log_sum_arr_full],
                min_support=min_support,
            )
            if prepared is None:
                continue

            t_arr, feat_arrays = prepared
            x_arr, log_sum_arr = feat_arrays

            F = np.column_stack([x_arr, log_sum_arr])
            feat_exprs: List[Expr] = [
                x_expr,
                log_sum_expr,
            ]

            conjs.extend(
                _fit_and_build_bounds(
                    t_arr=t_arr,
                    F=F,
                    feat_exprs=feat_exprs,
                    target_expr=target_expr,
                    hyp_pred=hyp.pred,
                    target_col=target_col,
                    hyp_name=f"{name_x}, log({name_x}+{name_y}) under {hyp.name}",
                    runner_tag="log-sum",
                    min_support=min_support,
                    max_denom=max_denom,
                    coef_bound=coef_bound,
                    zero_tol=zero_tol,
                    max_coef_abs=max_coef_abs,
                    max_intercept_abs=max_intercept_abs,
                )
            )

    return conjs
