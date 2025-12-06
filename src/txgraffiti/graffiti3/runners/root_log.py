# src/txgraffiti/graffiti3/runners/root_log.py

from __future__ import annotations

from typing import Dict, List, Sequence, TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.exprs import Expr, Const, sqrt as sqrt_expr, log as log_expr
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le
from txgraffiti.graffiti3.runners.utils import _build_affine_expr

# We reuse the LP helpers + canonicalization from the poly runner
from txgraffiti.graffiti3.runners.poly import (
    _solve_lp_upper,
    _solve_lp_lower,
)
from txgraffiti.graffiti3.runners.utils import (
    _rationalize_coeffs,
    _rationalize_scalar,
)

if TYPE_CHECKING:
    from txgraffiti.graffiti3.types import HypothesisInfo


# ───────────────────────── helpers ───────────────────────── #

def _prepare_valid_rows(
    t_arr: np.ndarray,
    cols: List[np.ndarray],
    *,
    min_support: int,
) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Restrict to rows with finite t and *all* given columns, enforce min_support.
    """
    mask = np.isfinite(t_arr)
    for c in cols:
        mask &= np.isfinite(c)
    if mask.sum() < min_support:
        return None
    return t_arr[mask], [c[mask] for c in cols]


def _canonical_nonlinear_rhs(
    beta: np.ndarray,
    c0: float,
    feat_exprs: List[Expr],
    *,
    zero_tol: float,
    max_coef_abs: float,
    max_intercept_abs: float,
    max_denom: int,
) -> Expr | None:
    """
    Generic version of the polynomial canonicalization:

    - Rationalize coefficients first.
    - Drop any feature whose *rationalized* coefficient is numerically zero.
    - Clean tiny intercepts.
    - Delegate to `_build_affine_expr` for final niceness checks.
    """
    beta = np.asarray(beta, dtype=float)
    beta_rat = _rationalize_coeffs(beta, max_denom=max_denom)
    c0_rat = _rationalize_scalar(float(c0), max_denom=max_denom)

    coefs: List[float] = []
    feats: List[Expr] = []

    for coef_rat, feat in zip(beta_rat, feat_exprs):
        coef_f = float(coef_rat)
        if abs(coef_f) >= zero_tol:
            coefs.append(coef_f)
            feats.append(feat)

    c0_val = float(c0_rat)
    if abs(c0_val) < zero_tol:
        c0_val = 0.0

    if not coefs and abs(c0_val) < zero_tol:
        return None

    return _build_affine_expr(
        const_val=c0_val,
        coefs=coefs,
        feats=feats,
        zero_tol=zero_tol,
        max_coef_abs=max_coef_abs,
        max_intercept_abs=max_intercept_abs,
        max_denom=max_denom,
    )

def _prepare_valid_rows_multi(
    t_arr: np.ndarray,
    features: Sequence[np.ndarray],
    *,
    min_support: int,
) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Restrict to rows where t and *all* feature arrays are finite.

    Parameters
    ----------
    t_arr : 1D array of target values
    features : sequence of 1D feature arrays (same length as t_arr)
    min_support : minimum #valid rows required

    Returns
    -------
    (t_valid, feats_valid) or None
    """
    if not features:
        return None

    mask = np.isfinite(t_arr)
    for f in features:
        mask &= np.isfinite(f)

    if mask.sum() < min_support:
        return None

    t_valid = t_arr[mask]
    feats_valid = [f[mask] for f in features]
    return t_valid, feats_valid


# ───────────────────────── sqrt runner ───────────────────────── #

from txgraffiti.graffiti3.exprs import Expr, sqrt  # make sure sqrt is imported
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le

def sqrt_single_runner(
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
    Square-root two-invariant bounds:

        H ⇒ t ≥ a x + b √y + c
        H ⇒ t ≤ a x + b √y + c

    for each hypothesis H and each pair (x, y) from `others`.

    Special case x = y recovers the older pattern
        a x + b √x + c
    that previously gave conjectures like
        p6 ≥ (35/9)·temperature(p6) + (13/15)·√temperature(p6).
    """
    conjs: List[Conjecture] = []


    t_all = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        # Outer loop: x for the linear term
        for name_x, x_expr in others.items():
            if name_x == target_col:
                continue

            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr_full = x_all[mask]
            if x_arr_full.size == 0:
                continue

            # Inner loop: y for the sqrt term
            for name_y, y_expr in others.items():
                if name_y == target_col:
                    continue
                # You *can* skip x=y here if you only want cross-terms; but
                # keeping x=y restores your earlier good conjectures.
                # if name_y == name_x:
                #     continue

                try:
                    sqrt_all = sqrt(y_expr).eval(df).to_numpy(dtype=float)
                except Exception:
                    continue

                sqrt_arr_full = sqrt_all[mask]
                if sqrt_arr_full.size == 0:
                    continue

                prepared = _prepare_valid_rows_multi(
                    t_arr_full,
                    [x_arr_full, sqrt_arr_full],
                    min_support=min_support,
                )
                if prepared is None:
                    continue

                t_arr, [x_arr, sqrt_arr] = prepared

                # Design matrix with features [x, √y]
                F = np.column_stack([x_arr, sqrt_arr])

                feat_exprs: List[Expr] = [
                    x_expr,
                    sqrt(y_expr),
                ]

                # ── Lower bound: t ≥ a x + b √y + c ───────────────────
                lo_res = _solve_lp_lower(
                    t_arr,
                    F,
                    coef_bound=coef_bound,
                )
                if lo_res is not None:
                    beta_lo, c0_lo = lo_res

                    rhs_lo = _canonical_nonlinear_rhs(
                        beta=beta_lo,
                        c0=c0_lo,
                        feat_exprs=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs_lo is not None:
                        conjs.append(
                            Conjecture(
                                relation=Ge(target_expr, rhs_lo),
                                condition=hyp.pred,
                                name=(
                                    f"[sqrt-single-lower] {target_col} "
                                    f"vs {name_x}, √({name_y}) under {hyp.name}"
                                ),
                            )
                        )

                # ── Upper bound: t ≤ a x + b √y + c ───────────────────
                up_res = _solve_lp_upper(
                    t_arr,
                    F,
                    coef_bound=coef_bound,
                )
                if up_res is not None:
                    beta_up, c0_up = up_res

                    rhs_up = _canonical_nonlinear_rhs(
                        beta=beta_up,
                        c0=c0_up,
                        feat_exprs=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs_up is not None:
                        conjs.append(
                            Conjecture(
                                relation=Le(target_expr, rhs_up),
                                condition=hyp.pred,
                                name=(
                                    f"[sqrt-single-upper] {target_col} "
                                    f"vs {name_x}, √({name_y}) under {hyp.name}"
                                ),
                            )
                        )

    return conjs



# ───────────────────────── log runner ───────────────────────── #

def log_single_runner(
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
    Logarithmic two-invariant bounds:

        H ⇒ t ≥ a x + b log(y) + c
        H ⇒ t ≤ a x + b log(y) + c

    for each hypothesis H and each pair of 'other' invariants (x, y).

    Parameters
    ----------
    log_base : float or None, optional
        Base of the log.  If None, use natural log.
    log_epsilon : float, optional
        Clamp argument to at least `log_epsilon` before applying log, so
        you can avoid -inf from small/zero values if you wish.
    """
    conjs: List[Conjecture] = []

    if _solve_lp_lower is None or _solve_lp_upper is None:
        return conjs

    t_all = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        # Outer loop: choose x for the linear term
        for name_x, x_expr in others.items():
            if name_x == target_col:
                continue

            # Evaluate x once per (hyp, x)
            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr_full = x_all[mask]
            if x_arr_full.size == 0:
                continue

            # Inner loop: choose y for the log term
            for name_y, y_expr in others.items():
                # Optional: skip using the target inside the log as well
                if name_y == target_col:
                    continue

                # (Optional) You can also skip the trivial x=y case if you like:
                # if name_y == name_x:
                #     continue

                try:
                    log_all = log_expr(
                        y_expr,
                        base=log_base,
                        epsilon=log_epsilon,
                    ).eval(df).to_numpy(dtype=float)
                except Exception:
                    continue

                log_arr_full = log_all[mask]
                if log_arr_full.size == 0:
                    continue

                prepared = _prepare_valid_rows(
                    t_arr_full,
                    [x_arr_full, log_arr_full],
                    min_support=min_support,
                )
                if prepared is None:
                    continue

                t_arr, [x_arr, log_arr] = prepared

                # Design matrix with features [x, log(y)]
                F = np.column_stack([x_arr, log_arr])

                feat_exprs: List[Expr] = [
                    x_expr,
                    log_expr(y_expr, base=log_base, epsilon=log_epsilon),
                ]

                # ── Lower bound: t ≥ a x + b log(y) + c ───────────────────
                lo_res = _solve_lp_lower(
                    t_arr,
                    F,
                    coef_bound=coef_bound,
                )
                if lo_res is not None:
                    beta_lo, c0_lo = lo_res

                    rhs_lo = _canonical_nonlinear_rhs(
                        beta=beta_lo,
                        c0=c0_lo,
                        feat_exprs=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs_lo is not None:
                        conjs.append(
                            Conjecture(
                                relation=Ge(target_expr, rhs_lo),
                                condition=hyp.pred,
                                name=(
                                    f"[log-single-lower] {target_col} "
                                    f"vs {name_x}, log({name_y}) under {hyp.name}"
                                ),
                            )
                        )

                # ── Upper bound: t ≤ a x + b log(y) + c ───────────────────
                up_res = _solve_lp_upper(
                    t_arr,
                    F,
                    coef_bound=coef_bound,
                )
                if up_res is not None:
                    beta_up, c0_up = up_res

                    rhs_up = _canonical_nonlinear_rhs(
                        beta=beta_up,
                        c0=c0_up,
                        feat_exprs=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs_up is not None:
                        conjs.append(
                            Conjecture(
                                relation=Le(target_expr, rhs_up),
                                condition=hyp.pred,
                                name=(
                                    f"[log-single-upper] {target_col} "
                                    f"vs {name_x}, log({name_y}) under {hyp.name}"
                                ),
                            )
                        )

    return conjs

from fractions import Fraction
from typing import Dict, List, Sequence, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.exprs import Expr, Const, sqrt as sqrt_expr
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le

if TYPE_CHECKING:
    from txgraffiti.graffiti3.types import HypothesisInfo

# assumes: linprog, _solve_lp_lower, _solve_lp_upper, _canonical_nonlinear_rhs,
# and _prepare_valid_rows_multi are defined in this module.


def quad_sqrt_runner(
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
    Quadratic + sqrt two-invariant bounds:

        H ⇒ t ≥ a x² + b √y + c
        H ⇒ t ≤ a x² + b √y + c

    for each hypothesis H and each pair (x, y) from `others`.

    Special case x = y yields bounds of the form

        t ≥ a x² + b √x + c

    which can capture mixed “curvature + spread” behavior.
    """
    conjs: List[Conjecture] = []

    t_all = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        # Outer loop: x for the quadratic term
        for name_x, x_expr in others.items():
            if name_x == target_col:
                continue

            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr_full = x_all[mask]
            if x_arr_full.size == 0:
                continue

            x_sq_full = np.square(x_arr_full, dtype=float)

            # Inner loop: y for the sqrt term
            for name_y, y_expr in others.items():
                if name_y == target_col:
                    continue

                try:
                    sqrt_all = sqrt_expr(y_expr).eval(df).to_numpy(dtype=float)
                except Exception:
                    continue

                sqrt_arr_full = sqrt_all[mask]
                if sqrt_arr_full.size == 0:
                    continue

                prepared = _prepare_valid_rows_multi(
                    t_arr_full,
                    [x_sq_full, sqrt_arr_full],
                    min_support=min_support,
                )
                if prepared is None:
                    continue

                t_arr, [x_sq_arr, sqrt_arr] = prepared

                # Design matrix with features [x², √y]
                F = np.column_stack([x_sq_arr, sqrt_arr])

                feat_exprs: List[Expr] = [
                    x_expr ** Const(Fraction(2, 1)),
                    sqrt_expr(y_expr),
                ]

                # ── Lower bound: t ≥ a x² + b √y + c ───────────────────
                lo_res = _solve_lp_lower(
                    t_arr,
                    F,
                    coef_bound=coef_bound,
                )
                if lo_res is not None:
                    beta_lo, c0_lo = lo_res

                    rhs_lo = _canonical_nonlinear_rhs(
                        beta=beta_lo,
                        c0=c0_lo,
                        feat_exprs=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs_lo is not None:
                        conjs.append(
                            Conjecture(
                                relation=Ge(target_expr, rhs_lo),
                                condition=hyp.pred,
                                name=(
                                    f"[quad-sqrt-lower] {target_col} "
                                    f"vs {name_x}², √({name_y}) under {hyp.name}"
                                ),
                            )
                        )

                # ── Upper bound: t ≤ a x² + b √y + c ───────────────────
                up_res = _solve_lp_upper(
                    t_arr,
                    F,
                    coef_bound=coef_bound,
                )
                if up_res is not None:
                    beta_up, c0_up = up_res

                    rhs_up = _canonical_nonlinear_rhs(
                        beta=beta_up,
                        c0=c0_up,
                        feat_exprs=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs_up is not None:
                        conjs.append(
                            Conjecture(
                                relation=Le(target_expr, rhs_up),
                                condition=hyp.pred,
                                name=(
                                    f"[quad-sqrt-upper] {target_col} "
                                    f"vs {name_x}², √({name_y}) under {hyp.name}"
                                ),
                            )
                        )

    return conjs

from txgraffiti.graffiti3.exprs import log as log_expr

def quad_log_runner(
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
    Quadratic + log two-invariant bounds:

        H ⇒ t ≥ a x² + b log(y) + c
        H ⇒ t ≤ a x² + b log(y) + c

    for each hypothesis H and each pair (x, y) from `others`.

    Special case x = y yields bounds of the form
        t ≥ a x² + b log(x) + c.
    """
    conjs: List[Conjecture] = []

    t_all = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr_full = t_all[mask]

        # Outer loop: x for the quadratic term
        for name_x, x_expr in others.items():
            if name_x == target_col:
                continue

            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr_full = x_all[mask]
            if x_arr_full.size == 0:
                continue

            x_sq_full = np.square(x_arr_full, dtype=float)

            # Inner loop: y for the log term
            for name_y, y_expr in others.items():
                if name_y == target_col:
                    continue

                try:
                    log_all = log_expr(
                        y_expr,
                        base=log_base,
                        epsilon=log_epsilon,
                    ).eval(df).to_numpy(dtype=float)
                except Exception:
                    continue

                log_arr_full = log_all[mask]
                if log_arr_full.size == 0:
                    continue

                prepared = _prepare_valid_rows_multi(
                    t_arr_full,
                    [x_sq_full, log_arr_full],
                    min_support=min_support,
                )
                if prepared is None:
                    continue

                t_arr, [x_sq_arr, log_arr] = prepared

                # Design matrix with features [x², log(y)]
                F = np.column_stack([x_sq_arr, log_arr])

                feat_exprs: List[Expr] = [
                    x_expr ** Const(Fraction(2, 1)),
                    log_expr(y_expr, base=log_base, epsilon=log_epsilon),
                ]

                # ── Lower bound: t ≥ a x² + b log(y) + c ───────────────
                lo_res = _solve_lp_lower(
                    t_arr,
                    F,
                    coef_bound=coef_bound,
                )
                if lo_res is not None:
                    beta_lo, c0_lo = lo_res

                    rhs_lo = _canonical_nonlinear_rhs(
                        beta=beta_lo,
                        c0=c0_lo,
                        feat_exprs=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs_lo is not None:
                        conjs.append(
                            Conjecture(
                                relation=Ge(target_expr, rhs_lo),
                                condition=hyp.pred,
                                name=(
                                    f"[quad-log-lower] {target_col} "
                                    f"vs {name_x}², log({name_y}) under {hyp.name}"
                                ),
                            )
                        )

                # ── Upper bound: t ≤ a x² + b log(y) + c ───────────────
                up_res = _solve_lp_upper(
                    t_arr,
                    F,
                    coef_bound=coef_bound,
                )
                if up_res is not None:
                    beta_up, c0_up = up_res

                    rhs_up = _canonical_nonlinear_rhs(
                        beta=beta_up,
                        c0=c0_up,
                        feat_exprs=feat_exprs,
                        zero_tol=zero_tol,
                        max_coef_abs=max_coef_abs,
                        max_intercept_abs=max_intercept_abs,
                        max_denom=max_denom,
                    )
                    if rhs_up is not None:
                        conjs.append(
                            Conjecture(
                                relation=Le(target_expr, rhs_up),
                                condition=hyp.pred,
                                name=(
                                    f"[quad-log-upper] {target_col} "
                                    f"vs {name_x}², log({name_y}) under {hyp.name}"
                                ),
                            )
                        )

    return conjs
