# src/txgraffiti/graffiti3/runners/poly.py

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.exprs import Expr, Const
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le
from txgraffiti.graffiti3.runners.utils import (
    _build_affine_expr,
    _rationalize_coeffs,
    _rationalize_scalar,
)

if TYPE_CHECKING:
    # Only for type hints; avoids circular imports at runtime.
    from txgraffiti.graffiti3.types import HypothesisInfo

# Try to get an LP solver
try:  # pragma: no cover
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    linprog = None


# ───────────────────────── helpers ───────────────────────── #

def _prepare_valid_rows(
    t_arr: np.ndarray,
    x_arr: np.ndarray,
    *,
    min_support: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Restrict to rows with finite t and x and enforce a minimum support.
    """
    mask = np.isfinite(t_arr) & np.isfinite(x_arr)
    if mask.sum() < min_support:
        return None
    return t_arr[mask], x_arr[mask]


def _solve_lp_upper(
    t_arr: np.ndarray,
    F: np.ndarray,
    *,
    coef_bound: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Solve for an upper bound t <= F beta + c0 via LP:

        minimize   sum_i (F_i · beta + c0)
        subject to F_i · beta + c0 >= t_i   for all i
                  |beta_j| <= coef_bound
                  |c0|     <= coef_bound
    """
    if linprog is None:
        return None

    n, k = F.shape
    if n == 0:
        return None

    # Variables: v = [beta_1, ..., beta_k, c0]
    # Objective: minimize sum_i (F_i · beta + c0)
    #           = sum_j beta_j * sum_i F_ij  +  c0 * n
    c_vec = np.zeros(k + 1, dtype=float)
    c_vec[:k] = F.sum(axis=0)
    c_vec[-1] = float(n)

    # Constraints: F_i · beta + c0 >= t_i  =>  -F_i · beta - c0 <= -t_i
    A_ub = np.empty((n, k + 1), dtype=float)
    A_ub[:, :k] = -F
    A_ub[:, -1] = -1.0
    b_ub = -t_arr.astype(float)

    bounds = [(-coef_bound, coef_bound)] * (k + 1)

    res = linprog(
        c=c_vec,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None

    v = res.x
    beta = v[:k]
    c0 = v[-1]

    # Numerical safety check
    rhs = F @ beta + c0
    if not np.all(t_arr <= rhs + 1e-7):
        return None

    return beta, c0


def _solve_lp_lower(
    t_arr: np.ndarray,
    F: np.ndarray,
    *,
    coef_bound: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Solve for a lower bound t >= F beta + c0 via LP:

        maximize   sum_i (F_i · beta + c0)
        subject to F_i · beta + c0 <= t_i   for all i
                  |beta_j| <= coef_bound
                  |c0|     <= coef_bound

    Implemented as:

        minimize  -sum_i (F_i · beta + c0)
        subject to F_i · beta + c0 <= t_i.
    """
    if linprog is None:
        return None

    n, k = F.shape
    if n == 0:
        return None

    # Variables: v = [beta_1, ..., beta_k, c0]
    # Objective: minimize -sum_i (F_i · beta + c0)
    #           = sum_j beta_j * (-sum_i F_ij)  +  c0 * (-n)
    c_vec = np.zeros(k + 1, dtype=float)
    c_vec[:k] = -F.sum(axis=0)
    c_vec[-1] = -float(n)

    # Constraints: F_i · beta + c0 <= t_i
    A_ub = np.empty((n, k + 1), dtype=float)
    A_ub[:, :k] = F
    A_ub[:, -1] = 1.0
    b_ub = t_arr.astype(float)

    bounds = [(-coef_bound, coef_bound)] * (k + 1)

    res = linprog(
        c=c_vec,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None

    v = res.x
    beta = v[:k]
    c0 = v[-1]

    # Numerical safety check
    rhs = F @ beta + c0
    if not np.all(t_arr >= rhs - 1e-7):
        return None

    return beta, c0


def _canonical_poly_rhs(
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
    Build a canonical polynomial RHS using `_build_affine_expr`, dropping any
    feature whose *rationalized* coefficient is numerically zero.

    This ensures that forms like a x + 0 x^2 + c normalize to the same
    expression as a x + c from other runners, and we never print `0 · (x)²`.
    """
    # Rationalize first, so tiny floats like 1e-12 become exact 0 if appropriate.
    beta = np.asarray(beta, dtype=float)
    beta_rat = _rationalize_coeffs(beta, max_denom=max_denom)
    c0_rat = _rationalize_scalar(float(c0), max_denom=max_denom)

    coefs: List[float] = []
    feats: List[Expr] = []

    for coef_rat, feat in zip(beta_rat, feat_exprs):
        coef_f = float(coef_rat)
        # Drop genuinely small / zero coefficients after rationalization
        if abs(coef_f) >= zero_tol:
            coefs.append(coef_f)
            feats.append(feat)

    # Clean tiny intercept as well
    c0_val = float(c0_rat)
    if abs(c0_val) < zero_tol:
        c0_val = 0.0

    # If everything is tiny and intercept is tiny, drop the form.
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


# ───────────────────────── main runner ───────────────────────── #


def poly_single_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence["HypothesisInfo"],
    df: pd.DataFrame,
    min_support: int = 8,
    max_denom: int = 20,
    # LP box constraint on |a|, |b|, |c0|
    coef_bound: float = 4.0,
    # “human niceness” / canonicalization controls (shared with other runners)
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_intercept_abs: float = 8.0,
) -> List[Conjecture]:
    """
    Polynomial single-invariant bounds:

        H ⇒ t ≥ a x + b x^2 + c
        H ⇒ t ≤ a x + b x^2 + c

    for each hypothesis H and each 'other' invariant x.

    All final expressions are built via the shared `_build_affine_expr`
    helper so that simple cases like

        a = 1, b = 0, c = 0

    normalize to the same representation as `t ≥ x` or `t ≤ x` coming from
    the ratio / LP runners (no spurious “1 · x” or “0 · x²” terms).

    Parameters
    ----------
    target_col : str
        Dependent variable column name t.
    target_expr : Expr
        Expr for t (usually to_expr(target_col)).
    others : dict[str, Expr]
        Candidate invariants x.
    hypotheses : sequence[HypothesisInfo]
        Hypotheses H, each with .mask (np.ndarray[bool]), .pred (Predicate), .name.
    df : DataFrame
        Numeric invariant table.
    min_support : int
        Minimum number of valid rows under a hypothesis.
    max_denom : int
        Max denominator for rational coefficients.
    coef_bound : float
        LP box constraint: |a|, |b|, |c0| <= coef_bound.
    zero_tol : float
        Coefficients with |c| < zero_tol are dropped in canonicalization.
    max_coef_abs : float
        If any |coef| > max_coef_abs after rationalization, drop the form.
    max_intercept_abs : float
        If |c0| > max_intercept_abs, drop the form.

    Returns
    -------
    list[Conjecture]
        List of lower and upper polynomial bounds.
    """
    conjs: List[Conjecture] = []

    if linprog is None:
        # No LP solver; gracefully return nothing.
        return conjs

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
            except Exception:
                continue

            x_arr_full = x_all[mask]
            if x_arr_full.size == 0:
                continue

            prepared = _prepare_valid_rows(
                t_arr_full,
                x_arr_full,
                min_support=min_support,
            )
            if prepared is None:
                continue

            t_arr, x_arr = prepared
            x_sq = np.square(x_arr, dtype=float)

            # Design matrix with features [x, x^2]
            F = np.column_stack([x_arr, x_sq])

            # Corresponding Expr list for canonical builder
            feat_exprs: List[Expr] = [
                x_expr,
                x_expr ** Const(Fraction(2, 1)),
            ]

            # 1) Lower bound via LP: t >= a x + b x^2 + c0
            lo_res = _solve_lp_lower(
                t_arr,
                F,
                coef_bound=coef_bound,
            )
            if lo_res is not None:
                beta_lo, c0_lo = lo_res

                rhs_lo = _canonical_poly_rhs(
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
                                f"[poly-single-lower] {target_col} "
                                f"vs {name}, x^2 under {hyp.name}"
                            ),
                        )
                    )

            # 2) Upper bound via LP: t <= a x + b x^2 + c0
            up_res = _solve_lp_upper(
                t_arr,
                F,
                coef_bound=coef_bound,
            )
            if up_res is not None:
                beta_up, c0_up = up_res

                rhs_up = _canonical_poly_rhs(
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
                                f"[poly-single-upper] {target_col} "
                                f"vs {name}, x^2 under {hyp.name}"
                            ),
                        )
                    )

    return conjs
