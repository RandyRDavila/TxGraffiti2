# src/txgraffiti2025/graffiti4_poly_single.py

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import Expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le

if TYPE_CHECKING:
    # Only for type hints; avoids circular imports at runtime.
    from txgraffiti2025.graffiti4_types import HypothesisInfo

# Try to get an LP solver
try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    linprog = None


# ───────────────────────── helpers ───────────────────────── #

def _to_const_fraction(x: float, max_denom: int) -> Const:
    """Return a Const representing a bounded-denominator rational close to x."""
    return Const(Fraction(x).limit_denominator(max_denom))


def _build_expr_from_coeffs(
    beta: np.ndarray,
    c0: float,
    feature_exprs: Sequence[Expr],
    *,
    max_denom: int,
    coef_tol: float = 1e-8,
) -> Expr:
    """
    Turn numeric coefficients + intercept into an Expr, dropping tiny coefficients
    and rationalizing the rest.
    """
    assert len(beta) == len(feature_exprs)

    terms: List[Expr] = []

    # linear terms a_i * feature_i
    for coeff, e in zip(beta, feature_exprs):
        if abs(coeff) < coef_tol:
            continue
        c_const = _to_const_fraction(float(coeff), max_denom)
        terms.append(c_const * e)

    # intercept term
    if abs(c0) >= coef_tol:
        terms.append(_to_const_fraction(float(c0), max_denom))

    if not terms:
        # fall back to a zero constant; caller can decide whether to keep/discard
        return Const(0)

    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr


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
    max_coef_abs: float = 4.0,
) -> List[Conjecture]:
    """
    Polynomial single-invariant bounds:

        H ⇒ t ≥ a x + b x^2 + c
        H ⇒ t ≤ a x + b x^2 + c

    for each hypothesis H and each 'other' invariant x.

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
    max_coef_abs : float
        Bound on |a|, |b|, |c| to avoid insane LP solutions.

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

            # 1) Lower bound via LP
            lo_res = _solve_lp_lower(
                t_arr,
                F,
                coef_bound=max_coef_abs,
            )
            if lo_res is not None:
                beta_lo, c0_lo = lo_res
                rhs_lo = _build_expr_from_coeffs(
                    beta_lo,
                    c0_lo,
                    feature_exprs=[
                        x_expr,
                        x_expr ** Const(Fraction(2, 1)),
                    ],
                    max_denom=max_denom,
                )
                conjs.append(
                    Conjecture(
                        relation=Ge(target_expr, rhs_lo),
                        condition=hyp.pred,
                        name=(
                            f"[poly-single-lower] {target_col} vs {name}, x^2 under {hyp.name}"
                        ),
                    )
                )

            # 2) Upper bound via LP
            up_res = _solve_lp_upper(
                t_arr,
                F,
                coef_bound=max_coef_abs,
            )
            if up_res is not None:
                beta_up, c0_up = up_res
                rhs_up = _build_expr_from_coeffs(
                    beta_up,
                    c0_up,
                    feature_exprs=[
                        x_expr,
                        x_expr ** Const(Fraction(2, 1)),
                    ],
                    max_denom=max_denom,
                )
                conjs.append(
                    Conjecture(
                        relation=Le(target_expr, rhs_up),
                        condition=hyp.pred,
                        name=(
                            f"[poly-single-upper] {target_col} vs {name}, x^2 under {hyp.name}"
                        ),
                    )
                )

    return conjs
