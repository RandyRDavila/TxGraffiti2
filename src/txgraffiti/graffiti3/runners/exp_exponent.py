# src/txgraffiti/graffiti3/runners/exp_exponent.py

from __future__ import annotations

from typing import Dict, List, Sequence, TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.exprs import (
    Expr,
    Const,
    log as log_expr,
)
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le

# Reuse the LP helpers and rationalizers from existing runners
from txgraffiti.graffiti3.runners.poly import _solve_lp_lower, _solve_lp_upper
from txgraffiti.graffiti3.runners.utils import (
    _rationalize_coeffs,
    _rationalize_scalar,
)

if TYPE_CHECKING:
    from txgraffiti.graffiti3.types import HypothesisInfo


# ───────────────────────── helpers ───────────────────────── #

def _build_exp_exponent_rhs(
    *,
    base_expr: Expr,
    chi_expr: Expr,
    beta: np.ndarray,
    c0: float,
    max_denom: int,
    zero_tol: float,
    max_exponent_coef_abs: float,
    max_log_intercept_abs: float,
) -> Optional[Expr]:
    """
    Given an LP solution for

        log t  ≲  c0 + beta[0] * log(base) + beta[1] * (chi · log(base)),

    construct an Expr representing

        RHS = C * base^(a0 + a1 * chi),

    where C = exp(c0), a0 = beta[0], a1 = beta[1] (after rationalization).

    Returns None if the coefficients are too large / degenerate.
    """
    beta = np.asarray(beta, dtype=float).ravel()
    if beta.size != 2:
        return None

    # Rationalize coefficients
    beta_rat = _rationalize_coeffs(beta, max_denom=max_denom)
    c0_rat = _rationalize_scalar(c0, max_denom=max_denom)

    a0, a1 = float(beta_rat[0]), float(beta_rat[1])

    # Drop tiny “noise” coefficients
    if abs(a0) < zero_tol:
        a0 = 0.0
    if abs(a1) < zero_tol:
        a1 = 0.0

    # Sanity / human-niceness bounds
    if abs(c0_rat) > max_log_intercept_abs:
        return None
    if max(abs(a0), abs(a1)) > max_exponent_coef_abs:
        return None

    # Build exponent expression: E(chi) = a0 + a1 * chi
    if abs(a0) < zero_tol and abs(a1) < zero_tol:
        # Exponent is essentially 0 → base^0 = 1, so RHS is just C
        exponent_expr: Expr = Const(0.0)
    else:
        terms: List[Expr] = []
        if abs(a0) >= zero_tol:
            terms.append(Const(a0))
        if abs(a1) >= zero_tol:
            terms.append(Const(a1) * chi_expr)

        exponent_expr = terms[0]
        for term in terms[1:]:
            exponent_expr = exponent_expr + term

    # Overall multiplicative constant C = exp(c0_rat)
    C = float(np.exp(c0_rat))
    C_expr = Const(C)

    # Final RHS: C * base_expr ** exponent_expr
    rhs = C_expr * (base_expr ** exponent_expr)
    return rhs


# ───────────────────────── main runner ───────────────────────── #

def exp_exponent_runner(
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
    max_exponent_coef_abs: float = 4.0,
    max_log_intercept_abs: float = 4.0,
    direction: str = "both",          # "upper", "lower", or "both"
    log_base: float | None = None,    # base for log(beta); None = natural log
    log_epsilon: float = 1e-9,        # clamp arg ≥ log_epsilon before log
) -> List[Conjecture]:
    """
    Exponent-type bounds via log-space LP.

    This runner searches for inequalities of the schematic form

        H ⇒ t ≥ C · base^{a0 + a1 · chi}
        H ⇒ t ≤ C · base^{a0 + a1 · chi}

    where:
      - `t` is the target invariant,
      - `base` and `chi` are drawn from `others`,
      - the exponent (a0 + a1 chi(G)) is *learned from data*,
      - C = exp(c0) is a learned multiplicative constant.

    Implementation details
    ----------------------
    We work in log-space:

        y(G) = log t(G)
        f1(G) = log base(G)
        f2(G) = chi(G) · log base(G)

    and solve LPs of the form

        upper:  y ≤ beta_1 f1 + beta_2 f2 + c0
        lower:  y ≥ beta_1 f1 + beta_2 f2 + c0

    using `_solve_lp_upper` / `_solve_lp_lower`.  Then we rewrite back to

        t ≤ exp(c0) · base^{beta_1 + beta_2 chi}

    and similarly for the lower bound.

    Parameters
    ----------
    target_col : str
        Name of the target column t.
    target_expr : Expr
        Expr referring to the target (usually to_expr(target_col)).
    others : dict[str, Expr]
        Candidate invariants; used both as bases and exponent modulators.
    hypotheses : sequence[HypothesisInfo]
        Hypotheses H with .mask, .pred, .name.
    df : DataFrame
        Invariant table.
    min_support : int
        Minimum number of valid rows under a hypothesis.
    max_denom : int
        Max denominator for rationalizing coefficients.
    coef_bound : float
        Box constraint for the LP solver: |beta_j|, |c0| ≤ coef_bound.
    zero_tol : float
        Threshold for treating coefficients as zero when building Exprs.
    max_exponent_coef_abs : float
        If |a0| or |a1| exceeds this after rationalization, drop the form.
    max_log_intercept_abs : float
        If |c0| exceeds this after rationalization, drop the form.
    direction : {"upper","lower","both"}
        Which side(s) of the inequality to generate.
    log_base : float or None
        Base for the logarithm of `base` (None = natural log).
    log_epsilon : float
        Clamp argument of logs to at least `log_epsilon` to avoid -inf/NaN.

    Returns
    -------
    list[Conjecture]
        Exponent-type conjectures for all hypotheses and (base, chi) pairs.
    """
    conjs: List[Conjecture] = []

    # If SciPy / LP is unavailable, the imported helpers will simply return None.
    if _solve_lp_lower is None or _solve_lp_upper is None:
        return conjs

    # Work in log-space for the target.
    log_t_expr = log_expr(target_expr, base=None, epsilon=log_epsilon)
    y_all = log_t_expr.eval(df).to_numpy(dtype=float)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        # For this hypothesis, we will slice y_all with a combined validity mask later.
        # Quick skip: require at least some finite y under the mask.
        if not np.isfinite(y_all[mask]).any():
            continue

        # Choose (base, chi) pairs from others.
        for base_name, base_expr in others.items():
            # Base going inside a power: we want it > 0 on the relevant rows.
            # `log_expr` will clamp via epsilon and yield NaN for truly bad values.
            log_base_expr = log_expr(base_expr, base=log_base, epsilon=log_epsilon)
            log_base_all = log_base_expr.eval(df).to_numpy(dtype=float)

            # If log_base is completely unusable under the hypothesis, skip early.
            if not np.isfinite(log_base_all[mask]).any():
                continue

            for chi_name, chi_expr in others.items():
                # You may optionally skip the trivial chi == base case; for now we allow it.
                # if chi_name == base_name:
                #     continue

                try:
                    chi_all = chi_expr.eval(df).to_numpy(dtype=float)
                except Exception:
                    continue

                # Valid rows: hypothesis holds, and all three quantities are finite.
                valid = (
                    mask
                    & np.isfinite(y_all)
                    & np.isfinite(log_base_all)
                    & np.isfinite(chi_all)
                )
                if valid.sum() < min_support:
                    continue

                y = y_all[valid]
                log_b = log_base_all[valid]
                chi_vals = chi_all[valid]

                # Build feature matrix F with columns [log_beta, chi * log_beta]
                f1 = log_b
                f2 = chi_vals * log_b
                F = np.column_stack([f1, f2])

                # Skip fully degenerate cases (no variation).
                if F.shape[0] == 0 or np.allclose(F, F[0, :], equal_nan=True):
                    continue

                # ── Upper bound: log t ≤ beta·F + c0 ─────────────────────────
                if direction in ("both", "upper"):
                    up_res = _solve_lp_upper(
                        t_arr=y,
                        F=F,
                        coef_bound=coef_bound,
                    )
                    if up_res is not None:
                        beta_up, c0_up = up_res

                        rhs_up = _build_exp_exponent_rhs(
                            base_expr=base_expr,
                            chi_expr=chi_expr,
                            beta=beta_up,
                            c0=c0_up,
                            max_denom=max_denom,
                            zero_tol=zero_tol,
                            max_exponent_coef_abs=max_exponent_coef_abs,
                            max_log_intercept_abs=max_log_intercept_abs,
                        )
                        if rhs_up is not None:
                            conjs.append(
                                Conjecture(
                                    relation=Le(target_expr, rhs_up),
                                    condition=hyp.pred,
                                    name=(
                                        f"[exp-exponent-upper] {target_col} "
                                        f"vs {base_name}^({chi_name}) under {hyp.name}"
                                    ),
                                )
                            )

                # ── Lower bound: log t ≥ beta·F + c0 ─────────────────────────
                if direction in ("both", "lower"):
                    lo_res = _solve_lp_lower(
                        t_arr=y,
                        F=F,
                        coef_bound=coef_bound,
                    )
                    if lo_res is not None:
                        beta_lo, c0_lo = lo_res

                        rhs_lo = _build_exp_exponent_rhs(
                            base_expr=base_expr,
                            chi_expr=chi_expr,
                            beta=beta_lo,
                            c0=c0_lo,
                            max_denom=max_denom,
                            zero_tol=zero_tol,
                            max_exponent_coef_abs=max_exponent_coef_abs,
                            max_log_intercept_abs=max_log_intercept_abs,
                        )
                        if rhs_lo is not None:
                            conjs.append(
                                Conjecture(
                                    relation=Ge(target_expr, rhs_lo),
                                    condition=hyp.pred,
                                    name=(
                                        f"[exp-exponent-lower] {target_col} "
                                        f"vs {base_name}^({chi_name}) under {hyp.name}"
                                    ),
                                )
                            )

    return conjs
