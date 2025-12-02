# src/txgraffiti2025/graffiti4_lp_single.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, List, TYPE_CHECKING

import numpy as np
import pandas as pd
from fractions import Fraction

from scipy.optimize import linprog  # assumes SciPy is available

from txgraffiti2025.forms.utils import Expr, to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge

# Only needed for type checking, not at runtime (avoids circular import)
if TYPE_CHECKING:
    from txgraffiti2025.graffiti4 import HypothesisInfo

# If you already have this helper in another LP module, adjust the import
from txgraffiti2025.graffiti4_lp import _build_affine_expr


def _rationalize_scalar(x: float, max_denom: int) -> float:
    """
    Map a float to a nearby rational with bounded denominator.
    Used to keep coefficients/intercepts sane.
    """
    return float(Fraction(x).limit_denominator(max_denom))


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
