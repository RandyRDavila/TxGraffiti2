# src/txgraffiti/graffiti3/runners/ratio.py

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.exprs import Expr
from txgraffiti.graffiti3.relations import (
    Conjecture,
    Ge,
    Le,
)
from txgraffiti.graffiti3.types import HypothesisInfo
from txgraffiti.graffiti3.runners.utils import _build_affine_expr


def ratio_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence[HypothesisInfo],
    df: pd.DataFrame,
    min_support: int = 5,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_denom: int = 20,
) -> List[Conjecture]:
    """
    Stage-1 generator: for each hypothesis h and each 'other' invariant x,
    compute ratios r_i = target_i / x_i on valid rows and set

        c_min = min r_i,  c_max = max r_i

    to get

        h ⇒ target ≥ c_min * x
        h ⇒ target ≤ c_max * x,

    with coefficients cleaned via the same affine builder as the LP runner.
    This ensures that, e.g.,

        target ≤ 1 · annihilation_number

    and

        target ≤ annihilation_number

    normalize to the *same* Expr and dedup works correctly.
    """
    conjs: List[Conjecture] = []
    target_vals = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        H = np.asarray(hyp.mask, dtype=bool)

        for other_name, other_expr in others.items():
            try:
                other_vals = other_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            valid = (
                H
                & np.isfinite(target_vals)
                & np.isfinite(other_vals)
                & (other_vals != 0.0)
            )
            if valid.sum() < min_support:
                continue

            r = target_vals[valid] / other_vals[valid]

            # raw extrema
            c_min_raw = float(np.min(r))
            c_max_raw = float(np.max(r))

            # rationalize a bit (same style as LP)
            c_min_rat = Fraction(c_min_raw).limit_denominator(max_denom)
            c_max_rat = Fraction(c_max_raw).limit_denominator(max_denom)
            c_min = float(c_min_rat)
            c_max = float(c_max_rat)

            # build RHS = c * x via the shared affine helper
            # (intercept is 0.0; we use same zero_tol+max_coef_abs logic)
            # rhs_min = _build_affine_expr(
            #     const_val=0.0,
            #     coefs=[c_min],
            #     feats=[other_expr],
            #     zero_tol=zero_tol,
            #     max_coef_abs=max_coef_abs,
            #     max_intercept_abs=0.0,  # 0 intercept, so this is safe
            # )
            # rhs_max = _build_affine_expr(
            #     const_val=0.0,
            #     coefs=[c_max],
            #     feats=[other_expr],
            #     zero_tol=zero_tol,
            #     max_coef_abs=max_coef_abs,
            #     max_intercept_abs=0.0,
            # )
            # ratio_runner
            rhs_min = _build_affine_expr(
                const_val=0.0,
                coefs=[c_min],
                feats=[other_expr],
                zero_tol=zero_tol,
                max_coef_abs=max_coef_abs,
                max_intercept_abs=0.0,
                max_denom=max_denom,
            )
            rhs_max = _build_affine_expr(
                const_val=0.0,
                coefs=[c_max],
                feats=[other_expr],
                zero_tol=zero_tol,
                max_coef_abs=max_coef_abs,
                max_intercept_abs=0.0,
                max_denom=max_denom,
            )

            # If coefficient is too small/too large, _build_affine_expr returns None
            if rhs_min is not None:
                rel_ge = Ge(left=target_expr, right=rhs_min)
                c_ge = Conjecture(
                    relation=rel_ge,
                    condition=hyp.pred,
                    name=f"[ratio-min] {target_col} vs {other_name} under {hyp.name}",
                )
                c_ge.target_name = target_col
                # c_le.target_name = target_col
                conjs.append(c_ge)


            if rhs_max is not None:
                rel_le = Le(left=target_expr, right=rhs_max)
                c_le = Conjecture(
                    relation=rel_le,
                    condition=hyp.pred,
                    name=f"[ratio-max] {target_col} vs {other_name} under {hyp.name}",
                )
                # c_ge.target_name = target_col
                c_le.target_name = target_col
                conjs.append(c_le)

    return conjs
