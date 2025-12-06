# src/txgraffiti/graffiti3/runners/ratio_lift.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Sequence, Optional

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.exprs import Expr
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le

from txgraffiti.graffiti3.types import HypothesisInfo  # has .name, .mask, .pred


@dataclass
class SlopeBoundSeed:
    """
    A simple one-variable bound suitable for RatioLift:

        hyp_name:   H_s
        direction:  "upper" for y <= c x_j, "lower" for y >= c x_j
        slope:      c
        intercept:  b (should be ~0 for RatioLift)
    """
    target_col: str
    target_expr: Expr
    hyp: HypothesisInfo
    base_name: str
    base_expr: Expr
    direction: str               # "upper" or "lower"
    slope: float                 # c
    intercept: float             # b (should be ~0)


# continue in src/txgraffiti/graffiti3/runners/ratio_lift.py

def _iter_ratio_templates(
    f_vals: np.ndarray,
    g_vals: np.ndarray,
    f_expr: Expr,
    g_expr: Expr,
    *,
    shifts: Sequence[int],
) -> List[Tuple[np.ndarray, Expr, str]]:
    """
    Numeric + symbolic versions of the templates listed in the LaTeX:

      f/g, f/(g+a), (f+a)/g,
      1/g, 1/(g+a), 1/(f+g),
      f/(f+g), (f+g)/g, (f+a)/(g+b).

    Returns [(E_vals, E_expr, name_str), ...].
    """
    out: List[Tuple[np.ndarray, Expr, str]] = []

    # Helper: safe divide
    def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        arr = np.full_like(num, np.nan, dtype=float)
        mask = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
        arr[mask] = num[mask] / den[mask]
        return arr

    # f/g
    E_fg = safe_div(f_vals, g_vals)
    out.append((E_fg, f_expr / g_expr, "f/g"))

    # f/(g + a), (f + a)/g, 1/g, 1/(g + a), 1/(f+g), f/(f+g), (f+g)/g, (f+a)/(g+b)
    for a in shifts:
        a_float = float(a)
        g_plus_a = g_vals + a_float

        # f/(g+a)
        E = safe_div(f_vals, g_plus_a)
        out.append((E, f_expr / (g_expr + a_float), f"f/(g+{a})"))

        # (f+a)/g
        f_plus_a = f_vals + a_float
        E2 = safe_div(f_plus_a, g_vals)
        out.append((E2, (f_expr + a_float) / g_expr, f"(f+{a})/g"))

        # 1/(g+a)
        E3 = safe_div(np.ones_like(g_vals), g_plus_a)
        out.append((E3, 1.0 / (g_expr + a_float), f"1/(g+{a})"))

    # 1/g
    E_1g = safe_div(np.ones_like(g_vals), g_vals)
    out.append((E_1g, 1.0 / g_expr, "1/g"))

    # 1/(f+g)
    f_plus_g = f_vals + g_vals
    E_1fg = safe_div(np.ones_like(f_vals), f_plus_g)
    out.append((E_1fg, 1.0 / (f_expr + g_expr), "1/(f+g)"))

    # f/(f+g)
    E_ffg = safe_div(f_vals, f_plus_g)
    out.append((E_ffg, f_expr / (f_expr + g_expr), "f/(f+g)"))

    # (f+g)/g
    E_fg_over_g = safe_div(f_plus_g, g_vals)
    out.append((E_fg_over_g, (f_expr + g_expr) / g_expr, "(f+g)/g"))

    # (f+a)/(g+b)
    for a in shifts:
        for b in shifts:
            fa = f_vals + float(a)
            gb = g_vals + float(b)
            E_ab = safe_div(fa, gb)
            out.append(
                (E_ab,
                 (f_expr + float(a)) / (g_expr + float(b)),
                 f"(f+{a})/(g+{b})")
            )

    return out


def _discover_coefficient_ratio_expressions(
    *,
    df: pd.DataFrame,
    hyp_mask: np.ndarray,
    slope: float,
    numeric_exprs: Dict[str, Expr],
    const_tol: float = 1e-3,
    coeff_tol: float = 1e-3,
    shifts: Sequence[int] = (-2, -1, 0, 1, 2),
) -> List[Tuple[Expr, str]]:
    """
    Search over ratio templates E(f,g,...) built from numeric columns, as in the LaTeX.

    Returns a list of (E_expr, name_str) whose numeric values E_i on H_s satisfy:
      max(E_i) - min(E_i) <= const_tol
      |mean(E_i) - slope| <= coeff_tol.
    """
    mask = np.asarray(hyp_mask, dtype=bool)

    # pre-extract numeric arrays
    col_vals: Dict[str, np.ndarray] = {}
    for name in numeric_exprs.keys():
        try:
            col_vals[name] = numeric_exprs[name].eval(df).to_numpy(dtype=float)
        except Exception:
            continue

    good: List[Tuple[Expr, str]] = []

    for f_name, f_expr in numeric_exprs.items():
        if f_name not in col_vals:
            continue
        f_all = col_vals[f_name]
        f = f_all[mask]

        for g_name, g_expr in numeric_exprs.items():
            if g_name not in col_vals:
                continue
            g_all = col_vals[g_name]
            g = g_all[mask]

            # Generate all template variants for this (f,g) pair
            for E_vals, E_expr, name_str in _iter_ratio_templates(
                f, g, f_expr, g_expr, shifts=shifts
            ):
                finite = np.isfinite(E_vals)
                if finite.sum() == 0:
                    continue

                E_vals_h = E_vals[finite]
                max_diff = float(E_vals_h.max() - E_vals_h.min())
                if max_diff > const_tol:
                    continue

                mean_E = float(E_vals_h.mean())
                if abs(mean_E - slope) > coeff_tol:
                    continue

                good.append((E_expr, f"{name_str}({f_name},{g_name})"))

    return good

def generalize_slope_bound_via_ratios(
    seed: SlopeBoundSeed,
    *,
    parent_hyp: HypothesisInfo,
    df: pd.DataFrame,
    numeric_exprs: Dict[str, Expr],
    const_tol: float = 1e-3,
    coeff_tol: float = 1e-3,
    inequality_tol: float = 1e-8,
) -> List[Conjecture]:
    """
    Implement the LaTeX procedure for a single seed:

      H_s ⇒ y ≤ c x_j  (or ≥)
      parent H_0  one step up in hypothesis lattice.

    Returns a list of lifted Conjectures on H_0.
    """
    # We only attempt RatioLift if intercept ~ 0 and it's truly one-variable.
    if abs(seed.intercept) > coeff_tol:
        return []

    Hs_mask = np.asarray(seed.hyp.mask, dtype=bool)
    H0_mask = np.asarray(parent_hyp.mask, dtype=bool)

    # Pre-extract numeric values
    y_all = df[seed.target_col].to_numpy(dtype=float)
    x_all = seed.base_expr.eval(df).to_numpy(dtype=float)

    # Candidate coefficient expressions E
    E_candidates = _discover_coefficient_ratio_expressions(
        df=df,
        hyp_mask=Hs_mask,
        slope=seed.slope,
        numeric_exprs=numeric_exprs,
        const_tol=const_tol,
        coeff_tol=coeff_tol,
    )

    lifted: List[Conjecture] = []

    for E_expr, E_name in E_candidates:
        try:
            E_vals_all = E_expr.eval(df).to_numpy(dtype=float)
        except Exception:
            continue

        # y_hat = E * x_j
        y_hat = E_vals_all * x_all

        valid = (
            H0_mask
            & np.isfinite(y_all)
            & np.isfinite(x_all)
            & np.isfinite(y_hat)
        )
        if not valid.any():
            continue

        y_sub = y_all[valid]
        y_hat_sub = y_hat[valid]

        if seed.direction == "upper":
            ok = np.all(y_sub <= y_hat_sub + inequality_tol)
        else:  # "lower"
            ok = np.all(y_sub + inequality_tol >= y_hat_sub)

        if not ok:
            continue

        # touch/support on H0
        support = int(valid.sum())
        touches = int(
            np.sum(np.isclose(y_sub, y_hat_sub, atol=inequality_tol))
        )

        # symbolic RHS = E * base_expr
        rhs_expr = E_expr * seed.base_expr

        if seed.direction == "upper":
            rel = Le(seed.target_expr, rhs_expr)
            tag = "ratio-lift-upper"
        else:
            rel = Ge(seed.target_expr, rhs_expr)
            tag = "ratio-lift-lower"

        conj = Conjecture(
            relation=rel,
            condition=parent_hyp.pred,
            name=(
                f"[{tag}] {seed.target_col} vs {seed.base_name} "
                f"under {parent_hyp.name} (E = {E_name})"
            ),
        )
        # Optional: attach metadata a la LaTeX
        conj.target_name = seed.target_col
        conj.source = "RatioLift"
        conj.touch = touches
        conj.support = support

        lifted.append(conj)

    return lifted
