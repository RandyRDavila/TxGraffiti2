from __future__ import annotations
"""
Ratio-based conjecture generator (R2) with sign-aware slicing.

Emits inequalities, under hypothesis H, of the form:

- If φ has mixed sign under H, split:
    H ∧ (φ > 0) ⇒ target ≥ c_min_pos · φ
    H ∧ (φ > 0) ⇒ target ≤ c_max_pos · φ
    H ∧ (φ < 0) ⇒ target ≥ c_for_lower_neg · φ   # uses max(r)
    H ∧ (φ < 0) ⇒ target ≤ c_for_upper_neg · φ   # uses min(r)

- If φ is strictly positive (or strictly negative) under H, we can drop the extra slice
  predicate to simplify H.

Features may be Expr or column names (str).
"""

from fractions import Fraction
from math import floor, ceil
from typing import Iterator, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import GT0, LT0, Predicate
from txgraffiti2025.forms.utils import Const, to_expr, Expr
from txgraffiti2025.utils.safe_generator import safe_generator

__all__ = ["ratios"]

FeatureLike = Union[Expr, str]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ratio_bounds(t: pd.Series, f: pd.Series, *, q_clip: Optional[float] = None) -> Tuple[float, float]:
    s = pd.to_numeric(t, errors="coerce") / pd.to_numeric(f, errors="coerce")
    s = s.replace([pd.NA, pd.NaT], np.nan).dropna()
    if len(s) == 0:
        raise ValueError("No finite ratios available.")
    if q_clip is None:
        return float(s.min()), float(s.max())
    q = float(q_clip)
    return float(s.quantile(q)), float(s.quantile(1.0 - q))


def _approx_one_sided(x: float, max_den: Optional[int], *, direction: str) -> float:
    if max_den is None:
        return float(x)

    if direction == "down":  # r <= x
        best = -float("inf")
        for q in range(1, max_den + 1):
            p = floor(x * q + 1e-15)
            cand = p / q
            if cand <= x and cand > best:
                best = cand
        return float(x) if best == -float("inf") else float(best)

    if direction == "up":  # r >= x
        best = float("inf")
        for q in range(1, max_den + 1):
            p = ceil(x * q - 1e-15)
            cand = p / q
            if cand >= x and cand < best:
                best = cand
        return float(x) if best == float("inf") else float(best)

    raise ValueError("direction must be 'down' or 'up'")


def _const_from_rational(x: float) -> Const:
    try:
        frac = Fraction(x).limit_denominator()
        if float(frac) == float(x):
            return Const(frac)
    except Exception:
        pass
    return Const(float(x))


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

@safe_generator
def ratios(
    df: pd.DataFrame,
    *,
    features: Sequence[FeatureLike],
    target: str,
    hypothesis: Predicate,
    max_denominator: int = 100,
    direction: str = "both",          # "both" | "upper" | "lower"
    q_clip: Optional[float] = None,   # e.g., 0.01 for robust bounds
    min_support: int = 2,             # rows per slice
    simplify_condition: bool = True,  # drop GT0/LT0 if φ uniform in sign
) -> Iterator[Conjecture]:
    # Base applicability
    H = hypothesis.mask(df).reindex(df.index, fill_value=False).astype(bool)
    if not H.any():
        return

    t_all = pd.to_numeric(df[target], errors="coerce")

    for φ_like in features:
        φ = to_expr(φ_like) if not isinstance(φ_like, Expr) else φ_like
        φ_vals = pd.to_numeric(φ.eval(df), errors="coerce")
        φ_name = repr(φ)

        # sign slices
        pos_mask = H & φ_vals.gt(0) & φ_vals.notna() & t_all.notna()
        neg_mask = H & φ_vals.lt(0) & φ_vals.notna() & t_all.notna()
        zer_mask = H & φ_vals.eq(0) & φ_vals.notna() & t_all.notna()

        pos_only = pos_mask.any() and not neg_mask.any() and not zer_mask.any()
        neg_only = neg_mask.any() and not pos_mask.any() and not zer_mask.any()

        # Positive slice
        if pos_mask.sum() >= min_support:
            rmin_pos, rmax_pos = _ratio_bounds(t_all[pos_mask], φ_vals[pos_mask], q_clip=q_clip)
            cmin_pos = _const_from_rational(_approx_one_sided(rmin_pos, max_denominator, direction="down"))
            cmax_pos = _const_from_rational(_approx_one_sided(rmax_pos, max_denominator, direction="up"))
            cond_pos = hypothesis if (simplify_condition and pos_only) else (hypothesis & GT0(φ))
            suffix = "" if (simplify_condition and pos_only) else "_pos"

            if direction in ("both", "lower"):
                yield Conjecture(
                    relation=Ge(to_expr(target), cmin_pos * φ),
                    condition=cond_pos,
                    name=f"ratio_lower_{target}_vs_{φ_name}{suffix}",
                )
            if direction in ("both", "upper"):
                yield Conjecture(
                    relation=Le(to_expr(target), cmax_pos * φ),
                    condition=cond_pos,
                    name=f"ratio_upper_{target}_vs_{φ_name}{suffix}",
                )

        # Negative slice
        if neg_mask.sum() >= min_support:
            rmin_neg, rmax_neg = _ratio_bounds(t_all[neg_mask], φ_vals[neg_mask], q_clip=q_clip)
            # For φ<0, c·φ decreases with c:
            #   lower bound (Ge) uses rmax (rounded UP)
            #   upper bound (Le) uses rmin (rounded DOWN)
            c_for_lower = _const_from_rational(_approx_one_sided(rmax_neg, max_denominator, direction="up"))
            c_for_upper = _const_from_rational(_approx_one_sided(rmin_neg, max_denominator, direction="down"))
            cond_neg = hypothesis if (simplify_condition and neg_only) else (hypothesis & LT0(φ))
            suffix = "" if (simplify_condition and neg_only) else "_neg"

            if direction in ("both", "lower"):
                yield Conjecture(
                    relation=Ge(to_expr(target), c_for_lower * φ),
                    condition=cond_neg,
                    name=f"ratio_lower_{target}_vs_{φ_name}{suffix}",
                )
            if direction in ("both", "upper"):
                yield Conjecture(
                    relation=Le(to_expr(target), c_for_upper * φ),
                    condition=cond_neg,
                    name=f"ratio_upper_{target}_vs_{φ_name}{suffix}",
                )
