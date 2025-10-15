"""
Ratio-based conjecture generator (R2 form family).

Emits inequalities of the form:
    H ⇒ target >= c_min * feature
    H ⇒ target <= c_max * feature
where H is a boolean Predicate mask over the DataFrame.
"""

from __future__ import annotations
import pandas as pd
from fractions import Fraction
from typing import List, Iterator

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Ge, Le, Conjecture
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.utils.safe_generator import safe_generator

__all__ = ["ratios"]

@safe_generator
def ratios(
    df: pd.DataFrame,
    *,
    features: List[str],
    target: str,
    hypothesis: Predicate,
    max_denominator: int = 100,
    direction: str = "both",  # "both" | "upper" | "lower"
) -> Iterator[Conjecture]:
    """
    Generate conjectures target >= c_min * feature and/or target <= c_max * feature under a hypothesis.

    - Skips features that are missing or zero on all applicable rows.
    - Uses rational approximation (limit_denominator) for nicer coefficients.
    """
    mask = hypothesis.mask(df)
    if not mask.any():
        return

    t_vals = pd.to_numeric(df.loc[mask, target], errors="coerce")

    for feat in features:
        if feat not in df.columns:
            continue

        f_vals = pd.to_numeric(df.loc[mask, feat], errors="coerce")
        nonzero = (f_vals != 0) & f_vals.notna() & t_vals.notna()
        if not nonzero.any():
            continue

        ratios = t_vals[nonzero] / f_vals[nonzero]
        cmin = Fraction(float(ratios.min())).limit_denominator(max_denominator)
        cmax = Fraction(float(ratios.max())).limit_denominator(max_denominator)

        if direction in ("both", "lower"):
            yield Conjecture(
                relation=Ge(to_expr(target), Const(cmin) * to_expr(feat)),
                condition=hypothesis,
                name=f"ratio_lower_{target}_vs_{feat}",
            )
        if direction in ("both", "upper"):
            yield Conjecture(
                relation=Le(to_expr(target), Const(cmax) * to_expr(feat)),
                condition=hypothesis,
                name=f"ratio_upper_{target}_vs_{feat}",
            )
