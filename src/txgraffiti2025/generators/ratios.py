"""
Ratio-based conjecture generator (R2 family) with sign-aware bounds and
automatic condition simplification.

Emits inequalities of the form, under a hypothesis H:

- If feature has mixed sign under H, split:
    H ∧ (feature > 0) ⇒ target ≥ c_min_pos * feature
    H ∧ (feature > 0) ⇒ target ≤ c_max_pos * feature
    H ∧ (feature < 0) ⇒ target ≥ c_for_lower_neg * feature   # uses max(r)
    H ∧ (feature < 0) ⇒ target ≤ c_for_upper_neg * feature   # uses min(r)

- If feature is strictly positive under H (and never zero), simplify to:
    H ⇒ target ≥ c_min * feature
    H ⇒ target ≤ c_max * feature

Likewise for strictly negative under H.

Parameters
----------
df : pd.DataFrame
features : list[str]
target : str
hypothesis : Predicate
max_denominator : int, default 100
    Rationalize coefficients via Fraction.limit_denominator. Use None for raw floats.
direction : {"both","upper","lower"}, default "both"
q_clip : float in (0, 0.5), optional
    Use quantiles [q_clip, 1-q_clip] instead of strict min/max (outlier-robust).
min_support : int, default 2
    Minimum rows required in a sign-slice to emit conjectures.
simplify_condition : bool, default True
    If True, drop redundant sign predicates from the condition when the feature
    has uniform sign under H and no zeros.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.predicates import Where
>>> df = pd.DataFrame({"H":[True,True,True], "x":[1,2,3], "y":[2,4.1,6.2]})
>>> H = Where(lambda d: d["H"])
>>> list(ratios(df, features=["x"], target="y", hypothesis=H, max_denominator=50))
[Conjecture(... 'y' >= 2 * 'x' ... | Where(...)),
 Conjecture(... 'y' <= 31/5 * 'x' ... | Where(...))]  # condition simplified (no (x>0))
"""

from __future__ import annotations
from fractions import Fraction
from typing import Iterator, List, Optional, Tuple

import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import GT0, LT0, Predicate
from txgraffiti2025.forms.utils import Const, to_expr
from txgraffiti2025.utils.safe_generator import safe_generator



def _approx_coeff(x: float, max_den: int | None):
    if max_den is None:
        return Const(float(x))
    return Const(Fraction(float(x)).limit_denominator(max_den))


def _ratio_bounds(
    t: pd.Series,
    f: pd.Series,
    *,
    q_clip: Optional[float] = None,
) -> Tuple[float, float]:
    s = pd.to_numeric(t, errors="coerce") / pd.to_numeric(f, errors="coerce")
    s = s.replace([pd.NA, pd.NaT], pd.NA).dropna()
    if len(s) == 0:
        raise ValueError("No finite ratios available.")
    if q_clip is None:
        return float(s.min()), float(s.max())
    q = float(q_clip)
    return float(s.quantile(q)), float(s.quantile(1.0 - q))


@safe_generator
def ratios(
    df: pd.DataFrame,
    *,
    features: List[str],
    target: str,
    hypothesis: Predicate,
    max_denominator: int = 100,
    direction: str = "both",          # "both" | "upper" | "lower"
    q_clip: Optional[float] = None,   # e.g., 0.01 for robust bounds
    min_support: int = 2,             # rows per slice
    simplify_condition: bool = True,  # drop redundant (feat>0)/(feat<0) if uniform
) -> Iterator[Conjecture]:
    # base applicability
    Hmask = hypothesis.mask(df).reindex(df.index, fill_value=False)
    if not Hmask.any():
        return

    t_all = pd.to_numeric(df[target], errors="coerce")

    for feat in features:
        if feat not in df.columns:
            continue
        f_all = pd.to_numeric(df[feat], errors="coerce")

        # sign slices
        pos_mask = Hmask & f_all.gt(0) & t_all.notna() & f_all.notna()
        neg_mask = Hmask & f_all.lt(0) & t_all.notna() & f_all.notna()
        zer_mask = Hmask & f_all.eq(0) & t_all.notna() & f_all.notna()

        # If requested, simplify condition when sign is uniform (and no zeros)
        pos_only = pos_mask.any() and not neg_mask.any() and not zer_mask.any()
        neg_only = neg_mask.any() and not pos_mask.any() and not zer_mask.any()

        # positive slice
        if pos_mask.sum() >= min_support:
            rmin_pos, rmax_pos = _ratio_bounds(t_all[pos_mask], f_all[pos_mask], q_clip=q_clip)
            cmin_pos = _approx_coeff(rmin_pos, max_denominator)
            cmax_pos = _approx_coeff(rmax_pos, max_denominator)
            cond_pos = hypothesis if (simplify_condition and pos_only) else (hypothesis & GT0(feat))
            suffix = "" if (simplify_condition and pos_only) else "_pos"

            if direction in ("both", "lower"):
                yield Conjecture(
                    relation=Ge(to_expr(target), cmin_pos * to_expr(feat)),
                    condition=cond_pos,
                    name=f"ratio_lower_{target}_vs_{feat}{suffix}",
                )
            if direction in ("both", "upper"):
                yield Conjecture(
                    relation=Le(to_expr(target), cmax_pos * to_expr(feat)),
                    condition=cond_pos,
                    name=f"ratio_upper_{target}_vs_{feat}{suffix}",
                )

        # negative slice
        if neg_mask.sum() >= min_support:
            rmin_neg, rmax_neg = _ratio_bounds(t_all[neg_mask], f_all[neg_mask], q_clip=q_clip)
            # For f<0: lower uses max(r), upper uses min(r)
            c_for_lower = _approx_coeff(rmax_neg, max_denominator)
            c_for_upper = _approx_coeff(rmin_neg, max_denominator)
            cond_neg = hypothesis if (simplify_condition and neg_only) else (hypothesis & LT0(feat))
            suffix = "" if (simplify_condition and neg_only) else "_neg"

            if direction in ("both", "lower"):
                yield Conjecture(
                    relation=Ge(to_expr(target), c_for_lower * to_expr(feat)),
                    condition=cond_neg,
                    name=f"ratio_lower_{target}_vs_{feat}{suffix}",
                )
            if direction in ("both", "upper"):
                yield Conjecture(
                    relation=Le(to_expr(target), c_for_upper * to_expr(feat)),
                    condition=cond_neg,
                    name=f"ratio_upper_{target}_vs_{feat}{suffix}",
                )
