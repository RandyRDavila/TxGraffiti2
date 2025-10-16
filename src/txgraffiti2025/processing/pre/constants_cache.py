"""
Precompute & cache constant ratios for many boolean hypotheses.

For each hypothesis H in a given list, we scan numeric columns and record every
ratio (A + a)/(B + b) that is (approximately) constant on rows where H holds.

Why?
- You can later look up constants instantly when processing many conjectures.
- Lets you propose generalizations: replace constant coefficients by structural
  expressions (ratios) and test on more general hypotheses.

Returned cache stores, for each hypothesis:
- a compact mask key,
- a list of ConstantRatio records (formula, value, support, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from fractions import Fraction

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.utils import Expr, to_expr, Const


# ---------------------------
# Utilities
# ---------------------------

def _mask_for(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
    if pred is None:
        return pd.Series(True, index=df.index)
    m = pred.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)

def _mask_key(mask: pd.Series) -> str:
    # Compact, index-order dependent fingerprint
    arr = mask.to_numpy(dtype=np.uint8, copy=False)
    return arr.tobytes().hex()

def _is_numeric_series(s: pd.Series) -> bool:
    return is_numeric_dtype(s) and not is_bool_dtype(s)

def _finite(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _ratio_series(num: pd.Series, den: pd.Series) -> pd.Series:
    n = _finite(num)
    d = _finite(den)
    d = d.replace(0.0, np.nan)
    return n / d

def _is_constant(
    ser: pd.Series, *, atol: float, rtol: float, min_support: int
) -> Tuple[bool, float, int]:
    v = _finite(ser).dropna()
    supp = int(v.size)
    if supp < min_support:
        return (False, np.nan, supp)
    vmax, vmin = float(v.max()), float(v.min())
    span = vmax - vmin
    tol = float(atol + rtol * max(1.0, np.abs(v).max()))
    if span <= tol:
        return (True, float(v.median()), supp)
    return (False, np.nan, supp)

def _rationalize(x: float, max_den: Optional[int]) -> Fraction | float:
    if max_den is None:
        return float(x)
    return Fraction(x).limit_denominator(max_den)


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class ConstantRatio:
    numerator: str
    denominator: str
    shift_num: int
    shift_den: int
    value_float: float
    value_display: str   # "1", "1/2", "2/3", or decimal
    support: int
    expr: Expr           # Expr for (num+shift_num)/(den+shift_den)

@dataclass
class HypothesisConstants:
    hypothesis: Optional[Predicate]
    mask_key: str
    constants: List[ConstantRatio]

@dataclass
class ConstantsCache:
    df_index_fingerprint: Tuple[int, ...]
    # map repr(hypothesis) -> mask_key
    hyp_to_key: Dict[str, str]
    # map mask_key -> HypothesisConstants
    key_to_constants: Dict[str, HypothesisConstants]

    def constants_for(self, hypothesis: Optional[Predicate]) -> List[ConstantRatio]:
        key = self.hyp_to_key.get(repr(hypothesis), None)
        if key is None:
            return []
        rec = self.key_to_constants.get(key)
        return rec.constants if rec is not None else []


# ---------------------------
# Main precompute API
# ---------------------------

def precompute_constant_ratios(
    df: pd.DataFrame,
    hypotheses: Sequence[Optional[Predicate]],
    *,
    numeric_cols: Optional[Sequence[str]] = None,
    shifts: Sequence[int] = (-2, -1, 0, 1, 2),
    atol: float = 1e-9,
    rtol: float = 1e-9,
    min_support: int = 8,
    max_denominator: Optional[int] = 50,
) -> ConstantsCache:
    """
    Precompute constant (A+a)/(B+b) on each hypothesis in `hypotheses`.

    Parameters
    ----------
    df : DataFrame
    hypotheses : sequence of Predicate or None
        The boolean classes to analyze (e.g., output of enumerate_boolean_hypotheses).
    numeric_cols : list[str], optional
        Which numeric columns to consider. Default: all numeric non-bool cols.
    shifts : iterable[int]
        Small integer shifts to try on numerator/denominator.
    atol, rtol : float
        Tolerances for constancy check.
    min_support : int
        Min usable rows under hypothesis for a constant to count.
    max_denominator : int or None
        Rationalize constants for display.

    Returns
    -------
    ConstantsCache
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if _is_numeric_series(df[c])]

    key_to_constants: Dict[str, HypothesisConstants] = {}
    hyp_to_key: Dict[str, str] = {}

    for hyp in hypotheses:
        mask = _mask_for(df, hyp)
        key = _mask_key(mask)
        hyp_to_key[repr(hyp)] = key

        data = df.loc[mask]
        consts: List[ConstantRatio] = []

        if not mask.any():
            key_to_constants[key] = HypothesisConstants(hypothesis=hyp, mask_key=key, constants=consts)
            continue

        # cache series per numeric col for speed
        col_data = {c: pd.to_numeric(data[c], errors="coerce") for c in numeric_cols}

        for i, num in enumerate(numeric_cols):
            sN_base = col_data[num]
            for j, den in enumerate(numeric_cols):
                if i == j:
                    continue
                sD_base = col_data[den]
                for a in shifts:
                    for b in shifts:
                        sN = sN_base + float(a)
                        sD = sD_base + float(b)
                        rr = _ratio_series(sN, sD)
                        ok, value, supp = _is_constant(rr, atol=atol, rtol=rtol, min_support=min_support)
                        if not ok:
                            continue
                        # display value
                        disp = _rationalize(value, max_denominator)
                        if isinstance(disp, Fraction):
                            value_display = f"{disp.numerator}" if disp.denominator == 1 else f"{disp.numerator}/{disp.denominator}"
                        else:
                            value_display = str(int(disp)) if float(disp).is_integer() else f"{disp}"

                        num_expr = to_expr(num) if a == 0 else (to_expr(num) + Const(a))
                        den_expr = to_expr(den) if b == 0 else (to_expr(den) + Const(b))
                        consts.append(ConstantRatio(
                            numerator=num,
                            denominator=den,
                            shift_num=int(a),
                            shift_den=int(b),
                            value_float=float(value),
                            value_display=value_display,
                            support=int(supp),
                            expr=num_expr / den_expr,
                        ))

        # sort by (support desc, short shifts, lexicographic)
        consts.sort(key=lambda r: (-r.support, abs(r.shift_num)+abs(r.shift_den), r.numerator, r.denominator, r.value_float))
        key_to_constants[key] = HypothesisConstants(hypothesis=hyp, mask_key=key, constants=consts)

    cache = ConstantsCache(
        df_index_fingerprint=tuple(range(len(df))),  # basic guard; assumes fixed order
        hyp_to_key=hyp_to_key,
        key_to_constants=key_to_constants,
    )
    return cache


# ---------------------------
# Query helpers
# ---------------------------

def constants_matching_coeff(
    cache: ConstantsCache,
    hypothesis: Optional[Predicate],
    coeff: float,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> List[ConstantRatio]:
    """Return constants under `hypothesis` whose numeric value ~== coeff."""
    consts = cache.constants_for(hypothesis)
    return [cr for cr in consts if np.isclose(cr.value_float, float(coeff), atol=atol, rtol=rtol)]


from typing import Sequence, Optional, List, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from txgraffiti2025.forms.predicates import Predicate, Where

def _detect_boolean_columns(
    df: pd.DataFrame,
    *,
    include_binary_numeric: bool = True,
    max_nan_frac: float = 0.2,
) -> List[str]:
    """
    Heuristic boolean column detector:
      - dtype bool, OR
      - numeric with unique subset of {0,1} (if include_binary_numeric)
    """
    cols: List[str] = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        if is_bool_dtype(s):
            cols.append(c); continue
        if include_binary_numeric and is_numeric_dtype(s):
            s2 = pd.to_numeric(s, errors="coerce")
            if s2.isna().mean() <= max_nan_frac:
                vals = set(pd.unique(s2.dropna()))
                if vals.issubset({0, 1}):
                    cols.append(c)
    return cols

def _pred_from_bool_col(col: str) -> Predicate:
    # Vectorized Where: returns boolean Series
    return Where(lambda d, col=col: d[col].astype(bool))

def build_pairwise_hypotheses(
    df: pd.DataFrame,
    base: Optional[Predicate],
    *,
    boolean_cols: Optional[Sequence[str]] = None,
) -> List[Optional[Predicate]]:
    """
    Return [base] + [base ∧ P_b] for each boolean column b, filtering out empties/duplicates.
    """
    if boolean_cols is None:
        boolean_cols = _detect_boolean_columns(df)

    H: List[Optional[Predicate]] = []
    # base mask (or TRUE)
    base_mask = (base.mask(df) if base is not None else pd.Series(True, index=df.index)).astype(bool)
    if base_mask.any():
        H.append(base)

    for b in boolean_cols:
        P = _pred_from_bool_col(b)
        mask = base_mask & P.mask(df).reindex(df.index, fill_value=False).astype(bool)
        if not mask.any():
            continue
        # avoid duplicates equal to base itself
        if np.array_equal(mask.to_numpy(), base_mask.to_numpy()):
            continue
        H.append(base & P if base is not None else P)

    return H

def precompute_constant_ratios_pairs(
    df: pd.DataFrame,
    base: Optional[Predicate],
    *,
    boolean_cols: Optional[Sequence[str]] = None,
    # pass-through to precompute_constant_ratios
    numeric_cols: Optional[Sequence[str]] = None,
    shifts: Sequence[int] = (-2, -1, 0, 1, 2),
    atol: float = 1e-9,
    rtol: float = 1e-9,
    min_support: int = 8,
    max_denominator: Optional[int] = 50,
):
    """
    Fast default: precompute constants for { base } ∪ { base ∧ each_boolean } only.
    """
    H_pairs = build_pairwise_hypotheses(df, base, boolean_cols=boolean_cols)
    return precompute_constant_ratios(
        df,
        hypotheses=H_pairs,
        numeric_cols=numeric_cols,
        shifts=shifts,
        atol=atol,
        rtol=rtol,
        min_support=min_support,
        max_denominator=max_denominator,
    )
