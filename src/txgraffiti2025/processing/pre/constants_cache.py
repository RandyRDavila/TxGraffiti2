# # src/txgraffiti2025/processing/pre/constants_cache.py

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple
# from fractions import Fraction

# import numpy as np
# import pandas as pd
# from pandas.api.types import is_numeric_dtype, is_bool_dtype

# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.forms.utils import Expr, to_expr, Const

# # ───── utilities ─────

# def _mask_for(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
#     if pred is None:
#         return pd.Series(True, index=df.index)
#     return pred.mask(df).reindex(df.index, fill_value=False).astype(bool)

# def _mask_key(mask: pd.Series) -> str:
#     return mask.to_numpy(dtype=np.uint8, copy=False).tobytes().hex()

# def _is_numeric_series(s: pd.Series) -> bool:
#     return is_numeric_dtype(s) and not is_bool_dtype(s)

# def _finite(x: pd.Series) -> pd.Series:
#     return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)

# def _ratio_series(num: pd.Series, den: pd.Series) -> pd.Series:
#     n = _finite(num); d = _finite(den)
#     d = d.replace(0.0, np.nan)
#     return n / d

# def _is_constant(
#     ser: pd.Series, *, atol: float, rtol: float, min_support: int
# ) -> Tuple[bool, float, int]:
#     v = _finite(ser).dropna()
#     supp = int(v.size)
#     if supp < min_support:
#         return (False, np.nan, supp)
#     vmax, vmin = float(v.max()), float(v.min())
#     span = vmax - vmin
#     tol = float(atol + rtol * max(1.0, np.abs(v).max()))
#     if span <= tol:
#         return (True, float(v.median()), supp)
#     return (False, np.nan, supp)

# def _rationalize(x: float, max_den: Optional[int]) -> Fraction | float:
#     if max_den is None:
#         return float(x)
#     return Fraction(x).limit_denominator(max_den)

# # ───── data structures ─────

# @dataclass
# class ConstantRatio:
#     numerator: str
#     denominator: str
#     shift_num: int
#     shift_den: int
#     value_float: float
#     value_display: str
#     support: int
#     expr: Expr

# @dataclass
# class HypothesisConstants:
#     hypothesis: Optional[Predicate]
#     mask_key: str
#     constants: List[ConstantRatio]

# @dataclass
# class ConstantsCache:
#     df_index_fingerprint: Tuple[int, ...]
#     hyp_to_key: Dict[str, str]
#     key_to_constants: Dict[str, HypothesisConstants]

#     def constants_for(self, hypothesis: Optional[Predicate]) -> List[ConstantRatio]:
#         key = self.hyp_to_key.get(repr(hypothesis), None)
#         if key is None:
#             return []
#         rec = self.key_to_constants.get(key)
#         return rec.constants if rec is not None else []

# def constants_matching_coeff(
#     cache: "ConstantsCache",
#     hypothesis: Optional["Predicate"],
#     coeff: float,
#     *,
#     atol: float = 1e-9,
#     rtol: float = 1e-9,
# ) -> List["ConstantRatio"]:
#     """
#     Return precomputed ConstantRatio records under `hypothesis`
#     whose numeric value is approximately equal to `coeff`.
#     """
#     consts = cache.constants_for(hypothesis)
#     target = float(coeff)
#     return [cr for cr in consts if np.isclose(cr.value_float, target, atol=atol, rtol=rtol)]

# # ───── main precompute ─────

# def precompute_constant_ratios(
#     df: pd.DataFrame,
#     hypotheses: Sequence[Optional[Predicate]],
#     *,
#     numeric_cols: Optional[Sequence[str]] = None,
#     shifts: Sequence[int] = (-2, -1, 0, 1, 2),
#     atol: float = 1e-9,
#     rtol: float = 1e-9,
#     min_support: int = 8,
#     max_denominator: Optional[int] = 50,
#     skip_identity: bool = True,      # ← allow shifted same-column, skip (N+a)/(N+a)
# ) -> ConstantsCache:
#     if numeric_cols is None:
#         numeric_cols = [c for c in df.columns if _is_numeric_series(df[c])]

#     key_to_constants: Dict[str, HypothesisConstants] = {}
#     hyp_to_key: Dict[str, str] = {}

#     for hyp in hypotheses:
#         mask = _mask_for(df, hyp)
#         key = _mask_key(mask)
#         hyp_to_key[repr(hyp)] = key

#         data = df.loc[mask]
#         consts: List[ConstantRatio] = []

#         if not mask.any():
#             key_to_constants[key] = HypothesisConstants(hypothesis=hyp, mask_key=key, constants=consts)
#             continue

#         col_data = {c: pd.to_numeric(data[c], errors="coerce") for c in numeric_cols}

#         for num in numeric_cols:
#             sN_base = col_data[num]
#             for den in numeric_cols:
#                 sD_base = col_data[den]
#                 same_col = (num == den)

#                 for a in shifts:
#                     for b in shifts:
#                         if skip_identity and same_col and (a == b):
#                             continue

#                         sN = sN_base + float(a)
#                         sD = sD_base + float(b)
#                         rr = _ratio_series(sN, sD)
#                         ok, value, supp = _is_constant(rr, atol=atol, rtol=rtol, min_support=min_support)
#                         if not ok:
#                             continue

#                         disp = _rationalize(value, max_denominator)
#                         if isinstance(disp, Fraction):
#                             value_display = f"{disp.numerator}" if disp.denominator == 1 else f"{disp.numerator}/{disp.denominator}"
#                         else:
#                             value_display = str(int(disp)) if float(disp).is_integer() else f"{disp}"

#                         num_expr = to_expr(num) if a == 0 else (to_expr(num) + Const(a))
#                         den_expr = to_expr(den) if b == 0 else (to_expr(den) + Const(b))
#                         consts.append(ConstantRatio(
#                             numerator=num,
#                             denominator=den,
#                             shift_num=int(a),
#                             shift_den=int(b),
#                             value_float=float(value),
#                             value_display=value_display,
#                             support=int(rr.dropna().shape[0]),
#                             expr=num_expr / den_expr,
#                         ))

#         consts.sort(key=lambda r: (-r.support, abs(r.shift_num)+abs(r.shift_den), r.numerator, r.denominator, r.value_float))
#         key_to_constants[key] = HypothesisConstants(hypothesis=hyp, mask_key=key, constants=consts)

#     return ConstantsCache(
#         df_index_fingerprint=tuple(range(len(df))),
#         hyp_to_key=hyp_to_key,
#         key_to_constants=key_to_constants,
#     )

# # ───── convenience: base + pairs ─────

# from typing import Sequence
# from pandas.api.types import is_bool_dtype, is_numeric_dtype
# from txgraffiti2025.forms.predicates import Where

# def _detect_boolean_columns(df: pd.DataFrame, *, include_binary_numeric: bool = True, max_nan_frac: float = 0.2) -> List[str]:
#     cols: List[str] = []
#     for c in df.columns:
#         s = df[c]
#         if is_bool_dtype(s):
#             cols.append(c); continue
#         if include_binary_numeric and is_numeric_dtype(s):
#             s2 = pd.to_numeric(s, errors="coerce")
#             if s2.isna().mean() <= max_nan_frac:
#                 vals = set(pd.unique(s2.dropna()))
#                 if vals.issubset({0, 1}):
#                     cols.append(c)
#     return cols

# def _pred_from_bool_col(col: str) -> Predicate:
#     return Where(lambda d, col=col: d[col].astype(bool), name=f"({col})")

# def build_pairwise_hypotheses(df: pd.DataFrame, base: Optional[Predicate], *, boolean_cols: Optional[Sequence[str]] = None) -> List[Optional[Predicate]]:
#     if boolean_cols is None:
#         boolean_cols = _detect_boolean_columns(df)
#     H: List[Optional[Predicate]] = []
#     base_mask = (base.mask(df) if base is not None else pd.Series(True, index=df.index)).astype(bool)
#     if base_mask.any():
#         H.append(base)
#     for b in boolean_cols:
#         P = _pred_from_bool_col(b)
#         mask = base_mask & P.mask(df).reindex(df.index, fill_value=False).astype(bool)
#         if not mask.any():
#             continue
#         if np.array_equal(mask.to_numpy(), base_mask.to_numpy()):
#             continue
#         H.append(base & P if base is not None else P)
#     return H

# def precompute_constant_ratios_pairs(
#     df: pd.DataFrame,
#     base: Optional[Predicate],
#     *,
#     boolean_cols: Optional[Sequence[str]] = None,
#     numeric_cols: Optional[Sequence[str]] = None,
#     shifts: Sequence[int] = (-2, -1, 0, 1, 2),
#     atol: float = 1e-9,
#     rtol: float = 1e-9,
#     min_support: int = 8,
#     max_denominator: Optional[int] = 50,
#     skip_identity: bool = True,
# ):
#     H_pairs = build_pairwise_hypotheses(df, base, boolean_cols=boolean_cols)
#     return precompute_constant_ratios(
#         df,
#         hypotheses=H_pairs,
#         numeric_cols=numeric_cols,
#         shifts=shifts,
#         atol=atol,
#         rtol=rtol,
#         min_support=min_support,
#         max_denominator=max_denominator,
#         skip_identity=skip_identity,
#     )

# src/txgraffiti2025/processing/pre/constants_cache.py
from __future__ import annotations

"""
Precompute and cache constant ratios of shifted numeric columns under class masks.

Given a DataFrame `df`, a set of hypotheses (Predicates) C, numeric columns X,
and integer shifts S, we seek pairs (num, den, a, b) such that the ratio

    (num + a) / (den + b)

is (approximately) constant over rows satisfying a hypothesis C. For each such
constant we store a record with its numeric value, a display string (optionally
rationalized), support count, and an Expr representation usable by the forms API.

Key points
----------
- Mask-based caching: hypotheses that induce the same boolean mask share results.
- Robust numerics: non-finite values are coerced to NaN; near-zero denominators
  can be masked with a tolerance.
- Constancy test: `span <= atol + rtol * max(1, |v|_max)` with median as the value.
- Same-column policy: control `(X+⋯)/(X+⋯)` via `same_column` parameter.
- Pairwise helper: automatically builds hypothesis pairs by ANDing a base
  predicate with each boolean column.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal
from fractions import Fraction

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.forms.utils import Expr, to_expr, Const

__all__ = [
    "ConstantRatio",
    "HypothesisConstants",
    "ConstantsCache",
    "precompute_constant_ratios",
    "constants_matching_coeff",
    "build_pairwise_hypotheses",
    "precompute_constant_ratios_pairs",
]

SameColumnMode = Literal["none", "mismatch_only", "all"]

# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────

def _mask_for(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
    """
    Return a boolean Series mask aligned to df.index (NA→False) for the predicate.
    If pred is None, returns an all-True mask.
    """
    if pred is None:
        return pd.Series(True, index=df.index, dtype=bool)
    m = pred.mask(df).reindex(df.index, fill_value=False)
    return m.fillna(False).astype(bool, copy=False)

def _mask_key(mask: pd.Series) -> str:
    """
    Produce a stable hex key for a boolean mask (row order sensitive).
    """
    return mask.to_numpy(dtype=np.uint8, copy=False).tobytes().hex()

def _is_numeric_series(s: pd.Series) -> bool:
    """True for numeric dtypes except booleans."""
    return is_numeric_dtype(s) and not is_bool_dtype(s)

def _finite(x: pd.Series) -> pd.Series:
    """Coerce to numeric, replace ±inf with NaN."""
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _ratio_series(
    num: pd.Series,
    den: pd.Series,
    *,
    zero_atol: float = 0.0,
) -> pd.Series:
    """
    Compute num/den with optional near-zero masking on denominator.

    zero_atol: if > 0, mask |den| <= zero_atol as NaN before division.
    """
    n = _finite(num)
    d = _finite(den)
    if zero_atol > 0.0:
        d = d.mask(d.abs() <= float(zero_atol), np.nan)
    else:
        d = d.replace(0.0, np.nan)
    return n / d

def _is_constant(
    ser: pd.Series,
    *,
    atol: float,
    rtol: float,
    min_support: int,
) -> Tuple[bool, float, int]:
    """
    Determine whether a Series is (approximately) constant on its finite values.

    Returns (ok, value, support) where:
      - ok: True if constant under the tolerance policy,
      - value: representative value (median) if ok else NaN,
      - support: number of finite, non-NaN observations considered.

    Tolerance policy:
      span := max(v) - min(v)
      tol  := atol + rtol * max(1, max(|v|))
    """
    v = _finite(ser).dropna()
    supp = int(v.size)
    if supp < int(min_support):
        return (False, float("nan"), supp)

    vmax = float(v.max())
    vmin = float(v.min())
    span = vmax - vmin
    absmax = float(np.abs(v).max()) if supp else 0.0
    tol = float(atol) + float(rtol) * max(1.0, absmax)

    if span <= tol:
        return (True, float(v.median()), supp)
    return (False, float("nan"), supp)

def _rationalize_display(x: float, max_den: Optional[int]) -> str:
    """
    Produce a compact display string for a numeric value.

    If max_den is provided, try Fraction(x).limit_denominator(max_den).
    Otherwise or if not helpful, fall back to compact float formatting.
    """
    if max_den is not None:
        frac = Fraction(x).limit_denominator(int(max_den))
        if frac.denominator == 1:
            return f"{frac.numerator}"
        if abs(frac.numerator) < 10**9 and frac.denominator < 10**9:
            return f"{frac.numerator}/{frac.denominator}"
    f = float(x)
    return f"{int(f)}" if f.is_integer() else f"{f:.10g}"

# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ConstantRatio:
    numerator: str
    denominator: str
    shift_num: int
    shift_den: int
    value_float: float
    value_display: str
    support: int
    expr: Expr  # (num + a) / (den + b) as an Expr (for downstream usage)

@dataclass
class HypothesisConstants:
    hypothesis: Optional[Predicate]
    mask_key: str
    constants: List[ConstantRatio]

@dataclass
class ConstantsCache:
    df_index_fingerprint: Tuple[object, ...]
    hyp_to_key: Dict[str, str]
    key_to_constants: Dict[str, HypothesisConstants]

    def constants_for(self, hypothesis: Optional[Predicate]) -> List[ConstantRatio]:
        """
        Retrieve precomputed ConstantRatio records for a hypothesis.
        """
        key = self.hyp_to_key.get(repr(hypothesis))
        if key is None:
            return []
        rec = self.key_to_constants.get(key)
        return rec.constants if rec is not None else []

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def constants_matching_coeff(
    cache: ConstantsCache,
    hypothesis: Optional[Predicate],
    coeff: float,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> List[ConstantRatio]:
    """
    Return ConstantRatio records under `hypothesis` whose numeric value
    is approximately equal to `coeff` per np.isclose(atol, rtol).
    """
    consts = cache.constants_for(hypothesis)
    target = float(coeff)
    return [cr for cr in consts if np.isclose(cr.value_float, target, atol=atol, rtol=rtol)]

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
    skip_identity: bool = True,                 # kept for back-compat
    same_column: SameColumnMode = "mismatch_only",  # NEW: 'none' | 'mismatch_only' | 'all'
) -> ConstantsCache:
    """
    Precompute constant ratios (num+a)/(den+b) for each hypothesis mask.

    Parameters
    ----------
    numeric_cols : sequence of str, optional
        If None, detected automatically from df (numeric, non-boolean).
    shifts : sequence of int
        Integer offsets 'a' and 'b' to try in (N+a)/(D+b).
    skip_identity : bool
        If num == den and a == b, skip (tautology). (Redundant when same_column='mismatch_only'.)
    same_column : {'none','mismatch_only','all'}
        Policy for same-column ratios:
          - 'none'           : disallow any (X+⋯)/(X+⋯)
          - 'mismatch_only'  : allow only when a != b (default)
          - 'all'            : allow all (subject to skip_identity)
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if _is_numeric_series(df[c])]

    key_to_constants: Dict[str, HypothesisConstants] = {}
    hyp_to_key: Dict[str, str] = {}

    # One pass to coerce numeric cols; reuse per slice
    df_num = {c: pd.to_numeric(df[c], errors="coerce") for c in numeric_cols}
    shifts_list = [int(s) for s in shifts]

    for hyp in hypotheses:
        mask = _mask_for(df, hyp)
        key = _mask_key(mask)
        hyp_to_key[repr(hyp)] = key

        if not mask.any():
            key_to_constants[key] = HypothesisConstants(hypothesis=hyp, mask_key=key, constants=[])
            continue

        # Slice data for hypothesis
        data = {c: s[mask] for c, s in df_num.items()}

        # Precompute shifted series for each (col, shift)
        shifted: Dict[str, Dict[int, pd.Series]] = {
            c: {s: (ser + float(s)) for s in shifts_list} for c, ser in data.items()
        }

        consts: List[ConstantRatio] = []

        for num in numeric_cols:
            sN_map = shifted.get(num)
            if sN_map is None:
                continue
            for den in numeric_cols:
                sD_map = shifted.get(den)
                if sD_map is None:
                    continue

                same_col = (num == den)
                if same_column == "none" and same_col:
                    continue

                for a in shifts_list:
                    for b in shifts_list:
                        # same-column policy
                        if same_col and same_column == "mismatch_only" and a == b:
                            continue
                        if skip_identity and same_col and a == b:
                            # back-compat guard; redundant when mismatch_only
                            continue

                        rr = _ratio_series(sN_map[a], sD_map[b], zero_atol=atol)
                        ok, value, supp = _is_constant(rr, atol=atol, rtol=rtol, min_support=min_support)
                        if not ok:
                            continue

                        value_display = _rationalize_display(value, max_denominator)

                        num_expr = to_expr(num) if a == 0 else (to_expr(num) + Const(a))
                        den_expr = to_expr(den) if b == 0 else (to_expr(den) + Const(b))

                        consts.append(
                            ConstantRatio(
                                numerator=num,
                                denominator=den,
                                shift_num=int(a),
                                shift_den=int(b),
                                value_float=float(value),
                                value_display=value_display,
                                support=int(supp),
                                expr=num_expr / den_expr,
                            )
                        )

        # Sort: highest support, then smaller shifts, then lexical by columns, then value
        consts.sort(
            key=lambda r: (
                -r.support,
                abs(r.shift_num) + abs(r.shift_den),
                r.numerator,
                r.denominator,
                r.value_float,
            )
        )

        key_to_constants[key] = HypothesisConstants(hypothesis=hyp, mask_key=key, constants=consts)

    fingerprint: Tuple[object, ...] = tuple(map(hash, df.index))
    return ConstantsCache(
        df_index_fingerprint=fingerprint,
        hyp_to_key=hyp_to_key,
        key_to_constants=key_to_constants,
    )

# ──────────────────────────────────────────────────────────────────────
# Convenience: base + boolean pairs
# ──────────────────────────────────────────────────────────────────────

def _detect_boolean_columns(
    df: pd.DataFrame,
    *,
    include_binary_numeric: bool = True,
    max_nan_frac: float = 0.2,
) -> List[str]:
    """
    Detect boolean columns; optionally include numeric columns that are effectively binary {0,1}.
    """
    cols: List[str] = []
    for c in df.columns:
        s = df[c]
        if is_bool_dtype(s):
            cols.append(c)
            continue
        if include_binary_numeric and is_numeric_dtype(s):
            s2 = pd.to_numeric(s, errors="coerce")
            if s2.isna().mean() <= float(max_nan_frac):
                vals = set(pd.unique(s2.dropna()))
                if vals.issubset({0, 1}):
                    cols.append(c)
    return cols

def _pred_from_bool_col(col: str) -> Predicate:
    """Build a predicate from a boolean-ish column with NA→False semantics."""
    return Where(lambda d, col=col: pd.Series(d[col]).fillna(False).astype(bool), name=f"({col})")

def build_pairwise_hypotheses(
    df: pd.DataFrame,
    base: Optional[Predicate],
    *,
    boolean_cols: Optional[Sequence[str]] = None,
) -> List[Optional[Predicate]]:
    """
    Construct hypothesis list [base] U [base ∧ P_b for b in boolean_cols] where masks are non-empty
    and strictly refine the base mask.
    """
    if boolean_cols is None:
        boolean_cols = _detect_boolean_columns(df)

    H: List[Optional[Predicate]] = []
    base_mask = _mask_for(df, base)

    if base_mask.any():
        H.append(base)

    for b in boolean_cols:
        P = _pred_from_bool_col(b)
        mask = base_mask & _mask_for(df, P)
        if not mask.any():
            continue
        if np.array_equal(mask.to_numpy(), base_mask.to_numpy()):
            continue
        H.append(base & P if base is not None else P)
    return H

def precompute_constant_ratios_pairs(
    df: pd.DataFrame,
    base: Optional[Predicate],
    *,
    boolean_cols: Optional[Sequence[str]] = None,
    numeric_cols: Optional[Sequence[str]] = None,
    shifts: Sequence[int] = (-2, -1, 0, 1, 2),
    atol: float = 1e-9,
    rtol: float = 1e-9,
    min_support: int = 8,
    max_denominator: Optional[int] = 50,
    skip_identity: bool = True,
    same_column: SameColumnMode = "mismatch_only",
) -> ConstantsCache:
    """
    Build pairwise hypotheses from boolean columns on top of an optional base,
    then precompute constant ratios for all of them.
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
        skip_identity=skip_identity,
        same_column=same_column,
    )
