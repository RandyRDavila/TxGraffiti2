# # # src/txgraffiti2025/processing/post/constant_ratios.py

# # from __future__ import annotations
# # from dataclasses import dataclass
# # from typing import List, Optional, Sequence, Tuple
# # from fractions import Fraction

# # import numpy as np
# # import pandas as pd
# # from pandas.api.types import is_numeric_dtype, is_bool_dtype

# # from txgraffiti2025.forms.utils import Expr, Const, ColumnTerm, BinOp, to_expr
# # from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq

# # def _is_numeric_series(s: pd.Series) -> bool:
# #     return is_numeric_dtype(s) and not is_bool_dtype(s)

# # def _finite(series: pd.Series) -> pd.Series:
# #     return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

# # def _is_constant(series: pd.Series, *, atol: float, rtol: float, min_support: int) -> Tuple[bool, float]:
# #     v = _finite(series).dropna()
# #     if len(v) < min_support:
# #         return (False, np.nan)
# #     span = float(v.max() - v.min())
# #     tol = float(atol + rtol * max(1.0, np.abs(v).max()))
# #     if span <= tol:
# #         return (True, float(v.median()))
# #     return (False, np.nan)

# # def _ratio_series(num: pd.Series, den: pd.Series) -> pd.Series:
# #     x = _finite(num); y = _finite(den)
# #     y = y.replace(0.0, np.nan)
# #     return x / y

# # def _rationalize(x: float, max_den: Optional[int]) -> Fraction | float:
# #     if max_den is None:
# #         return float(x)
# #     return Fraction(x).limit_denominator(max_den)

# # @dataclass
# # class RatioPattern:
# #     kind: str
# #     target: str
# #     feature: str
# #     coefficient: float

# # def _extract_ratio_pattern(conj: Conjecture) -> Optional[RatioPattern]:
# #     rel = conj.relation
# #     kind = rel.__class__.__name__
# #     if isinstance(rel, Eq):
# #         return None
# #     left = getattr(rel, "left", None)
# #     right = getattr(rel, "right", None)

# #     def _match_mul(expr: Expr) -> Optional[Tuple[float, str]]:
# #         if isinstance(expr, BinOp) and getattr(expr, "fn", None) is np.multiply:  # type: ignore
# #             L, R = expr.left, expr.right
# #             if isinstance(L, Const) and isinstance(R, ColumnTerm):
# #                 return (float(L.value), R.col)
# #             if isinstance(R, Const) and isinstance(L, ColumnTerm):
# #                 return (float(R.value), L.col)
# #         return None

# #     from txgraffiti2025.forms.utils import ColumnTerm as CT
# #     if isinstance(left, CT):
# #         m = _match_mul(right)
# #         if m: coeff, feat = m; return RatioPattern(kind, left.col, feat, float(coeff))
# #     if isinstance(right, CT):
# #         m = _match_mul(left)
# #         if m:
# #             coeff, feat = m
# #             new_kind = "Ge" if kind == "Le" else "Le" if kind == "Ge" else kind
# #             return RatioPattern(new_kind, right.col, feat, float(coeff))
# #     return None

# # @dataclass
# # class ConstantRatio:
# #     numerator: str
# #     denominator: str
# #     shift_num: int
# #     shift_den: int
# #     value_float: float
# #     value_display: str
# #     support: int
# #     formula_expr: Expr
# #     matches_conj_coeff: bool

# # def find_constant_ratios_for_conjecture(
# #     df: pd.DataFrame,
# #     conj: Conjecture,
# #     *,
# #     numeric_cols: Optional[Sequence[str]] = None,
# #     shifts: Sequence[int] = (-2, -1, 0, 1, 2),
# #     atol: float = 1e-9,
# #     rtol: float = 1e-9,
# #     min_support: int = 8,
# #     max_denominator: Optional[int] = 50,
# # ) -> List[ConstantRatio]:
# #     mask = (conj.condition.mask(df) if conj.condition is not None else pd.Series(True, index=df.index))
# #     mask = mask.reindex(df.index, fill_value=False).astype(bool)
# #     if not mask.any():
# #         return []

# #     data = df.loc[mask]
# #     if numeric_cols is None:
# #         numeric_cols = [c for c in data.columns if _is_numeric_series(data[c])]

# #     patt = _extract_ratio_pattern(conj)
# #     out: List[ConstantRatio] = []

# #     for num in numeric_cols:
# #         sN_all = pd.to_numeric(data[num], errors="coerce")
# #         for den in numeric_cols:
# #             sD_all = pd.to_numeric(data[den], errors="coerce")
# #             same_col = (num == den)

# #             for a in shifts:
# #                 for b in shifts:
# #                     if same_col and (a == b):
# #                         continue  # skip identity (N+a)/(N+a)

# #                     sN = sN_all + float(a)
# #                     sD = sD_all + float(b)
# #                     rr = _ratio_series(sN, sD)
# #                     is_const, val = _is_constant(rr, atol=atol, rtol=rtol, min_support=min_support)
# #                     if not is_const:
# #                         continue

# #                     disp = _rationalize(val, max_denominator)
# #                     if isinstance(disp, Fraction):
# #                         value_display = f"{disp.numerator}" if disp.denominator == 1 else f"{disp.numerator}/{disp.denominator}"
# #                     else:
# #                         value_display = str(int(disp)) if float(disp).is_integer() else f"{disp}"

# #                     num_expr = to_expr(num) if a == 0 else (to_expr(num) + Const(a))
# #                     den_expr = to_expr(den) if b == 0 else (to_expr(den) + Const(b))
# #                     formula_expr = num_expr / den_expr

# #                     matches = False
# #                     if patt is not None:
# #                         matches = np.isclose(val, patt.coefficient, atol=atol, rtol=rtol)

# #                     out.append(ConstantRatio(
# #                         numerator=num,
# #                         denominator=den,
# #                         shift_num=int(a),
# #                         shift_den=int(b),
# #                         value_float=float(val),
# #                         value_display=value_display,
# #                         support=int(rr.dropna().shape[0]),
# #                         formula_expr=formula_expr,
# #                         matches_conj_coeff=bool(matches),
# #                     ))

# #     out.sort(key=lambda cr: (
# #         not cr.matches_conj_coeff,
# #         -cr.support,
# #         abs(cr.shift_num) + abs(cr.shift_den),
# #         cr.numerator, cr.denominator, cr.value_float
# #     ))
# #     return out

# # src/txgraffiti2025/processing/post/constant_ratios.py

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Optional, Sequence, Tuple
# from fractions import Fraction

# import numpy as np
# import pandas as pd
# from pandas.api.types import is_numeric_dtype, is_bool_dtype

# from txgraffiti2025.forms.utils import Expr, Const, ColumnTerm, BinOp, to_expr
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq


# def _is_numeric_series(s: pd.Series) -> bool:
#     """True iff a series is numeric (excluding boolean)."""
#     return is_numeric_dtype(s) and not is_bool_dtype(s)


# def _finite(series: pd.Series) -> pd.Series:
#     """Coerce to numeric and replace ±inf with NaN."""
#     return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


# def _is_constant(
#     series: pd.Series, *, atol: float, rtol: float, min_support: int
# ) -> Tuple[bool, float]:
#     """
#     Check if a series is approximately constant under a robust tolerance.

#     Returns (is_constant, representative_value). The representative is the median.
#     """
#     v = _finite(series).dropna()
#     if len(v) < min_support:
#         return (False, np.nan)
#     span = float(v.max() - v.min())
#     tol = float(atol + rtol * max(1.0, np.abs(v).max()))
#     if span <= tol:
#         return (True, float(v.median()))
#     return (False, np.nan)


# def _ratio_series(num: pd.Series, den: pd.Series) -> pd.Series:
#     """Compute num/den with zero-denominator masked to NaN."""
#     x = _finite(num)
#     y = _finite(den).replace(0.0, np.nan)
#     return x / y


# def _rationalize(x: float, max_den: Optional[int]) -> Fraction | float:
#     """Return Fraction(x) limited by max_den, or the float itself if None."""
#     if max_den is None:
#         return float(x)
#     return Fraction(x).limit_denominator(max_den)


# @dataclass
# class RatioPattern:
#     kind: str          # "Le" or "Ge"
#     target: str        # column name on the inequality side that carries the bound
#     feature: str       # the feature multiplied by a constant
#     coefficient: float # positive finite coefficient


# def _extract_ratio_pattern(conj: Conjecture) -> Optional[RatioPattern]:
#     """
#     Try to interpret a conjecture as `target <= coeff * feature` (or >=).

#     If the target is on the right, we flip the inequality direction when moving it
#     to the left. Only positive, finite coefficients are accepted; negative or NaN
#     coefficients are skipped (they require sign assumptions we don't have).
#     """
#     rel = conj.relation
#     kind = rel.__class__.__name__
#     if isinstance(rel, Eq):
#         return None

#     left = getattr(rel, "left", None)
#     right = getattr(rel, "right", None)

#     def _match_mul(expr: Expr) -> Optional[Tuple[float, str]]:
#         if isinstance(expr, BinOp) and getattr(expr, "fn", None) is np.multiply:  # type: ignore
#             L, R = expr.left, expr.right
#             if isinstance(L, Const) and isinstance(R, ColumnTerm):
#                 return (float(L.value), R.col)
#             if isinstance(R, Const) and isinstance(L, ColumnTerm):
#                 return (float(R.value), L.col)
#         return None

#     from txgraffiti2025.forms.utils import ColumnTerm as CT

#     # Case: target is on the left:  target (<= or >=) coeff * feature
#     if isinstance(left, CT):
#         m = _match_mul(right)
#         if m:
#             coeff, feat = m
#             if not np.isfinite(coeff) or coeff <= 0:
#                 return None
#             return RatioPattern(kind, left.col, feat, float(coeff))

#     # Case: target is on the right:  coeff * feature (<= or >=) target
#     if isinstance(right, CT):
#         m = _match_mul(left)
#         if m:
#             coeff, feat = m
#             if not np.isfinite(coeff) or coeff <= 0:
#                 return None
#             # Move target to the left: flip direction for inequalities
#             new_kind = "Ge" if kind == "Le" else "Le" if kind == "Ge" else kind
#             return RatioPattern(new_kind, right.col, feat, float(coeff))

#     return None


# @dataclass
# class ConstantRatio:
#     numerator: str
#     denominator: str
#     shift_num: int
#     shift_den: int
#     value_float: float
#     value_display: str
#     support: int
#     formula_expr: Expr
#     matches_conj_coeff: bool


# def find_constant_ratios_for_conjecture(
#     df: pd.DataFrame,
#     conj: Conjecture,
#     *,
#     numeric_cols: Optional[Sequence[str]] = None,
#     shifts: Sequence[int] = (-2, -1, 0, 1, 2),
#     atol: float = 1e-9,
#     rtol: float = 1e-9,
#     min_support: int = 8,
#     max_denominator: Optional[int] = 50,
# ) -> List[ConstantRatio]:
#     """
#     Search for approximately constant ratios (N+a)/(D+b) over the conjecture's base mask.
#     Includes rational approximation for display and flags ratios close to the conjecture's
#     learned coefficient.
#     """
#     mask = (
#         conj.condition.mask(df)
#         if conj.condition is not None
#         else pd.Series(True, index=df.index)
#     )
#     mask = mask.reindex(df.index, fill_value=False).astype(bool)
#     if not mask.any():
#         return []

#     data = df.loc[mask]
#     if numeric_cols is None:
#         numeric_cols = [c for c in data.columns if _is_numeric_series(data[c])]

#     patt = _extract_ratio_pattern(conj)
#     out: List[ConstantRatio] = []

#     for num in numeric_cols:
#         sN_all = pd.to_numeric(data[num], errors="coerce")
#         for den in numeric_cols:
#             sD_all = pd.to_numeric(data[den], errors="coerce")
#             same_col = (num == den)

#             for a in shifts:
#                 for b in shifts:
#                     # Avoid trivial identity (N+a)/(N+a)
#                     if same_col and (a == b):
#                         continue

#                     sN = sN_all + float(a)
#                     sD = sD_all + float(b)
#                     rr = _ratio_series(sN, sD)
#                     is_const, val = _is_constant(
#                         rr, atol=atol, rtol=rtol, min_support=min_support
#                     )
#                     if not is_const:
#                         continue

#                     disp = _rationalize(val, max_denominator)
#                     if isinstance(disp, Fraction):
#                         value_display = (
#                             f"{disp.numerator}"
#                             if disp.denominator == 1
#                             else f"{disp.numerator}/{disp.denominator}"
#                         )
#                     else:
#                         value_display = (
#                             str(int(disp))
#                             if float(disp).is_integer()
#                             else f"{disp}"
#                         )

#                     num_expr = to_expr(num) if a == 0 else (to_expr(num) + Const(a))
#                     den_expr = to_expr(den) if b == 0 else (to_expr(den) + Const(b))
#                     formula_expr = num_expr / den_expr

#                     matches = False
#                     if patt is not None:
#                         matches = np.isclose(val, patt.coefficient, atol=atol, rtol=rtol)

#                     out.append(
#                         ConstantRatio(
#                             numerator=num,
#                             denominator=den,
#                             shift_num=int(a),
#                             shift_den=int(b),
#                             value_float=float(val),
#                             value_display=value_display,
#                             support=int(rr.dropna().shape[0]),
#                             formula_expr=formula_expr,
#                             matches_conj_coeff=bool(matches),
#                         )
#                     )

#     out.sort(
#         key=lambda cr: (
#             not cr.matches_conj_coeff,               # prefer those matching learned coeff
#             -cr.support,                              # more support first
#             abs(cr.shift_num) + abs(cr.shift_den),    # smaller total shift
#             cr.numerator, cr.denominator, cr.value_float,
#         )
#     )
#     return out

# # src/txgraffiti2025/processing/post/constant_ratios.py
# from __future__ import annotations

# """
# constant_ratios
# ---------------

# Goal
# ====
# Given a DataFrame and a collection of *subclasses* (hypotheses/predicates),
# discover (approximately) constant ratios of the form (N + a) / (D + b)
# inside those subclasses. Optionally highlight constants that match the
# coefficient in a ratio-style conjecture, e.g.  target (<=/>=) c * feature.

# Key entrypoints
# ---------------
# - find_constant_ratios_over_hypotheses(df, hypotheses, ...)
#     Mine constant ratios on each hypothesis slice.

# - find_constant_ratios_for_conjecture(df, conj, hypotheses, ...)
#     Same as above, but also parses conj as ratio-style and flags
#     matches to its learned coefficient.

# Robustness
# ----------
# Constancy may be tested by:
# - 'cv'    : robust coefficient-of-variation via MAD/median (default)
# - 'qspan' : middle quantile span (q10..q90) below an absolute tolerance
# - 'span'  : legacy strict max-min <= atol + rtol * scale

# Back-compat
# -----------
# We export `extract_ratio_pattern(conj)` and a compatibility alias
# `_extract_ratio_pattern(conj)` so older imports continue to work.
# """

# from dataclasses import dataclass
# from fractions import Fraction
# from typing import Iterable, List, Optional, Sequence, Tuple, Dict

# import numpy as np
# import pandas as pd
# from pandas.api.types import is_numeric_dtype, is_bool_dtype

# from txgraffiti2025.forms.utils import Expr, Const, ColumnTerm, BinOp, to_expr
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq
# from txgraffiti2025.forms.predicates import Predicate, AndPred  # for type and repr in results


# # ---------------------------------------------------------------------
# # Basic helpers
# # ---------------------------------------------------------------------

# def _finite(series: pd.Series) -> pd.Series:
#     """Coerce to numeric and replace ±inf with NaN."""
#     return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

# def _is_numeric_series(s: pd.Series) -> bool:
#     """True iff a series is numeric (excluding boolean dtype)."""
#     return is_numeric_dtype(s) and not is_bool_dtype(s)

# def _ratio_series(num: pd.Series, den: pd.Series) -> pd.Series:
#     """Compute num/den with zero-denominator masked to NaN."""
#     x = _finite(num)
#     y = _finite(den).replace(0.0, np.nan)
#     return x / y

# def _rationalize(x: float, max_den: Optional[int]) -> Fraction | float:
#     """
#     Return a simple Fraction approximation of x limited by max_den.
#     If max_den is None or approximation fails, return float(x).
#     """
#     if max_den is None:
#         return float(x)
#     try:
#         return Fraction(x).limit_denominator(max_den)
#     except Exception:
#         return float(x)

# def _mask(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
#     """Aligned boolean mask for a predicate (None means TRUE)."""
#     if pred is None:
#         return pd.Series(True, index=df.index)
#     return pred.mask(df).reindex(df.index, fill_value=False).astype(bool)


# # ---------------------------------------------------------------------
# # Ratio-style conjecture pattern
# # ---------------------------------------------------------------------

# @dataclass(frozen=True)
# class RatioPattern:
#     """
#     Canonical form of a ratio-style inequality:

#         target (Le/Ge) coefficient * feature

#     If the original conjecture had `target` on the right-hand side,
#     the direction is flipped accordingly when moved to the left.
#     """
#     kind: str           # "Le" or "Ge"
#     target: str         # bounded quantity (column name)
#     feature: str        # multiplier base (column name)
#     coefficient: float  # positive finite coefficient


# def extract_ratio_pattern(conj: Conjecture) -> Optional[RatioPattern]:
#     """
#     Try to parse a conjecture as 'target <= c * feature' (or '>=').
#     Returns a RatioPattern or None if not recognized.

#     Notes
#     -----
#     * Skips Eq (equalities).
#     * Only positive, finite coefficients are accepted.
#     """
#     rel = conj.relation
#     if isinstance(rel, Eq):
#         return None

#     left = getattr(rel, "left", None)
#     right = getattr(rel, "right", None)

#     def _const_float(x) -> float:
#         try:
#             return float(x)
#         except Exception:
#             return float("nan")

#     def _match_mul(expr: Expr) -> Optional[Tuple[float, str]]:
#         if isinstance(expr, BinOp) and getattr(expr, "fn", None) is np.multiply:
#             L, R = expr.left, expr.right
#             if isinstance(L, Const) and isinstance(R, ColumnTerm):
#                 return (_const_float(L.value), R.col)
#             if isinstance(R, Const) and isinstance(L, ColumnTerm):
#                 return (_const_float(R.value), L.col)
#         return None

#     # Case A: target on the left: target (<=/>=) c * feature
#     if isinstance(left, ColumnTerm):
#         m = _match_mul(right)
#         if m:
#             coeff, feat = m
#             if np.isfinite(coeff) and coeff > 0:
#                 return RatioPattern(rel.__class__.__name__, left.col, feat, float(coeff))

#     # Case B: target on the right: c * feature (<=/>=) target  → flip direction
#     if isinstance(right, ColumnTerm):
#         m = _match_mul(left)
#         if m:
#             coeff, feat = m
#             if np.isfinite(coeff) and coeff > 0:
#                 flipped = "Ge" if isinstance(rel, Le) else "Le" if isinstance(rel, Ge) else rel.__class__.__name__
#                 return RatioPattern(flipped, right.col, feat, float(coeff))

#     return None


# # Back-compat for older code importing the private name
# _extract_ratio_pattern = extract_ratio_pattern


# # ---------------------------------------------------------------------
# # Robust constancy tests
# # ---------------------------------------------------------------------

# def _robust_cv(series: pd.Series) -> float:
#     """
#     Robust coefficient of variation via MAD/median.
#     Returns +inf if median≈0 or empty.
#     """
#     v = _finite(series).dropna()
#     if v.empty:
#         return float("inf")
#     med = float(v.median())
#     if med == 0.0:
#         return float("inf")
#     mad = float((v - med).abs().median())
#     return abs(mad / med)

# def _robust_qspan(series: pd.Series, qlo: float = 0.1, qhi: float = 0.9) -> float:
#     """
#     Robust middle spread (quantile span). Returns +inf if insufficient data.
#     """
#     v = _finite(series).dropna()
#     if len(v) < 3:
#         return float("inf")
#     return float(v.quantile(qhi) - v.quantile(qlo))

# def _is_constant_span(series: pd.Series, *, atol: float, rtol: float, min_support: int) -> Tuple[bool, float, Dict[str, float]]:
#     v = _finite(series).dropna()
#     if len(v) < min_support:
#         return (False, float("nan"), {"support": float(len(v))})
#     span = float(v.max() - v.min())
#     tol = float(atol + rtol * max(1.0, np.abs(v).max()))
#     return (span <= tol, float(v.median()), {"support": float(len(v)), "span": span, "tol": tol})

# def _is_constant_cv(series: pd.Series, *, cv_tol: float, min_support: int) -> Tuple[bool, float, Dict[str, float]]:
#     v = _finite(series).dropna()
#     if len(v) < min_support:
#         return (False, float("nan"), {"support": float(len(v))})
#     cv = _robust_cv(v)
#     return (cv <= cv_tol, float(v.median()), {"support": float(len(v)), "cv": cv})

# def _is_constant_qspan(series: pd.Series, *, q_tol: float, min_support: int) -> Tuple[bool, float, Dict[str, float]]:
#     v = _finite(series).dropna()
#     if len(v) < min_support:
#         return (False, float("nan"), {"support": float(len(v))})
#     span = _robust_qspan(v)
#     return (span <= q_tol, float(v.median()), {"support": float(len(v)), "qspan": span})


# # ---------------------------------------------------------------------
# # Result records
# # ---------------------------------------------------------------------

# @dataclass
# class ConstantRatio:
#     """
#     A discovered approximately-constant ratio (N+a)/(D+b) on a specific hypothesis slice.
#     """
#     numerator: str
#     denominator: str
#     shift_num: int
#     shift_den: int

#     value_float: float              # representative value (median)
#     value_display: str              # rationalized display (e.g. '3/2')
#     support: int                    # number of non-NaN ratios used (within slice)
#     cv: float                       # robust MAD/median coefficient of variation
#     qspan: float                    # middle 80% span (absolute units)

#     formula_expr: Expr              # Expr for (N+a)/(D+b)
#     hypothesis: Optional[Predicate] # subclass slice on which constant holds
#     matches_conj_coeff: bool        # True if close to conjecture’s coefficient


# # ---------------------------------------------------------------------
# # Core mining routine over a *single* slice
# # ---------------------------------------------------------------------

# def _mine_constant_ratios_on_slice(
#     data: pd.DataFrame,
#     *,
#     numeric_cols: Sequence[str],
#     shifts: Sequence[int],
#     constancy: str,
#     min_support: int,
#     # tolerances
#     atol: float,
#     rtol: float,
#     cv_tol: float,
#     qspan_tol: float,
#     # matching
#     coeff_to_match: Optional[float],
#     max_denominator: Optional[int],
# ) -> List[ConstantRatio]:
#     """
#     Scan (N+a)/(D+b) over 'data' (already sliced by hypothesis mask).
#     Returns ConstantRatio records with dispersion stats and rationalized display.
#     """
#     results: List[ConstantRatio] = []

#     # choose constancy test
#     if constancy == "cv":
#         def _const_test(rr):
#             ok, val, info = _is_constant_cv(rr, cv_tol=cv_tol, min_support=min_support)
#             # for summary fields (cv, qspan)
#             cv = info.get("cv", float("inf"))
#             qsp = _robust_qspan(rr)  # compute anyway for ranking tie-breaks
#             return ok, val, cv, qsp
#     elif constancy == "qspan":
#         def _const_test(rr):
#             ok, val, info = _is_constant_qspan(rr, q_tol=qspan_tol, min_support=min_support)
#             cv = _robust_cv(rr)
#             qsp = info.get("qspan", float("inf"))
#             return ok, val, cv, qsp
#     else:  # 'span'
#         def _const_test(rr):
#             ok, val, info = _is_constant_span(rr, atol=atol, rtol=rtol, min_support=min_support)
#             cv = _robust_cv(rr)
#             qsp = _robust_qspan(rr)
#             return ok, val, cv, qsp

#     for num in numeric_cols:
#         sN_all = pd.to_numeric(data[num], errors="coerce")
#         for den in numeric_cols:
#             sD_all = pd.to_numeric(data[den], errors="coerce")
#             same_col = (num == den)

#             for a in shifts:
#                 for b in shifts:
#                     # avoid trivial identity (N+a)/(N+a)
#                     if same_col and (a == b):
#                         continue

#                     rr = _ratio_series(sN_all + float(a), sD_all + float(b))
#                     rr_f = _finite(rr).dropna()
#                     if rr_f.size < min_support:
#                         continue

#                     ok, val, cv, qsp = _const_test(rr_f)
#                     if not ok:
#                         continue

#                     disp = _rationalize(val, max_denominator)
#                     if isinstance(disp, Fraction):
#                         value_display = f"{disp.numerator}" if disp.denominator == 1 else f"{disp.numerator}/{disp.denominator}"
#                     else:
#                         value_display = str(int(disp)) if float(disp).is_integer() else f"{disp}"

#                     num_expr = to_expr(num) if a == 0 else (to_expr(num) + Const(a))
#                     den_expr = to_expr(den) if b == 0 else (to_expr(den) + Const(b))
#                     formula_expr = num_expr / den_expr

#                     matches = False
#                     if coeff_to_match is not None:
#                         matches = np.isclose(val, float(coeff_to_match), atol=atol, rtol=rtol)

#                     results.append(
#                         ConstantRatio(
#                             numerator=num,
#                             denominator=den,
#                             shift_num=int(a),
#                             shift_den=int(b),
#                             value_float=float(val),
#                             value_display=value_display,
#                             support=int(rr_f.size),
#                             cv=float(cv),
#                             qspan=float(qsp),
#                             formula_expr=formula_expr,
#                             hypothesis=None,        # filled by caller
#                             matches_conj_coeff=bool(matches),
#                         )
#                     )
#     return results


# # ---------------------------------------------------------------------
# # Public APIs
# # ---------------------------------------------------------------------

# def find_constant_ratios_over_hypotheses(
#     df: pd.DataFrame,
#     *,
#     hypotheses: Sequence[Optional[Predicate]],
#     numeric_cols: Optional[Sequence[str]] = None,
#     shifts: Sequence[int] = (-2, -1, 0, 1, 2),
#     constancy: str = "cv",            # 'cv' | 'qspan' | 'span'
#     cv_tol: float = 0.08,             # ~8% MAD/median variability
#     qspan_tol: float = 0.5,           # absolute span (tune to dataset scale)
#     atol: float = 1e-6,               # used for coeff matching (and 'span' mode)
#     rtol: float = 5e-2,
#     min_support: int = 8,
#     max_denominator: Optional[int] = 50,
#     top_k_per_hypothesis: Optional[int] = 50,
# ) -> List[ConstantRatio]:
#     """
#     Discover (approximately) constant ratios on each provided hypothesis slice.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Dataset with numeric columns.
#     hypotheses : sequence of Predicate | None
#         Candidate subclasses (slices). None means TRUE (whole dataset).
#     numeric_cols : sequence of str, optional
#         Which columns may serve as numerator/denominator. Defaults to all
#         non-boolean numeric columns on the *slice*.
#     shifts : sequence of int
#         Integer offsets 'a' and 'b' to try in (N+a)/(D+b).
#     constancy : {'cv','qspan','span'}
#         Robust criterion. 'cv' (MAD/median) is recommended.
#     cv_tol, qspan_tol, atol, rtol : float
#         Tolerances for the constancy decision (see module docstring).
#     min_support : int
#         Minimum number of finite ratio values needed on the slice.
#     max_denominator : int or None
#         Fraction limit for human-friendly display.
#     top_k_per_hypothesis : int or None
#         If set, keep only the top-K candidates for each hypothesis slice.

#     Returns
#     -------
#     List[ConstantRatio]
#         Each record contains (N, D, a, b, value, support, cv, qspan, formula_expr)
#         and the *hypothesis* (predicate) for the slice on which it was mined.
#     """
#     all_hits: List[ConstantRatio] = []

#     for H in hypotheses:
#         m = _mask(df, H)
#         if not m.any():
#             continue

#         data = df.loc[m]
#         if numeric_cols is None:
#             cols = [c for c in data.columns if _is_numeric_series(data[c])]
#         else:
#             # keep only those present and numeric on this slice
#             cols = [c for c in numeric_cols if c in data.columns and _is_numeric_series(data[c])]

#         if len(cols) == 0:
#             continue

#         hits = _mine_constant_ratios_on_slice(
#             data,
#             numeric_cols=cols,
#             shifts=shifts,
#             constancy=constancy,
#             min_support=min_support,
#             atol=atol,
#             rtol=rtol,
#             cv_tol=cv_tol,
#             qspan_tol=qspan_tol,
#             coeff_to_match=None,
#             max_denominator=max_denominator,
#         )

#         # Attach the hypothesis and rank locally
#         for h in hits:
#             h.hypothesis = H

#         hits.sort(
#             key=lambda cr: (
#                 cr.cv,                                  # tighter dispersion first
#                 -cr.support,                            # more support
#                 abs(cr.shift_num) + abs(cr.shift_den),  # simpler shifts
#                 cr.numerator, cr.denominator, cr.value_float,
#             )
#         )
#         if top_k_per_hypothesis is not None:
#             hits = hits[:top_k_per_hypothesis]

#         all_hits.extend(hits)

#     # Global, stable ordering (primarily by dispersion, then support)
#     all_hits.sort(
#         key=lambda cr: (
#             cr.cv,
#             -cr.support,
#             abs(cr.shift_num) + abs(cr.shift_den),
#             repr(cr.hypothesis) if cr.hypothesis is not None else "",
#             cr.numerator, cr.denominator, cr.value_float,
#         )
#     )
#     return all_hits


# def find_constant_ratios_for_conjecture(
#     df: pd.DataFrame,
#     conj: Conjecture,
#     *,
#     hypotheses: Sequence[Optional[Predicate]],
#     numeric_cols: Optional[Sequence[str]] = None,
#     shifts: Sequence[int] = (-2, -1, 0, 1, 2),
#     constancy: str = "cv",
#     cv_tol: float = 0.08,
#     qspan_tol: float = 0.5,
#     atol: float = 1e-6,
#     rtol: float = 5e-2,
#     min_support: int = 8,
#     max_denominator: Optional[int] = 50,
#     top_k_per_hypothesis: Optional[int] = 50,
# ) -> List[ConstantRatio]:
#     """
#     Same as `find_constant_ratios_over_hypotheses`, but if `conj` is ratio-style,
#     also flags candidates where the discovered constant is close to conj's
#     learned coefficient.

#     The `conj.condition` is *not* used to filter hypotheses; instead, pass the
#     subclasses you want to mine in `hypotheses`. (You can, of course, include
#     conj.condition among them.)
#     """
#     patt = extract_ratio_pattern(conj)
#     coeff = patt.coefficient if patt is not None else None

#     all_hits: List[ConstantRatio] = []

#     for H in hypotheses:
#         m = _mask(df, H)
#         if not m.any():
#             continue

#         data = df.loc[m]
#         if numeric_cols is None:
#             cols = [c for c in data.columns if _is_numeric_series(data[c])]
#         else:
#             cols = [c for c in numeric_cols if c in data.columns and _is_numeric_series(data[c])]

#         if len(cols) == 0:
#             continue

#         hits = _mine_constant_ratios_on_slice(
#             data,
#             numeric_cols=cols,
#             shifts=shifts,
#             constancy=constancy,
#             min_support=min_support,
#             atol=atol,
#             rtol=rtol,
#             cv_tol=cv_tol,
#             qspan_tol=qspan_tol,
#             coeff_to_match=coeff,
#             max_denominator=max_denominator,
#         )

#         for h in hits:
#             h.hypothesis = H

#         # Prefer coefficient matches within the slice, then dispersion/support/shift simplicity
#         hits.sort(
#             key=lambda cr: (
#                 not cr.matches_conj_coeff,
#                 cr.cv,
#                 -cr.support,
#                 abs(cr.shift_num) + abs(cr.shift_den),
#                 cr.numerator, cr.denominator, cr.value_float,
#             )
#         )
#         if top_k_per_hypothesis is not None:
#             hits = hits[:top_k_per_hypothesis]

#         all_hits.extend(hits)

#     # Global ordering: prioritize matches overall, then dispersion/support
#     all_hits.sort(
#         key=lambda cr: (
#             not cr.matches_conj_coeff,
#             cr.cv,
#             -cr.support,
#             abs(cr.shift_num) + abs(cr.shift_den),
#             repr(cr.hypothesis) if cr.hypothesis is not None else "",
#             cr.numerator, cr.denominator, cr.value_float,
#         )
#     )
#     return all_hits

# src/txgraffiti2025/processing/post/constant_ratios.py
from __future__ import annotations

"""
constant_ratios
---------------

Goal
====
Given a DataFrame and a collection of subclasses (predicates), discover
(approximately) constant ratios of the form (N + a) / (D + b) inside those
subclasses. Optionally highlight constants that match the coefficient in a
ratio-style conjecture, e.g.  target (<=/>=) c * feature.

Key entrypoints
---------------
- find_constant_ratios_over_hypotheses(df, hypotheses, ...)
    Mine constant ratios on each hypothesis slice.

- find_constant_ratios_for_conjecture(df, conj, hypotheses, ...)
    Same as above, but also parses `conj` as ratio-style and flags
    candidates whose constant ≈ the conjecture’s slope.

Constancy criteria
------------------
- 'cv'    : robust coefficient-of-variation via MAD/median (default).
- 'qspan' : middle quantile span (q10..q90) must be <= qspan_tol.
- 'span'  : strict max-min <= (atol + rtol * scale).

Notes
-----
- All tests ignore NaNs/±inf and require `min_support` finite ratio values.
- In 'cv' mode, if the median≈0, the CV is treated as +inf (won’t pass).
- Same-column policy:
    * "none"           : disallow any (X+…)/(X+…)
    * "mismatch_only"  : allow same-column only when shifts differ (default)
    * "all"            : allow any same-column ratio

Back-compat
-----------
We export `_extract_ratio_pattern` as an alias of `extract_ratio_pattern`.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from txgraffiti2025.forms.utils import Expr, Const, ColumnTerm, BinOp, to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq
from txgraffiti2025.forms.predicates import Predicate

SameColumnMode = Literal["none", "mismatch_only", "all"]

# ──────────────────────────────────────────────────────────────────────
# Basic helpers
# ──────────────────────────────────────────────────────────────────────

def _finite(series: pd.Series) -> pd.Series:
    """Coerce to numeric and replace ±inf with NaN."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _is_numeric_series(s: pd.Series) -> bool:
    """True iff a series is numeric (excluding boolean dtype)."""
    return is_numeric_dtype(s) and not is_bool_dtype(s)

def _ratio_series(num: pd.Series, den: pd.Series, *, zero_atol: float = 0.0) -> pd.Series:
    """Compute num/den with near-zero denominator masked to NaN."""
    x = _finite(num)
    y = _finite(den)
    if zero_atol > 0.0:
        y = y.mask(y.abs() <= zero_atol, np.nan)
    else:
        y = y.replace(0.0, np.nan)
    return x / y

def _rationalize(x: float, max_den: Optional[int]) -> Fraction | float:
    """Return a simple Fraction approximation of x limited by max_den; else float."""
    if max_den is None:
        return float(x)
    try:
        return Fraction(x).limit_denominator(max_den)
    except Exception:
        return float(x)

def _mask(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
    """Aligned boolean mask for a predicate (None means TRUE)."""
    if pred is None:
        return pd.Series(True, index=df.index, dtype=bool)
    return pred.mask(df).reindex(df.index, fill_value=False).astype(bool)

# ──────────────────────────────────────────────────────────────────────
# Ratio-style conjecture pattern
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RatioPattern:
    """
    Canonical form of a ratio-style inequality:

        target (Le/Ge) coefficient * feature

    If the original conjecture had `target` on the right-hand side,
    the direction is flipped accordingly.
    """
    kind: str           # "Le" or "Ge"
    target: str         # bounded quantity (column name)
    feature: str        # feature label (column name or repr(expr))
    coefficient: float  # positive finite coefficient

def extract_ratio_pattern(conj: Conjecture) -> Optional[RatioPattern]:
    """
    Parse a conjecture as 'target <= c * feature' or 'target >= c * feature'.

    - Accepts any Expr on the feature side (not just a ColumnTerm).
    - Extracts coefficient only when product is unambiguously Const * Expr.
    - Stores feature as ColumnTerm.col if present, else repr(expr).
    """
    rel = conj.relation
    if isinstance(rel, Eq):
        return None

    left = getattr(rel, "left", None)
    right = getattr(rel, "right", None)

    def _const_float_safe(v) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _feat_label(e: Expr) -> str:
        return e.col if isinstance(e, ColumnTerm) else repr(e)

    def _match_mul_const_expr(expr: Expr) -> Optional[Tuple[float, str]]:
        # Accept BinOp(np.multiply, Const, Expr) or (Expr, Const)
        if isinstance(expr, BinOp) and getattr(expr, "fn", None) is np.multiply:
            L, R = expr.left, expr.right
            if isinstance(L, Const) and isinstance(R, Expr):
                return _const_float_safe(L.value), _feat_label(R)
            if isinstance(R, Const) and isinstance(L, Expr):
                return _const_float_safe(R.value), _feat_label(L)
        return None

    # Case A: target on the left
    if isinstance(left, ColumnTerm):
        m = _match_mul_const_expr(right)
        if m:
            coeff, feat = m
            if np.isfinite(coeff) and coeff > 0:
                return RatioPattern(rel.__class__.__name__, left.col, feat, float(coeff))

    # Case B: target on the right → flip direction
    if isinstance(right, ColumnTerm):
        m = _match_mul_const_expr(left)
        if m:
            coeff, feat = m
            if np.isfinite(coeff) and coeff > 0:
                flipped = "Ge" if isinstance(rel, Le) else "Le" if isinstance(rel, Ge) else rel.__class__.__name__
                return RatioPattern(flipped, right.col, feat, float(coeff))

    return None

# Back-compat private alias
_extract_ratio_pattern = extract_ratio_pattern

# ──────────────────────────────────────────────────────────────────────
# Robust constancy tests
# ──────────────────────────────────────────────────────────────────────

def _robust_cv(series: pd.Series) -> float:
    """Robust coefficient of variation via MAD/median. +inf if median≈0 or empty."""
    v = _finite(series).dropna()
    if v.empty:
        return float("inf")
    med = float(v.median())
    if med == 0.0:
        return float("inf")
    mad = float((v - med).abs().median())
    return abs(mad / med)

def _robust_qspan(series: pd.Series, qlo: float = 0.1, qhi: float = 0.9) -> float:
    """Robust middle spread (quantile span). +inf if insufficient data."""
    v = _finite(series).dropna()
    if len(v) < 3:
        return float("inf")
    return float(v.quantile(qhi) - v.quantile(qlo))

def _is_constant_span(series: pd.Series, *, atol: float, rtol: float, min_support: int) -> Tuple[bool, float, Dict[str, float]]:
    v = _finite(series).dropna()
    n = len(v)
    if n < min_support:
        return (False, float("nan"), {"support": float(n)})
    span = float(v.max() - v.min())
    tol = float(atol + rtol * max(1.0, np.abs(v).max()))
    return (span <= tol, float(v.median()), {"support": float(n), "span": span, "tol": tol})

def _is_constant_cv(series: pd.Series, *, cv_tol: float, min_support: int) -> Tuple[bool, float, Dict[str, float]]:
    v = _finite(series).dropna()
    n = len(v)
    if n < min_support:
        return (False, float("nan"), {"support": float(n)})
    cv = _robust_cv(v)
    return (cv <= cv_tol, float(v.median()), {"support": float(n), "cv": cv})

def _is_constant_qspan(series: pd.Series, *, q_tol: float, min_support: int) -> Tuple[bool, float, Dict[str, float]]:
    v = _finite(series).dropna()
    n = len(v)
    if n < min_support:
        return (False, float("nan"), {"support": float(n)})
    span = _robust_qspan(v)
    return (span <= q_tol, float(v.median()), {"support": float(n), "qspan": span})

# ──────────────────────────────────────────────────────────────────────
# Result records
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ConstantRatio:
    """
    A discovered approximately-constant ratio (N+a)/(D+b) on a specific hypothesis slice.
    """
    numerator: str
    denominator: str
    shift_num: int
    shift_den: int

    value_float: float              # representative value (median or center)
    value_display: str              # rationalized display (e.g. '3/2')
    support: int                    # number of finite ratios used (within slice)
    cv: float                       # robust MAD/median coefficient of variation
    qspan: float                    # middle 80% span (absolute units)

    formula_expr: Expr              # Expr for (N+a)/(D+b)
    hypothesis: Optional[Predicate] # subclass slice on which constant holds
    matches_conj_coeff: bool        # True if close to conjecture’s coefficient

# ──────────────────────────────────────────────────────────────────────
# Core mining routine over a *single* slice
# ──────────────────────────────────────────────────────────────────────

def _mine_constant_ratios_on_slice(
    data: pd.DataFrame,
    *,
    numeric_cols: Sequence[str],
    shifts: Sequence[int],
    constancy: str,
    min_support: int,
    atol: float,
    rtol: float,
    cv_tol: float,
    qspan_tol: float,
    coeff_to_match: Optional[float],
    max_denominator: Optional[int],
    same_column: SameColumnMode,
) -> List[ConstantRatio]:
    """
    Scan (N+a)/(D+b) over 'data' (already sliced by hypothesis mask).
    Returns ConstantRatio records with dispersion stats and rationalized display.
    """
    results: List[ConstantRatio] = []

    # choose constancy test
    if constancy == "cv":
        def _const_test(rr: pd.Series) -> Tuple[bool, float, float, float]:
            ok, val, info = _is_constant_cv(rr, cv_tol=cv_tol, min_support=min_support)
            cv = info.get("cv", float("inf"))
            qsp = _robust_qspan(rr)  # compute anyway for ranking tie-breaks
            return ok, val, cv, qsp
    elif constancy == "qspan":
        def _const_test(rr: pd.Series) -> Tuple[bool, float, float, float]:
            ok, val, info = _is_constant_qspan(rr, q_tol=qspan_tol, min_support=min_support)
            cv = _robust_cv(rr)
            qsp = info.get("qspan", float("inf"))
            return ok, val, cv, qsp
    else:  # 'span'
        def _const_test(rr: pd.Series) -> Tuple[bool, float, float, float]:
            ok, val, info = _is_constant_span(rr, atol=atol, rtol=rtol, min_support=min_support)
            cv = _robust_cv(rr)
            qsp = _robust_qspan(rr)
            return ok, val, cv, qsp

    # pre-coerce numeric columns once for this slice
    numeric_data: Dict[str, pd.Series] = {c: pd.to_numeric(data[c], errors="coerce") for c in numeric_cols}

    for num in numeric_cols:
        sN_all = numeric_data[num]
        for den in numeric_cols:
            sD_all = numeric_data[den]
            same_col = (num == den)

            if same_column == "none" and same_col:
                continue

            for a in shifts:
                for b in shifts:
                    # same-column policy: allow only mismatched shifts unless 'all'
                    if same_col and same_column == "mismatch_only" and a == b:
                        continue

                    rr = _ratio_series(sN_all + float(a), sD_all + float(b))
                    rr_f = _finite(rr).dropna()
                    if rr_f.size < min_support:
                        continue

                    ok, val, cv, qsp = _const_test(rr_f)
                    if not ok:
                        continue

                    disp = _rationalize(val, max_denominator)
                    if isinstance(disp, Fraction):
                        value_display = f"{disp.numerator}" if disp.denominator == 1 else f"{disp.numerator}/{disp.denominator}"
                    else:
                        value_display = str(int(disp)) if float(disp).is_integer() else f"{disp}"

                    num_expr = to_expr(num) if a == 0 else (to_expr(num) + Const(a))
                    den_expr = to_expr(den) if b == 0 else (to_expr(den) + Const(b))
                    formula_expr = num_expr / den_expr

                    matches = False
                    if coeff_to_match is not None:
                        matches = np.isclose(val, float(coeff_to_match), atol=atol, rtol=rtol)

                    results.append(
                        ConstantRatio(
                            numerator=num,
                            denominator=den,
                            shift_num=int(a),
                            shift_den=int(b),
                            value_float=float(val),
                            value_display=value_display,
                            support=int(rr_f.size),
                            cv=float(cv),
                            qspan=float(qsp),
                            formula_expr=formula_expr,
                            hypothesis=None,        # filled by caller
                            matches_conj_coeff=bool(matches),
                        )
                    )
    return results

# ──────────────────────────────────────────────────────────────────────
# Public APIs
# ──────────────────────────────────────────────────────────────────────

def find_constant_ratios_over_hypotheses(
    df: pd.DataFrame,
    *,
    hypotheses: Sequence[Optional[Predicate]],
    numeric_cols: Optional[Sequence[str]] = None,
    shifts: Sequence[int] = (-2, -1, 0, 1, 2),
    constancy: str = "cv",            # 'cv' | 'qspan' | 'span'
    cv_tol: float = 0.08,             # ~8% MAD/median variability
    qspan_tol: float = 0.5,           # absolute span (tune to dataset scale)
    atol: float = 1e-6,               # used for coeff matching (and 'span' mode)
    rtol: float = 5e-2,
    min_support: int = 8,
    max_denominator: Optional[int] = 50,
    top_k_per_hypothesis: Optional[int] = 50,
    same_column: SameColumnMode = "mismatch_only",
) -> List[ConstantRatio]:
    """
    Discover (approximately) constant ratios on each provided hypothesis slice.
    """
    all_hits: List[ConstantRatio] = []

    for H in hypotheses:
        m = _mask(df, H)
        if not m.any():
            continue

        data = df.loc[m]

        if numeric_cols is None:
            cols = [c for c in data.columns if _is_numeric_series(data[c])]
        else:
            cols = [c for c in numeric_cols if c in data.columns and _is_numeric_series(data[c])]
        if len(cols) == 0:
            continue

        hits = _mine_constant_ratios_on_slice(
            data,
            numeric_cols=cols,
            shifts=shifts,
            constancy=constancy,
            min_support=min_support,
            atol=atol,
            rtol=rtol,
            cv_tol=cv_tol,
            qspan_tol=qspan_tol,
            coeff_to_match=None,
            max_denominator=max_denominator,
            same_column=same_column,
        )

        # Attach the hypothesis and rank locally
        for h in hits:
            h.hypothesis = H

        hits.sort(
            key=lambda cr: (
                cr.cv,                                  # tighter dispersion first
                -cr.support,                            # then more support
                abs(cr.shift_num) + abs(cr.shift_den),  # simpler shifts
                cr.numerator, cr.denominator, cr.value_float,
            )
        )
        if top_k_per_hypothesis is not None:
            hits = hits[:top_k_per_hypothesis]

        all_hits.extend(hits)

    # Global, stable ordering
    all_hits.sort(
        key=lambda cr: (
            cr.cv,
            -cr.support,
            abs(cr.shift_num) + abs(cr.shift_den),
            repr(cr.hypothesis) if cr.hypothesis is not None else "",
            cr.numerator, cr.denominator, cr.value_float,
        )
    )
    return all_hits

def find_constant_ratios_for_conjecture(
    df: pd.DataFrame,
    conj: Conjecture,
    *,
    hypotheses: Sequence[Optional[Predicate]],
    numeric_cols: Optional[Sequence[str]] = None,
    shifts: Sequence[int] = (-2, -1, 0, 1, 2),
    constancy: str = "cv",
    cv_tol: float = 0.08,
    qspan_tol: float = 0.5,
    atol: float = 1e-6,
    rtol: float = 5e-2,
    min_support: int = 8,
    max_denominator: Optional[int] = 50,
    top_k_per_hypothesis: Optional[int] = 50,
    same_column: SameColumnMode = "mismatch_only",
) -> List[ConstantRatio]:
    """
    Same as `find_constant_ratios_over_hypotheses`, but if `conj` is ratio-style,
    also flags candidates where the discovered constant is close to conj's
    learned coefficient.
    """
    patt = extract_ratio_pattern(conj)
    coeff = patt.coefficient if patt is not None else None

    all_hits: List[ConstantRatio] = []

    for H in hypotheses:
        m = _mask(df, H)
        if not m.any():
            continue

        data = df.loc[m]

        if numeric_cols is None:
            cols = [c for c in data.columns if _is_numeric_series(data[c])]
        else:
            cols = [c for c in numeric_cols if c in data.columns and _is_numeric_series(data[c])]
        if len(cols) == 0:
            continue

        hits = _mine_constant_ratios_on_slice(
            data,
            numeric_cols=cols,
            shifts=shifts,
            constancy=constancy,
            min_support=min_support,
            atol=atol,
            rtol=rtol,
            cv_tol=cv_tol,
            qspan_tol=qspan_tol,
            coeff_to_match=coeff,
            max_denominator=max_denominator,
            same_column=same_column,
        )

        for h in hits:
            h.hypothesis = H

        # Prefer coefficient matches within the slice, then dispersion/support/shift simplicity
        hits.sort(
            key=lambda cr: (
                not cr.matches_conj_coeff,
                cr.cv,
                -cr.support,
                abs(cr.shift_num) + abs(cr.shift_den),
                cr.numerator, cr.denominator, cr.value_float,
            )
        )
        if top_k_per_hypothesis is not None:
            hits = hits[:top_k_per_hypothesis]

        all_hits.extend(hits)

    # Global ordering: prioritize matches overall, then dispersion/support
    all_hits.sort(
        key=lambda cr: (
            not cr.matches_conj_coeff,
            cr.cv,
            -cr.support,
            abs(cr.shift_num) + abs(cr.shift_den),
            repr(cr.hypothesis) if cr.hypothesis is not None else "",
            cr.numerator, cr.denominator, cr.value_float,
        )
    )
    return all_hits
