# # src/txgraffiti2025/graffiti_lp_list_integer_aware.py

# from __future__ import annotations
# from typing import Iterable, List, Optional, Tuple
# import numpy as np
# import pandas as pd

# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE
# from txgraffiti2025.forms.utils import to_expr, Const, ceil, floor, BinOp

# def _mask_for(cond, gcr, nrows: int) -> np.ndarray:
#     if cond is None or cond is TRUE:
#         return np.ones(nrows, dtype=bool)
#     return gcr._mask_cached(cond)

# def _arr_expr(expr, df: pd.DataFrame) -> np.ndarray:
#     s = to_expr(expr).eval(df)
#     return s.to_numpy(dtype=float, copy=False)

# def _is_integer_series(vals: np.ndarray, tol: float = 1e-12) -> bool:
#     m = np.isfinite(vals)
#     if not m.any():
#         return False
#     frac = np.abs(vals[m] - np.round(vals[m]))
#     return bool(np.all(frac <= tol))

# def _touch(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> Tuple[int, float, int]:
#     m = mask & np.isfinite(lhs) & np.isfinite(rhs)
#     n = int(m.sum())
#     if n == 0:
#         return 0, 0.0, 0
#     d = np.abs(lhs[m] - rhs[m])
#     t = atol + rtol * np.abs(rhs[m])
#     tc = int((d <= t).sum())
#     return tc, (tc / n), n

# def _valid_ge(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> bool:
#     m = mask & np.isfinite(lhs) & np.isfinite(rhs)
#     if not m.any():
#         return False
#     return bool(np.all(lhs[m] + (atol + rtol * np.abs(rhs[m])) >= rhs[m]))

# def _valid_le(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> bool:
#     m = mask & np.isfinite(lhs) & np.isfinite(rhs)
#     if not m.any():
#         return False
#     return bool(np.all(lhs[m] - (atol + rtol * np.abs(rhs[m])) <= rhs[m]))

# def _build_affine_arrays(df: pd.DataFrame, cj: Conjecture):
#     """
#     Returns (t_arr, terms: List[Tuple[label, arr, expr]], b_arr, b_expr)
#     where RHS = sum(terms) + b_arr
#     """
#     rel = cj.relation
#     lhs_e = rel.left
#     rhs_e = rel.right

#     # If coefficients are attached, prefer those (fast, unambiguous).
#     pairs = getattr(cj, "coefficient_pairs", None)
#     intercept = getattr(cj, "intercept", 0.0)

#     t_arr = lhs_e.eval(df).to_numpy(dtype=float, copy=False)

#     if pairs is not None:
#         terms = []
#         for name, coef in pairs:
#             col_e = to_expr(name)
#             arr = coef * col_e.eval(df).to_numpy(dtype=float, copy=False)
#             terms.append((name, arr, Const(coef) * col_e))
#         b_arr = np.full(len(df), float(intercept), dtype=float)
#         b_expr = Const(float(intercept))
#         return t_arr, terms, b_arr, b_expr

#     # Fallback: evaluate whole RHS and treat as a single term.
#     rhs_arr = rhs_e.eval(df).to_numpy(dtype=float, copy=False)
#     return t_arr, [("rhs", rhs_arr, rhs_e)], np.zeros(len(df), dtype=float), Const(0.0)

# def _sum_expr(exprs):
#     acc = None
#     for e in exprs:
#         acc = e if acc is None else (acc + e)
#     return acc or Const(0.0)

# def _ceil_expr(e):  return ceil(e)
# def _floor_expr(e): return floor(e)

# def _make_rhs_expr_from_terms(terms_exprs, b_expr):
#     exprs = [te for te in terms_exprs]
#     if getattr(b_expr, "__class__", None) is not None:
#         exprs.append(b_expr)
#     return _sum_expr(exprs)

# def lift_integer_aware(
#     *,
#     df: pd.DataFrame,
#     gcr,
#     conjectures: Iterable[Conjecture],
#     touch_atol: float = 0.0,
#     touch_rtol: float = 0.0,
#     intercept_zero_eps: float = 1e-12,
# ) -> List[Conjecture]:
#     """
#     Integer-aware tightening using term-level ceil/floor splits.

#     For each affine conjecture (RHS = Σ a_j x_j + b) with Ge/Le relation and condition H:
#       • If target (LHS) is integer on H-support:
#           Ge: try ceil(RHS) and the split: Σ ceil(a_j x_j) + [ceil(b) if b≠0] − (m_eff − 1)
#           Le: try floor(RHS) and the split: Σ floor(a_j x_j) + [floor(b) if b≠0] − (m_eff − 1)

#         where m_eff = (#nonzero terms) + (1 if b≠0 else 0).

#       • Keep the best candidate that:
#           (a) remains valid on H (no violations under tolerance), and
#           (b) increases touch_count, or ties on touches but is strictly tighter on average,
#               or ties on both but increases touch_rate.
#     """
#     out: List[Conjecture] = []
#     nrows = len(df)

#     for cj in conjectures:
#         rel = cj.relation
#         if not isinstance(rel, (Ge, Le)):
#             out.append(cj)
#             continue

#         # Condition mask
#         mask = _mask_for(cj.condition, gcr, nrows)

#         # Decompose RHS into arrays/expr parts and build the base arrays
#         # _build_affine_arrays(df, cj) must return:
#         #   lhs_arr: np.ndarray
#         #   terms:   List[Tuple[str, np.ndarray, Expr]]    # (label, arr, expr) per term a_j x_j
#         #   b_arr:   np.ndarray                             # intercept array (possibly all zeros)
#         #   b_expr:  Expr                                   # intercept expression (Const(0) if zero)
#         lhs_arr, terms, b_arr, b_expr = _build_affine_arrays(df, cj)

#         # Base RHS array (no lifting)
#         rhs_arr0 = np.zeros_like(lhs_arr)
#         for _, arr, _ in terms:
#             rhs_arr0 += arr
#         rhs_arr0 = rhs_arr0 + b_arr

#         # Base touches/rate for current conjecture
#         tc0, tr0, _ = _touch(lhs_arr, rhs_arr0, mask, atol=touch_atol, rtol=touch_rtol)
#         best = cj
#         best_key = (tc0, tr0, 0.0)  # (touches, touch_rate, avg_gain)

#         # Only attempt lifting if LHS is integer on H-support
#         integer_target = _is_integer_series(lhs_arr[mask])
#         if not integer_target:
#             out.append(best)
#             continue

#         # Figure out #effective terms and whether b is nonzero on support
#         # Note: we only count b if it's actually nonzero on H-support.
#         b_is_nonzero = bool(np.any(np.abs(b_arr[mask]) > intercept_zero_eps))
#         m_terms = len(terms)
#         m_eff = m_terms + (1 if b_is_nonzero else 0)

#         # Build candidate RHS (arrays + exprs)
#         candidates: List[Tuple[str, np.ndarray, Expr]] = []

#         # Whole ceil/floor on the full RHS
#         if isinstance(rel, Ge):
#             rhs_arr_whole = np.ceil(rhs_arr0)
#             rhs_expr_whole = ceil(_make_rhs_expr_from_terms([te for _, _, te in terms], b_expr))
#             candidates.append(("ceil_whole", rhs_arr_whole, rhs_expr_whole))
#         else:
#             rhs_arr_whole = np.floor(rhs_arr0)
#             rhs_expr_whole = floor(_make_rhs_expr_from_terms([te for _, _, te in terms], b_expr))
#             candidates.append(("floor_whole", rhs_arr_whole, rhs_expr_whole))

#         # Split lifts with correct (m_eff - 1) adjustment
#         if m_eff >= 1:
#             if isinstance(rel, Ge):
#                 # Σ ceil(term) + [ceil(b) if b≠0] − (m_eff − 1)
#                 term_ceils_arr = [np.ceil(arr) for _, arr, _ in terms]
#                 rhs_arr_split = np.sum(term_ceils_arr, axis=0)
#                 rhs_expr_parts = [_ceil_expr(te) for _, _, te in terms]

#                 if b_is_nonzero:
#                     rhs_arr_split = rhs_arr_split + np.ceil(b_arr)
#                     rhs_expr_parts.append(_ceil_expr(b_expr))

#                 if m_eff >= 2:
#                     rhs_arr_split = rhs_arr_split - (m_eff - 1)
#                     rhs_expr_split = _make_rhs_expr_from_terms(rhs_expr_parts, Const(0.0)) - Const(m_eff - 1)
#                 else:
#                     rhs_expr_split = _make_rhs_expr_from_terms(rhs_expr_parts, Const(0.0))

#                 candidates.append(("ceil_split", rhs_arr_split, rhs_expr_split))

#             else:
#                 # Σ floor(term) + [floor(b) if b≠0] − (m_eff − 1)
#                 term_floors_arr = [np.floor(arr) for _, arr, _ in terms]
#                 rhs_arr_split = np.sum(term_floors_arr, axis=0)
#                 rhs_expr_parts = [_floor_expr(te) for _, _, te in terms]

#                 if b_is_nonzero:
#                     rhs_arr_split = rhs_arr_split + np.floor(b_arr)
#                     rhs_expr_parts.append(_floor_expr(b_expr))

#                 if m_eff >= 2:
#                     rhs_arr_split = rhs_arr_split - (m_eff - 1)
#                     rhs_expr_split = _make_rhs_expr_from_terms(rhs_expr_parts, Const(0.0)) - Const(m_eff - 1)
#                 else:
#                     rhs_expr_split = _make_rhs_expr_from_terms(rhs_expr_parts, Const(0.0))

#                 candidates.append(("floor_split", rhs_arr_split, rhs_expr_split))

#         # Evaluate candidates: keep valid ones that improve touches or tightness
#         for label, rhs_arr1, rhs_expr1 in candidates:
#             if isinstance(rel, Ge):
#                 if not _valid_ge(lhs_arr, rhs_arr1, mask, atol=touch_atol, rtol=touch_rtol):
#                     continue
#             else:
#                 if not _valid_le(lhs_arr, rhs_arr1, mask, atol=touch_atol, rtol=touch_rtol):
#                     continue

#             tc1, tr1, _ = _touch(lhs_arr, rhs_arr1, mask, atol=touch_atol, rtol=touch_rtol)

#             # Tightness gain on support (average improvement in the “right” direction)
#             sup = mask & np.isfinite(lhs_arr) & np.isfinite(rhs_arr0) & np.isfinite(rhs_arr1)
#             if not sup.any():
#                 continue
#             if isinstance(rel, Ge):
#                 gain = float(np.mean(np.maximum(0.0, rhs_arr1[sup] - rhs_arr0[sup])))
#                 new_rel = Ge(rel.left, rhs_expr1)
#             else:
#                 gain = float(np.mean(np.maximum(0.0, rhs_arr0[sup] - rhs_arr1[sup])))
#                 new_rel = Le(rel.left, rhs_expr1)

#             # Keep if strictly more touches; or equal touches but strictly tighter (or higher touch_rate)
#             if (tc1 > best_key[0]) or (tc1 == best_key[0] and gain > 1e-12) or (tc1 == best_key[0] and tr1 > best_key[1]):
#                 cand = Conjecture(relation=new_rel, condition=cj.condition)
#                 # Preserve LP metadata for downstream tooling
#                 setattr(cand, "coefficient_pairs", getattr(cj, "coefficient_pairs", None))
#                 setattr(cand, "intercept", getattr(cj, "intercept", None))
#                 setattr(cand, "touch_count", tc1)
#                 setattr(cand, "touch_rate", tr1)
#                 setattr(cand, "support_n", int(mask.sum()))
#                 best = cand
#                 best_key = (tc1, tr1, gain)

#         out.append(best)

#     return out

# src/txgraffiti2025/graffiti_lp_list_integer_aware.py
# src/txgraffiti2025/graffiti_lp_list_integer_aware.py

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE
from txgraffiti2025.forms.utils import to_expr, Const, ceil, floor

try:
    # Optional pruning/dominance filter; ok if missing.
    from txgraffiti2025.processing.post import morgan_filter as _morgan_filter
except Exception:
    _morgan_filter = None  # type: ignore

# ───────────────────────────── mask helpers ───────────────────────────── #

def _mask_for(cond, gcr, nrows: int) -> np.ndarray:
    """
    Public, tolerant mask accessor:
    - Prefer gcr.mask_for / gcr.condition_mask_for
    - Fall back to _mask_cached if still present
    - TRUE/None ⇒ all-True mask
    """
    if cond is None or cond is TRUE:
        return np.ones(nrows, dtype=bool)

    if hasattr(gcr, "mask_for"):
        return gcr.mask_for(cond)
    if hasattr(gcr, "condition_mask_for"):
        return gcr.condition_mask_for(cond)
    if hasattr(gcr, "_mask_cached"):
        return gcr._mask_cached(cond)  # legacy shim
    if hasattr(cond, "mask"):
        s = cond.mask(gcr.df).reindex(gcr.df.index, fill_value=False)
        if s.dtype is not bool:
            s = s.fillna(False).astype(bool, copy=False)
        return s.to_numpy(dtype=bool, copy=False)

    raise AttributeError(
        "GraffitiClassRelations must expose a public mask method (mask_for/condition_mask_for), "
        "or Predicate must provide .mask(df)."
    )

# ───────────────────────────── array helpers ───────────────────────────── #

def _arr_expr(expr, df: pd.DataFrame) -> np.ndarray:
    s = to_expr(expr).eval(df)
    return s.to_numpy(dtype=float, copy=False)

def _is_integer_series(vals: np.ndarray, tol: float = 1e-12) -> bool:
    m = np.isfinite(vals)
    if not m.any():
        return False
    frac = np.abs(vals[m] - np.round(vals[m]))
    return bool(np.all(frac <= tol))

def _touch(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> Tuple[int, float, int]:
    m = mask & np.isfinite(lhs) & np.isfinite(rhs)
    n = int(m.sum())
    if n == 0:
        return 0, 0.0, 0
    d = np.abs(lhs[m] - rhs[m])
    t = atol + rtol * np.abs(rhs[m])
    tc = int((d <= t).sum())
    return tc, (tc / n), n

def _valid_ge(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> bool:
    m = mask & np.isfinite(lhs) & np.isfinite(rhs)
    if not m.any():
        return False
    return bool(np.all(lhs[m] + (atol + rtol * np.abs(rhs[m])) >= rhs[m]))

def _valid_le(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> bool:
    m = mask & np.isfinite(lhs) & np.isfinite(rhs)
    if not m.any():
        return False
    return bool(np.all(lhs[m] - (atol + rtol * np.abs(rhs[m])) <= rhs[m]))

def _valid_eq(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> bool:
    m = mask & np.isfinite(lhs) & np.isfinite(rhs)
    if not m.any():
        return False
    d = np.abs(lhs[m] - rhs[m])
    t = atol + rtol * np.abs(rhs[m])
    return bool(np.all(d <= t))

# ───────────────────── RHS decomposition (affine) ───────────────────── #

def _build_affine_arrays(df: pd.DataFrame, cj: Conjecture):
    """
    Returns (lhs_arr, terms: List[Tuple[label, arr, expr]], b_arr, b_expr)
    where RHS = sum(terms) + b_arr and expr mirrors the same structure.

    Prefers attached LP metadata (coefficient_pairs, intercept) if present:
      - coefficient_pairs: List[Tuple[str, float]]   # (feature, coef)
      - intercept: float
    """
    rel = cj.relation
    lhs_e = rel.left
    rhs_e = rel.right

    pairs = getattr(cj, "coefficient_pairs", None)
    intercept = getattr(cj, "intercept", 0.0)

    lhs_arr = lhs_e.eval(df).to_numpy(dtype=float, copy=False)

    if pairs is not None:
        terms = []
        for name, coef in pairs:
            col_e = to_expr(name)
            arr = float(coef) * col_e.eval(df).to_numpy(dtype=float, copy=False)
            terms.append((str(name), arr, Const(float(coef)) * col_e))
        b_arr = np.full(len(df), float(intercept), dtype=float)
        b_expr = Const(float(intercept))
        return lhs_arr, terms, b_arr, b_expr

    # Fallback: treat whole RHS as a single “term”; no intercept.
    rhs_arr = rhs_e.eval(df).to_numpy(dtype=float, copy=False)
    return lhs_arr, [("rhs", rhs_arr, rhs_e)], np.zeros(len(df), dtype=float), Const(0.0)

def _sum_expr(exprs):
    acc = None
    for e in exprs:
        acc = e if acc is None else (acc + e)
    return acc or Const(0.0)

def _ceil_expr(e):  return ceil(e)
def _floor_expr(e): return floor(e)

def _make_rhs_expr_from_terms(terms_exprs, b_expr):
    exprs = [te for te in terms_exprs]
    exprs.append(b_expr)
    return _sum_expr(exprs)

# ─────────────────────── integer-aware lifting ─────────────────────── #

def lift_integer_aware(
    *,
    df: pd.DataFrame,
    gcr,
    conjectures: Iterable[Conjecture],
    touch_atol: float = 0.0,
    touch_rtol: float = 0.0,
    intercept_zero_eps: float = 1e-12,
) -> List[Conjecture]:
    """
    Integer-aware tightening using term-level ceil/floor splits.

    For each affine conjecture (RHS = Σ a_j x_j + b) with Ge/Le relation and condition H:
      • If target (LHS) is integer on H-support:
          Ge: try ceil(RHS) and the split: Σ ceil(a_j x_j) + [ceil(b) if b≠0] − (m_eff − 1)
          Le: try floor(RHS) and the split: Σ floor(a_j x_j) + [floor(b) if b≠0] − (m_eff − 1)

        where m_eff = (#nonzero terms) + (1 if b≠0 else 0).

      • Keep the best candidate that:
          (a) remains valid on H (no violations under tolerance), and
          (b) increases touch_count, or ties on touches but is strictly tighter on average,
              or ties on both but increases touch_rate.
    """
    out: List[Conjecture] = []
    nrows = len(df)

    for cj in conjectures:
        rel = cj.relation
        if not isinstance(rel, (Ge, Le)):
            out.append(cj)
            continue

        # Condition mask
        mask = _mask_for(cj.condition, gcr, nrows)

        # Decompose RHS into arrays/expr parts and build the base arrays
        lhs_arr, terms, b_arr, b_expr = _build_affine_arrays(df, cj)

        # Base RHS array (no lifting)
        rhs_arr0 = np.zeros_like(lhs_arr)
        for _, arr, _ in terms:
            rhs_arr0 += arr
        rhs_arr0 = rhs_arr0 + b_arr

        # Base touches/rate for current conjecture
        tc0, tr0, _ = _touch(lhs_arr, rhs_arr0, mask, atol=touch_atol, rtol=touch_rtol)
        best = cj
        best_key = (tc0, tr0, 0.0)  # (touches, touch_rate, avg_gain)

        # Only attempt lifting if LHS is integer on H-support
        integer_target = _is_integer_series(lhs_arr[mask])
        if not integer_target:
            # Still ensure metrics are attached for downstream sorting
            setattr(best, "touch_count", getattr(best, "touch_count", tc0))
            setattr(best, "touch_rate", getattr(best, "touch_rate", tr0))
            setattr(best, "support_n", getattr(best, "support_n", int(mask.sum())))
            out.append(best)
            continue

        # Determine effective term count; include b only if it's nonzero on support
        b_is_nonzero = bool(np.any(np.abs(b_arr[mask]) > intercept_zero_eps))
        m_terms = len(terms)
        m_eff = m_terms + (1 if b_is_nonzero else 0)

        # Build candidate RHS (arrays + exprs)
        candidates: List[Tuple[str, np.ndarray, object]] = []

        # Whole ceil/floor on the full RHS
        if isinstance(rel, Ge):
            rhs_arr_whole = np.ceil(rhs_arr0)
            rhs_expr_whole = ceil(_make_rhs_expr_from_terms([te for _, _, te in terms], b_expr))
            candidates.append(("ceil_whole", rhs_arr_whole, rhs_expr_whole))
        else:
            rhs_arr_whole = np.floor(rhs_arr0)
            rhs_expr_whole = floor(_make_rhs_expr_from_terms([te for _, _, te in terms], b_expr))
            candidates.append(("floor_whole", rhs_arr_whole, rhs_expr_whole))

        # Split lifts with correct (m_eff − 1) adjustment
        if m_eff >= 1:
            if isinstance(rel, Ge):
                # Σ ceil(term) + [ceil(b) if b≠0] − (m_eff − 1)
                term_ceils_arr = [np.ceil(arr) for _, arr, _ in terms]
                rhs_arr_split = np.sum(term_ceils_arr, axis=0)
                rhs_expr_parts = [_ceil_expr(te) for _, _, te in terms]

                if b_is_nonzero:
                    rhs_arr_split = rhs_arr_split + np.ceil(b_arr)
                    rhs_expr_parts.append(_ceil_expr(b_expr))

                if m_eff >= 2:
                    rhs_arr_split = rhs_arr_split - (m_eff - 1)
                    rhs_expr_split = _sum_expr(rhs_expr_parts) - Const(m_eff - 1)
                else:
                    rhs_expr_split = _sum_expr(rhs_expr_parts)

                candidates.append(("ceil_split", rhs_arr_split, rhs_expr_split))

            else:
                # Σ floor(term) + [floor(b) if b≠0] − (m_eff − 1)
                term_floors_arr = [np.floor(arr) for _, arr, _ in terms]
                rhs_arr_split = np.sum(term_floors_arr, axis=0)
                rhs_expr_parts = [_floor_expr(te) for _, _, te in terms]

                if b_is_nonzero:
                    rhs_arr_split = rhs_arr_split + np.floor(b_arr)
                    rhs_expr_parts.append(_floor_expr(b_expr))

                if m_eff >= 2:
                    rhs_arr_split = rhs_arr_split - (m_eff - 1)
                    rhs_expr_split = _sum_expr(rhs_expr_parts) - Const(m_eff - 1)
                else:
                    rhs_expr_split = _sum_expr(rhs_expr_parts)

                candidates.append(("floor_split", rhs_arr_split, rhs_expr_split))

        # Evaluate candidates: keep valid ones that improve touches or tightness
        for label, rhs_arr1, rhs_expr1 in candidates:
            if isinstance(rel, Ge):
                if not _valid_ge(lhs_arr, rhs_arr1, mask, atol=touch_atol, rtol=touch_rtol):
                    continue
            else:
                if not _valid_le(lhs_arr, rhs_arr1, mask, atol=touch_atol, rtol=touch_rtol):
                    continue

            tc1, tr1, _ = _touch(lhs_arr, rhs_arr1, mask, atol=touch_atol, rtol=touch_rtol)

            # Tightness gain on support (avg improvement in the “right” direction)
            sup = mask & np.isfinite(lhs_arr) & np.isfinite(rhs_arr0) & np.isfinite(rhs_arr1)
            if not sup.any():
                continue
            if isinstance(rel, Ge):
                gain = float(np.mean(np.maximum(0.0, rhs_arr1[sup] - rhs_arr0[sup])))
                new_rel = Ge(rel.left, rhs_expr1)
            else:
                gain = float(np.mean(np.maximum(0.0, rhs_arr0[sup] - rhs_arr1[sup])))
                new_rel = Le(rel.left, rhs_expr1)

            # Keep if strictly more touches; or equal touches but strictly tighter (or higher touch_rate)
            if (tc1 > best_key[0]) or (tc1 == best_key[0] and gain > 1e-12) or (tc1 == best_key[0] and tr1 > best_key[1]):
                cand = Conjecture(relation=new_rel, condition=cj.condition)
                # Preserve LP metadata for downstream tooling
                setattr(cand, "coefficient_pairs", getattr(cj, "coefficient_pairs", None))
                setattr(cand, "intercept", getattr(cj, "intercept", None))
                setattr(cand, "touch_count", tc1)
                setattr(cand, "touch_rate", tr1)
                setattr(cand, "support_n", int(mask.sum()))
                best = cand
                best_key = (tc1, tr1, gain)

        # Ensure metrics on best
        if not hasattr(best, "touch_count"):
            setattr(best, "touch_count", best_key[0])
            setattr(best, "touch_rate", best_key[1])
            setattr(best, "support_n", int(mask.sum()))

        out.append(best)
    return out

# ───────────────────── post-process & sort helpers ───────────────────── #

def _ensure_metrics_and_validate(
    df: pd.DataFrame,
    gcr,
    c: Conjecture,
    *,
    touch_atol: float = 0.0,
    touch_rtol: float = 0.0,
) -> Optional[Conjecture]:
    """
    Make sure touch_count/touch_rate/support_n are present and that the
    conjecture is valid on its condition mask. Returns the conjecture or None.
    """
    cond = getattr(c, "condition", None)
    mask = _mask_for(cond, gcr, len(df))

    lhs_arr = c.relation.left.eval(df).to_numpy(dtype=float, copy=False)
    rhs_arr = c.relation.right.eval(df).to_numpy(dtype=float, copy=False)

    # validity check
    rel = c.relation
    if isinstance(rel, Ge):
        valid = _valid_ge(lhs_arr, rhs_arr, mask, atol=touch_atol, rtol=touch_rtol)
    elif isinstance(rel, Le):
        valid = _valid_le(lhs_arr, rhs_arr, mask, atol=touch_atol, rtol=touch_rtol)
    elif isinstance(rel, Eq):
        valid = _valid_eq(lhs_arr, rhs_arr, mask, atol=touch_atol, rtol=touch_rtol)
    else:
        valid = True

    if not valid:
        return None

    # ensure metrics
    tc, tr, n = _touch(lhs_arr, rhs_arr, mask, atol=touch_atol, rtol=touch_rtol)
    if not hasattr(c, "touch_count"): setattr(c, "touch_count", tc)
    if not hasattr(c, "touch_rate"):  setattr(c, "touch_rate", tr)
    if not hasattr(c, "support_n"):   setattr(c, "support_n", n)
    return c

def _dedup_str(lst: List[Conjecture]) -> List[Conjecture]:
    seen, out = set(), []
    for c in lst:
        s = str(c)
        if s in seen:
            continue
        seen.add(s); out.append(c)
    return out

def process_and_sort_conjectures(
    *,
    df: pd.DataFrame,
    gcr,
    conjectures: Iterable[Conjecture],
    touch_atol: float = 0.0,
    touch_rtol: float = 0.0,
    use_morgan_filter: bool = True,
    top_k: Optional[int] = None,
    per_condition: bool = False,
) -> List[Conjecture]:
    """
    Validate, deduplicate, (optionally) prune, and sort conjectures.

    Sorting key: (touch_count desc, support_n desc, touch_rate desc).
    If per_condition=True, the top_k is taken per (stringified) condition.

    Returns a flat, sorted list.
    """
    # 1) validate + ensure metrics
    kept: List[Conjecture] = []
    for c in conjectures:
        cc = _ensure_metrics_and_validate(df, gcr, c, touch_atol=touch_atol, touch_rtol=touch_rtol)
        if cc is not None:
            kept.append(cc)

    if not kept:
        return []

    # 2) dedup
    kept = _dedup_str(kept)

    # 3) optional pruning via Morgan filter
    if use_morgan_filter and _morgan_filter is not None:
        kept = _morgan_filter(df, kept).kept

    # 4) sort
    def _key(c: Conjecture):
        return (getattr(c, "touch_count", 0), getattr(c, "support_n", 0), getattr(c, "touch_rate", 0.0))

    if per_condition and top_k is not None and top_k > 0:
        buckets: Dict[str, List[Conjecture]] = {}
        for c in kept:
            cond_str = "TRUE" if (getattr(c, "condition", None) is None or getattr(c, "condition", None) is TRUE) else str(getattr(c, "condition"))
            buckets.setdefault(cond_str, []).append(c)
        out: List[Conjecture] = []
        for _, grp in buckets.items():
            grp.sort(key=_key, reverse=True)
            out.extend(grp[:int(top_k)])
        return out

    kept.sort(key=_key, reverse=True)
    if top_k is not None and top_k > 0:
        return kept[:int(top_k)]
    return kept

# ───────────────────────── one-shot convenience ───────────────────────── #

def lift_and_process_sorted(
    *,
    df: pd.DataFrame,
    gcr,
    conjectures: Iterable[Conjecture],
    touch_atol: float = 0.0,
    touch_rtol: float = 0.0,
    intercept_zero_eps: float = 1e-12,
    use_morgan_filter: bool = True,
    top_k: Optional[int] = None,
    per_condition: bool = False,
) -> List[Conjecture]:
    """
    Convenience: integer-aware lift followed by validate/dedup/prune/sort.
    """
    lifted = lift_integer_aware(
        df=df,
        gcr=gcr,
        conjectures=conjectures,
        touch_atol=touch_atol,
        touch_rtol=touch_rtol,
        intercept_zero_eps=intercept_zero_eps,
    )
    return process_and_sort_conjectures(
        df=df,
        gcr=gcr,
        conjectures=lifted,
        touch_atol=touch_atol,
        touch_rtol=touch_rtol,
        use_morgan_filter=use_morgan_filter,
        top_k=top_k,
        per_condition=per_condition,
    )
