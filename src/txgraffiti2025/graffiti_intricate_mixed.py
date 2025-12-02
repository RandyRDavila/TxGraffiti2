# src/txgraffiti2025/generators/graffiti_intricate_mixed.py
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import (
    to_expr, Expr, Const, floor, ceil, sqrt,
)
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Ge, Le, TRUE
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)
from txgraffiti2025.processing.post import morgan_filter

__all__ = ["GraffitiLPIntricate", "IntricateConfig"]


# ───────────────────────── small helpers ───────────────────────── #

from fractions import Fraction

def _as_fraction(x, max_denom: int) -> Fraction:
    # Accept Const, Fraction, int, float
    if hasattr(x, "value") and isinstance(x.value, Fraction):  # Const(Fraction)
        return x.value
    if isinstance(x, Fraction):
        return x
    return Fraction(float(x)).limit_denominator(max_denom)

def mul_consts(*vals, max_denom: int = 300) -> Const:
    """
    Multiply several Const/numeric values into a single Const with a reduced Fraction.
    """
    f = Fraction(1, 1)
    for v in vals:
        f *= _as_fraction(v, max_denom)
    return Const(f.limit_denominator(max_denom))

def to_frac_const(val: float, max_denom: int = 300) -> Const:
    """Rationalize to improve pretty-printing and stability in Expr trees."""
    return Const(Fraction(val).limit_denominator(max_denom))


def _pred_cache_key(p: Predicate | None) -> str:
    if p is None or p is TRUE:
        return "TRUE"
    n = getattr(p, "name", None)
    return f"name:{n}" if n else f"repr:{repr(p)}"


def _expr_cache_key(e: Expr) -> str:
    return repr(e)


class _MaskCache:
    """Memoizes boolean masks per-predicate for a fixed DataFrame."""
    __slots__ = ("df", "cache")

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cache: dict[str, np.ndarray] = {}

    def get(self, pred: Predicate | None) -> np.ndarray:
        if pred is None or pred is TRUE:
            return np.ones(len(self.df), dtype=bool)
        key = _pred_cache_key(pred)
        a = self.cache.get(key)
        if a is not None:
            return a
        s = pred.mask(self.df).reindex(self.df.index, fill_value=False)
        if s.dtype is not bool:
            s = s.fillna(False).astype(bool, copy=False)
        a = s.to_numpy(copy=False)
        self.cache[key] = a
        return a


class _ExprEvalCache:
    """
    Evaluate Exprs once on the FULL df (aligned to df.index), then slice by masks.
    Avoids repeated expr.eval() calls.
    """
    __slots__ = ("df", "array_cache")

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.array_cache: dict[str, np.ndarray] = {}

    def arr(self, expr: Expr) -> np.ndarray:
        k = _expr_cache_key(expr)
        a = self.array_cache.get(k)
        if a is not None:
            return a
        s = expr.eval(self.df)
        if hasattr(s, "to_numpy"):
            a = s.to_numpy(dtype=float, copy=False)
        else:
            a = np.asarray(s, dtype=float)
            if a.ndim == 0:
                a = np.full(len(self.df), float(a), dtype=float)
        self.array_cache[k] = a
        return a


def _batch_touch_counts(
    df: pd.DataFrame,
    conjs: Sequence[Conjecture],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> dict[int, int]:
    """
    Vectorized touch counting:
    groups by (condition, left, right) to eval each (L,R,mask) once.
    """
    if not conjs:
        return {}

    mcache = _MaskCache(df)
    ecache = _ExprEvalCache(df)

    groups: dict[tuple[str, str, str], list[int]] = {}
    for i, c in enumerate(conjs):
        Hk = _pred_cache_key(c.condition)
        Lk = _expr_cache_key(c.relation.left)
        Rk = _expr_cache_key(c.relation.right)
        groups.setdefault((Hk, Lk, Rk), []).append(i)

    out: dict[int, int] = {}
    for (Hk, Lk, Rk), idxs in groups.items():
        rep = conjs[idxs[0]]
        mask = mcache.get(rep.condition)
        if not np.any(mask):
            for j in idxs:
                out[j] = 0
            continue
        L = ecache.arr(rep.relation.left)
        R = ecache.arr(rep.relation.right)
        ok = np.isfinite(L) & np.isfinite(R) & mask
        cnt = int(np.isclose(L[ok], R[ok], rtol=rtol, atol=atol).sum()) if np.any(ok) else 0
        for j in idxs:
            out[j] = cnt
    return out


def rank_and_filter_fast(df: pd.DataFrame, conjs: List[Conjecture], min_touch: int) -> List[Conjecture]:
    """Sort by touch desc (batch computed) and apply Morgan filter for validity."""
    if not conjs:
        return []
    counts = _batch_touch_counts(df, conjs)
    order = sorted(range(len(conjs)), key=lambda i: counts.get(i, 0), reverse=True)
    prelim = [conjs[i] for i in order if counts.get(i, 0) >= int(min_touch)]
    return list(morgan_filter(df, prelim).kept)


# ───────────────────────── config ───────────────────────── #

@dataclass
class IntricateConfig:
    """
    Tunables used by GraffitiLPIntricate.
    """
    min_touch_keep: int = 3
    max_denom: int = 30
    # Hypothesis enumeration / pruning
    min_support_hyp: int = 10
    treat_binary_ints: bool = True
    include_pairwise_hyps: bool = True


# ──────────────────────── main class ──────────────────────── #

class GraffitiLPIntricate:
    """
    Intricate (mixed) inequality generator using the classic technique:

      For each hypothesis H, primary x, secondary y:

        sqrt-mix:
          lower:  y ≥ w * (cmin_x * x + cmin_sqrt * sqrt(y))   using best of
                  { base, ceil(whole), ceil-split }
          upper:  y ≤ w * (cmax_x * x + cmax_sqrt * sqrt(y))   using best of
                  { base, floor(whole), floor-split }

        square-mix:
          lower:  y ≥ w * (cmin_x * x + cmin_sq * y^2)         best of { base, ceil(whole) }
          upper:  y ≤ w * (cmax_x * x + cmax_sq * y^2)         best of { base, floor(whole) }

    Design goals:
      • Single _ExprEvalCache per hypothesis (no repeated expr.eval()).
      • Pure array selection for validity/tightness; mirror with symbolic Expr builders.
      • Keep the surrounding API similar to GraffitiLP (hyps_kept, numeric_columns, rank_and_filter).
    """

    # ------------- lifecycle -------------

    def __init__(self, df: pd.DataFrame, *, config: Optional[IntricateConfig] = None):
        self.df = df
        self.config = config or IntricateConfig()

        # hypotheses
        self.base_hyp: Predicate = detect_base_hypothesis(df)
        hyps_all = enumerate_boolean_hypotheses(
            df,
            treat_binary_ints=self.config.treat_binary_ints,
            include_base=True,
            include_pairs=self.config.include_pairwise_hyps,
            skip_always_false=True,
        )
        self.hyps_kept, _ = simplify_and_dedup_hypotheses(
            df,
            hyps_all,
            min_support=self.config.min_support_hyp,
            treat_binary_ints=self.config.treat_binary_ints,
        )

        # columns split
        self.bool_columns, self.numeric_columns = self._split_columns(df)

    # ------------- columns & masks -------------

    @staticmethod
    def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        bool_cols: List[str] = []
        for c in df.columns:
            s = df[c]
            if s.dtype == bool:
                bool_cols.append(c)
            elif pd.api.types.is_integer_dtype(s):
                vals = pd.unique(s.dropna())
                try:
                    ints = set(int(v) for v in vals)
                except Exception:
                    continue
                if len(ints) <= 2 and ints.issubset({0, 1}):
                    bool_cols.append(c)
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in bool_cols]
        return bool_cols, num_cols

    @staticmethod
    def _mask_for(df: pd.DataFrame, hypothesis) -> np.ndarray:
        if hypothesis is None or hypothesis is TRUE:
            return np.ones(len(df), dtype=bool)
        return np.asarray(hypothesis.mask(df), dtype=bool)

    # ------------- ranking -------------

    def rank_and_filter(self, conjs: List[Conjecture], *, min_touch: Optional[int] = None) -> List[Conjecture]:
        m = int(min_touch if min_touch is not None else self.config.min_touch_keep)
        return rank_and_filter_fast(self.df, conjs, m)

    # ------------- core generator -------------

    def generate_intricate_mixed_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable[Predicate]] = None,
        primary: Optional[Iterable[str]] = None,   # x
        secondary: Optional[Iterable[str]] = None, # y
        weight: float = 0.5,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Reintroduces the classic 'intricate inequalities' technique:
        y ≥ w*(cmin_x·x + cmin_sqrt·sqrt(y)) with best-of {base, ceil-whole, ceil-split}
        y ≤ w*(cmax_x·x + cmax_sqrt·sqrt(y)) with best-of {base, floor-whole, floor-split}
        and the analogous square-mix using y^2.

        Notes
        -----
        • Optimized: one _ExprEvalCache per hypothesis; arrays sliced by mask.
        • Validity checked numerically (all rows on H-support must satisfy).
        • The selected array form is mirrored by a symbolic Expr for the Conjecture.
        """
        assert 0 < weight <= 1.0, "weight must be in (0,1]"

        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        prim_cols = list(primary or self.numeric_columns)
        sec_cols  = list(secondary or self.numeric_columns)

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        w = float(weight)
        w_const = to_frac_const(weight, self.config.max_denom)

        def _pick_best_ge(t_arr, rhs_variants):
            # choose valid (all t ≥ rhs) with largest mean(rhs)
            best = None; best_score = -np.inf
            for _, rhs, make_expr in rhs_variants:
                ok = np.isfinite(t_arr) & np.isfinite(rhs)
                if not np.any(ok):
                    continue
                if np.all(t_arr[ok] >= rhs[ok]):
                    score = float(np.mean(rhs[ok]))
                    if score > best_score:
                        best = make_expr; best_score = score
            return best

        def _pick_best_le(t_arr, rhs_variants):
            # choose valid (all t ≤ rhs) with smallest mean(rhs)
            best = None; best_score = np.inf
            for _, rhs, make_expr in rhs_variants:
                ok = np.isfinite(t_arr) & np.isfinite(rhs)
                if not np.any(ok):
                    continue
                if np.all(t_arr[ok] <= rhs[ok]):
                    score = float(np.mean(rhs[ok]))
                    if score < best_score:
                        best = make_expr; best_score = score
            return best

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask):
                continue

            ecache = _ExprEvalCache(self.df)
            t_arr = ecache.arr(target)[mask]

            for xname in prim_cols:
                if xname == target_col:
                    continue

                x_arr = ecache.arr(to_expr(xname))[mask]
                if x_arr.size == 0 or np.nanmin(x_arr) <= 0:
                    continue

                rx = t_arr / x_arr
                f_rx = np.isfinite(rx)
                if not np.any(f_rx):
                    continue
                cmin_f = float(np.min(rx[f_rx]))
                cmax_f = float(np.max(rx[f_rx]))

                for yname in sec_cols:
                    if yname == target_col:
                        continue

                    y_arr = ecache.arr(to_expr(yname))[mask]
                    if y_arr.size == 0 or np.nanmin(y_arr) <= 0:
                        continue

                    # ---------- sqrt mix ----------
                    sqrt_y_arr = np.sqrt(y_arr, dtype=float)
                    r_sqrt = t_arr / sqrt_y_arr
                    f_sq = np.isfinite(r_sqrt)
                    if np.any(f_sq):
                        s_cmin_f = float(np.min(r_sqrt[f_sq]))
                        s_cmax_f = float(np.max(r_sqrt[f_sq]))

                        base_lo     = w * (cmin_f * x_arr + s_cmin_f * sqrt_y_arr)
                        base_up     = w * (cmax_f * x_arr + s_cmax_f * sqrt_y_arr)
                        ceil_whole  = np.ceil(base_lo)
                        floor_whole = np.floor(base_up)
                        ceil_split  = np.ceil(w * cmin_f * x_arr) + np.ceil(w * s_cmin_f * sqrt_y_arr) - 1.0
                        floor_split = np.floor(w * cmax_f * x_arr) + np.floor(w * s_cmax_f * sqrt_y_arr)

                        # symbolic mirrors
                        def _lo_base():
                            return (w_const * to_frac_const(cmin_f, self.config.max_denom) * to_expr(xname)
                                    + w_const * to_frac_const(s_cmin_f, self.config.max_denom) * sqrt(to_expr(yname)))

                        def _lo_ceil_whole():
                            return ceil(_lo_base())

                        def _lo_ceil_split():
                            return (ceil(w_const * to_frac_const(cmin_f, self.config.max_denom) * to_expr(xname))
                                    + ceil(w_const * to_frac_const(s_cmin_f, self.config.max_denom) * sqrt(to_expr(yname)))
                                    - Const(1))

                        def _up_base():
                            return (w_const * to_frac_const(cmax_f, self.config.max_denom) * to_expr(xname)
                                    + w_const * to_frac_const(s_cmax_f, self.config.max_denom) * sqrt(to_expr(yname)))

                        def _up_floor_whole():
                            return floor(_up_base())

                        def _up_floor_split():
                            return (floor(w_const * to_frac_const(cmax_f, self.config.max_denom) * to_expr(xname))
                                    + floor(w_const * to_frac_const(s_cmax_f, self.config.max_denom) * sqrt(to_expr(yname))))

                        lo_choice = _pick_best_ge(t_arr, [
                            ("base",        base_lo,     _lo_base),
                            ("ceil_whole",  ceil_whole,  _lo_ceil_whole),
                            ("ceil_split",  ceil_split,  _lo_ceil_split),
                        ])
                        if lo_choice is not None:
                            lowers.append(Conjecture(Ge(target, lo_choice()), H))

                        up_choice = _pick_best_le(t_arr, [
                            ("base",        base_up,      _up_base),
                            ("floor_whole", floor_whole,  _up_floor_whole),
                            ("floor_split", floor_split,  _up_floor_split),
                        ])
                        if up_choice is not None:
                            uppers.append(Conjecture(Le(target, up_choice()), H))

                    # ---------- square mix ----------
                    y_sq_arr = np.square(y_arr, dtype=float)
                    r_sq = t_arr / y_sq_arr
                    f_rsq = np.isfinite(r_sq)
                    if np.any(f_rsq):
                        q_cmin_f = float(np.min(r_sq[f_rsq]))
                        q_cmax_f = float(np.max(r_sq[f_rsq]))

                        base_lo_sq     = w * (cmin_f * x_arr + q_cmin_f * y_sq_arr)
                        base_up_sq     = w * (cmax_f * x_arr + q_cmax_f * y_sq_arr)
                        ceil_whole_sq  = np.ceil(base_lo_sq)
                        floor_whole_sq = np.floor(base_up_sq)

                        md = self.config.max_denom


                        # ---- sqrt mix symbolic builders (replace old ones) ----
                        def _lo_base():
                            kx = mul_consts(w_const, to_frac_const(cmin_f, md), max_denom=md)          # (w * cmin_f)
                            ky = mul_consts(w_const, to_frac_const(s_cmin_f, md), max_denom=md)        # (w * s_cmin_f)
                            return kx * to_expr(xname) + ky * sqrt(to_expr(yname))

                        def _lo_ceil_whole():
                            return ceil(_lo_base())

                        def _lo_ceil_split():
                            kx = mul_consts(w_const, to_frac_const(cmin_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(s_cmin_f, md), max_denom=md)
                            # ceil(kx * x) + ceil(ky * sqrt(y)) - 1
                            return ceil(kx * to_expr(xname)) + ceil(ky * sqrt(to_expr(yname))) - Const(1)

                        def _up_base():
                            kx = mul_consts(w_const, to_frac_const(cmax_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(s_cmax_f, md), max_denom=md)
                            return kx * to_expr(xname) + ky * sqrt(to_expr(yname))

                        def _up_floor_whole():
                            return floor(_up_base())

                        def _up_floor_split():
                            kx = mul_consts(w_const, to_frac_const(cmax_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(s_cmax_f, md), max_denom=md)
                            # floor(kx * x) + floor(ky * sqrt(y))
                            return floor(kx * to_expr(xname)) + floor(ky * sqrt(to_expr(yname)))


                        def _lo_sq_base():
                            kx = mul_consts(w_const, to_frac_const(cmin_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(q_cmin_f, md), max_denom=md)
                            return kx * to_expr(xname) + ky * (to_expr(yname) ** Const(Fraction(2, 1)))

                        def _lo_sq_ceil_whole():
                            return ceil(_lo_sq_base())

                        def _up_sq_base():
                            kx = mul_consts(w_const, to_frac_const(cmax_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(q_cmax_f, md), max_denom=md)
                            return kx * to_expr(xname) + ky * (to_expr(yname) ** Const(Fraction(2, 1)))

                        def _up_sq_floor_whole():
                            return floor(_up_sq_base())


                        lo_sq_choice = _pick_best_ge(t_arr, [
                            ("base",       base_lo_sq,     _lo_sq_base),
                            ("ceil_whole", ceil_whole_sq,  _lo_sq_ceil_whole),
                        ])
                        if lo_sq_choice is not None:
                            lowers.append(Conjecture(Ge(target, lo_sq_choice()), H))

                        up_sq_choice = _pick_best_le(t_arr, [
                            ("base",        base_up_sq,     _up_sq_base),
                            ("floor_whole", floor_whole_sq, _up_sq_floor_whole),
                        ])
                        if up_sq_choice is not None:
                            uppers.append(Conjecture(Le(target, up_sq_choice()), H))

        return lowers, uppers

    def _mask_for(self, df: pd.DataFrame, H) -> np.ndarray:
        if H is None or H is TRUE:
            return np.ones(len(df), dtype=bool)
        s = H.mask(df) if hasattr(H, "mask") else H(df)
        return np.asarray(s, dtype=bool)

    def _promote_equalities(
        self,
        df: pd.DataFrame,
        conjs: list[Conjecture],
        *,
        rtol: float = 1e-9,
        atol: float = 1e-9,
    ) -> tuple[list[Conjecture], list[Conjecture], list[Conjecture]]:
        """
        Split into (lowers, uppers, equals) by promoting any Ge/Le that are equal
        on the entire hypothesis support to Eq(left, right).
        """
        lowers, uppers, equals = [], [], []
        ecache = _ExprEvalCache(df)

        for c in conjs:
            rel = c.relation
            Hm = self._mask_for(df, c.condition)
            if not np.any(Hm):
                if isinstance(rel, Ge): lowers.append(c)
                elif isinstance(rel, Le): uppers.append(c)
                else: equals.append(c)
                continue

            L = ecache.arr(rel.left)
            R = ecache.arr(rel.right)
            ok = np.isfinite(L) & np.isfinite(R) & Hm
            if not np.any(ok):
                if isinstance(rel, Ge): lowers.append(c)
                elif isinstance(rel, Le): uppers.append(c)
                else: equals.append(c)
                continue

            eq_mask = np.isclose(L[ok], R[ok], rtol=rtol, atol=atol)
            if np.all(eq_mask):
                eq = Conjecture(Eq(rel.left, rel.right), c.condition)
                for k in ("coefficient_pairs", "intercept", "support_n", "touch_count", "touch_rate"):
                    if hasattr(c, k): setattr(eq, k, getattr(c, k))
                equals.append(eq)
            else:
                if isinstance(rel, Ge): lowers.append(c)
                elif isinstance(rel, Le): uppers.append(c)
                else: equals.append(c)

        return lowers, uppers, equals

    def _postprocess_triplet(
        self,
        df: pd.DataFrame,
        lowers: list[Conjecture],
        uppers: list[Conjecture],
        equals: list[Conjecture],
    ) -> dict[str, list[Conjecture]]:
        # 1) dedup + sort (with numeric attrs computed)
        lowers = self._dedup_sort(lowers)
        uppers = self._dedup_sort(uppers)
        equals = self._dedup_sort(equals)

        # 2) Morgan filter on each bucket
        lowers = list(morgan_filter(df, lowers).kept)
        uppers = list(morgan_filter(df, uppers).kept)
        equals = list(morgan_filter(df, equals).kept)

        return {"lowers": lowers, "uppers": uppers, "equals": equals}

    def run_intricate_mixed_pipeline(
        self,
        target_col: str,
        *,
        weight: float = 0.5,
        primary: Optional[Iterable[str]] = None,
        secondary: Optional[Iterable[str]] = None,
        min_touch: Optional[int] = None,
    ) -> dict[str, list[Conjecture]]:
        """
        Full pipeline:
        1) generate intricate mixed bounds
        2) rank/filter by touches (fast batch counting)
        3) promote equalities on full support
        4) dedup/sort + Morgan filter per bucket
        5) return {lowers, uppers, equals}
        """
        lows, ups = self.generate_intricate_mixed_bounds(
            target_col,
            weight=weight,
            primary=primary,
            secondary=secondary,
        )

        # Rank/filter (touch threshold)
        m = int(min_touch if min_touch is not None else self.config.min_touch_keep)
        lows = rank_and_filter_fast(self.df, lows, m)
        ups  = rank_and_filter_fast(self.df, ups,  m)

        # Promote Ge/Le to Eq when true on H-support
        lows, ups, eqs = self._promote_equalities(self.df, (lows + ups))

        # Final pass: dedup + Morgan filter per bucket
        return self._postprocess_triplet(self.df, lows, ups, eqs)

    # ────────────── sorting helpers (robust against callables/None) ────────────── #

    def _as_number(self, v, default=None):
        """
        Return a numeric value from v, calling it if callable.
        If v is None or cannot be coerced, return `default` (which may be None).
        """
        if v is None:
            return default
        try:
            v = v() if callable(v) else v
            if hasattr(v, "item") and callable(getattr(v, "item")):
                v = v.item()
            return int(v) if isinstance(v, (int, bool)) else float(v)
        except Exception:
            return default

    def _compute_touch_support(self, c, *, rtol: float = 1e-9, atol: float = 1e-9):
        """
        Ensure numeric c.touch_count, c.support_n, c.touch_rate exist.
        Returns (touch_count, support_n, touch_rate).
        """
        tc = self._as_number(getattr(c, "touch_count", None), default=None)
        sn = self._as_number(getattr(c, "support_n",  None), default=None)
        tr = self._as_number(getattr(c, "touch_rate",  None), default=None)

        if tc is not None and sn is not None and tr is not None:
            return int(tc), int(sn), float(tr)

        df = self.df
        Hm = self._mask_for(df, c.condition)
        if not Hm.any():
            tc_num, sn_num, tr_num = 0, 0, 0.0
        else:
            ecache = _ExprEvalCache(df)
            L = ecache.arr(c.relation.left)
            R = ecache.arr(c.relation.right)
            ok = np.isfinite(L) & np.isfinite(R) & Hm
            sup = int(ok.sum())
            if sup == 0:
                tc_num, sn_num, tr_num = 0, 0, 0.0
            else:
                eq = np.isclose(L[ok], R[ok], rtol=rtol, atol=atol)
                tc_num = int(eq.sum())
                sn_num  = sup
                tr_num  = float(tc_num / sup)

        setattr(c, "touch_count", int(tc_num))
        setattr(c, "support_n", int(sn_num))
        setattr(c, "touch_rate", float(tr_num))
        return int(tc_num), int(sn_num), float(tr_num)

    def _dedup_sort(self, lst: List[Conjecture]) -> List[Conjecture]:
        seen, tmp = set(), []
        for c in lst:
            key = str(c)  # use string only as dedup key
            if key in seen:
                continue
            seen.add(key)
            self._compute_touch_support(c)  # ensure numeric attrs exist
            tmp.append(c)

        def _key(c):
            tc = self._as_number(getattr(c, "touch_count", 0), default=0) or 0
            sn = self._as_number(getattr(c, "support_n",  0), default=0) or 0
            return (int(tc), int(sn))

        tmp.sort(key=_key, reverse=True)
        return tmp
