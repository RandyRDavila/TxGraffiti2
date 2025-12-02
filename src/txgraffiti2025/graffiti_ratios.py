# src/txgraffiti2025/generators/graffiti_ratios.py
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Tuple, Optional, Sequence, Dict, Union

import numpy as np
import pandas as pd

from txgraffiti2025.graffiti_base import GraffitiBase
from txgraffiti2025.graffiti_class_logic import GraffitiClassLogic
from txgraffiti2025.graffiti_comparable import GraffitiComparable  # optional; not required but handy

from txgraffiti2025.forms.utils import (
    to_expr, Expr, Const, floor, ceil, sqrt,
)
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Ge, Le, TRUE
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.post import morgan_filter

__all__ = ["GraffitiRatios", "RatiosConfig", "GraffitiLPIntricate"]


# ───────────────────────── helpers (fractions & caching) ───────────────────────── #

def _as_fraction(x, max_denom: int) -> Fraction:
    """Accept Const/Fraction/int/float and return a reduced Fraction."""
    if hasattr(x, "value") and isinstance(x.value, Fraction):  # Const(Fraction)
        return x.value
    if isinstance(x, Fraction):
        return x
    return Fraction(float(x)).limit_denominator(max_denom)

def mul_consts(*vals, max_denom: int = 300) -> Const:
    """Multiply many numbers into a single rationalized Const."""
    f = Fraction(1, 1)
    for v in vals:
        f *= _as_fraction(v, max_denom)
    return Const(f.limit_denominator(max_denom))

def to_frac_const(val: float, max_denom: int = 300) -> Const:
    """Rationalize to improve pretty-printing and Expr stability."""
    return Const(Fraction(val).limit_denominator(max_denom))


def _pred_cache_key(p: Predicate | None) -> str:
    if p is None or p is TRUE:
        return "TRUE"
    n = getattr(p, "name", None)
    return f"name:{n}" if n else f"repr:{repr(p)}"

def _expr_cache_key(e: Expr) -> str:
    return repr(e)


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
    groups by (condition, left, right) so each (L,R,mask) is evaluated once.
    """
    if not conjs:
        return {}

    # We rely on Predicate.mask here (GraffitiBase uses the same semantics).
    mask_cache: Dict[str, np.ndarray] = {}
    def _mask_for(pred: Optional[Predicate]) -> np.ndarray:
        key = _pred_cache_key(pred)
        a = mask_cache.get(key)
        if a is not None:
            return a
        if pred is None or pred is TRUE:
            a = np.ones(len(df), dtype=bool)
        else:
            s = pred.mask(df).reindex(df.index, fill_value=False)
            if s.dtype is not bool:
                s = s.fillna(False).astype(bool, copy=False)
            a = s.to_numpy(copy=False)
        mask_cache[key] = a
        return a

    ecache = _ExprEvalCache(df)

    # group indices by identical (pred,left,right) “signatures”
    groups: dict[tuple[str, str, str], list[int]] = {}
    for i, c in enumerate(conjs):
        Hk = _pred_cache_key(c.condition)
        Lk = _expr_cache_key(c.relation.left)
        Rk = _expr_cache_key(c.relation.right)
        groups.setdefault((Hk, Lk, Rk), []).append(i)

    out: dict[int, int] = {}
    for (Hk, Lk, Rk), idxs in groups.items():
        rep = conjs[idxs[0]]
        mask = _mask_for(rep.condition)
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


def _rank_and_filter_fast(df: pd.DataFrame, conjs: List[Conjecture], min_touch: int) -> List[Conjecture]:
    """Sort by touch desc (batch computed) and apply Morgan filter for validity."""
    if not conjs:
        return []
    counts = _batch_touch_counts(df, conjs)
    order = sorted(range(len(conjs)), key=lambda i: counts.get(i, 0), reverse=True)
    prelim = [conjs[i] for i in order if counts.get(i, 0) >= int(min_touch)]
    return list(morgan_filter(df, prelim).kept)


# ───────────────────────── config ───────────────────────── #

@dataclass
class RatiosConfig:
    """
    Tunables used by GraffitiRatios.
    """
    min_touch_keep: int = 3
    max_denom: int = 30

    # Hypothesis selection
    use_sorted_conjunctions: bool = True
    conjunction_limit: Optional[int] = None  # None = all
    include_base_hypothesis: bool = True

    # Column selection
    exclude_nonpositive_x: bool = True
    exclude_nonpositive_y: bool = True


# ──────────────────────── main class ──────────────────────── #

class GraffitiRatios(GraffitiBase):
    """
    Inequality generator for “intricate mixed” ratio-style bounds:

      For each hypothesis H, primary x, secondary y:

        sqrt-mix:
          lower:  t ≥ w * (cmin_x * x + cmin_sqrt * √y)   via best of
                  { base, ceil(whole), ceil-split }
          upper:  t ≤ w * (cmax_x * x + cmax_sqrt * √y)   via best of
                  { base, floor(whole), floor-split }

        square-mix:
          lower:  t ≥ w * (cmin_x * x + cmin_sq * y²)     via best of { base, ceil(whole) }
          upper:  t ≤ w * (cmax_x * x + cmax_sq * y²)     via best of { base, floor(whole) }

    Design:
      • Inherits prepared state & cache hooks from GraffitiBase (or adopts from GraffitiClassLogic).
      • Single _ExprEvalCache per hypothesis (avoids repeated expr.eval()).
      • Validity checked numerically; symbolic Expr mirrors are built for Conjectures.
      • Compact pipeline: generate → rank/filter by touches → promote equalities → postprocess.
    """

    # ------------- lifecycle -------------

    def __init__(
        self,
        base_or_df_or_logic: Union[pd.DataFrame, GraffitiBase, GraffitiClassLogic],
        *,
        config: Optional[RatiosConfig] = None,
    ):
        if isinstance(base_or_df_or_logic, GraffitiClassLogic):
            self._adopt_base(base_or_df_or_logic)
            self._logic: Optional[GraffitiClassLogic] = base_or_df_or_logic
        elif isinstance(base_or_df_or_logic, GraffitiBase):
            self._adopt_base(base_or_df_or_logic)
            self._logic = None
        else:
            super().__init__(base_or_df_or_logic)  # fresh base from DataFrame
            self._logic = None

        self.config = config or RatiosConfig()

    # ---- adopt/copy-from-base helper ------------------------------------- #
    def _adopt_base(self, base: GraffitiBase) -> None:
        """
        Adopt state from an existing GraffitiBase/GraffitiClassLogic without
        mutating the source's caches.
        """
        # Core refs
        self.df = base.df

        # Partitions and wrappers
        self.boolean_cols = list(base.boolean_cols)
        self.expr_cols = list(base.expr_cols)
        self.predicates = dict(base.predicates)
        self.base_predicates = dict(base.base_predicates)
        self.exprs = dict(base.exprs)

        # Base hypothesis
        self.base_hypothesis = base.base_hypothesis
        self.base_hypothesis_name = base.base_hypothesis_name

        # Fresh local cache bound to this df
        self._mask_cache = {}
        df_id = id(self.df)
        self._mask_cache_version = df_id

        # Bookkeeping hooks consistent with GraffitiBase
        self.synthetic_expr_names_ = getattr(base, "synthetic_expr_names_", set())
        self.abs_exprs = getattr(base, "abs_exprs", [])

    # ------------- hypothesis/column selection -------------

    def _iter_hypotheses(self) -> Iterable[Predicate]:
        """
        Yield hypotheses to evaluate according to config:
          • base hypothesis (optional),
          • sorted conjunctions from attached GraffitiClassLogic (optional).
        """
        C = self.config
        emitted = set()
        if C.include_base_hypothesis:
            yield self.base_hypothesis
            emitted.add(repr(self.base_hypothesis))

        if C.use_sorted_conjunctions and self._logic is not None:
            items = getattr(self._logic, "sorted_conjunctions_", []) or []
            if C.conjunction_limit is not None:
                items = items[:max(0, int(C.conjunction_limit))]
            for name, pred in items:
                sig = repr(pred)
                if sig in emitted:
                    continue
                emitted.add(sig)
                yield pred

    def _mask_for(self, H: Optional[Predicate]) -> np.ndarray:
        """Base-aware mask (uses GraffitiBase cache)."""
        if H is None or H is TRUE:
            # respect base hypothesis if it’s not TRUE
            if self.base_hypothesis is not TRUE:
                return self.mask(self.base_hypothesis)
            return np.ones(len(self.df), dtype=bool)
        return self.mask(H)

    def _choose_columns(
        self,
        *,
        primary: Optional[Iterable[str]],
        secondary: Optional[Iterable[str]],
        target_col: str,
    ) -> tuple[list[str], list[str]]:
        """Resolve primary/secondary column lists, excluding the target."""
        prim_cols = [c for c in (primary or self.expr_cols) if c in self.expr_cols and c != target_col]
        sec_cols  = [c for c in (secondary or self.expr_cols) if c in self.expr_cols and c != target_col]
        return prim_cols, sec_cols

    # ------------- ranking & promotion -------------

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
            Hm = self._mask_for(c.condition)
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
                # copy light metadata if present
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
        lowers = self._dedup_sort(df, lowers)
        uppers = self._dedup_sort(df, uppers)
        equals = self._dedup_sort(df, equals)

        # 2) Morgan filter on each bucket
        lowers = list(morgan_filter(df, lowers).kept)
        uppers = list(morgan_filter(df, uppers).kept)
        equals = list(morgan_filter(df, equals).kept)

        return {"lowers": lowers, "uppers": uppers, "equals": equals}

    def _as_number(self, v, default=None):
        """Numeric value from v, calling if callable; fallback to default."""
        if v is None:
            return default
        try:
            v = v() if callable(v) else v
            if hasattr(v, "item") and callable(getattr(v, "item")):
                v = v.item()
            return int(v) if isinstance(v, (int, bool)) else float(v)
        except Exception:
            return default

    def _compute_touch_support(self, df: pd.DataFrame, c, *, rtol: float = 1e-9, atol: float = 1e-9):
        """Ensure numeric c.touch_count, c.support_n, c.touch_rate exist."""
        tc = self._as_number(getattr(c, "touch_count", None), default=None)
        sn = self._as_number(getattr(c, "support_n",  None), default=None)
        tr = self._as_number(getattr(c, "touch_rate",  None), default=None)

        if tc is not None and sn is not None and tr is not None:
            return int(tc), int(sn), float(tr)

        Hm = self._mask_for(c.condition)
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
                sn_num = sup
                tr_num = float(tc_num / sup)

        setattr(c, "touch_count", int(tc_num))
        setattr(c, "support_n", int(sn_num))
        setattr(c, "touch_rate", float(tr_num))
        return int(tc_num), int(sn_num), float(tr_num)

    def _dedup_sort(self, df: pd.DataFrame, lst: List[Conjecture]) -> List[Conjecture]:
        seen, tmp = set(), []
        for c in lst:
            key = str(c)  # use string repr as dedup key
            if key in seen:
                continue
            seen.add(key)
            self._compute_touch_support(df, c)  # ensure numeric attrs exist
            tmp.append(c)

        def _key(c):
            tc = self._as_number(getattr(c, "touch_count", 0), default=0) or 0
            sn = self._as_number(getattr(c, "support_n",  0), default=0) or 0
            return (int(tc), int(sn))

        tmp.sort(key=_key, reverse=True)
        return tmp

    # ------------- core generator -------------

    def generate_intricate_mixed_bounds(
        self,
        target_col: str,
        *,
        primary: Optional[Iterable[str]] = None,    # x
        secondary: Optional[Iterable[str]] = None,  # y
        weight: float = 0.5,
        hyps: Optional[Iterable[Predicate]] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Mixed ratio-style bounds on each selected hypothesis.

        Parameters
        ----------
        target_col : str
            Dependent variable t.
        primary : iterable[str] or None
            Candidate x columns (defaults to all numeric except t).
        secondary : iterable[str] or None
            Candidate y columns (defaults to all numeric except t).
        weight : float in (0,1]
            Leading weight w applied to the mixture.
        hyps : iterable[Predicate] or None
            Hypotheses to evaluate; defaults to self._iter_hypotheses().

        Returns
        -------
        (lowers, uppers) : tuple[list[Conjecture], list[Conjecture]]
        """
        assert 0 < weight <= 1.0, "weight must be in (0,1]"
        md = self.config.max_denom

        target = to_expr(target_col)
        prim_cols, sec_cols = self._choose_columns(primary=primary, secondary=secondary, target_col=target_col)
        hyps_iter = list(hyps) if hyps is not None else list(self._iter_hypotheses())

        lows: List[Conjecture] = []
        ups:  List[Conjecture] = []

        w = float(weight)
        w_const = to_frac_const(weight, md)

        # pickers choose among array rhs-variants; return a callable that builds an Expr
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
            mask = self._mask_for(H)
            if not np.any(mask):
                continue

            ecache = _ExprEvalCache(self.df)
            t_arr = ecache.arr(target)[mask]

            for xname in prim_cols:
                if xname == target_col:
                    continue
                x_expr = to_expr(xname)
                x_arr = ecache.arr(x_expr)[mask]
                if x_arr.size == 0:
                    continue
                if self.config.exclude_nonpositive_x and (np.nanmin(x_arr) <= 0):
                    continue

                # ratios t/x
                rx = t_arr / x_arr
                f_rx = np.isfinite(rx)
                if not np.any(f_rx):
                    continue
                cmin_f = float(np.min(rx[f_rx]))
                cmax_f = float(np.max(rx[f_rx]))

                for yname in sec_cols:
                    if yname == target_col:
                        continue
                    y_expr = to_expr(yname)
                    y_arr = ecache.arr(y_expr)[mask]
                    if y_arr.size == 0:
                        continue
                    if self.config.exclude_nonpositive_y and (np.nanmin(y_arr) <= 0):
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
                            kx = mul_consts(w_const, to_frac_const(cmin_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(s_cmin_f, md), max_denom=md)
                            return kx * x_expr + ky * sqrt(y_expr)

                        def _lo_ceil_whole():
                            return ceil(_lo_base())

                        def _lo_ceil_split():
                            kx = mul_consts(w_const, to_frac_const(cmin_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(s_cmin_f, md), max_denom=md)
                            return ceil(kx * x_expr) + ceil(ky * sqrt(y_expr)) - Const(1)

                        def _up_base():
                            kx = mul_consts(w_const, to_frac_const(cmax_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(s_cmax_f, md), max_denom=md)
                            return kx * x_expr + ky * sqrt(y_expr)

                        def _up_floor_whole():
                            return floor(_up_base())

                        def _up_floor_split():
                            kx = mul_consts(w_const, to_frac_const(cmax_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(s_cmax_f, md), max_denom=md)
                            return floor(kx * x_expr) + floor(ky * sqrt(y_expr))

                        lo_choice = _pick_best_ge(t_arr, [
                            ("base",        base_lo,     _lo_base),
                            ("ceil_whole",  ceil_whole,  _lo_ceil_whole),
                            ("ceil_split",  ceil_split,  _lo_ceil_split),
                        ])
                        if lo_choice is not None:
                            lows.append(Conjecture(Ge(target, lo_choice()), H))

                        up_choice = _pick_best_le(t_arr, [
                            ("base",        base_up,      _up_base),
                            ("floor_whole", floor_whole,  _up_floor_whole),
                            ("floor_split", floor_split,  _up_floor_split),
                        ])
                        if up_choice is not None:
                            ups.append(Conjecture(Le(target, up_choice()), H))

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

                        def _lo_sq_base():
                            kx = mul_consts(w_const, to_frac_const(cmin_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(q_cmin_f, md), max_denom=md)
                            return kx * x_expr + ky * (y_expr ** Const(Fraction(2, 1)))

                        def _lo_sq_ceil_whole():
                            return ceil(_lo_sq_base())

                        def _up_sq_base():
                            kx = mul_consts(w_const, to_frac_const(cmax_f, md), max_denom=md)
                            ky = mul_consts(w_const, to_frac_const(q_cmax_f, md), max_denom=md)
                            return kx * x_expr + ky * (y_expr ** Const(Fraction(2, 1)))

                        def _up_sq_floor_whole():
                            return floor(_up_sq_base())

                        lo_sq_choice = _pick_best_ge(t_arr, [
                            ("base",       base_lo_sq,     _lo_sq_base),
                            ("ceil_whole", ceil_whole_sq,  _lo_sq_ceil_whole),
                        ])
                        if lo_sq_choice is not None:
                            lows.append(Conjecture(Ge(target, lo_sq_choice()), H))

                        up_sq_choice = _pick_best_le(t_arr, [
                            ("base",        base_up_sq,     _up_sq_base),
                            ("floor_whole", floor_whole_sq, _up_sq_floor_whole),
                        ])
                        if up_sq_choice is not None:
                            ups.append(Conjecture(Le(target, up_sq_choice()), H))

        return lows, ups

    # ------------- public pipeline -------------

    def run_pipeline(
        self,
        target_col: str,
        *,
        primary: Optional[Iterable[str]] = None,
        secondary: Optional[Iterable[str]] = None,
        weight: float = 0.5,
        min_touch: Optional[int] = None,
        hyps: Optional[Iterable[Predicate]] = None,
    ) -> dict[str, list[Conjecture]]:
        """
        Full pipeline:
        1) generate intricate mixed bounds across configured hypotheses
        2) rank/filter by touches (fast batch counting)
        3) promote equalities on full H-support
        4) dedup/sort + Morgan filter per bucket
        5) return {lowers, uppers, equals}
        """
        lows, ups = self.generate_intricate_mixed_bounds(
            target_col,
            primary=primary,
            secondary=secondary,
            weight=weight,
            hyps=hyps,
        )

        # Rank/filter (touch threshold)
        m = int(min_touch if min_touch is not None else self.config.min_touch_keep)
        lows = _rank_and_filter_fast(self.df, lows, m)
        ups  = _rank_and_filter_fast(self.df, ups,  m)

        # Promote Ge/Le to Eq when true on H-support
        lows, ups, eqs = self._promote_equalities(self.df, (lows + ups))

        # Final pass: dedup + Morgan filter per bucket
        return self._postprocess_triplet(self.df, lows, ups, eqs)

    def summarize(
        self,
        results: dict[str, list],
        *,
        max_per_bucket: int = 12,
        group_by_hypothesis: bool = True,
        include_counts_bar: bool = True,
        title: str = "GraffitiRatios • Intricate Mixed Bounds",
        verbose: bool = True,
    ) -> str:
        """
        Pretty-print a concise catalog of ratio-style conjectures.

        Parameters
        ----------
        results : dict
            Output of `run_pipeline(...)`, i.e. {"lowers": [...], "uppers": [...], "equals": [...] }.
        max_per_bucket : int, default=12
            Maximum printed per (hypothesis × bucket).
        group_by_hypothesis : bool, default=True
            If True, group by condition (hypothesis). Otherwise just list buckets.
        include_counts_bar : bool, default=True
            Include a header section with bucket counts.
        title : str
            Title line.
        verbose : bool, default=True
            If True, print; otherwise return the formatted string.

        Returns
        -------
        str
            The formatted summary block.
        """

        lowers = list(results.get("lowers", []) or [])
        uppers = list(results.get("uppers", []) or [])
        equals = list(results.get("equals", []) or [])

        # ---------- helpers ----------
        def _cj_pretty(c):
            rel = getattr(c, "relation", None)
            cond = getattr(c, "condition", None)
            rel_s = rel.pretty() if hasattr(rel, "pretty") else repr(rel)
            cond_s = cond.pretty() if hasattr(cond, "pretty") else (getattr(cond, "name", None) or (repr(cond) if cond is not None else "TRUE"))
            return rel_s, cond_s

        def _bucket_counts():
            return {
                "lowers": len(lowers),
                "uppers": len(uppers),
                "equals": len(equals),
                "total":  len(lowers) + len(uppers) + len(equals),
            }

        def _group_by_hyp(lst):
            byH: dict[str, list] = {}
            for c in lst:
                _, cond_s = _cj_pretty(c)
                byH.setdefault(cond_s, []).append(c)
            return byH

        # ---------- header ----------
        lines: list[str] = []
        lines.append("──────────────────────────────────────────────")
        lines.append(title)
        lines.append("──────────────────────────────────────────────")
        lines.append(f"Base hypothesis: {getattr(self, 'base_hypothesis_name', 'TRUE')}")
        lines.append(f"DataFrame: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
        lines.append("")

        if include_counts_bar:
            counts = _bucket_counts()
            lines.append("Counts:")
            lines.append(f"  Lowers (t ≥ ·): {counts['lowers']}")
            lines.append(f"  Uppers (t ≤ ·): {counts['uppers']}")
            lines.append(f"  Equals (t = ·): {counts['equals']}")
            lines.append(f"  Total:          {counts['total']}")
            lines.append("")

        # ---------- body ----------
        def _emit_bucket(title: str, lst: list):
            lines.append(title + ":")
            if not lst:
                lines.append("  (none)")
                lines.append("")
                return

            if group_by_hypothesis:
                byH = _group_by_hyp(lst)
                # sort groups by size desc, then name
                items = sorted(byH.items(), key=lambda kv: (-len(kv[1]), kv[0]))
                for cond_s, group in items:
                    lines.append(f"  • {cond_s} — {len(group)}")
                    for c in group[:max_per_bucket]:
                        rel_s, _ = _cj_pretty(c)
                        # Show touches if available
                        tc = getattr(c, "touch_count", None)
                        sn = getattr(c, "support_n", None)
                        touch_txt = f"  [touch={tc}/{sn}]" if (tc is not None and sn is not None) else ""
                        lines.append(f"      {rel_s}{touch_txt}")
                    extra = max(0, len(group) - max_per_bucket)
                    if extra:
                        lines.append(f"      … +{extra} more")
                lines.append("")
            else:
                # flat list, highest-touch first
                def _key(c):
                    tc = getattr(c, "touch_count", 0) or 0
                    sn = getattr(c, "support_n", 0) or 0
                    return (-int(tc), -int(sn), str(c))
                lst_sorted = sorted(lst, key=_key)
                for c in lst_sorted[:max_per_bucket]:
                    rel_s, cond_s = _cj_pretty(c)
                    tc = getattr(c, "touch_count", None)
                    sn = getattr(c, "support_n", None)
                    touch_txt = f"  [touch={tc}/{sn}]" if (tc is not None and sn is not None) else ""
                    lines.append(f"  • {rel_s} | {cond_s}{touch_txt}")
                extra = max(0, len(lst_sorted) - max_per_bucket)
                if extra:
                    lines.append(f"  … +{extra} more")
                lines.append("")

        _emit_bucket("Lowers (t ≥ RHS)", lowers)
        _emit_bucket("Uppers (t ≤ RHS)", uppers)
        _emit_bucket("Equalities (t = RHS)", equals)

        lines.append("──────────────────────────────────────────────")
        s = "\n".join(lines)
        if verbose:
            print(s)
        return s


# ───────────────────────── Back-compat alias ─────────────────────────
# (so old imports like from ...intricate_mixed import GraffitiLPIntricate keep working)
GraffitiLPIntricate = GraffitiRatios
