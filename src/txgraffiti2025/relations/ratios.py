from __future__ import annotations

from dataclasses import dataclass, field

from fractions import Fraction
from typing import Iterable, List, Tuple, Optional, Sequence, Dict

import numpy as np
import pandas as pd

from .core import DataModel, MaskCache
from ..forms.utils import to_expr, Expr, Const, floor, ceil, sqrt
from ..forms.generic_conjecture import Conjecture, Eq, Ge, Le, TRUE
from ..forms.predicates import Predicate
from ..processing.post import morgan_filter

__all__ = ["RatiosConfig", "RatiosMiner"]


# ───────────────────────── helpers (fractions & caching) ───────────────────────── #

def _as_fraction(x, max_denom: int) -> Fraction:
    if hasattr(x, "value") and isinstance(x.value, Fraction):  # Const(Fraction)
        return x.value
    if isinstance(x, Fraction):
        return x
    return Fraction(float(x)).limit_denominator(max_denom)

def mul_consts(*vals, max_denom: int = 300) -> Const:
    f = Fraction(1, 1)
    for v in vals:
        f *= _as_fraction(v, max_denom)
    return Const(f.limit_denominator(max_denom))

def to_frac_const(val: float, max_denom: int = 300) -> Const:
    return Const(Fraction(val).limit_denominator(max_denom))


def _expr_key(e: Expr) -> str:
    return repr(e)


class _ExprEvalCache:
    """
    Evaluate Exprs once on the FULL df, then slice by masks. Avoids repeated expr.eval().
    """
    __slots__ = ("df", "cache")

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cache: dict[str, np.ndarray] = {}

    def arr(self, expr: Expr) -> np.ndarray:
        k = _expr_key(expr)
        a = self.cache.get(k)
        if a is not None:
            return a
        s = expr.eval(self.df)
        if hasattr(s, "to_numpy"):
            a = s.to_numpy(dtype=float, copy=False)
        else:
            a = np.asarray(s, dtype=float)
            if a.ndim == 0:
                a = np.full(len(self.df), float(a), dtype=float)
        self.cache[k] = a
        return a


# ───────────────────────── config ───────────────────────── #

@dataclass(slots=True)
class RatiosConfig:
    """
    Tunables used by RatiosMiner.
    """
    min_touch_keep: int = 3
    max_denom: int = 30

    # Hypothesis selection
    include_base_hypothesis: bool = True
    use_sorted_conjunctions: bool = True
    conjunction_limit: Optional[int] = None  # None = all

    # Column selection
    exclude_nonpositive_x: bool = True
    exclude_nonpositive_y: bool = True


# ──────────────────────── main miner ──────────────────────── #

@dataclass(slots=True)
class RatiosMiner:
    """
    Inequality generator for “intricate mixed” ratio-style bounds over the new core:

      For each hypothesis H, primary x, secondary y:

        sqrt-mix:
          lower:  t ≥ w * (cmin_x * x + cmin_sqrt * √y)   via best of
                  { base, ceil(whole), ceil-split }
          upper:  t ≤ w * (cmax_x * x + cmax_sqrt * √y)   via best of
                  { base, floor(whole), floor-split }

        square-mix:
          lower:  t ≥ w * (cmin_x * x + cmin_sq * y²)     via best of { base, ceil(whole) }
          upper:  t ≤ w * (cmax_x * x + cmax_sq * y²)     via best of { base, floor(whole) }
    """

    model: DataModel
    cache: MaskCache
    # Optional: a ClassLogic-like object exposing sorted conjunctions as [(name, Predicate), ...]
    class_logic: Optional[object] = None
    config: RatiosConfig = field(default_factory=RatiosConfig)  # ✅

    # ───────── hypothesis utilities ───────── #

    def _iter_hypotheses(self) -> Iterable[Tuple[str, Predicate]]:
        C = self.config
        emitted = set()

        # Base hypothesis first
        base = getattr(self.model, "base_pred", TRUE)
        base_name = getattr(self.model, "base_name", None) or (base.name if hasattr(base, "name") else "BASE")
        if C.include_base_hypothesis and base is not TRUE:
            sig = repr(base)
            if sig not in emitted:
                emitted.add(sig)
                yield base_name, base

        # Sorted conjunctions from class_logic (if present)
        if C.use_sorted_conjunctions and self.class_logic is not None:
            items = getattr(self.class_logic, "sorted_conjunctions_", []) or []
            if C.conjunction_limit is not None:
                items = items[: max(0, int(C.conjunction_limit))]
            for name, pred in items:
                sig = repr(pred)
                if sig in emitted:
                    continue
                emitted.add(sig)
                yield name, pred

        # If nothing emitted, yield TRUE over full domain
        if not emitted:
            yield "TRUE", TRUE

    def _mask_for(self, H: Optional[Predicate]) -> np.ndarray:
        if H is None or H is TRUE:
            base = getattr(self.model, "base_pred", TRUE)
            if base is not TRUE:
                return self.cache.mask(base)
            return np.ones(len(self.model.df), dtype=bool)
        return self.cache.mask(H)

    def _choose_columns(
        self,
        *,
        primary: Optional[Iterable[str]],
        secondary: Optional[Iterable[str]],
        target_col: str,
    ) -> tuple[list[str], list[str]]:
        prim_cols = [c for c in (primary or self.model.numeric_cols) if c in self.model.numeric_cols and c != target_col]
        sec_cols  = [c for c in (secondary or self.model.numeric_cols) if c in self.model.numeric_cols and c != target_col]
        return prim_cols, sec_cols

    # ───────── ranking & promotion ───────── #

    def _compute_touch_support(self, df: pd.DataFrame, c: Conjecture, *, rtol: float = 1e-9, atol: float = 1e-9):
        tc = getattr(c, "touch_count", None)
        sn = getattr(c, "support_n", None)
        tr = getattr(c, "touch_rate", None)
        if (tc is not None) and (sn is not None) and (tr is not None):
            return int(tc), int(sn), float(tr)

        Hm = self._mask_for(c.condition)
        if not np.any(Hm):
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
        seen, out = set(), []
        for c in lst:
            key = str(c)  # stable repr key
            if key in seen:
                continue
            seen.add(key)
            self._compute_touch_support(df, c)
            out.append(c)

        def _key(c):
            return (int(getattr(c, "touch_count", 0) or 0), int(getattr(c, "support_n", 0) or 0))
        out.sort(key=_key, reverse=True)
        return out

    def _rank_and_filter(self, df: pd.DataFrame, conjs: List[Conjecture], min_touch: int) -> List[Conjecture]:
        if not conjs:
            return []
        tmp = self._dedup_sort(df, conjs)
        tmp = [c for c in tmp if int(getattr(c, "touch_count", 0) or 0) >= int(min_touch)]
        return list(morgan_filter(df, tmp).kept)

    def _promote_equalities(
        self, df: pd.DataFrame, conjs: list[Conjecture], *, rtol: float = 1e-9, atol: float = 1e-9
    ) -> tuple[list[Conjecture], list[Conjecture], list[Conjecture]]:
        lowers, uppers, equals = [], [], []
        ecache = _ExprEvalCache(df)
        for c in conjs:
            rel = c.relation
            Hm = self._mask_for(c.condition)
            if not np.any(Hm):
                (lowers if isinstance(rel, Ge) else uppers if isinstance(rel, Le) else equals).append(c)
                continue
            L = ecache.arr(rel.left); R = ecache.arr(rel.right)
            ok = np.isfinite(L) & np.isfinite(R) & Hm
            if not np.any(ok):
                (lowers if isinstance(rel, Ge) else uppers if isinstance(rel, Le) else equals).append(c)
                continue
            if np.all(np.isclose(L[ok], R[ok], rtol=rtol, atol=atol)):
                eq = Conjecture(Eq(rel.left, rel.right), c.condition)
                for k in ("coefficient_pairs", "intercept", "support_n", "touch_count", "touch_rate"):
                    if hasattr(c, k):
                        setattr(eq, k, getattr(c, k))
                equals.append(eq)
            else:
                (lowers if isinstance(rel, Ge) else uppers if isinstance(rel, Le) else equals).append(c)
        return lowers, uppers, equals

    # ───────── core generator ───────── #

    def generate_intricate_mixed_bounds(
        self,
        target_col: str,
        *,
        primary: Optional[Iterable[str]] = None,
        secondary: Optional[Iterable[str]] = None,
        weight: float = 0.5,
        hyps: Optional[Iterable[Predicate]] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        assert 0 < weight <= 1.0, "weight must be in (0,1]"
        md = self.config.max_denom
        df = self.model.df

        target = to_expr(target_col)
        prim_cols, sec_cols = self._choose_columns(primary=primary, secondary=secondary, target_col=target_col)
        hyps_iter: List[Tuple[str, Predicate]]
        if hyps is not None:
            hyps_iter = [(getattr(h, "name", None) or repr(h), h) for h in hyps]
        else:
            hyps_iter = list(self._iter_hypotheses())

        lows: List[Conjecture] = []
        ups:  List[Conjecture] = []

        w = float(weight)
        w_const = to_frac_const(weight, md)
        ecache = _ExprEvalCache(df)

        def _pick_best_ge(t_arr, rhs_variants):
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

        for name, H in hyps_iter:
            mask = self._mask_for(H)
            if not np.any(mask):
                continue

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

    # ───────── public pipeline ───────── #

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
        1) generate intricate mixed bounds across hypotheses
        2) rank/filter by touches
        3) promote equalities on full H-support
        4) dedup/sort + Morgan filter per bucket
        """
        lows, ups = self.generate_intricate_mixed_bounds(
            target_col,
            primary=primary,
            secondary=secondary,
            weight=weight,
            hyps=hyps,
        )
        m = int(min_touch if min_touch is not None else self.config.min_touch_keep)
        df = self.model.df
        lows = self._rank_and_filter(df, lows, m)
        ups  = self._rank_and_filter(df, ups,  m)
        lows, ups, eqs = self._promote_equalities(df, (lows + ups))
        # Final pass
        lows = self._dedup_sort(df, list(morgan_filter(df, lows).kept))
        ups  = self._dedup_sort(df, list(morgan_filter(df, ups).kept))
        eqs  = self._dedup_sort(df, list(morgan_filter(df, eqs).kept))
        return {"lowers": lows, "uppers": ups, "equals": eqs}

    def summarize(
        self,
        results: dict[str, list[Conjecture]],
        *,
        max_per_bucket: int = 12,
        group_by_hypothesis: bool = True,
        include_counts_bar: bool = True,
        title: str = "Ratios • Intricate Mixed Bounds",
        verbose: bool = True,
    ) -> str:
        lowers = list(results.get("lowers", []) or [])
        uppers = list(results.get("uppers", []) or [])
        equals = list(results.get("equals", []) or [])

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

        lines: list[str] = []
        lines.append("──────────────────────────────────────────────")
        lines.append(title)
        lines.append("──────────────────────────────────────────────")
        lines.append(f"Base hypothesis: {getattr(self.model, 'base_name', 'TRUE')}")
        lines.append(f"DataFrame: {self.model.df.shape[0]} rows × {self.model.df.shape[1]} cols")
        lines.append("")

        if include_counts_bar:
            counts = _bucket_counts()
            lines.append("Counts:")
            lines.append(f"  Lowers (t ≥ ·): {counts['lowers']}")
            lines.append(f"  Uppers (t ≤ ·): {counts['uppers']}")
            lines.append(f"  Equals (t = ·): {counts['equals']}")
            lines.append(f"  Total:          {counts['total']}")
            lines.append("")

        def _emit_bucket(title: str, lst: list):
            lines.append(title + ":")
            if not lst:
                lines.append("  (none)")
                lines.append("")
                return
            if group_by_hypothesis:
                byH = _group_by_hyp(lst)
                items = sorted(byH.items(), key=lambda kv: (-len(kv[1]), kv[0]))
                for cond_s, group in items:
                    lines.append(f"  • {cond_s} — {len(group)}")
                    for c in group[:max_per_bucket]:
                        rel_s, _ = _cj_pretty(c)
                        tc = getattr(c, "touch_count", None)
                        sn = getattr(c, "support_n", None)
                        touch_txt = f"  [touch={tc}/{sn}]" if (tc is not None and sn is not None) else ""
                        lines.append(f"      {rel_s}{touch_txt}")
                    extra = max(0, len(group) - max_per_bucket)
                    if extra:
                        lines.append(f"      … +{extra} more")
                lines.append("")
            else:
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
