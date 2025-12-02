# src/txgraffiti2025/graffiti_comparable.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .forms.generic_conjecture import Lt, Gt
from .graffiti_base import GraffitiBase
from .graffiti_class_logic import GraffitiClassLogic
from .forms.predicates import Predicate
from .forms.generic_conjecture import TRUE, Le, Ge
from .forms.utils import Expr, to_expr, abs_, Const  # ensure Const is imported


__all__ = ["GraffitiComparable", "PairCompareResult"]


# ───────────────────────────────────────────────────────────────────────────────
# Result struct
# ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairCompareResult:
    x: str
    y: str
    n: int
    le_all: bool
    ge_all: bool
    eq_all: bool
    lt_any: bool
    gt_any: bool
    incomparable: bool

    def relation_symbol(self) -> str:
        if self.eq_all:
            return "=="
        if self.le_all and not self.ge_all:
            return "<="
        if self.ge_all and not self.le_all:
            return ">="
        return "↕" if self.incomparable else "?"


# ───────────────────────────────────────────────────────────────────────────────
# Main class (minimal)
# ───────────────────────────────────────────────────────────────────────────────

class GraffitiComparable(GraffitiBase):
    """
    Minimal comparable / non-comparable analysis that:
      • Works with arbitrary datasets (no column assumptions).
      • Treats features as Exprs (columns or synthesized).
      • Finds top incomparable pairs on the base mask.
      • Synthesizes and registers |x−y| Exprs so they behave like real features.
    """

    def __init__(
        self,
        base_or_df_or_logic: Union[pd.DataFrame, GraffitiBase, GraffitiClassLogic],
        *,
        top_k: int = 20,
        tol: float = 1e-9,
        min_witness: int = 2,
        auto_build_abs: bool = True,
        verbose_init: bool = True,
    ):
        # Adopt a prepared base/logic or build from a raw DataFrame
        if isinstance(base_or_df_or_logic, GraffitiClassLogic):
            self._adopt_base(base_or_df_or_logic)
            self._logic: Optional[GraffitiClassLogic] = base_or_df_or_logic
        elif isinstance(base_or_df_or_logic, GraffitiBase):
            self._adopt_base(base_or_df_or_logic)
            self._logic = None
        else:
            super().__init__(base_or_df_or_logic)
            self._logic = None

        # Hooks used by higher layers; ensure presence
        if not hasattr(self, "synthetic_expr_names_"):
            self.synthetic_expr_names_: set[str] = set()
        if not hasattr(self, "abs_exprs"):
            self.abs_exprs: List[Tuple[str, Expr]] = []

        self.last_debug: Dict[str, object] = {}

        if auto_build_abs:
            _, dbg = self.top_incomparable_pairs_on_base(
                top_k=top_k, tol=tol, min_witness=min_witness, return_debug=True
            )
            self.last_debug = dbg
            self.build_abs_exprs_for_top_incomparables(
                top_k=top_k, tol=tol, min_witness=min_witness, register=True
            )
            if verbose_init:
                self.summary_abs_exprs(verbose=True)

    # ───────────────────────── adopt/copy (no shared caches) ───────────────────────── #

    def _adopt_base(self, base: GraffitiBase) -> None:
        self.df = base.df
        self.boolean_cols = list(base.boolean_cols)
        self.expr_cols = list(base.expr_cols)
        self.predicates = dict(base.predicates)
        self.base_predicates = dict(base.base_predicates)
        self.exprs = dict(base.exprs)
        self.base_hypothesis = base.base_hypothesis
        self.base_hypothesis_name = base.base_hypothesis_name
        self._mask_cache = {}
        self._mask_cache_version = id(self.df)
        self.synthetic_expr_names_ = set(getattr(base, "synthetic_expr_names_", set()))
        self.abs_exprs = list(getattr(base, "abs_exprs", []))

    # ───────────────────────────── small utilities ───────────────────────────── #

    def _scope_mask(self, cond: Optional[Predicate]) -> np.ndarray:
        if cond in (None, TRUE):
            return self.mask(self.base_hypothesis) if self.base_hypothesis is not TRUE else np.ones(len(self.df), dtype=bool)
        return self.mask(cond)

    @staticmethod
    def _finite_mask(a: np.ndarray) -> np.ndarray:
        return np.isfinite(a)

    def _arr(self, key: Union[str, Expr]) -> np.ndarray:
        """
        Resolve either a column/Expr name or an Expr into a float numpy array.
        Prefers evaluating self.exprs[name] when available so synthesized Exprs work.
        """
        if isinstance(key, str):
            if key in self.exprs:
                return self.exprs[key].eval(self.df).to_numpy(dtype=float, copy=False)
            return self.df[key].to_numpy(dtype=float, copy=False)
        return key.eval(self.df).to_numpy(dtype=float, copy=False)

    # ───────────────────────────── pair comparison core ───────────────────────────── #

    def _pair_compare(
        self,
        xcol: str,
        ycol: str,
        base_mask: Optional[np.ndarray] = None,
        *,
        tol: float,
    ) -> PairCompareResult:
        x = self._arr(xcol)
        y = self._arr(ycol)

        m = base_mask if base_mask is not None else np.ones(len(self.df), dtype=bool)
        m = m.copy()
        m &= self._finite_mask(x)
        m &= self._finite_mask(y)

        if not np.any(m):
            return PairCompareResult(xcol, ycol, 0, True, True, True, False, False, False)

        diff = x[m] - y[m]
        le_all = bool(np.all(diff <=  tol))
        ge_all = bool(np.all(-diff <= tol))
        eq_all = bool(np.all(np.abs(diff) <= tol))
        lt_any = bool(np.any(diff < -tol))
        gt_any = bool(np.any(diff >  tol))
        incomparable = lt_any and gt_any

        return PairCompareResult(xcol, ycol, int(m.sum()), le_all, ge_all, eq_all, lt_any, gt_any, incomparable)

    # ───────────────────────────── discovery: incomparables ───────────────────────────── #

    def find_noncomparable_pairs(
        self,
        *,
        scope: str = "base",
        max_pairs: Optional[int] = None,
        tol: float = 0.0,
        include_equal_as_comparable: bool = True,
        use_sorted_conjunctions: bool = False,
        conjunction_limit: Optional[int] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[PairCompareResult]]:
        if columns is None:
            cols = list(self.expr_cols)
        else:
            cols = [c for c in columns if c in self.expr_cols]

        out: Dict[str, List[PairCompareResult]] = {}

        def consider_scope(mask: Optional[np.ndarray], label: str) -> None:
            results: List[PairCompareResult] = []
            for i, xi in enumerate(cols):
                for yj in cols[i + 1:]:
                    pr = self._pair_compare(xi, yj, base_mask=mask, tol=tol)
                    if pr.incomparable or (not include_equal_as_comparable and pr.eq_all):
                        results.append(pr)
                        if max_pairs is not None and len(results) >= max_pairs:
                            out[label] = results
                            return
            out[label] = results

        if scope == "base":
            base_mask = self._scope_mask(self.base_hypothesis if self.base_hypothesis is not TRUE else None)
            consider_scope(base_mask, f"BASE[{self.base_hypothesis_name}]")
        elif scope == "global":
            consider_scope(None, "BASE[GLOBAL]")
        else:
            raise ValueError("scope must be one of {'base','global'}")

        if use_sorted_conjunctions and self._logic is not None:
            items = list(getattr(self._logic, "sorted_conjunctions_", []) or [])
            if conjunction_limit is not None:
                items = items[:max(0, int(conjunction_limit))]
            for name, pred in items:
                m = self._scope_mask(pred)
                consider_scope(m, f"COND[{name}]")

        return out

    # ───────────────── top-k incomparables on base → |x−y| synthesis ───────────────── #

    def _witness_stats(
        self,
        xcol: str,
        ycol: str,
        mask: Optional[np.ndarray],
        *,
        tol: float,
    ) -> tuple[int, int, float, int]:
        x = self._arr(xcol)
        y = self._arr(ycol)
        m = np.ones(len(self.df), dtype=bool) if mask is None else mask.copy()
        m &= np.isfinite(x)
        m &= np.isfinite(y)
        if not np.any(m):
            return (0, 0, float("nan"), 0)
        diff = x[m] - y[m]
        cnt_lt = int(np.sum(diff < -tol))
        cnt_gt = int(np.sum(diff >  tol))
        mean_abs = float(np.mean(np.abs(diff))) if int(m.sum()) else float("nan")
        return (cnt_lt, cnt_gt, mean_abs, int(m.sum()))

    def top_incomparable_pairs_on_base(
        self,
        *,
        top_k: int = 20,
        tol: float = 0.0,
        min_witness: int = 1,
        columns: Optional[Iterable[str]] = None,
        return_debug: bool = False,
    ) -> Union[List[Tuple[str, str, dict]], Tuple[List[Tuple[str, str, dict]], dict]]:
        if columns is None:
            cols = list(self.expr_cols)
        else:
            cols = [c for c in columns if c in self.expr_cols]

        base_mask = self._scope_mask(self.base_hypothesis if self.base_hypothesis is not TRUE else None)
        debug = {"base_support": int(base_mask.sum()), "tol": tol, "considered_pairs": 0, "kept_pairs": 0}

        ranked: List[Tuple[int, float, str, str, dict]] = []
        for i, xi in enumerate(cols):
            for yj in cols[i + 1:]:
                debug["considered_pairs"] += 1
                cnt_lt, cnt_gt, mean_abs, n_used = self._witness_stats(xi, yj, base_mask, tol=tol)
                if cnt_lt > 0 and cnt_gt > 0:
                    wmin = min(cnt_lt, cnt_gt)
                    if wmin >= int(min_witness):
                        meta = {"witness_min": wmin, "cnt_lt": cnt_lt, "cnt_gt": cnt_gt, "mean_abs": mean_abs, "n": n_used}
                        ranked.append((wmin, (mean_abs if np.isfinite(mean_abs) else -np.inf), xi, yj, meta))

        ranked.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
        out: List[Tuple[str, str, dict]] = [(x, y, meta) for (_, _, x, y, meta) in ranked[:max(0, int(top_k))]]
        debug["kept_pairs"] = len(out)
        return (out, debug) if return_debug else out

    def build_abs_exprs_for_top_incomparables(
        self,
        *,
        top_k: int = 20,
        tol: float = 0.0,
        min_witness: int = 1,
        columns: Optional[Iterable[str]] = None,
        register: bool = True,
        name_fmt: str = "abs({x}-{y})",
    ) -> List[Tuple[str, Expr]]:
        pairs = self.top_incomparable_pairs_on_base(
            top_k=top_k, tol=tol, min_witness=min_witness, columns=columns
        )
        built: List[Tuple[str, Expr]] = []
        for x, y, _meta in pairs:
            ex = self.exprs[x] if isinstance(x, str) else x
            ey = self.exprs[y] if isinstance(y, str) else y
            abs_expr: Expr = abs_(ex - ey)
            name = name_fmt.format(x=x, y=y)
            built.append((name, abs_expr))

        if register and built:
            for nm, ex in built:
                self.exprs[nm] = ex
                if nm not in self.expr_cols:
                    self.expr_cols.append(nm)
                self.synthetic_expr_names_.add(nm)
            self.abs_exprs.extend(built)

        return built

    # ───────────────────── surprise measures (optional but handy) ───────────────────── #

    @staticmethod
    def _incomparability_index_from_diff(diff: np.ndarray, *, tol: float) -> tuple[float, int, int, int]:
        n = diff.size
        if n == 0:
            return 0.0, 0, 0, 0
        cnt_lt = int(np.sum(diff < -tol))
        cnt_gt = int(np.sum(diff >  tol))
        I = 2.0 * float(min(cnt_lt, cnt_gt)) / float(n)
        return float(I), cnt_lt, cnt_gt, n

    def _masked_xy_pair(self, xcol: str, ycol: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self._arr(xcol)
        y = self._arr(ycol)
        m = mask.copy()
        m &= np.isfinite(x)
        m &= np.isfinite(y)
        return x[m], y[m]

    def surprising_measure_pair(
        self,
        xcol: str,
        ycol: str,
        *,
        hypothesis: Optional[Predicate] = None,
        tol: float = 1e-9,
        score: str = "diff",
        eps: float = 1e-12,
    ) -> dict:
        base_mask = self._scope_mask(self.base_hypothesis if self.base_hypothesis is not TRUE else None)
        H_mask = self._scope_mask(hypothesis)

        xb, yb = self._masked_xy_pair(xcol, ycol, base_mask)
        Ib, blt, bgt, bn = self._incomparability_index_from_diff(xb - yb, tol=tol)

        xh, yh = self._masked_xy_pair(xcol, ycol, H_mask)
        Ih, hlt, hgt, hn = self._incomparability_index_from_diff(xh - yh, tol=tol)

        surprise = (Ib - Ih) / max(Ib, eps) if score == "ratio" else (Ib - Ih)

        return {
            "surprise": float(surprise),
            "score": score,
            "I_base": float(Ib),
            "I_H": float(Ih),
            "counts_base": {"lt": blt, "gt": bgt, "n": bn},
            "counts_H": {"lt": hlt, "gt": hgt, "n": hn},
            "x": xcol, "y": ycol, "tol": float(tol),
            "hypothesis": repr(hypothesis) if hypothesis is not None else "BASE",
        }

    def surprising_top_pairs(
        self,
        *,
        hypothesis: Optional[Predicate],
        columns: Optional[Iterable[str]] = None,
        top_k: int = 25,
        score: str = "ratio",
        tol: float = 1e-9,
        min_n_base: int = 20,
        min_n_H: int = 20,
    ) -> pd.DataFrame:
        if columns is None:
            cols = list(self.expr_cols)
        else:
            cols = [c for c in columns if c in self.expr_cols]

        base_mask = self._scope_mask(self.base_hypothesis if self.base_hypothesis is not TRUE else None)
        H_mask = self._scope_mask(hypothesis)

        rows = []
        for i, xi in enumerate(cols):
            for yj in cols[i + 1:]:
                xb, yb = self._masked_xy_pair(xi, yj, base_mask)
                Ib, blt, bgt, bn = self._incomparability_index_from_diff(xb - yb, tol=tol)
                if bn < min_n_base:
                    continue

                xh, yh = self._masked_xy_pair(xi, yj, H_mask)
                Ih, hlt, hgt, hn = self._incomparability_index_from_diff(xh - yh, tol=tol)
                if hn < min_n_H:
                    continue

                surprise = (Ib - Ih) / max(Ib, 1e-12) if score == "ratio" else (Ib - Ih)
                rows.append({
                    "x": xi, "y": yj, "surprise": float(surprise), "score": score,
                    "I_base": float(Ib), "I_H": float(Ih),
                    "base_lt": blt, "base_gt": bgt, "n_base": bn,
                    "H_lt": hlt, "H_gt": hgt, "n_H": hn,
                    "witness_min_H": min(hlt, hgt),
                })

        if not rows:
            return pd.DataFrame(columns=[
                "x","y","surprise","score","I_base","I_H",
                "base_lt","base_gt","n_base","H_lt","H_gt","n_H","witness_min_H"
            ])

        df = pd.DataFrame(rows)
        df.sort_values(
            by=["surprise", "I_base", "witness_min_H", "x", "y"],
            ascending=[False, False, False, True, True],
            inplace=True
        )
        if top_k is not None and top_k >= 0:
            df = df.head(top_k).reset_index(drop=True)
        return df

    # ───────────────────── propose hypotheses from incomparables ───────────────────── #

    def propose_pair_hypotheses(
        self,
        *,
        top_k_pairs: int = 40,
        tol: float = 1e-9,
        min_support: int = 10,
        min_ratio: float = 0.20,
        columns: Optional[Iterable[str]] = None,
        register: bool = True,
        name_fmt_lt: str = "{x}<{y}",
        name_fmt_gt: str = "{x}>{y}",
        return_table: bool = True,
        auto_relax_if_empty: bool = True,
    ) -> Union[List[Tuple[str, Predicate, dict]], Tuple[List[Tuple[str, Predicate, dict]], pd.DataFrame]]:
        """
        Propose *strict* hypotheses from top incomparable pairs on BASE:

            H_xy^lt:  x < y  (with margin tol → x < y - tol)
            H_xy^gt:  x > y  (with margin tol → x > y + tol)

        Keep side S ∈ {lt,gt} when:
            support_S ≥ min_support  and  support_S / n ≥ min_ratio.

        Returns a list of (name, Predicate, meta), and optionally a diagnostics DataFrame.
        """
        # 1) Candidate pairs (incomparables) on base
        pairs = self.top_incomparable_pairs_on_base(
            top_k=top_k_pairs, tol=tol, min_witness=min_support, columns=columns
        )
        if isinstance(pairs, tuple):
            pairs, _dbg = pairs

        base_mask = self._scope_mask(self.base_hypothesis if self.base_hypothesis is not TRUE else None)

        # 2) Diagnostics per pair
        rows = []
        for x, y, meta in pairs:
            cnt_lt, cnt_gt, mean_abs, n_used = self._witness_stats(x, y, base_mask, tol=tol)
            rows.append({
                "x": x, "y": y, "n": n_used,
                "cnt_lt": cnt_lt, "cnt_gt": cnt_gt,
                "ratio_lt": (cnt_lt / n_used) if n_used else 0.0,
                "ratio_gt": (cnt_gt / n_used) if n_used else 0.0,
                "mean_abs": meta.get("mean_abs", float("nan")),
            })

        diag = pd.DataFrame(rows, columns=["x","y","n","cnt_lt","cnt_gt","ratio_lt","ratio_gt","mean_abs"])
        if not diag.empty:
            diag.sort_values(by=["n","mean_abs","x","y"], ascending=[False, False, True, True], inplace=True, kind="mergesort")

        def _select(min_sup: int, min_rat: float) -> List[Tuple[str, Predicate, dict]]:
            out: List[Tuple[str, Predicate, dict]] = []
            for _, r in diag.iterrows():
                x, y, n = str(r["x"]), str(r["y"]), int(r["n"])
                if n <= 0:
                    continue
                cnt_lt, cnt_gt = int(r["cnt_lt"]), int(r["cnt_gt"])
                ratio_lt, ratio_gt = float(r["ratio_lt"]), float(r["ratio_gt"])

                keep_lt = (cnt_lt >= min_sup) and (ratio_lt >= min_rat)
                keep_gt = (cnt_gt >= min_sup) and (ratio_gt >= min_rat)

                if not (keep_lt or keep_gt):
                    continue

                ex = self.exprs[x]
                ey = self.exprs[y]
                d  = ex - ey  # Expr difference

                if keep_lt:
                    # strict: d < -tol
                    rel_lt = Lt(d, Const(-tol), tol=0.0)
                    pred_lt: Predicate = self.relation_to_predicate(rel_lt)
                    name_lt = name_fmt_lt.format(x=x, y=y)
                    meta_lt = {"x": x, "y": y, "side": "x<y", "n": n,
                            "cnt_lt": cnt_lt, "cnt_gt": cnt_gt,
                            "ratio_lt": ratio_lt, "ratio_gt": ratio_gt, "tol": tol}
                    out.append((name_lt, pred_lt, meta_lt))

                if keep_gt:
                    # strict: d > +tol
                    rel_gt = Gt(d, Const(+tol), tol=0.0)
                    pred_gt: Predicate = self.relation_to_predicate(rel_gt)
                    name_gt = name_fmt_gt.format(x=x, y=y)
                    meta_gt = {"x": x, "y": y, "side": "x>y", "n": n,
                            "cnt_lt": cnt_lt, "cnt_gt": cnt_gt,
                            "ratio_lt": ratio_lt, "ratio_gt": ratio_gt, "tol": tol}
                    out.append((name_gt, pred_gt, meta_gt))
            return out

        cands = _select(min_support, min_ratio)

        # Optional auto-relax if nothing survives
        if auto_relax_if_empty and not cands and not diag.empty:
            looser_sup = max(1, min_support // 2)
            looser_rat = min(0.5, min_ratio * 0.5)
            cands = _select(looser_sup, looser_rat)

        # 3) Optionally register into base_predicates for GraffitiClassLogic to reason about redundancy
        if register and cands:
            for name, pred, _meta in cands:
                # don’t overwrite; only add new
                if name not in self.base_predicates:
                    self.base_predicates[name] = pred
                    # also keep a string name → Predicate for convenience if you have a main registry
                    self.predicates[name] = pred

        # 4) Pretty summary
        lines = []
        lines.append("──────────────────────────────────────────────")
        lines.append("GraffitiComparable • Candidate Hypotheses (strict)")
        lines.append("──────────────────────────────────────────────")
        lines.append(f"Total: {len(cands)}")
        lines.append("")
        for name, _pred, meta in cands[:20]:
            lines.append(f"  • {name}  [n={meta['n']}; lt={meta['cnt_lt']}, gt={meta['cnt_gt']}; "
                        f"r_lt={meta['ratio_lt']:.3f}, r_gt={meta['ratio_gt']:.3f}; tol={meta['tol']:.1e}]")
        print("\n".join(lines))

        return (cands, diag) if return_table else cands


    def summarize_candidate_hypotheses(
        self,
        *,
        max_list: int = 30,
        verbose: bool = True,
    ) -> str:
        """
        Pretty print the last batch of proposed hypotheses, if any.
        """
        items: List[Tuple[str, Predicate, dict]] = getattr(self, "candidate_hypotheses_", [])
        lines: List[str] = []
        lines.append("──────────────────────────────────────────────")
        lines.append("GraffitiComparable • Candidate Hypotheses (from incomparables)")
        lines.append("──────────────────────────────────────────────")
        lines.append(f"Total: {len(items)}")
        lines.append("")
        for name, _pred, meta in items[:max_list]:
            x, y = meta["x"], meta["y"]
            ratio_lt = meta["ratio_lt"]
            ratio_gt = meta["ratio_gt"]
            n = meta["n"]
            lines.append(f"  • {name}   [pair=({x},{y}); n={n}; lt={ratio_lt:.2%}; gt={ratio_gt:.2%}]")
        if len(items) > max_list:
            lines.append(f"  … +{len(items) - max_list} more")
        lines.append("──────────────────────────────────────────────")
        s = "\n".join(lines)
        if verbose:
            print(s)
        return s


    # ─────────────────────────────── summaries ─────────────────────────────── #

    def summary_abs_exprs(self, *, max_list: int = 20, verbose: bool = True) -> str:
        names = sorted(getattr(self, "synthetic_expr_names_", set()))
        lines: List[str] = []
        lines.append("──────────────────────────────────────────────")
        lines.append("GraffitiComparable • Registered |x−y| Expressions")
        lines.append("──────────────────────────────────────────────")
        lines.append(f"Total: {len(names)}")
        lines.append("")
        if not names:
            lines.append("(none)")
        else:
            for nm in names[:max_list]:
                lines.append(f"  • {nm}")
            if len(names) > max_list:
                lines.append(f"  … +{len(names) - max_list} more")
        lines.append("──────────────────────────────────────────────")
        s = "\n".join(lines)
        if verbose:
            print(s)
        return s

    def summary_noncomparables(
        self,
        *,
        scope: str = "base",
        tol: float = 0.0,
        max_pairs: Optional[int] = 20,
        use_sorted_conjunctions: bool = False,
        conjunction_limit: Optional[int] = 8,
        columns: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> str:
        findings = self.find_noncomparable_pairs(
            scope=scope,
            max_pairs=max_pairs,
            tol=tol,
            include_equal_as_comparable=True,
            use_sorted_conjunctions=use_sorted_conjunctions,
            conjunction_limit=conjunction_limit,
            columns=columns,
        )

        lines: List[str] = []
        lines.append("──────────────────────────────────────────────")
        lines.append("GraffitiComparable • Non-Comparable Numeric Pairs")
        lines.append("──────────────────────────────────────────────")
        lines.append(f"Base hypothesis: {getattr(self, 'base_hypothesis_name', 'TRUE')}")
        lines.append(f"DataFrame: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
        lines.append(f"Tolerance: {tol:g}")
        lines.append("")
        if not findings:
            lines.append("(no scopes evaluated)")
        else:
            for scope_name, pairs in findings.items():
                lines.append(f"{scope_name}: {len(pairs)} pair(s)")
                if not pairs:
                    lines.append("  (none)")
                else:
                    for pr in pairs:
                        sym = pr.relation_symbol()
                        witness = []
                        if pr.lt_any:
                            witness.append("x<y")
                        if pr.gt_any:
                            witness.append("x>y")
                        wtxt = ", ".join(witness) if witness else "-"
                        lines.append(f"  • {pr.x} {sym} {pr.y}   [n={pr.n}; witness: {wtxt}]")
                lines.append("")
        lines.append("──────────────────────────────────────────────")
        s = "\n".join(lines)
        if verbose:
            print(s)
        return s
