# src/txgraffiti/graffiti3/graffiti3.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .class_relations import GraffitiClassRelations
from .exprs import (
    Expr,
    to_expr,
    abs_ as abs_expr,
    min_ as min_expr,
    max_ as max_expr,
)
from .relations import Conjecture
from .utils import (
    _filter_by_touch,
    _dedup_conjectures,
    _annotate_and_sort_conjectures,
)
from .types import HypothesisInfo, NonComparablePair, Graffiti3Result, SophieCondition
from .sophie import (
    discover_sophie_from_inequalities,
    rank_sophie_conditions_global,
    print_sophie_conditions,
)


# Runners
from txgraffiti.graffiti3.runners.constant import constant_runner
from txgraffiti.graffiti3.runners.ratio import ratio_runner
from txgraffiti.graffiti3.runners.lp import lp_single_runner, lp_runner
from txgraffiti.graffiti3.runners.mixed import mixed_runner
from txgraffiti.graffiti3.runners.poly import poly_single_runner
from txgraffiti.graffiti3.runners.heuristic import heuristic_runner


# ───────────────────────── main workspace ───────────────────────── #

class Graffiti3:
    """
    Refactored, modular conjecturing workspace.

    Construction:
        g4 = Graffiti4(df)

    Precomputes:
        - base_property, base_name, base_mask
        - invariants: Exprs for numeric columns
        - invariant_products: pairwise products of invariants
        - properties: atomic boolean columns
        - hypotheses: nonredundant conjunctions of properties with base
        - non_comparable_invariants
        - abs_exprs, min_max_exprs on non-comparable pairs

    Usage:
        result = g4.conjecture(
            target="independence_number",
            complexity=1,
            include_invariant_products=True,
            include_abs=True,
            include_min_max=True,
        )
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_boolean_arity: int = 2,
        morgan_filter=None,
        dalmatian_filter=None,
        sophie_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.df = df.copy()
        self.gcr = GraffitiClassRelations(self.df)

        # optional heuristic hooks
        self.morgan_filter = morgan_filter
        self.dalmatian_filter = dalmatian_filter

        self.sophie_cfg = sophie_cfg or {}

        # base hypothesis & mask
        self.base_property = self.gcr.base_hypothesis
        self.base_name = self.gcr.base_hypothesis_name
        self.base_mask = np.asarray(
            self.gcr._mask_cached(self.gcr.base_hypothesis), dtype=bool
        )

        # numeric invariants as Exprs
        self.invariants: Dict[str, Expr] = self._build_invariants()

        # boolean properties from df / GCR
        self.properties: List[str] = list(self.gcr.boolean_cols)

        # hypotheses: nonredundant conjunctions (arity 1–2) intersected with base
        self.hypotheses: List[HypothesisInfo] = self._build_hypotheses(
            max_boolean_arity=max_boolean_arity
        )

        # pairwise products of invariants
        self.invariant_products: Dict[str, Expr] = self._build_invariant_products()

        # non-comparable invariant pairs (w.r.t. base_mask)
        self.non_comparable_pairs: List[NonComparablePair] = (
            self._find_non_comparable_pairs()
        )

        # abs(x - y) columns for non-comparable pairs
        self.abs_exprs: Dict[str, Expr] = self._build_abs_exprs()

        # min(x,y) / max(x,y) for non-comparable pairs
        self.min_max_exprs: Dict[str, Expr] = self._build_min_max_exprs()

    # ── internal construction helpers ──

    def _build_invariants(self) -> Dict[str, Expr]:
        """Numeric columns → Expr."""
        out: Dict[str, Expr] = {}
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                out[col] = to_expr(col)
        return out



    def _build_hypotheses(self, max_boolean_arity: int) -> List[HypothesisInfo]:
        """
        Build nontrivial hypotheses inside the base universe, avoiding
        trivial conjunctions like (bipartite & triangle_free) when they
        define the same set of graphs as a simpler property.

        Strategy
        --------
        1. Ask GraffitiClassRelations for nonredundant conjunctions
           (arity ≤ max_boolean_arity).
        2. Restrict each mask to the base universe.
        3. Discard empty masks and those equal to the base mask.
        4. For identical masks, keep the *simplest* label
           (fewest '&' parts).
        5. Return HypothesisInfo objects, with base first.
        """
        # Step 1: get GCR's nonredundant boolean conjunctions
        self.gcr.enumerate_conjunctions(max_arity=max_boolean_arity)
        nonred, _, _ = self.gcr.find_redundant_conjunctions()

        hyps: List[HypothesisInfo] = []

        # Base hypothesis first
        base_info = HypothesisInfo(
            name=self.base_name,
            pred=self.base_property,
            mask=self.base_mask.copy(),
        )
        hyps.append(base_info)

        # We track masks via bytes to detect identical classes
        # and keep the simplest label for each.
        def _complexity(label: str) -> int:
            # crude complexity: number of '&'-parts
            # "bipartite" → 1, "bipartite & triangle_free" → 2, etc.
            parts = [p.strip() for p in label.split("&") if p.strip()]
            return max(1, len(parts))

        seen: Dict[bytes, Tuple[int, HypothesisInfo]] = {}

        # Register the base itself so anything equal to base is ignored
        base_key = self.base_mask.tobytes()
        seen[base_key] = (0, base_info)  # complexity 0 = special base

        # Step 2–4: process all nonredundant conjunctions
        for name, pred in nonred:
            raw_mask = np.asarray(self.gcr._mask_cached(pred), dtype=bool)

            # Restrict to base universe
            mask = raw_mask & self.base_mask
            if not mask.any():
                # empty class
                continue

            key = mask.tobytes()
            comp = _complexity(name)

            # Skip those that *are* the base (or already represented
            # by some simpler label).
            if key in seen:
                old_comp, old_info = seen[key]
                if comp >= old_comp:
                    # existing representative is simpler or equal — drop new
                    continue
                # otherwise, new one is simpler: replace old
                seen[key] = (comp, HypothesisInfo(name=name, pred=pred, mask=mask))
            else:
                seen[key] = (comp, HypothesisInfo(name=name, pred=pred, mask=mask))

        # Step 5: collect all but the base (which we already added)
        for key, (_comp, info) in seen.items():
            if info.name == self.base_name:
                continue  # base already in hyps[0]
            hyps.append(info)

        return hyps


    def _build_invariant_products(self) -> Dict[str, Expr]:
        """All pairwise products of numeric invariants."""
        cols = list(self.invariants.keys())
        out: Dict[str, Expr] = {}
        for i in range(len(cols)):
            for j in range(i, len(cols)):
                a, b = cols[i], cols[j]
                name = f"{a}·{b}" if a != b else f"{a}²"
                out[name] = self.invariants[a] * self.invariants[b]
        return out

    def _find_non_comparable_pairs(self) -> List[NonComparablePair]:
        """
        Find pairs (x, y) such that, restricted to the base universe,
        there exist rows with x < y and rows with x > y.
        """
        cols = list(self.invariants.keys())
        vals = {c: self.df[c].to_numpy(dtype=float) for c in cols}
        bm = self.base_mask
        out: List[NonComparablePair] = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                xa = vals[a][bm]
                xb = vals[b][bm]

                finite_mask = np.isfinite(xa) & np.isfinite(xb)
                if finite_mask.sum() == 0:
                    continue

                x = xa[finite_mask]
                y = xb[finite_mask]

                lt = (x < y).any()
                gt = (x > y).any()
                if lt and gt:
                    out.append(
                        NonComparablePair(
                            left=self.invariants[a],
                            right=self.invariants[b],
                            left_name=a,
                            right_name=b,
                        )
                    )
        return out

    def _build_abs_exprs(self) -> Dict[str, Expr]:
        """abs(x - y) for each non-comparable pair (x, y)."""
        out: Dict[str, Expr] = {}
        for pair in self.non_comparable_pairs:
            name = f"|{pair.left_name} - {pair.right_name}|"
            out[name] = abs_expr(pair.left - pair.right)
        return out

    def _build_min_max_exprs(self) -> Dict[str, Expr]:
        """min(x,y) and max(x,y) for each non-comparable pair (x, y)."""
        out: Dict[str, Expr] = {}
        for pair in self.non_comparable_pairs:
            name_min = f"min({pair.left_name}, {pair.right_name})"
            name_max = f"max({pair.left_name}, {pair.right_name})"
            out[name_min] = min_expr(pair.left, pair.right)
            out[name_max] = max_expr(pair.left, pair.right)
        return out

    # ── helper: build others pool from flags ──

    def _build_others_pool(
        self,
        target: str,
        *,
        include_invariant_products: bool,
        include_abs: bool,
        include_min_max: bool,
    ) -> Dict[str, Expr]:
        """
        Construct the dictionary of available "other" expressions, respecting
        the boolean flags and removing anything that obviously depends on
        the target column.
        """
        target_str = target

        def _uses_target(expr: Expr) -> bool:
            # Quick string-based detection; swap in structural helper if desired.
            return target_str in repr(expr)

        pool: Dict[str, Expr] = {}

        # base numeric invariants
        for name, expr in self.invariants.items():
            if name == target:
                continue
            pool[name] = expr

        if include_invariant_products:
            for name, expr in self.invariant_products.items():
                if _uses_target(expr):
                    continue
                pool[name] = expr

        if include_abs:
            for name, expr in self.abs_exprs.items():
                if _uses_target(expr):
                    continue
                pool[name] = expr

        if include_min_max:
            for name, expr in self.min_max_exprs.items():
                if _uses_target(expr):
                    continue
                pool[name] = expr

        return pool

    # ── public conjecturing API ──

    def conjecture(
        self,
        target: str,
        complexity: int,
        *,
        include_invariant_products: bool = False,
        include_abs: bool = False,
        include_min_max: bool = False,
    ) -> Graffiti3Result:
        """
        Main driver, matching your staged design:

            Stage 0: constant bounds on target per hypothesis
            Stage 1: ratio bounds target vs each other invariant
            Stage 2: LP bounds with 1 other
            Stage 3: LP bounds with up to 2 others
            Stage 4: mixed intricate bounds

        Before applying Morgan/Dalmatian, we now *filter* conjectures
        by touch_count to avoid wasting time on weak statements.
        """
        if target not in self.df.columns:
            raise KeyError(f"Target column '{target}' not found.")

        target_expr = to_expr(target)

        # Minimum touches required to even *enter* heuristics/Sophie.
        # You can move this to __init__ or make it an argument.
        min_touches = 3

        # Build pool of "other" expressions (excluding the target itself)
        others: Dict[str, Expr] = self._build_others_pool(
            target=target,
            include_invariant_products=include_invariant_products,
            include_abs=include_abs,
            include_min_max=include_min_max,
        )


        all_conjectures: List[Conjecture] = []
        all_sophie: List[SophieCondition] = []
        stage_info: Dict[str, Any] = {}

        # ── Stage 0: constant bounds on target per hypothesis ────────────
        const_conjs = constant_runner(
            target_col=target,
            target_expr=target_expr,
            hypotheses=self.hypotheses,
            df=self.df,
        )
        # early touch-based filter (before heuristics)
        const_conjs = _filter_by_touch(self.df, const_conjs, min_touches)

        const_conjs = heuristic_runner(
            const_conjs,
            df=self.df,
            morgan_filter=self.morgan_filter,
            dalmatian_filter=self.dalmatian_filter,
        )
        const_sophie = sophie_runner(
            conjectures=const_conjs,
            g4=self,
        )

        all_conjectures.extend(const_conjs)
        all_conjectures = _dedup_conjectures(all_conjectures)
        all_sophie.extend(const_sophie)

        stage_info["constant"] = dict(
            conjectures=len(const_conjs),
            sophie=len(const_sophie),
        )

        # ── Stage 1: ratio-based bounds target vs other ────────────────
        if complexity >= 1:
            ratio_conjs = ratio_runner(
                target_col=target,
                target_expr=target_expr,
                others=others,
                hypotheses=self.hypotheses,
                df=self.df,
            )
            ratio_conjs = _filter_by_touch(self.df, ratio_conjs, min_touches)


            ratio_conjs = heuristic_runner(
                ratio_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )

            ratio_sophie = sophie_runner(
                conjectures=ratio_conjs,
                g4=self,
            )

            all_conjectures.extend(ratio_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(ratio_sophie)

            stage_info["ratio"] = dict(
                conjectures=len(ratio_conjs),
                sophie=len(ratio_sophie),
            )

            # 1-variable LP: t <= m*x + b, t >= m*x + b
            lp1_conjs = lp_single_runner(
                target_col=target,
                target_expr=target_expr,
                others=others,
                hypotheses=self.hypotheses,
                df=self.df,
                direction="both",  # upper + lower
            )
            lp1_conjs = _filter_by_touch(self.df, lp1_conjs, min_touches)

            lp1_conjs = heuristic_runner(
                lp1_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )
            lp1_sophie = sophie_runner(
                conjectures=lp1_conjs,
                g4=self,
            )

            all_conjectures.extend(lp1_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(lp1_sophie)

            stage_info["lp1"] = dict(
                conjectures=len(lp1_conjs),
                sophie=len(lp1_sophie),
            )

        # ── Stage 2: LP with up to 2 features ────────────────────────────
        if complexity >= 2:
            lp_conjs = lp_runner(
                target_col=target,
                target_expr=target_expr,
                others=others,
                hypotheses=self.hypotheses,
                df=self.df,
                max_features=2,
                max_denom=20,
                coef_bound=10.0,
                direction="both",
            )
            lp_conjs = _filter_by_touch(self.df, lp_conjs, min_touches)

            lp_conjs = heuristic_runner(
                lp_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )
            lp_sophie = sophie_runner(conjectures=lp_conjs, g4=self)

            all_conjectures.extend(lp_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(lp_sophie)

            stage_info["lp"] = dict(
                conjectures=len(lp_conjs),
                sophie=len(lp_sophie),
            )

            poly_conjs = poly_single_runner(
                target_col=target,
                target_expr=target_expr,
                others=others,
                hypotheses=self.hypotheses,
                df=self.df,
                min_support=8,
                max_denom=20,
                max_coef_abs=4.0,
            )
            poly_conjs = heuristic_runner(
                poly_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )
            poly_sophie = sophie_runner(conjectures=poly_conjs, g4=self)

            all_conjectures.extend(poly_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(poly_sophie)
            stage_info["poly_single"] = dict(
                conjectures=len(poly_conjs),
                sophie=len(poly_sophie),
            )

        if complexity >= 3:
            mixed_conjs = mixed_runner(
                target_col=target,
                target_expr=target_expr,
                primaries=self.invariants,
                secondaries=self.invariants,
                hypotheses=self.hypotheses,
                df=self.df,
                weight=0.5,
            )
            mixed_conjs = heuristic_runner(
                mixed_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )
            mixed_sophie = sophie_runner(conjectures=mixed_conjs, g4=self)

            all_conjectures.extend(mixed_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(mixed_sophie)
            stage_info["mixed"] = dict(
                conjectures=len(mixed_conjs),
                sophie=len(mixed_sophie),
            )

        # ── Final pass: annotate & sort conjectures by touches/support ──
        all_conjectures = _annotate_and_sort_conjectures(self.df, all_conjectures)

        # ── Sophie: deduplicate and globally rank ───────────────────────
        all_sophie_ranked = rank_sophie_conditions_global(
            _group_sophie_by_property(all_sophie)
        )

        return Graffiti3Result(
            target=target,
            conjectures=all_conjectures,
            sophie_conditions=all_sophie_ranked,
            stage_breakdown=stage_info,
        )




def sophie_runner(
    *,
    conjectures: Sequence[Conjecture],
    g4: Graffiti3,
) -> List[SophieCondition]:
    """
    Thin wrapper around your Sophie module: take a list of inequality
    conjectures and return a *flat list* of SophieCondition objects.
    """
    if not conjectures:
        return []

    df_num = g4.df
    bool_cols = list(g4.gcr.boolean_cols)
    bool_df = df_num[bool_cols].copy() if bool_cols else pd.DataFrame(index=df_num.index)

    base_mask = g4.base_mask
    base_name = g4.base_name

    sophie_by_prop = discover_sophie_from_inequalities(
        df_num=df_num,
        bool_df=bool_df,
        base_mask=base_mask,
        base_name=base_name,
        inequality_conjectures=conjectures,
        **g4.sophie_cfg,
    )

    flat: List[SophieCondition] = []
    for _, conds in sophie_by_prop.items():
        flat.extend(conds)
    return flat


def _group_sophie_by_property(
    conds: Sequence[SophieCondition],
) -> Dict[str, List[SophieCondition]]:
    """
    Helper: take a flat list of SophieCondition and group them into the
    dictionary format expected by rank_sophie_conditions_global.
    """
    out: Dict[str, List[SophieCondition]] = {}
    for sc in conds:
        out.setdefault(sc.property_name, []).append(sc)
    return out


# ───────────────────────── top-level pretty printer ───────────────────────── #

def print_g3_result(
    result: Graffiti3Result,
    *,
    k_conjectures: int = 20,
    k_sophie: int = 20,
) -> None:
    """
    Convenience printer for Graffiti4Result: stage breakdown, top conjectures
    (sorted by touches), and top Sophie conditions.
    """
    print("Stage breakdown:", result.stage_breakdown)
    print(f"Total conjectures: {len(result.conjectures)}")
    print(f"Total Sophie conditions: {len(result.sophie_conditions)}\n")

    print("=== Top conjectures (by touch_count, then support) ===\n")
    for i, c in enumerate(result.conjectures[:k_conjectures], 1):
        touches = getattr(c, "touch_count", getattr(c, "touch", "?"))
        support = getattr(c, "support_n", getattr(c, "support", "?"))
        print(f"Conjecture {i}. {c.pretty()}   [touches={touches}, support={support}]\n")

    print_sophie_conditions(result.sophie_conditions, top_n=k_sophie)
