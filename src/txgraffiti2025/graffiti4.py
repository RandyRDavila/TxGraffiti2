# src/txgraffiti2025/graffiti4.py

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.graffiti_relations import GraffitiClassRelations
from txgraffiti2025.forms.utils import (
    Expr,
    to_expr,
    abs_ as abs_expr,
    min_ as min_expr,
    max_ as max_expr,
)
from txgraffiti2025.forms.generic_conjecture import (
    Conjecture,
    Ge,
    Le,
)
from txgraffiti2025.graffiti4_types import HypothesisInfo, NonComparablePair, Graffiti4Result
from txgraffiti2025.graffiti4_mixed import mixed_runner
from txgraffiti2025.graffiti4_poly_single import poly_single_runner

from txgraffiti2025.graffiti4_lp import lp_runner
from txgraffiti2025.graffiti4_lp_single import lp_single_runner
from txgraffiti2025.sophie import (
    SophieCondition,
    discover_sophie_from_inequalities,
    rank_sophie_conditions_global,
    print_sophie_conditions,
)

# inside Graffiti4 (just as a temporary shim)
from txgraffiti2025.graffiti4_runner import Graffiti4Runner, Graffiti4RunnerConfig
# ───────────────────────── helper data classes ───────────────────────── #

def _filter_by_touch(
    df: pd.DataFrame,
    conjectures: Sequence[Conjecture],
    min_touches: int,
) -> List[Conjecture]:
    """
    Compute touch_count for each conjecture and discard those with
    touch_count < min_touches.

    This is intended to be used *before* expensive heuristics
    (Morgan, Dalmatian), so that they only see reasonably “tight”
    candidates.
    """
    if min_touches <= 0:
        # Nothing to filter; still ensure touch_count is computed once.
        out: List[Conjecture] = []
        for c in conjectures:
            try:
                c.touch_count(df)
            except Exception:
                setattr(c, "touch_count", 0)
            out.append(c)
        return out

    kept: List[Conjecture] = []
    for c in conjectures:
        try:
            t = c.touch_count(df)  # sets c.touch and c.touch_count
        except Exception:
            t = 0
            setattr(c, "touch_count", 0)
        if t >= min_touches:
            kept.append(c)
    return kept


def _dedup_conjectures(conjs: Sequence[Conjecture]) -> List[Conjecture]:
    """
    Stable dedup by conjecture.signature().

    Keeps the first occurrence of each signature and drops later duplicates.
    """
    seen: set[str] = set()
    out: List[Conjecture] = []
    for c in conjs:
        sig = c.signature()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


# @dataclass
# class HypothesisInfo:
#     """Metadata for one hypothesis h."""
#     name: str              # printable name, e.g. "connected & planar"
#     pred: Any              # Predicate used as Conjecture.condition
#     mask: np.ndarray       # boolean mask over df.index


# @dataclass
# class NonComparablePair:
#     """Pair (x, y) of invariants that cross: sometimes x < y and sometimes x > y."""
#     left: Expr
#     right: Expr
#     left_name: str
#     right_name: str


# @dataclass
# class Graffiti4Result:
#     """Aggregated result of a call to Graffiti4.conjecture."""
#     target: str
#     conjectures: List[Conjecture]
#     sophie_conditions: List[SophieCondition]
#     stage_breakdown: Dict[str, Any]


# ───────────────────────── numeric helpers ───────────────────────── #

def _nice_fraction(
    x: float,
    *,
    max_denom: int = 50,
    max_numer: int = 200,
) -> Optional[Fraction]:
    """
    Approximate x by a "nice" rational p/q with small numerator/denominator.

    Returns None if:
      - x is not finite, or
      - |p| > max_numer or q > max_denom.

    This is what prevents coefficients like 4740631186705785/8 from appearing.
    """
    if not np.isfinite(x):
        return None

    frac = Fraction(x).limit_denominator(max_denom)
    if abs(frac.numerator) > max_numer or abs(frac.denominator) > max_denom:
        return None
    return frac


def _annotate_and_sort_conjectures(
    df: pd.DataFrame,
    conjs: Sequence[Conjecture],
) -> List[Conjecture]:
    """
    Compute touch_count and support_n for each conjecture, deduplicate by
    signature, and sort by (touch_count, support_n) descending.
    """
    unique: List[Conjecture] = []
    seen: set[str] = set()

    for c in conjs:
        sig = c.signature()
        if sig in seen:
            continue
        seen.add(sig)

        # Compute touch_count once; Conjecture.touch_count mutates itself
        touch_attr = getattr(c, "touch_count", None)
        if callable(touch_attr):
            try:
                val = c.touch_count(df, auto_base=False)
            except TypeError:
                # Fallback if signature differs
                val = c.touch_count(df)
        else:
            # Already materialized as an int
            val = touch_attr if isinstance(touch_attr, int) else 0

        setattr(c, "touch_count", int(val))
        setattr(c, "touch", int(val))  # for backward compatibility

        # Compute support_n: how many rows are in the hypothesis class
        try:
            applicable, _, _ = c.check(df, auto_base=False)
            support = int(applicable.sum())
        except Exception:
            support = 0

        setattr(c, "support_n", support)
        setattr(c, "support", support)

        unique.append(c)

    unique.sort(
        key=lambda cc: (
            int(getattr(cc, "touch_count", 0)),
            int(getattr(cc, "support_n", 0)),
        ),
        reverse=True,
    )
    return unique


# ───────────────────────── main workspace ───────────────────────── #

class Graffiti4:
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
    ) -> Graffiti4Result:
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

        # ── Stage 3: mixed intricate bounds ─────────────────────────────
        # if complexity >= 3:
        #     mixed_conjs = mixed_runner(
        #         target_col=target,
        #         target_expr=target_expr,
        #         primaries=self.invariants,   # or a curated subset
        #         secondaries=self.invariants, # idem
        #         hypotheses=self.hypotheses,
        #         df=self.df,
        #         weight=0.5,
        #     )
        #     mixed_conjs = _filter_by_touch(self.df, mixed_conjs, min_touches)

        #     mixed_conjs = heuristic_runner(
        #         mixed_conjs,
        #         df=self.df,
        #         morgan_filter=self.morgan_filter,
        #         dalmatian_filter=self.dalmatian_filter,
        #     )
        #     mixed_sophie = sophie_runner(conjectures=mixed_conjs, g4=self)

        #     all_conjectures.extend(mixed_conjs)
        #     all_conjectures = _dedup_conjectures(all_conjectures)
        #     all_sophie.extend(mixed_sophie)

        #     stage_info["mixed"] = dict(
        #         conjectures=len(mixed_conjs),
        #         sophie=len(mixed_sophie),
        #     )

        # if complexity >= 3:
        #     mixed_conjs = mixed_runner(
        #         target_col=target,
        #         target_expr=target_expr,
        #         primaries=self.invariants,       # or a curated subset
        #         secondaries=self.invariants,     # idem
        #         hypotheses=self.hypotheses,
        #         df=self.df,
        #         min_support=10,
        #         coef_bound=10.0,
        #         max_denom=20,
        #     )
        #     mixed_conjs = heuristic_runner(
        #         mixed_conjs,
        #         df=self.df,
        #         morgan_filter=self.morgan_filter,
        #         dalmatian_filter=self.dalmatian_filter,
        #     )
        #     mixed_sophie = sophie_runner(conjectures=mixed_conjs, g4=self)

        #     all_conjectures.extend(mixed_conjs)
        #     all_conjectures = _dedup_conjectures(all_conjectures)
        #     all_sophie.extend(mixed_sophie)
        #     stage_info["mixed_ratio_lp"] = dict(
        #         conjectures=len(mixed_conjs),
        #         sophie=len(mixed_sophie),
        #     )


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

        return Graffiti4Result(
            target=target,
            conjectures=all_conjectures,
            sophie_conditions=all_sophie_ranked,
            stage_breakdown=stage_info,
        )



    def conjecture_v2(self, target: str, complexity: int = 2):
        cfg = Graffiti4RunnerConfig(
            min_touch=3,
            complexity=complexity,
            include_invariant_products=False,
            include_abs=True,
            include_min_max=True,
        )
        runner = Graffiti4Runner(self, cfg)
        return runner.run(target)


# ───────────────────────── runner functions ───────────────────────── #

def constant_runner(
    *,
    target_col: str,
    target_expr: Expr,
    hypotheses: Sequence[HypothesisInfo],
    df: pd.DataFrame,
) -> List[Conjecture]:
    """
    Stage-0 generator: for each hypothesis h, produce constant bounds

        h ⇒ target ≥ c_min
        h ⇒ target ≤ c_max

    where c_min, c_max are taken over rows satisfying h & base.
    """
    conjs: List[Conjecture] = []
    vals = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        finite = mask & np.isfinite(vals)

        if finite.sum() == 0:
            continue

        v = vals[finite]
        # For constants, we expect integers; enforce that explicitly.
        c_min = int(np.min(v))
        c_max = int(np.max(v))

        rhs_min = to_expr(c_min)
        rhs_max = to_expr(c_max)

        rel_ge = Ge(left=target_expr, right=rhs_min)
        rel_le = Le(left=target_expr, right=rhs_max)

        c_ge = Conjecture(
            relation=rel_ge,
            condition=hyp.pred,
            name=f"[const-min] {target_col} under {hyp.name}",
        )

        c_le = Conjecture(
            relation=rel_le,
            condition=hyp.pred,
            name=f"[const-max] {target_col} under {hyp.name}",
        )
        c_ge.target_name = target_col
        c_le.target_name = target_col
        conjs.extend([c_ge, c_le])

    return conjs


from fractions import Fraction

def _nice_ratio_coef(
    c: float,
    *,
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_denom: int = 20,
) -> Optional[float]:
    """
    Clean up a ratio coefficient:

      - If |c| < zero_tol, treat as 0 and skip (too trivial).
      - If |c| > max_coef_abs, skip (too extreme to be useful).
      - Otherwise, project to a nearby rational with small denominator.
    """
    c = float(c)
    if not np.isfinite(c):
        return None
    if abs(c) < zero_tol:
        return None
    if abs(c) > max_coef_abs:
        return None
    frac = Fraction(c).limit_denominator(max_denom)
    return float(frac)

from .graffiti4_lp import _build_affine_expr, _rationalize_scalar

def ratio_runner(
    *,
    target_col: str,
    target_expr: Expr,
    others: Dict[str, Expr],
    hypotheses: Sequence[HypothesisInfo],
    df: pd.DataFrame,
    min_support: int = 5,
    # niceness parameters should match your LP runner
    zero_tol: float = 1e-8,
    max_coef_abs: float = 4.0,
    max_denom: int = 20,
) -> List[Conjecture]:
    """
    Stage-1 generator: for each hypothesis h and each 'other' invariant x,
    compute ratios r_i = target_i / x_i on valid rows and set

        c_min = min r_i,  c_max = max r_i

    to get

        h ⇒ target ≥ c_min * x
        h ⇒ target ≤ c_max * x,

    with coefficients cleaned via the same affine builder as the LP runner.
    This ensures that, e.g.,

        target ≤ 1 · annihilation_number

    and

        target ≤ annihilation_number

    normalize to the *same* Expr and dedup works correctly.
    """
    conjs: List[Conjecture] = []
    target_vals = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        H = np.asarray(hyp.mask, dtype=bool)

        for other_name, other_expr in others.items():
            try:
                other_vals = other_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            valid = (
                H
                & np.isfinite(target_vals)
                & np.isfinite(other_vals)
                & (other_vals != 0.0)
            )
            if valid.sum() < min_support:
                continue

            r = target_vals[valid] / other_vals[valid]

            # raw extrema
            c_min_raw = float(np.min(r))
            c_max_raw = float(np.max(r))

            # rationalize a bit (same style as LP)
            c_min_rat = Fraction(c_min_raw).limit_denominator(max_denom)
            c_max_rat = Fraction(c_max_raw).limit_denominator(max_denom)
            c_min = float(c_min_rat)
            c_max = float(c_max_rat)

            # build RHS = c * x via the shared affine helper
            # (intercept is 0.0; we use same zero_tol+max_coef_abs logic)
            rhs_min = _build_affine_expr(
                const_val=0.0,
                coefs=[c_min],
                feats=[other_expr],
                zero_tol=zero_tol,
                max_coef_abs=max_coef_abs,
                max_intercept_abs=0.0,  # 0 intercept, so this is safe
            )
            rhs_max = _build_affine_expr(
                const_val=0.0,
                coefs=[c_max],
                feats=[other_expr],
                zero_tol=zero_tol,
                max_coef_abs=max_coef_abs,
                max_intercept_abs=0.0,
            )

            # If coefficient is too small/too large, _build_affine_expr returns None
            if rhs_min is not None:
                rel_ge = Ge(left=target_expr, right=rhs_min)
                c_ge = Conjecture(
                    relation=rel_ge,
                    condition=hyp.pred,
                    name=f"[ratio-min] {target_col} vs {other_name} under {hyp.name}",
                )
                c_ge.target_name = target_col
                # c_le.target_name = target_col
                conjs.append(c_ge)

            if rhs_max is not None:
                rel_le = Le(left=target_expr, right=rhs_max)
                c_le = Conjecture(
                    relation=rel_le,
                    condition=hyp.pred,
                    name=f"[ratio-max] {target_col} vs {other_name} under {hyp.name}",
                )
                # c_ge.target_name = target_col
                c_le.target_name = target_col
                conjs.append(c_le)

    return conjs



def heuristic_runner(
    conjectures: Sequence[Conjecture],
    *,
    df: pd.DataFrame,
    morgan_filter=None,
    dalmatian_filter=None,
) -> List[Conjecture]:
    """
    Apply Morgan and Dalmatian-style filters in sequence, if provided.
    Expected signatures:

        morgan_filter(df, conjectures) -> list[Conjecture]
        dalmatian_filter(df, conjectures) -> list[Conjecture]

    (You can plug your existing implementations here; for now the default
    is to just return the input list sorted by touch/support.)
    """
    out = list(conjectures)

    if morgan_filter is not None:
        out = morgan_filter(df, out)

    if dalmatian_filter is not None:
        out = dalmatian_filter(df, out)

    # As a fallback, we lightly annotate and sort here as well,
    # though Graffiti4.conjecture will do a full annotation later.
    out = _annotate_and_sort_conjectures(df, out)
    return out


def sophie_runner(
    *,
    conjectures: Sequence[Conjecture],
    g4: Graffiti4,
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

def print_g4_result(
    result: Graffiti4Result,
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
