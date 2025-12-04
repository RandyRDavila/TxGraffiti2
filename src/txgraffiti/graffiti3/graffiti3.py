# from __future__ import annotations

# from enum import Enum
# from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# import numpy as np
# import pandas as pd

# from .class_relations import GraffitiClassRelations
# from .exprs import (
#     Expr,
#     to_expr,
#     abs_ as abs_expr,
#     min_ as min_expr,
#     max_ as max_expr,
#     log as log_expr,
# )
# from .relations import Conjecture
# from .utils import (
#     _filter_by_touch,
#     _dedup_conjectures,
#     _annotate_and_sort_conjectures,
# )
# from .types import HypothesisInfo, NonComparablePair, Graffiti3Result, SophieCondition
# from .sophie import (
#     discover_sophie_from_inequalities,
#     rank_sophie_conditions_global,
#     print_sophie_conditions,
# )

# # Runners
# from txgraffiti.graffiti3.runners.constant import constant_runner
# from txgraffiti.graffiti3.runners.ratio import ratio_runner
# from txgraffiti.graffiti3.runners.lp import lp_single_runner, lp_runner
# from txgraffiti.graffiti3.runners.mixed import mixed_runner
# from txgraffiti.graffiti3.runners.poly import poly_single_runner
# from txgraffiti.graffiti3.runners.heuristic import heuristic_runner

# # Assume sophie_runner + _group_sophie_by_property are defined in this module
# # or imported elsewhere, as in your existing code.


# # ───────────────────────── stage enum ───────────────────────── #

# class Stage(str, Enum):
#     """Named stages in the Graffiti3 conjecturing pipeline."""
#     CONSTANT = "constant"
#     RATIO = "ratio"
#     LP1 = "lp1"
#     LP = "lp"
#     LP3 = "lp3"
#     POLY_SINGLE = "poly_single"
#     MIXED = "mixed"


# StageLike = Union[Stage, str]


# # ───────────────────────── main workspace ───────────────────────── #

# class Graffiti3:
#     """
#     Refactored, modular conjecturing workspace.

#     Construction
#     ------------
#         g3 = Graffiti3(df)

#     Precomputes
#     -----------
#     - base_property, base_name, base_mask
#     - invariants: Exprs for numeric columns
#     - invariant_products: pairwise products of invariants
#     - properties: atomic boolean columns
#     - hypotheses: nonredundant conjunctions of properties with base
#     - non_comparable_invariants
#     - abs_exprs, min_max_exprs on non-comparable pairs

#     Conjecturing API
#     ----------------
#     The main entry point is

#         result = g3.conjecture(
#             target="independence_number",
#             complexity=2,                     # or stages=[Stage.RATIO, Stage.LP]
#             include_invariant_products=True,
#             include_abs=True,
#             include_min_max=True,
#             enable_sophie=True,
#             # sophie_stages=[Stage.LP, Stage.POLY_SINGLE],
#         )

#     Stages are:

#         Stage.CONSTANT     : constant bounds per hypothesis
#         Stage.RATIO        : ratio bounds between target and each other invariant
#         Stage.LP1          : 1-feature LP bounds (t ≶ m x + b)
#         Stage.LP           : 2-feature LP bounds
#         Stage.POLY_SINGLE  : polynomial-ish single-feature bounds
#         Stage.MIXED        : mixed intricate bounds

#     You can either provide an explicit list of stages or use `complexity`
#     for a simple, backward-compatible knob:

#         complexity=0  →  [CONSTANT]
#         complexity=1  →  [CONSTANT, RATIO, LP1]
#         complexity=2  →  [CONSTANT, RATIO, LP1, LP, POLY_SINGLE]
#         complexity≥3  →  all of the above + [MIXED]
#     """

#     # ──────────────────────── construction ────────────────────────

#     def __init__(
#         self,
#         df: pd.DataFrame,
#         *,
#         max_boolean_arity: int = 2,
#         morgan_filter=None,
#         dalmatian_filter=None,
#         sophie_cfg: Optional[Dict[str, Any]] = None,
#     ) -> None:
#         self.df = df.copy()
#         self.gcr = GraffitiClassRelations(self.df)

#         # optional heuristic hooks
#         self.morgan_filter = morgan_filter
#         self.dalmatian_filter = dalmatian_filter

#         # Sophie configuration (passed through to runner/discovery)
#         self.sophie_cfg = sophie_cfg or {}

#         # base hypothesis & mask
#         self.base_property = self.gcr.base_hypothesis
#         self.base_name = self.gcr.base_hypothesis_name
#         self.base_mask = np.asarray(
#             self.gcr._mask_cached(self.gcr.base_hypothesis), dtype=bool
#         )

#         # numeric invariants as Exprs
#         self.invariants: Dict[str, Expr] = self._build_invariants()

#         # boolean properties from df / GCR
#         self.properties: List[str] = list(self.gcr.boolean_cols)

#         # hypotheses: nonredundant conjunctions (arity 1–2) intersected with base
#         self.hypotheses: List[HypothesisInfo] = self._build_hypotheses(
#             max_boolean_arity=max_boolean_arity
#         )

#         # pairwise products of invariants
#         self.invariant_products: Dict[str, Expr] = self._build_invariant_products()

#         # non-comparable invariant pairs (w.r.t. base_mask)
#         self.non_comparable_pairs: List[NonComparablePair] = (
#             self._find_non_comparable_pairs()
#         )

#         # abs(x - y) columns for non-comparable pairs
#         self.abs_exprs: Dict[str, Expr] = self._build_abs_exprs()

#         # min(x,y) / max(x,y) for non-comparable pairs
#         self.min_max_exprs: Dict[str, Expr] = self._build_min_max_exprs()

#         # log(x) for numeric invariants that are safely > 1 on the base universe
#         self.log_exprs: Dict[str, Expr] = self._build_log_exprs(
#             min_value=1.0,
#             base=None,   # natural log; you can make this a parameter later
#         )

#     # ──────────────────────── internal helpers ────────────────────────

#     def _build_invariants(self) -> Dict[str, Expr]:
#         """Numeric DataFrame columns → Expr."""
#         out: Dict[str, Expr] = {}
#         for col in self.df.columns:
#             if np.issubdtype(self.df[col].dtype, np.number):
#                 out[col] = to_expr(col)
#         return out

#     def _build_hypotheses(self, max_boolean_arity: int) -> List[HypothesisInfo]:
#         """
#         Build nontrivial hypotheses inside the base universe, avoiding
#         trivial or redundant conjunctions.

#         Strategy
#         --------
#         1. Ask GraffitiClassRelations for nonredundant conjunctions
#            (arity ≤ max_boolean_arity).
#         2. Restrict each mask to the base universe.
#         3. Discard empty masks and those equal to the base mask.
#         4. For identical masks, keep the *simplest* label
#            (fewest '&' parts).
#         5. Return HypothesisInfo objects, with base first.
#         """
#         # Step 1: get GCR's nonredundant boolean conjunctions
#         self.gcr.enumerate_conjunctions(max_arity=max_boolean_arity)
#         nonred, _, _ = self.gcr.find_redundant_conjunctions()

#         hyps: List[HypothesisInfo] = []

#         # Base hypothesis first
#         base_info = HypothesisInfo(
#             name=self.base_name,
#             pred=self.base_property,
#             mask=self.base_mask.copy(),
#         )
#         hyps.append(base_info)

#         def _complexity(label: str) -> int:
#             parts = [p.strip() for p in label.split("&") if p.strip()]
#             return max(1, len(parts))

#         seen: Dict[bytes, Tuple[int, HypothesisInfo]] = {}

#         # Register the base itself so anything equal to base is ignored
#         base_key = self.base_mask.tobytes()
#         seen[base_key] = (0, base_info)  # complexity 0 = special base

#         # Step 2–4: process all nonredundant conjunctions
#         for name, pred in nonred:
#             raw_mask = np.asarray(self.gcr._mask_cached(pred), dtype=bool)

#             # Restrict to base universe
#             mask = raw_mask & self.base_mask
#             if not mask.any():
#                 continue

#             key = mask.tobytes()
#             comp = _complexity(name)

#             if key in seen:
#                 old_comp, old_info = seen[key]
#                 if comp >= old_comp:
#                     continue
#                 seen[key] = (comp, HypothesisInfo(name=name, pred=pred, mask=mask))
#             else:
#                 seen[key] = (comp, HypothesisInfo(name=name, pred=pred, mask=mask))

#         # Step 5: collect all but the base (which we already added)
#         for _key, (_comp, info) in seen.items():
#             if info.name == self.base_name:
#                 continue
#             hyps.append(info)

#         return hyps

#     def _build_invariant_products(self) -> Dict[str, Expr]:
#         """All pairwise products of numeric invariants."""
#         cols = list(self.invariants.keys())
#         out: Dict[str, Expr] = {}
#         for i in range(len(cols)):
#             for j in range(i, len(cols)):
#                 a, b = cols[i], cols[j]
#                 name = f"{a}·{b}" if a != b else f"{a}²"
#                 out[name] = self.invariants[a] * self.invariants[b]
#         return out

#     def _build_log_exprs(
#         self,
#         *,
#         min_value: float = 1.0,
#         base: float | None = None,
#     ) -> Dict[str, Expr]:
#         """
#         Build log invariants log(x) for numeric columns x that are safely > min_value
#         on the base universe.

#         We check values only on rows where base_mask is True, and require:
#             - finite
#             - strictly greater than `min_value`

#         This ensures log(x) is well-defined without relying on epsilon clamping.
#         """
#         out: Dict[str, Expr] = {}
#         bm = self.base_mask

#         for name, expr in self.invariants.items():
#             col = self.df[name].to_numpy(dtype=float)

#             # Restrict to base universe
#             vals = col[bm]

#             # Require at least one finite value and all > min_value
#             finite = np.isfinite(vals)
#             if finite.sum() == 0:
#                 continue

#             safe_vals = vals[finite]
#             if np.all(safe_vals > min_value):
#                 # Use a simple ASCII-safe label; Expr.pretty() will show ln(x)
#                 if base is None:
#                     log_name = f"ln_{name}"
#                 else:
#                     log_name = f"log{int(base)}_{name}" if float(base).is_integer() else f"log_{base}_{name}"

#                 out[log_name] = log_expr(name, base=base, epsilon=0.0)

#         return out

#     def _find_non_comparable_pairs(self) -> List[NonComparablePair]:
#         """
#         Find pairs (x, y) such that, restricted to the base universe,
#         there exist rows with x < y and rows with x > y.
#         """
#         cols = list(self.invariants.keys())
#         vals = {c: self.df[c].to_numpy(dtype=float) for c in cols}
#         bm = self.base_mask
#         out: List[NonComparablePair] = []

#         for i in range(len(cols)):
#             for j in range(i + 1, len(cols)):
#                 a, b = cols[i], cols[j]
#                 xa = vals[a][bm]
#                 xb = vals[b][bm]

#                 finite_mask = np.isfinite(xa) & np.isfinite(xb)
#                 if finite_mask.sum() == 0:
#                     continue

#                 x = xa[finite_mask]
#                 y = xb[finite_mask]

#                 lt = (x < y).any()
#                 gt = (x > y).any()
#                 if lt and gt:
#                     out.append(
#                         NonComparablePair(
#                             left=self.invariants[a],
#                             right=self.invariants[b],
#                             left_name=a,
#                             right_name=b,
#                         )
#                     )
#         return out

#     def _build_abs_exprs(self) -> Dict[str, Expr]:
#         """abs(x - y) for each non-comparable pair (x, y)."""
#         out: Dict[str, Expr] = {}
#         for pair in self.non_comparable_pairs:
#             name = f"|{pair.left_name} - {pair.right_name}|"
#             out[name] = abs_expr(pair.left - pair.right)
#         return out

#     def _build_min_max_exprs(self) -> Dict[str, Expr]:
#         """min(x,y) and max(x,y) for each non-comparable pair (x, y)."""
#         out: Dict[str, Expr] = {}
#         for pair in self.non_comparable_pairs:
#             name_min = f"min({pair.left_name}, {pair.right_name})"
#             name_max = f"max({pair.left_name}, {pair.right_name})"
#             out[name_min] = min_expr(pair.left, pair.right)
#             out[name_max] = max_expr(pair.left, pair.right)
#         return out

#     # ── helper: build others pool from flags ──

#     def _build_others_pool(
#         self,
#         target: str,
#         *,
#         include_invariant_products: bool,
#         include_abs: bool,
#         include_min_max: bool,
#         include_log: bool,
#     ) -> Dict[str, Expr]:
#         """
#         Construct the dictionary of available "other" expressions, respecting
#         the boolean flags and removing anything that obviously depends on
#         the target column.
#         """
#         target_str = target

#         def _uses_target(expr: Expr) -> bool:
#             # Quick string-based detection; swap in structural helper if desired.
#             return target_str in repr(expr)

#         pool: Dict[str, Expr] = {}

#         # base numeric invariants
#         for name, expr in self.invariants.items():
#             if name == target:
#                 continue
#             pool[name] = expr

#         if include_invariant_products:
#             for name, expr in self.invariant_products.items():
#                 if _uses_target(expr):
#                     continue
#                 pool[name] = expr

#         if include_abs:
#             for name, expr in self.abs_exprs.items():
#                 if _uses_target(expr):
#                     continue
#                 pool[name] = expr

#         if include_min_max:
#             for name, expr in self.min_max_exprs.items():
#                 if _uses_target(expr):
#                     continue
#                 pool[name] = expr

#         if include_log:
#             for name, expr in self.log_exprs.items():
#                 if _uses_target(expr):
#                     continue
#                 pool[name] = expr

#         return pool

#     # ── helper: map complexity ↦ default stages ──

#     @staticmethod
#     def _stages_from_complexity(complexity: int) -> List[Stage]:
#         """
#         Map an integer complexity level to a default list of stages.
#         """
#         stages: List[Stage] = [Stage.CONSTANT]
#         if complexity >= 1:
#             stages += [Stage.RATIO, Stage.LP1]
#         if complexity >= 2:
#             stages += [Stage.LP, Stage.POLY_SINGLE]
#         if complexity >= 3:
#             stages += [Stage.MIXED]
#         return stages

#     @staticmethod
#     def _normalize_stage_list(
#         raw: Sequence[StageLike],
#         *,
#         param_name: str,
#     ) -> List[Stage]:
#         """
#         Normalize a sequence of Stage / string into a deduplicated list of Stage.
#         Raises on unknown strings.
#         """
#         out: List[Stage] = []
#         seen: set[Stage] = set()
#         for item in raw:
#             if isinstance(item, Stage):
#                 stage = item
#             else:
#                 try:
#                     stage = Stage(item)
#                 except ValueError as e:
#                     allowed = ", ".join(s.value for s in Stage)
#                     raise ValueError(
#                         f"Unknown stage in `{param_name}`: {item!r}. "
#                         f"Allowed: {allowed}"
#                     ) from e
#             if stage not in seen:
#                 seen.add(stage)
#                 out.append(stage)
#         return out

#     def _select_relevant_others(
#         self,
#         target: str,
#         others: Dict[str, Expr],
#         *,
#         min_corr: float = 0.1,
#         max_candidates: int | None = 200,
#     ) -> Dict[str, Expr]:
#         """
#         Cheap pre-filter: keep only those 'other' expressions that show
#         at least `min_corr` absolute Pearson correlation with the target
#         over the base universe, up to `max_candidates` features.

#         This is a heuristic to avoid feeding completely irrelevant abs/min/max/log/products
#         into the expensive LP/poly/mixed stages.
#         """
#         y_full = pd.to_numeric(self.df[target], errors="coerce").astype(float)
#         y = y_full[self.base_mask]

#         scores: list[tuple[float, str, Expr]] = []

#         for name, expr in others.items():
#             vals = expr.eval(self.df)[self.base_mask]

#             # Drop NaNs
#             m = y.notna() & vals.notna()
#             if m.sum() < 5:
#                 continue

#             yv = y[m].to_numpy(dtype=float)
#             gv = vals[m].to_numpy(dtype=float)

#             if np.allclose(gv, gv[0]):
#                 # constant or nearly-constant; not helpful
#                 continue

#             try:
#                 c = np.corrcoef(yv, gv)[0, 1]
#             except Exception:
#                 continue
#             if not np.isfinite(c):
#                 continue

#             score = float(abs(c))
#             if score >= min_corr:
#                 scores.append((score, name, expr))

#         # Sort by score descending
#         scores.sort(key=lambda t: t[0], reverse=True)

#         filtered: Dict[str, Expr] = {}
#         for score, name, expr in scores:
#             filtered[name] = expr
#             if max_candidates is not None and len(filtered) >= max_candidates:
#                 break

#         return filtered


#     # ──────────────────────── public API ────────────────────────

#     def conjecture(
#         self,
#         target: str,
#         complexity: int = 2,
#         *,
#         stages: Optional[Sequence[StageLike]] = None,
#         include_invariant_products: bool = False,
#         include_abs: bool = False,
#         include_min_max: bool = False,
#         include_log: bool = False,
#         enable_sophie: bool = True,
#         sophie_stages: Optional[Sequence[StageLike]] = None,
#         quick: bool = False,
#     ) -> Graffiti3Result:
#         """
#         Main conjecturing driver.

#         Parameters
#         ----------
#         target : str
#             Target invariant column.
#         complexity : int, default=2
#             Backwards-compatible knob for enabling groups of stages.
#             Ignored if `stages` is provided.
#         stages : sequence of Stage or str, optional
#             Explicit list of stages to run. If given, overrides `complexity`.
#         include_invariant_products : bool, default=False
#             Whether to allow pairwise products of invariants as candidate features.
#         include_abs : bool, default=False
#             Whether to include abs(x - y) for non-comparable pairs.
#         include_min_max : bool, default=False
#             Whether to include min(x,y) and max(x,y) for non-comparable pairs.
#         include_log : bool, default=False
#             Whether to include log(x) invariants for numeric columns x that are
#             safely > 1 on the base universe.
#         enable_sophie : bool, default=True
#             If False, skip all Sophie mining and return an empty list of
#             Sophie conditions.
#         sophie_stages : sequence of Stage or str, optional
#             If provided, Sophie conditions are mined only from these stages.
#             If None, Sophie is mined from all active stages.
#         quick : bool, default=False
#             If True, perform a cheap pre-filter over the `others` pool
#             via `self._select_relevant_others(...)` before running
#             the expensive LP/poly/mixed stages. This is a heuristic
#             “quick mode” for faster conjecturing with fewer, more
#             target-related derived features.

#         Returns
#         -------
#         Graffiti3Result
#             Aggregated conjectures, Sophie conditions, and per-stage counts.
#         """
#         if target not in self.df.columns:
#             raise KeyError(f"Target column '{target}' not found.")

#         # Decide which stages to run
#         if stages is not None:
#             stages_to_run: List[Stage] = self._normalize_stage_list(
#                 stages, param_name="stages"
#             )
#         else:
#             stages_to_run = self._stages_from_complexity(complexity)

#         stages_to_run_set = set(stages_to_run)

#         # Sophie stage selection
#         if sophie_stages is not None:
#             sophie_stage_list = self._normalize_stage_list(
#                 sophie_stages, param_name="sophie_stages"
#             )
#             sophie_stage_set = set(sophie_stage_list)
#         else:
#             # by default, allow Sophie on all stages that actually run
#             sophie_stage_set = stages_to_run_set

#         target_expr = to_expr(target)

#         # Minimum touches required to enter heuristics/Sophie.
#         min_touches = 3

#         # Build pool of "other" expressions (excluding the target itself)
#         others: Dict[str, Expr] = self._build_others_pool(
#             target=target,
#             include_invariant_products=include_invariant_products,
#             include_abs=include_abs,
#             include_min_max=include_min_max,
#             include_log=include_log,
#         )

#         # Optional quick pre-filter to keep only target-related others
#         if quick and others:
#             # _select_relevant_others should have internal defaults
#             # (e.g., min_corr, max_candidates) so we don't need extra knobs here.
#             others = self._select_relevant_others(
#                 target=target,
#                 others=others,
#             )

#         all_conjectures: List[Conjecture] = []
#         all_sophie: List[SophieCondition] = []
#         stage_info: Dict[str, Any] = {}

#         # Small helper to run Sophie condition mining conditionally
#         def _maybe_sophie(stage: Stage, conjs: List[Conjecture]) -> List[SophieCondition]:
#             if not enable_sophie:
#                 return []
#             if stage not in sophie_stage_set:
#                 return []
#             return sophie_runner(conjectures=conjs, g4=self)

#         # ── Stage CONSTANT ──────────────────────────────────────────────
#         if Stage.CONSTANT in stages_to_run_set:
#             const_conjs = constant_runner(
#                 target_col=target,
#                 target_expr=target_expr,
#                 hypotheses=self.hypotheses,
#                 df=self.df,
#             )
#             const_conjs = _filter_by_touch(self.df, const_conjs, min_touches)

#             const_conjs = heuristic_runner(
#                 const_conjs,
#                 df=self.df,
#                 morgan_filter=self.morgan_filter,
#                 dalmatian_filter=self.dalmatian_filter,
#             )

#             const_sophie = _maybe_sophie(Stage.CONSTANT, const_conjs)

#             all_conjectures.extend(const_conjs)
#             all_conjectures = _dedup_conjectures(all_conjectures)
#             all_sophie.extend(const_sophie)

#             stage_info[Stage.CONSTANT.value] = dict(
#                 conjectures=len(const_conjs),
#                 sophie=len(const_sophie),
#             )

#         # ── Stage RATIO ─────────────────────────────────────────────────
#         if Stage.RATIO in stages_to_run_set:
#             ratio_conjs = ratio_runner(
#                 target_col=target,
#                 target_expr=target_expr,
#                 others=others,
#                 hypotheses=self.hypotheses,
#                 df=self.df,
#             )
#             ratio_conjs = _filter_by_touch(self.df, ratio_conjs, min_touches)
#             ratio_conjs = heuristic_runner(
#                 ratio_conjs,
#                 df=self.df,
#                 morgan_filter=self.morgan_filter,
#                 dalmatian_filter=self.dalmatian_filter,
#             )

#             ratio_sophie = _maybe_sophie(Stage.RATIO, ratio_conjs)
#             all_conjectures.extend(ratio_conjs)
#             all_conjectures = _dedup_conjectures(all_conjectures)
#             all_sophie.extend(ratio_sophie)

#             stage_info[Stage.RATIO.value] = dict(
#                 conjectures=len(ratio_conjs),
#                 sophie=len(ratio_sophie),
#             )

#         # ── Stage LP1 ───────────────────────────────────────────────────
#         if Stage.LP1 in stages_to_run_set:
#             lp1_conjs = lp_single_runner(
#                 target_col=target,
#                 target_expr=target_expr,
#                 others=others,
#                 hypotheses=self.hypotheses,
#                 df=self.df,
#                 direction="both",  # upper + lower
#             )
#             lp1_conjs = _filter_by_touch(self.df, lp1_conjs, min_touches)
#             lp1_conjs = heuristic_runner(
#                 lp1_conjs,
#                 df=self.df,
#                 morgan_filter=self.morgan_filter,
#                 dalmatian_filter=self.dalmatian_filter,
#             )

#             lp1_sophie = _maybe_sophie(Stage.LP1, lp1_conjs)

#             all_conjectures.extend(lp1_conjs)
#             all_conjectures = _dedup_conjectures(all_conjectures)
#             all_sophie.extend(lp1_sophie)

#             stage_info[Stage.LP1.value] = dict(
#                 conjectures=len(lp1_conjs),
#                 sophie=len(lp1_sophie),
#             )

#         # ── Stage LP ────────────────────────────────────────────────────
#         if Stage.LP in stages_to_run_set:
#             lp_conjs = lp_runner(
#                 target_col=target,
#                 target_expr=target_expr,
#                 others=others,
#                 hypotheses=self.hypotheses,
#                 df=self.df,
#                 max_features=2,
#                 max_denom=20,
#                 coef_bound=10.0,
#                 direction="both",
#             )
#             lp_conjs = _filter_by_touch(self.df, lp_conjs, min_touches)

#             lp_conjs = heuristic_runner(
#                 lp_conjs,
#                 df=self.df,
#                 morgan_filter=self.morgan_filter,
#                 dalmatian_filter=self.dalmatian_filter,
#             )

#             lp_sophie = _maybe_sophie(Stage.LP, lp_conjs)

#             all_conjectures.extend(lp_conjs)
#             all_conjectures = _dedup_conjectures(all_conjectures)
#             all_sophie.extend(lp_sophie)

#             stage_info[Stage.LP.value] = dict(
#                 conjectures=len(lp_conjs),
#                 sophie=len(lp_sophie),
#             )

#         # ── Stage LP ────────────────────────────────────────────────────
#         if Stage.LP3 in stages_to_run_set:
#             lp_conjs = lp_runner(
#                 target_col=target,
#                 target_expr=target_expr,
#                 others=others,
#                 hypotheses=self.hypotheses,
#                 df=self.df,
#                 max_features=3,
#                 max_denom=20,
#                 coef_bound=10.0,
#                 direction="both",
#             )
#             lp_conjs = _filter_by_touch(self.df, lp_conjs, min_touches)

#             lp_conjs = heuristic_runner(
#                 lp_conjs,
#                 df=self.df,
#                 morgan_filter=self.morgan_filter,
#                 dalmatian_filter=self.dalmatian_filter,
#             )

#             lp_sophie = _maybe_sophie(Stage.LP, lp_conjs)

#             all_conjectures.extend(lp_conjs)
#             all_conjectures = _dedup_conjectures(all_conjectures)
#             all_sophie.extend(lp_sophie)

#             stage_info[Stage.LP.value] = dict(
#                 conjectures=len(lp_conjs),
#                 sophie=len(lp_sophie),
#             )

#         # ── Stage POLY_SINGLE ──────────────────────────────────────────
#         if Stage.POLY_SINGLE in stages_to_run_set:
#             poly_conjs = poly_single_runner(
#                 target_col=target,
#                 target_expr=target_expr,
#                 others=others,
#                 hypotheses=self.hypotheses,
#                 df=self.df,
#                 min_support=8,
#                 max_denom=20,
#                 max_coef_abs=4.0,
#             )
#             poly_conjs = heuristic_runner(
#                 poly_conjs,
#                 df=self.df,
#                 morgan_filter=self.morgan_filter,
#                 dalmatian_filter=self.dalmatian_filter,
#             )

#             poly_sophie = _maybe_sophie(Stage.POLY_SINGLE, poly_conjs)

#             all_conjectures.extend(poly_conjs)
#             all_conjectures = _dedup_conjectures(all_conjectures)
#             all_sophie.extend(poly_sophie)

#             stage_info[Stage.POLY_SINGLE.value] = dict(
#                 conjectures=len(poly_conjs),
#                 sophie=len(poly_sophie),
#             )

#         # ── Stage MIXED ─────────────────────────────────────────────────
#         if Stage.MIXED in stages_to_run_set:
#             mixed_conjs = mixed_runner(
#                 target_col=target,
#                 target_expr=target_expr,
#                 primaries=self.invariants,
#                 secondaries=self.invariants,
#                 hypotheses=self.hypotheses,
#                 df=self.df,
#                 weight=0.5,
#             )
#             mixed_conjs = heuristic_runner(
#                 mixed_conjs,
#                 df=self.df,
#                 morgan_filter=self.morgan_filter,
#                 dalmatian_filter=self.dalmatian_filter,
#             )

#             mixed_sophie = _maybe_sophie(Stage.MIXED, mixed_conjs)

#             all_conjectures.extend(mixed_conjs)
#             all_conjectures = _dedup_conjectures(all_conjectures)
#             all_sophie.extend(mixed_sophie)

#             stage_info[Stage.MIXED.value] = dict(
#                 conjectures=len(mixed_conjs),
#                 sophie=len(mixed_sophie),
#             )

#         # ── Final pass: annotate & sort conjectures by touches/support ──
#         all_conjectures = _annotate_and_sort_conjectures(self.df, all_conjectures)

#         # ── Sophie: deduplicate and globally rank ───────────────────────
#         if enable_sophie and all_sophie:
#             all_sophie_ranked = rank_sophie_conditions_global(_group_sophie_by_property(all_sophie))
#         else:
#             all_sophie_ranked: List[SophieCondition] = []

#         return Graffiti3Result(
#             target=target,
#             conjectures=all_conjectures,
#             sophie_conditions=all_sophie_ranked,
#             stage_breakdown=stage_info,
#         )

#     def list_conjecture(
#         self,
#         targets: List[str],
#         complexity: int = 2,
#         *,
#         stages: Optional[Sequence[StageLike]] = None,
#         include_invariant_products: bool = False,
#         include_abs: bool = False,
#         include_min_max: bool = False,
#         include_log: bool = False,
#         enable_sophie: bool = True,
#         sophie_stages: Optional[Sequence[StageLike]] = None,
#         quick: bool = False,
#     ) -> Graffiti3Result:
#         """
#         Main conjecturing driver.

#         Parameters
#         ----------
#         target : str
#             Target invariant column.
#         complexity : int, default=2
#             Backwards-compatible knob for enabling groups of stages.
#             Ignored if `stages` is provided.
#         stages : sequence of Stage or str, optional
#             Explicit list of stages to run. If given, overrides `complexity`.
#         include_invariant_products : bool, default=False
#             Whether to allow pairwise products of invariants as candidate features.
#         include_abs : bool, default=False
#             Whether to include abs(x - y) for non-comparable pairs.
#         include_min_max : bool, default=False
#             Whether to include min(x,y) and max(x,y) for non-comparable pairs.
#         include_log : bool, default=False
#             Whether to include log(x) invariants for numeric columns x that are
#             safely > 1 on the base universe.
#         enable_sophie : bool, default=True
#             If False, skip all Sophie mining and return an empty list of
#             Sophie conditions.
#         sophie_stages : sequence of Stage or str, optional
#             If provided, Sophie conditions are mined only from these stages.
#             If None, Sophie is mined from all active stages.
#         quick : bool, default=False
#             If True, perform a cheap pre-filter over the `others` pool
#             via `self._select_relevant_others(...)` before running
#             the expensive LP/poly/mixed stages. This is a heuristic
#             “quick mode” for faster conjecturing with fewer, more
#             target-related derived features.

#         Returns
#         -------
#         Graffiti3Result
#             Aggregated conjectures, Sophie conditions, and per-stage counts.
#         """
#         list_all_conjectures = []
#         list_all_sophie = []
#         for target in targets:
#             if target not in self.df.columns:
#                 raise KeyError(f"Target column '{target}' not found.")

#             # Decide which stages to run
#             if stages is not None:
#                 stages_to_run: List[Stage] = self._normalize_stage_list(
#                     stages, param_name="stages"
#                 )
#             else:
#                 stages_to_run = self._stages_from_complexity(complexity)

#             stages_to_run_set = set(stages_to_run)

#             # Sophie stage selection
#             if sophie_stages is not None:
#                 sophie_stage_list = self._normalize_stage_list(
#                     sophie_stages, param_name="sophie_stages"
#                 )
#                 sophie_stage_set = set(sophie_stage_list)
#             else:
#                 # by default, allow Sophie on all stages that actually run
#                 sophie_stage_set = stages_to_run_set

#             target_expr = to_expr(target)

#             # Minimum touches required to enter heuristics/Sophie.
#             min_touches = 3

#             # Build pool of "other" expressions (excluding the target itself)
#             others: Dict[str, Expr] = self._build_others_pool(
#                 target=target,
#                 include_invariant_products=include_invariant_products,
#                 include_abs=include_abs,
#                 include_min_max=include_min_max,
#                 include_log=include_log,
#             )

#             # Optional quick pre-filter to keep only target-related others
#             if quick and others:
#                 # _select_relevant_others should have internal defaults
#                 # (e.g., min_corr, max_candidates) so we don't need extra knobs here.
#                 others = self._select_relevant_others(
#                     target=target,
#                     others=others,
#                 )

#             all_conjectures: List[Conjecture] = []
#             all_sophie: List[SophieCondition] = []
#             stage_info: Dict[str, Any] = {}

#             # Small helper to run Sophie condition mining conditionally
#             def _maybe_sophie(stage: Stage, conjs: List[Conjecture]) -> List[SophieCondition]:
#                 if not enable_sophie:
#                     return []
#                 if stage not in sophie_stage_set:
#                     return []
#                 return sophie_runner(conjectures=conjs, g4=self)

#             # ── Stage CONSTANT ──────────────────────────────────────────────
#             if Stage.CONSTANT in stages_to_run_set:
#                 const_conjs = constant_runner(
#                     target_col=target,
#                     target_expr=target_expr,
#                     hypotheses=self.hypotheses,
#                     df=self.df,
#                 )
#                 const_conjs = _filter_by_touch(self.df, const_conjs, min_touches)

#                 const_conjs = heuristic_runner(
#                     const_conjs,
#                     df=self.df,
#                     morgan_filter=self.morgan_filter,
#                     dalmatian_filter=self.dalmatian_filter,
#                 )

#                 const_sophie = _maybe_sophie(Stage.CONSTANT, const_conjs)

#                 all_conjectures.extend(const_conjs)
#                 all_conjectures = _dedup_conjectures(all_conjectures)
#                 all_sophie.extend(const_sophie)

#                 stage_info[Stage.CONSTANT.value] = dict(
#                     conjectures=len(const_conjs),
#                     sophie=len(const_sophie),
#                 )

#             # ── Stage RATIO ─────────────────────────────────────────────────
#             if Stage.RATIO in stages_to_run_set:
#                 ratio_conjs = ratio_runner(
#                     target_col=target,
#                     target_expr=target_expr,
#                     others=others,
#                     hypotheses=self.hypotheses,
#                     df=self.df,
#                 )
#                 ratio_conjs = _filter_by_touch(self.df, ratio_conjs, min_touches)
#                 ratio_conjs = heuristic_runner(
#                     ratio_conjs,
#                     df=self.df,
#                     morgan_filter=self.morgan_filter,
#                     dalmatian_filter=self.dalmatian_filter,
#                 )

#                 ratio_sophie = _maybe_sophie(Stage.RATIO, ratio_conjs)
#                 all_conjectures.extend(ratio_conjs)
#                 all_conjectures = _dedup_conjectures(all_conjectures)
#                 all_sophie.extend(ratio_sophie)

#                 stage_info[Stage.RATIO.value] = dict(
#                     conjectures=len(ratio_conjs),
#                     sophie=len(ratio_sophie),
#                 )

#             # ── Stage LP1 ───────────────────────────────────────────────────
#             if Stage.LP1 in stages_to_run_set:
#                 lp1_conjs = lp_single_runner(
#                     target_col=target,
#                     target_expr=target_expr,
#                     others=others,
#                     hypotheses=self.hypotheses,
#                     df=self.df,
#                     direction="both",  # upper + lower
#                 )
#                 lp1_conjs = _filter_by_touch(self.df, lp1_conjs, min_touches)
#                 lp1_conjs = heuristic_runner(
#                     lp1_conjs,
#                     df=self.df,
#                     morgan_filter=self.morgan_filter,
#                     dalmatian_filter=self.dalmatian_filter,
#                 )

#                 lp1_sophie = _maybe_sophie(Stage.LP1, lp1_conjs)

#                 all_conjectures.extend(lp1_conjs)
#                 all_conjectures = _dedup_conjectures(all_conjectures)
#                 all_sophie.extend(lp1_sophie)

#                 stage_info[Stage.LP1.value] = dict(
#                     conjectures=len(lp1_conjs),
#                     sophie=len(lp1_sophie),
#                 )

#             # ── Stage LP ────────────────────────────────────────────────────
#             if Stage.LP in stages_to_run_set:
#                 lp_conjs = lp_runner(
#                     target_col=target,
#                     target_expr=target_expr,
#                     others=others,
#                     hypotheses=self.hypotheses,
#                     df=self.df,
#                     max_features=2,
#                     max_denom=20,
#                     coef_bound=10.0,
#                     direction="both",
#                 )
#                 lp_conjs = _filter_by_touch(self.df, lp_conjs, min_touches)

#                 lp_conjs = heuristic_runner(
#                     lp_conjs,
#                     df=self.df,
#                     morgan_filter=self.morgan_filter,
#                     dalmatian_filter=self.dalmatian_filter,
#                 )

#                 lp_sophie = _maybe_sophie(Stage.LP, lp_conjs)

#                 all_conjectures.extend(lp_conjs)
#                 all_conjectures = _dedup_conjectures(all_conjectures)
#                 all_sophie.extend(lp_sophie)

#                 stage_info[Stage.LP.value] = dict(
#                     conjectures=len(lp_conjs),
#                     sophie=len(lp_sophie),
#                 )

#             if Stage.LP3 in stages_to_run_set:
#                 lp_conjs = lp_runner(
#                     target_col=target,
#                     target_expr=target_expr,
#                     others=others,
#                     hypotheses=self.hypotheses,
#                     df=self.df,
#                     max_features=3,
#                     max_denom=20,
#                     coef_bound=10.0,
#                     direction="both",
#                 )
#                 lp_conjs = _filter_by_touch(self.df, lp_conjs, min_touches)

#                 lp_conjs = heuristic_runner(
#                     lp_conjs,
#                     df=self.df,
#                     morgan_filter=self.morgan_filter,
#                     dalmatian_filter=self.dalmatian_filter,
#                 )

#                 lp_sophie = _maybe_sophie(Stage.LP, lp_conjs)

#                 all_conjectures.extend(lp_conjs)
#                 all_conjectures = _dedup_conjectures(all_conjectures)
#                 all_sophie.extend(lp_sophie)

#                 stage_info[Stage.LP.value] = dict(
#                     conjectures=len(lp_conjs),
#                     sophie=len(lp_sophie),
#                 )

#             # ── Stage POLY_SINGLE ──────────────────────────────────────────
#             if Stage.POLY_SINGLE in stages_to_run_set:
#                 poly_conjs = poly_single_runner(
#                     target_col=target,
#                     target_expr=target_expr,
#                     others=others,
#                     hypotheses=self.hypotheses,
#                     df=self.df,
#                     min_support=8,
#                     max_denom=20,
#                     max_coef_abs=4.0,
#                 )
#                 poly_conjs = heuristic_runner(
#                     poly_conjs,
#                     df=self.df,
#                     morgan_filter=self.morgan_filter,
#                     dalmatian_filter=self.dalmatian_filter,
#                 )

#                 poly_sophie = _maybe_sophie(Stage.POLY_SINGLE, poly_conjs)

#                 all_conjectures.extend(poly_conjs)
#                 all_conjectures = _dedup_conjectures(all_conjectures)
#                 all_sophie.extend(poly_sophie)

#                 stage_info[Stage.POLY_SINGLE.value] = dict(
#                     conjectures=len(poly_conjs),
#                     sophie=len(poly_sophie),
#                 )

#             # ── Stage MIXED ─────────────────────────────────────────────────
#             if Stage.MIXED in stages_to_run_set:
#                 mixed_conjs = mixed_runner(
#                     target_col=target,
#                     target_expr=target_expr,
#                     primaries=self.invariants,
#                     secondaries=self.invariants,
#                     hypotheses=self.hypotheses,
#                     df=self.df,
#                     weight=0.5,
#                 )
#                 mixed_conjs = heuristic_runner(
#                     mixed_conjs,
#                     df=self.df,
#                     morgan_filter=self.morgan_filter,
#                     dalmatian_filter=self.dalmatian_filter,
#                 )

#                 mixed_sophie = _maybe_sophie(Stage.MIXED, mixed_conjs)

#                 all_conjectures.extend(mixed_conjs)
#                 all_conjectures = _dedup_conjectures(all_conjectures)
#                 all_sophie.extend(mixed_sophie)

#                 stage_info[Stage.MIXED.value] = dict(
#                     conjectures=len(mixed_conjs),
#                     sophie=len(mixed_sophie),
#                 )

#             # ── Final pass: annotate & sort conjectures by touches/support ──
#             all_conjectures = _annotate_and_sort_conjectures(self.df, all_conjectures)
#             list_all_conjectures.extend(all_conjectures)
#             # ── Sophie: deduplicate and globally rank ───────────────────────
#             if enable_sophie and all_sophie:
#                 all_sophie_ranked = rank_sophie_conditions_global(_group_sophie_by_property(all_sophie))
#             else:
#                 all_sophie_ranked: List[SophieCondition] = []
#             list_all_sophie.extend(all_sophie_ranked)

#         list_all_conjectures = heuristic_runner(
#                     list_all_conjectures,
#                     df=self.df,
#                     morgan_filter=self.morgan_filter,
#                     dalmatian_filter=self.dalmatian_filter,
#                 )
#         return Graffiti3Result(
#             target=target,
#             conjectures=list_all_conjectures,
#             sophie_conditions=list_all_sophie,
#             stage_breakdown={'results': 'test'},
#         )

# def sophie_runner(
#     *,
#     conjectures: Sequence[Conjecture],
#     g4: Graffiti3,
# ) -> List[SophieCondition]:
#     """
#     Thin wrapper around your Sophie module: take a list of inequality
#     conjectures and return a *flat list* of SophieCondition objects.
#     """
#     if not conjectures:
#         return []

#     df_num = g4.df
#     bool_cols = list(g4.gcr.boolean_cols)
#     bool_df = df_num[bool_cols].copy() if bool_cols else pd.DataFrame(index=df_num.index)

#     base_mask = g4.base_mask
#     base_name = g4.base_name

#     sophie_by_prop = discover_sophie_from_inequalities(
#         df_num=df_num,
#         bool_df=bool_df,
#         base_mask=base_mask,
#         base_name=base_name,
#         inequality_conjectures=conjectures,
#         **g4.sophie_cfg,
#     )

#     flat: List[SophieCondition] = []
#     for _, conds in sophie_by_prop.items():
#         flat.extend(conds)
#     return flat


# def _group_sophie_by_property(
#     conds: Sequence[SophieCondition],
# ) -> Dict[str, List[SophieCondition]]:
#     """
#     Helper: take a flat list of SophieCondition and group them into the
#     dictionary format expected by rank_sophie_conditions_global.
#     """
#     out: Dict[str, List[SophieCondition]] = {}
#     for sc in conds:
#         out.setdefault(sc.property_name, []).append(sc)
#     return out


# # ───────────────────────── top-level pretty printer ───────────────────────── #

# def print_g3_result(
#     result: Graffiti3Result,
#     *,
#     k_conjectures: int = 20,
#     k_sophie: int = 20,
# ) -> None:
#     """
#     Convenience printer for Graffiti4Result: stage breakdown, top conjectures
#     (sorted by touches), and top Sophie conditions.
#     """
#     print("Stage breakdown:", result.stage_breakdown)
#     print(f"Total conjectures: {len(result.conjectures)}")
#     print(f"Total Sophie conditions: {len(result.sophie_conditions)}\n")

#     print("=== Top conjectures (by touch_count, then support) ===\n")
#     for i, c in enumerate(result.conjectures[:k_conjectures], 1):
#         touches = getattr(c, "touch_count", getattr(c, "touch", "?"))
#         support = getattr(c, "support_n", getattr(c, "support", "?"))
#         print(f"Conjecture {i}. {c.pretty()}   [touches={touches}, support={support}]\n")

#     print_sophie_conditions(result.sophie_conditions, top_n=k_sophie)

# src/txgraffiti/graffiti3/graffiti3.py

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .class_relations import GraffitiClassRelations
from .exprs import (
    Expr,
    to_expr,
    abs_ as abs_expr,
    min_ as min_expr,
    max_ as max_expr,
    log as log_expr,
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


# ───────────────────────── enums ───────────────────────── #

class Stage(str, Enum):
    """Named stages in the Graffiti3 conjecturing pipeline."""
    CONSTANT = "constant"
    RATIO = "ratio"
    LP1 = "lp1"              # 1-feature LP (t ≶ m x + b)
    LP = "lp"                # 2-feature LP
    LP3 = "lp3"              # 3-feature LP
    POLY_SINGLE = "poly_single"
    MIXED = "mixed"


StageLike = Union[Stage, str]


class Mode(str, Enum):
    """
    High level search presets.

    - FAST:    quick, coarse pass with simple features.
    - STANDARD: balanced default (good for most use).
    - DEEP:    more stages & derived features; heavier but richer.
    """
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


ModeLike = Union[Mode, str]


# ───────────────────────── main workspace ───────────────────────── #

class Graffiti3:
    """
    Refactored, modular conjecturing workspace.

    Typical usage
    -------------
        g3 = Graffiti3(df)

        # simplest: one target, standard search
        result = g3.conjecture("independence_number")

        # fast / deep presets
        result_fast  = g3.conjecture("independence_number", mode=Mode.FAST)
        result_deep  = g3.conjecture("independence_number", mode=Mode.DEEP)

        # batch over several targets
        results = g3.list_conjecture(["independence_number", "domination_number"])

    What Graffiti3 precomputes
    --------------------------
    - base_property, base_name, base_mask
    - invariants: Exprs for numeric columns
    - invariant_products: pairwise products of invariants
    - log_exprs: log(x) for safely >1 numeric invariants
    - properties: atomic boolean columns
    - hypotheses: nonredundant conjunctions of properties with base
    - non_comparable_invariants: (x, y) where both x<y and x>y occur
    - abs_exprs, min_max_exprs on non-comparable pairs
    """

    # ──────────────────────── construction ────────────────────────

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_boolean_arity: int = 2,
        morgan_filter=None,
        dalmatian_filter=None,
        sophie_cfg: Optional[Dict[str, Any]] = None,
        min_touches: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        df : DataFrame
            Invariant / property table (rows = objects, columns = invariants + booleans).
        max_boolean_arity : int, default=2
            Max size of conjunctions used to form hypotheses.
        morgan_filter, dalmatian_filter
            Optional callable heuristics used by `heuristic_runner`.
        sophie_cfg : dict, optional
            Optional overrides for Sophie mining config (eq_tol, min_target_support, etc.).
        min_touches : int, default=3
            Minimum #equalities required for a conjecture to enter heuristics/Sophie.
        """
        self.df = df.copy()
        self.gcr = GraffitiClassRelations(self.df)

        # heuristic hooks
        self.morgan_filter = morgan_filter
        self.dalmatian_filter = dalmatian_filter

        # Sophie configuration: sensible defaults + user overrides
        cfg = self._default_sophie_cfg()
        if sophie_cfg:
            cfg.update(sophie_cfg)
        self.sophie_cfg: Dict[str, Any] = cfg

        # low-touch pruning threshold
        self.min_touches = int(min_touches)

        # base universe
        self.base_property = self.gcr.base_hypothesis
        self.base_name = self.gcr.base_hypothesis_name
        self.base_mask = np.asarray(
            self.gcr._mask_cached(self.gcr.base_hypothesis), dtype=bool
        )

        # numeric invariants as Exprs
        self.invariants: Dict[str, Expr] = self._build_invariants()

        # boolean properties
        self.properties: List[str] = list(self.gcr.boolean_cols)

        # hypotheses (arity ≤ max_boolean_arity), base first
        self.hypotheses: List[HypothesisInfo] = self._build_hypotheses(
            max_boolean_arity=max_boolean_arity
        )

        # derived numeric Expr pools
        self.invariant_products: Dict[str, Expr] = self._build_invariant_products()
        self.non_comparable_pairs: List[NonComparablePair] = (
            self._find_non_comparable_pairs()
        )
        self.abs_exprs: Dict[str, Expr] = self._build_abs_exprs()
        self.min_max_exprs: Dict[str, Expr] = self._build_min_max_exprs()
        self.log_exprs: Dict[str, Expr] = self._build_log_exprs(
            min_value=1.0,
            base=None,  # natural log by default
        )

    # ──────────────────────── defaults / config ────────────────────────

    @staticmethod
    def _default_sophie_cfg() -> Dict[str, Any]:
        """Baseline settings for Sophie-style condition discovery."""
        return dict(
            eq_tol=1e-4,
            min_target_support=5,
            min_h_support=3,
            max_violations=0,
            min_new_coverage=1,
        )

    # ──────────────────────── internal builders ────────────────────────

    def _build_invariants(self) -> Dict[str, Expr]:
        """Numeric DataFrame columns → Expr."""
        out: Dict[str, Expr] = {}
        for col in self.df.columns:
            if np.issubdtype(self.df[col].dtype, np.number):
                out[col] = to_expr(col)
        return out

    def _build_hypotheses(self, max_boolean_arity: int) -> List[HypothesisInfo]:
        """
        Build nontrivial hypotheses inside the base universe, avoiding
        trivial or redundant conjunctions.

        1. Ask GraffitiClassRelations for nonredundant conjunctions (arity ≤ max_boolean_arity).
        2. Restrict each mask to the base universe.
        3. Discard empty masks and those equal to the base mask.
        4. For identical masks, keep the simplest label (fewest '&' parts).
        5. Return HypothesisInfo objects, with base first.
        """
        self.gcr.enumerate_conjunctions(max_arity=max_boolean_arity)
        nonred, _, _ = self.gcr.find_redundant_conjunctions()

        hyps: List[HypothesisInfo] = []

        base_info = HypothesisInfo(
            name=self.base_name,
            pred=self.base_property,
            mask=self.base_mask.copy(),
        )
        hyps.append(base_info)

        def _complexity(label: str) -> int:
            parts = [p.strip() for p in label.split("&") if p.strip()]
            return max(1, len(parts))

        seen: Dict[bytes, Tuple[int, HypothesisInfo]] = {}
        base_key = self.base_mask.tobytes()
        seen[base_key] = (0, base_info)

        for name, pred in nonred:
            raw_mask = np.asarray(self.gcr._mask_cached(pred), dtype=bool)
            mask = raw_mask & self.base_mask
            if not mask.any():
                continue

            key = mask.tobytes()
            comp = _complexity(name)

            if key in seen:
                old_comp, _old_info = seen[key]
                if comp >= old_comp:
                    continue
            seen[key] = (comp, HypothesisInfo(name=name, pred=pred, mask=mask))

        for _key, (_comp, info) in seen.items():
            if info.name == self.base_name:
                continue
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

    def _build_log_exprs(
        self,
        *,
        min_value: float = 1.0,
        base: float | None = None,
    ) -> Dict[str, Expr]:
        """
        Build log invariants log(x) for numeric columns x that are safely > min_value
        on the base universe.
        """
        out: Dict[str, Expr] = {}
        bm = self.base_mask

        for name, _expr in self.invariants.items():
            col = self.df[name].to_numpy(dtype=float)
            vals = col[bm]

            finite = np.isfinite(vals)
            if finite.sum() == 0:
                continue

            safe_vals = vals[finite]
            if np.all(safe_vals > min_value):
                if base is None:
                    log_name = f"ln_{name}"
                else:
                    b = float(base)
                    if b.is_integer():
                        log_name = f"log{int(b)}_{name}"
                    else:
                        log_name = f"log_{b}_{name}"

                out[log_name] = log_expr(name, base=base, epsilon=0.0)

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
        include_log: bool,
    ) -> Dict[str, Expr]:
        """
        Construct the dictionary of available "other" expressions, respecting
        the boolean flags and removing anything that obviously depends on
        the target column.
        """
        target_str = target

        def _uses_target(expr: Expr) -> bool:
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

        if include_log:
            for name, expr in self.log_exprs.items():
                if _uses_target(expr):
                    continue
                pool[name] = expr

        return pool

    # ── helper: map complexity ↦ default stages ──

    @staticmethod
    def _stages_from_complexity(complexity: int) -> List[Stage]:
        """
        Map an integer complexity level to a default list of stages.

        0 → [CONSTANT]
        1 → + [RATIO, LP1]
        2 → + [LP, POLY_SINGLE]
        3 → + [LP3]
        ≥4 → + [MIXED]
        """
        stages: List[Stage] = [Stage.CONSTANT]
        if complexity >= 1:
            stages += [Stage.RATIO, Stage.LP1]
        if complexity >= 2:
            stages += [Stage.LP, Stage.POLY_SINGLE]
        if complexity >= 3:
            stages += [Stage.LP3]
        if complexity >= 4:
            stages += [Stage.MIXED]
        return stages

    @staticmethod
    def _normalize_stage_list(
        raw: Sequence[StageLike],
        *,
        param_name: str,
    ) -> List[Stage]:
        """
        Normalize a sequence of Stage / string into a deduplicated list of Stage.
        Raises on unknown strings.
        """
        out: List[Stage] = []
        seen: set[Stage] = set()
        for item in raw:
            if isinstance(item, Stage):
                stage = item
            else:
                try:
                    stage = Stage(item)
                except ValueError as e:
                    allowed = ", ".join(s.value for s in Stage)
                    raise ValueError(
                        f"Unknown stage in `{param_name}`: {item!r}. "
                        f"Allowed: {allowed}"
                    ) from e
            if stage not in seen:
                seen.add(stage)
                out.append(stage)
        return out

    @staticmethod
    def _normalize_mode(mode: ModeLike) -> Mode:
        """Normalize Mode / string to a Mode enum."""
        if isinstance(mode, Mode):
            return mode
        try:
            return Mode(mode)
        except ValueError as e:
            allowed = ", ".join(m.value for m in Mode)
            raise ValueError(
                f"Unknown mode {mode!r}. Allowed modes: {allowed}"
            ) from e

    @staticmethod
    def _presets_for_mode(mode: Mode) -> Dict[str, Any]:
        """
        Map Mode → presets for complexity / feature flags / quick.
        """
        if mode is Mode.FAST:
            return dict(
                complexity=1,
                include_invariant_products=False,
                include_abs=False,
                include_min_max=False,
                include_log=False,
                quick=True,
            )
        if mode is Mode.STANDARD:
            return dict(
                complexity=2,
                include_invariant_products=False,
                include_abs=True,
                include_min_max=True,
                include_log=True,
                quick=False,
            )
        if mode is Mode.DEEP:
            return dict(
                complexity=3,
                include_invariant_products=True,
                include_abs=True,
                include_min_max=True,
                include_log=True,
                quick=False,
            )
        raise AssertionError("unreachable")

    def _select_relevant_others(
        self,
        target: str,
        others: Dict[str, Expr],
        *,
        min_corr: float = 0.1,
        max_candidates: int | None = 200,
    ) -> Dict[str, Expr]:
        """
        Cheap pre-filter: keep only those 'other' expressions that show
        at least `min_corr` absolute Pearson correlation with the target
        over the base universe, up to `max_candidates` features.
        """
        y_full = pd.to_numeric(self.df[target], errors="coerce").astype(float)
        y = y_full[self.base_mask]

        scores: List[Tuple[float, str, Expr]] = []

        for name, expr in others.items():
            vals = expr.eval(self.df)[self.base_mask]

            m = y.notna() & vals.notna()
            if m.sum() < 5:
                continue

            yv = y[m].to_numpy(dtype=float)
            gv = vals[m].to_numpy(dtype=float)

            if np.allclose(gv, gv[0]):
                continue

            try:
                c = np.corrcoef(yv, gv)[0, 1]
            except Exception:
                continue
            if not np.isfinite(c):
                continue

            score = float(abs(c))
            if score >= min_corr:
                scores.append((score, name, expr))

        scores.sort(key=lambda t: t[0], reverse=True)

        filtered: Dict[str, Expr] = {}
        for _score, name, expr in scores:
            filtered[name] = expr
            if max_candidates is not None and len(filtered) >= max_candidates:
                break

        return filtered

    # ──────────────────────── public API ────────────────────────

    def conjecture(
        self,
        target: str,
        complexity: Optional[int] = None,
        *,
        mode: Optional[ModeLike] = None,
        stages: Optional[Sequence[StageLike]] = None,
        include_invariant_products: Optional[bool] = None,
        include_abs: Optional[bool] = None,
        include_min_max: Optional[bool] = None,
        include_log: Optional[bool] = None,
        enable_sophie: bool = True,
        sophie_stages: Optional[Sequence[StageLike]] = None,
        quick: Optional[bool] = None,
    ) -> Graffiti3Result:
        """
        Main conjecturing driver for a single target.

        Parameters
        ----------
        target : str
            Target invariant column.
        complexity : int, optional
            Coarse knob for enabling groups of stages; ignored if `stages` is provided.
            If None, defaults to 2 (constant + ratio + LP1 + LP + poly_single)
            unless overridden by `mode`.
        mode : Mode or {"fast","standard","deep"}, optional
            High-level preset. If provided, it sets *defaults* for complexity and
            feature flags; you can override them explicitly by passing the
            corresponding arguments.
        stages : sequence of Stage or str, optional
            Explicit list of stages to run. If given, overrides `complexity`.
        include_invariant_products / include_abs / include_min_max / include_log : bool, optional
            Feature toggles. If None, they are inferred from `mode` (if given) or
            fall back to sensible defaults.
        enable_sophie : bool, default=True
            If False, skip all Sophie mining and return an empty list.
        sophie_stages : sequence of Stage or str, optional
            If provided, Sophie is mined only from these stages (subset of `stages`).
            If None, Sophie is mined from all active stages.
        quick : bool, optional
            If True, run a cheap correlation-based prefilter over the `others` pool
            before expensive LP/poly/mixed stages. If None, inferred from `mode`.

        Returns
        -------
        Graffiti3Result
        """
        if target not in self.df.columns:
            raise KeyError(f"Target column '{target}' not found.")

        # ── 1. Apply mode presets to fill in missing knobs ──────────────
        if mode is not None:
            mode_enum = self._normalize_mode(mode)
            presets = self._presets_for_mode(mode_enum)

            if complexity is None:
                complexity = presets["complexity"]
            if include_invariant_products is None:
                include_invariant_products = presets["include_invariant_products"]
            if include_abs is None:
                include_abs = presets["include_abs"]
            if include_min_max is None:
                include_min_max = presets["include_min_max"]
            if include_log is None:
                include_log = presets["include_log"]
            if quick is None:
                quick = presets["quick"]

        # Fallback defaults when no mode / partial overrides
        if complexity is None:
            complexity = 2
        if include_invariant_products is None:
            include_invariant_products = False
        if include_abs is None:
            include_abs = True
        if include_min_max is None:
            include_min_max = True
        if include_log is None:
            include_log = True
        if quick is None:
            quick = False

        # ── 2. Decide stages ────────────────────────────────────────────
        if stages is not None:
            stages_to_run: List[Stage] = self._normalize_stage_list(
                stages, param_name="stages"
            )
        else:
            stages_to_run = self._stages_from_complexity(int(complexity))

        stages_to_run_set = set(stages_to_run)

        # Sophie stage selection
        if sophie_stages is not None:
            sophie_stage_list = self._normalize_stage_list(
                sophie_stages, param_name="sophie_stages"
            )
            sophie_stage_set = set(sophie_stage_list)
        else:
            sophie_stage_set = stages_to_run_set

        target_expr = to_expr(target)
        min_touches = self.min_touches

        # ── 3. Build others pool (and maybe prefilter) ─────────────────
        others: Dict[str, Expr] = self._build_others_pool(
            target=target,
            include_invariant_products=bool(include_invariant_products),
            include_abs=bool(include_abs),
            include_min_max=bool(include_min_max),
            include_log=bool(include_log),
        )

        if quick and others:
            others = self._select_relevant_others(target=target, others=others)

        all_conjectures: List[Conjecture] = []
        all_sophie: List[SophieCondition] = []
        stage_info: Dict[str, Any] = {}

        def _maybe_sophie(stage: Stage, conjs: List[Conjecture]) -> List[SophieCondition]:
            if not enable_sophie or not conjs:
                return []
            if stage not in sophie_stage_set:
                return []
            return sophie_runner(conjectures=conjs, g4=self)

        # ── Stage: CONSTANT ────────────────────────────────────────────
        if Stage.CONSTANT in stages_to_run_set:
            const_conjs = constant_runner(
                target_col=target,
                target_expr=target_expr,
                hypotheses=self.hypotheses,
                df=self.df,
            )
            const_conjs = _filter_by_touch(self.df, const_conjs, min_touches)
            const_conjs = heuristic_runner(
                const_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )
            const_sophie = _maybe_sophie(Stage.CONSTANT, const_conjs)

            all_conjectures.extend(const_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(const_sophie)

            stage_info[Stage.CONSTANT.value] = dict(
                conjectures=len(const_conjs),
                sophie=len(const_sophie),
            )

        # ── Stage: RATIO ───────────────────────────────────────────────
        if Stage.RATIO in stages_to_run_set:
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
            ratio_sophie = _maybe_sophie(Stage.RATIO, ratio_conjs)

            all_conjectures.extend(ratio_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(ratio_sophie)

            stage_info[Stage.RATIO.value] = dict(
                conjectures=len(ratio_conjs),
                sophie=len(ratio_sophie),
            )

        # ── Stage: LP1 ────────────────────────────────────────────────
        if Stage.LP1 in stages_to_run_set:
            lp1_conjs = lp_single_runner(
                target_col=target,
                target_expr=target_expr,
                others=others,
                hypotheses=self.hypotheses,
                df=self.df,
                direction="both",
            )
            lp1_conjs = _filter_by_touch(self.df, lp1_conjs, min_touches)
            lp1_conjs = heuristic_runner(
                lp1_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )
            lp1_sophie = _maybe_sophie(Stage.LP1, lp1_conjs)

            all_conjectures.extend(lp1_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(lp1_sophie)

            stage_info[Stage.LP1.value] = dict(
                conjectures=len(lp1_conjs),
                sophie=len(lp1_sophie),
            )

        # ── Stage: LP (2 features) ────────────────────────────────────
        if Stage.LP in stages_to_run_set:
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
            lp_sophie = _maybe_sophie(Stage.LP, lp_conjs)

            all_conjectures.extend(lp_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(lp_sophie)

            stage_info[Stage.LP.value] = dict(
                conjectures=len(lp_conjs),
                sophie=len(lp_sophie),
            )

        # ── Stage: LP3 (3 features) ───────────────────────────────────
        if Stage.LP3 in stages_to_run_set:
            lp3_conjs = lp_runner(
                target_col=target,
                target_expr=target_expr,
                others=others,
                hypotheses=self.hypotheses,
                df=self.df,
                max_features=3,
                max_denom=20,
                coef_bound=10.0,
                direction="both",
            )
            lp3_conjs = _filter_by_touch(self.df, lp3_conjs, min_touches)
            lp3_conjs = heuristic_runner(
                lp3_conjs,
                df=self.df,
                morgan_filter=self.morgan_filter,
                dalmatian_filter=self.dalmatian_filter,
            )
            lp3_sophie = _maybe_sophie(Stage.LP3, lp3_conjs)

            all_conjectures.extend(lp3_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(lp3_sophie)

            stage_info[Stage.LP3.value] = dict(
                conjectures=len(lp3_conjs),
                sophie=len(lp3_sophie),
            )

        # ── Stage: POLY_SINGLE ────────────────────────────────────────
        if Stage.POLY_SINGLE in stages_to_run_set:
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
            poly_sophie = _maybe_sophie(Stage.POLY_SINGLE, poly_conjs)

            all_conjectures.extend(poly_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(poly_sophie)

            stage_info[Stage.POLY_SINGLE.value] = dict(
                conjectures=len(poly_conjs),
                sophie=len(poly_sophie),
            )

        # ── Stage: MIXED ──────────────────────────────────────────────
        if Stage.MIXED in stages_to_run_set:
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
            mixed_sophie = _maybe_sophie(Stage.MIXED, mixed_conjs)

            all_conjectures.extend(mixed_conjs)
            all_conjectures = _dedup_conjectures(all_conjectures)
            all_sophie.extend(mixed_sophie)

            stage_info[Stage.MIXED.value] = dict(
                conjectures=len(mixed_conjs),
                sophie=len(mixed_sophie),
            )

        # ── Final pass: annotate & sort ───────────────────────────────
        all_conjectures = _annotate_and_sort_conjectures(self.df, all_conjectures)

        # ── Sophie: deduplicate and globally rank ─────────────────────
        if enable_sophie and all_sophie:
            all_sophie_ranked = rank_sophie_conditions_global(
                _group_sophie_by_property(all_sophie)
            )
        else:
            all_sophie_ranked = []

        return Graffiti3Result(
            target=target,
            conjectures=all_conjectures,
            sophie_conditions=all_sophie_ranked,
            stage_breakdown=stage_info,
        )

    def list_conjecture(
        self,
        targets: Sequence[str],
        *,
        complexity: Optional[int] = None,
        mode: Optional[ModeLike] = None,
        stages: Optional[Sequence[StageLike]] = None,
        include_invariant_products: Optional[bool] = None,
        include_abs: Optional[bool] = None,
        include_min_max: Optional[bool] = None,
        include_log: Optional[bool] = None,
        enable_sophie: bool = True,
        sophie_stages: Optional[Sequence[StageLike]] = None,
        quick: Optional[bool] = None,
    ) -> Graffiti3Result:
        """
        Batch version of `conjecture` over multiple targets.

        Returns a single Graffiti3Result aggregating all conjectures and Sophie
        conditions, with `stage_breakdown` containing a per-target summary.
        """
        all_conjs: List[Conjecture] = []
        all_sophie: List[SophieCondition] = []
        breakdown: Dict[str, Any] = {}

        for t in targets:
            res = self.conjecture(
                t,
                complexity=complexity,
                mode=mode,
                stages=stages,
                include_invariant_products=include_invariant_products,
                include_abs=include_abs,
                include_min_max=include_min_max,
                include_log=include_log,
                enable_sophie=enable_sophie,
                sophie_stages=sophie_stages,
                quick=quick,
            )
            all_conjs.extend(res.conjectures)
            all_sophie.extend(res.sophie_conditions)
            breakdown[t] = res.stage_breakdown

        # Re-run annotation / sorting globally
        all_conjs = _annotate_and_sort_conjectures(self.df, all_conjs)

        if enable_sophie and all_sophie:
            all_sophie_ranked = rank_sophie_conditions_global(
                _group_sophie_by_property(all_sophie)
            )
        else:
            all_sophie_ranked = []

        # This "target" is just a label in the combined result
        return Graffiti3Result(
            target=",".join(targets),
            conjectures=all_conjs,
            sophie_conditions=all_sophie_ranked,
            stage_breakdown=breakdown,
        )


# ───────────────── Sophie helpers / pretty printer ───────────────── #

def sophie_runner(
    *,
    conjectures: Sequence[Conjecture],
    g4: Graffiti3,
) -> List[SophieCondition]:
    """
    Thin wrapper around the Sophie module: take a list of inequality
    conjectures and return a flat list of SophieCondition objects.
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
    Group a flat list of SophieCondition into the dictionary format expected
    by rank_sophie_conditions_global.
    """
    out: Dict[str, List[SophieCondition]] = {}
    for sc in conds:
        out.setdefault(sc.property_name, []).append(sc)
    return out


def print_g3_result(
    result: Graffiti3Result,
    *,
    k_conjectures: int = 20,
    k_sophie: int = 20,
) -> None:
    """
    Convenience printer for Graffiti3Result: stage breakdown, top conjectures
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
