# # src/txgraffiti2025/math_discovery.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, Iterable, List, Optional, Tuple, Any

# import numpy as np
# import pandas as pd

# from txgraffiti2025.graffiti_relations import GraffitiClassRelations
# from txgraffiti2025.graffiti_lp import GraffitiLP, LPFitConfig
# from txgraffiti2025.graffiti_intricate_mixed import GraffitiLPIntricate
# from txgraffiti2025.graffiti_lp_lift_integer_aware import lift_integer_aware
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE

# # Optional global pruning
# try:
#     from txgraffiti2025.processing.post import morgan_filter
# except Exception:  # pragma: no cover
#     morgan_filter = None  # type: ignore


# # ───────────────────────── configs & metrics ───────────────────────── #

# @dataclass
# class MathDiscoveryConfig:
#     """
#     Tunables controlling metric computation and dominance checks.
#     """
#     touch_atol: float = 0.0
#     touch_rtol: float = 0.0

#     # how strict we are about "meaningful improvement"
#     improvement_margin_eps: float = 1e-9      # margin tolerance
#     improvement_touch_eps: int = 0           # allow equal or ≥ this more touches
#     improvement_rate_eps: float = 1e-9       # min improvement in touch_rate

#     # final bank trimming
#     top_k_per_bucket: Optional[int] = None   # None ⇒ keep all


# @dataclass
# class ConjectureMetrics:
#     """
#     Summary statistics used for refinement dominance.
#     """
#     support_n: int
#     touch_count: int
#     touch_rate: float
#     avg_margin: float
#     complexity: float


# # ───────────────────────── main class ───────────────────────── #

# class MathDiscovery:
#     r"""
#     High-level orchestrator for conjectures of the form

#         IF hypothesis H THEN (target relation RHS)

#     where relation ∈ { ≥, ≤, = }.

#     Responsibilities
#     ----------------
#     • Wrap GraffitiClassRelations, GraffitiLP, GraffitiLPIntricate.
#     • Run LP-based and intricate mixed generators.
#     • Integrate integer-aware lifting.
#     • Maintain a conjecture bank with *dominance-based refinement*:
#       a new conjecture only replaces an old one in the same context
#       if it is meaningfully better (support, touches, rate, tightness, simplicity).
#     """

#     def __init__(
#         self,
#         df: pd.DataFrame,
#         target: str,
#         config: Optional[MathDiscoveryConfig] = None,
#     ) -> None:
#         self.df = df
#         self.target = target
#         self.config = config or MathDiscoveryConfig()

#         # core engines
#         self.gcr = GraffitiClassRelations(df)
#         self.lp = GraffitiLP(self.gcr)
#         self.intricate = GraffitiLPIntricate(df)

#         # main conjecture bank; categorized by direction
#         self.bank: Dict[str, List[Conjecture]] = {
#             "lowers": [],
#             "uppers": [],
#             "equals": [],
#         }

#         # lightweight log of what we did
#         self.history: List[Dict[str, Any]] = []

#     # ───────────────────────── helpers: masks & metrics ───────────────────────── #

#     def _condition_mask(self, cond) -> np.ndarray:
#         """
#         Return mask for a condition (Predicate or None) on self.df.
#         Uses public GCR methods if available, otherwise legacy shim.
#         """
#         if cond is None or cond is TRUE:
#             return np.ones(len(self.df), dtype=bool)
#         if hasattr(self.gcr, "mask_for"):
#             return self.gcr.mask_for(cond)
#         if hasattr(self.gcr, "condition_mask_for"):
#             return self.gcr.condition_mask_for(cond)
#         if hasattr(self.gcr, "_mask_cached"):
#             return self.gcr._mask_cached(cond)  # legacy
#         # final fallback if Predicate exposes .mask(df)
#         if hasattr(cond, "mask"):
#             s = cond.mask(self.df).reindex(self.df.index, fill_value=False)
#             if s.dtype is not bool:
#                 s = s.fillna(False).astype(bool, copy=False)
#             return s.to_numpy(dtype=bool, copy=False)
#         raise AttributeError("Cannot compute mask for condition")

#     def _metrics_for(self, c: Conjecture) -> ConjectureMetrics:
#         """
#         Compute & attach summary metrics (support_n, touch_count, touch_rate,
#         avg_margin, complexity) for a conjecture on its hypothesis support.
#         """
#         cfg = self.config
#         cond = getattr(c, "condition", None)
#         mask = self._condition_mask(cond)

#         lhs = c.relation.left.eval(self.df).to_numpy(dtype=float, copy=False)
#         rhs = c.relation.right.eval(self.df).to_numpy(dtype=float, copy=False)

#         m = mask & np.isfinite(lhs) & np.isfinite(rhs)
#         n = int(m.sum())
#         if n == 0:
#             metrics = ConjectureMetrics(
#                 support_n=0,
#                 touch_count=0,
#                 touch_rate=0.0,
#                 avg_margin=float("inf"),
#                 complexity=float("inf"),
#             )
#         else:
#             diff = lhs[m] - rhs[m]
#             tol_arr = cfg.touch_atol + cfg.touch_rtol * np.abs(rhs[m])
#             touches = np.abs(diff) <= tol_arr
#             touch_count = int(touches.sum())
#             touch_rate = float(touch_count / n)

#             # slack in the "correct" direction
#             if isinstance(c.relation, Ge):
#                 margin = np.maximum(0.0, diff)        # y - rhs (≥0 ideally)
#             elif isinstance(c.relation, Le):
#                 margin = np.maximum(0.0, -diff)       # rhs - y
#             else:  # Eq
#                 margin = np.abs(diff)                 # closeness

#             avg_margin = float(np.mean(margin))
#             complexity = float(len(str(c.relation.right)))

#             metrics = ConjectureMetrics(
#                 support_n=n,
#                 touch_count=touch_count,
#                 touch_rate=touch_rate,
#                 avg_margin=avg_margin,
#                 complexity=complexity,
#             )

#         # Attach for downstream tools & printing
#         setattr(c, "support_n", metrics.support_n)
#         setattr(c, "touch_count", metrics.touch_count)
#         setattr(c, "touch_rate", metrics.touch_rate)
#         setattr(c, "avg_margin", metrics.avg_margin)
#         setattr(c, "complexity", metrics.complexity)

#         return metrics

#     # ───────────────────────── dominance & refinement ───────────────────────── #

#     def _same_context(self, c1: Conjecture, c2: Conjecture) -> bool:
#         """
#         Two conjectures are in the *same context* if:
#           • they have the same relation type (Ge, Le, Eq),
#           • the same LHS (target Expr),
#           • the same condition/hypothesis (up to string repr).
#         """
#         if type(c1.relation) is not type(c2.relation):
#             return False
#         if str(c1.relation.left) != str(c2.relation.left):
#             return False

#         cond1 = getattr(c1, "condition", None)
#         cond2 = getattr(c2, "condition", None)
#         return str(cond1) == str(cond2)

#     def _dominates(
#         self,
#         old: Conjecture,
#         new: Conjecture,
#         m_old: ConjectureMetrics,
#         m_new: ConjectureMetrics,
#     ) -> bool:
#         """
#         Return True if `new` is a *meaningful* refinement of `old`,
#         i.e., at least as good on all primary axes, and strictly
#         better on at least one axis (support, touches, rate, tightness, simplicity).
#         """
#         cfg = self.config

#         # Not allowed to be strictly worse on any primary axis
#         if m_new.support_n < m_old.support_n:
#             return False
#         if m_new.touch_count < m_old.touch_count - cfg.improvement_touch_eps:
#             return False
#         if m_new.touch_rate < m_old.touch_rate - cfg.improvement_rate_eps:
#             return False

#         # Must be strictly better somehow
#         better_support = m_new.support_n > m_old.support_n
#         better_touches = m_new.touch_count > m_old.touch_count
#         better_rate = m_new.touch_rate > m_old.touch_rate + cfg.improvement_rate_eps
#         tighter = m_new.avg_margin + cfg.improvement_margin_eps < m_old.avg_margin
#         simpler = (
#             m_new.avg_margin <= m_old.avg_margin + cfg.improvement_margin_eps
#             and m_new.complexity < m_old.complexity
#         )

#         return bool(better_support or better_touches or better_rate or tighter or simpler)

#     def _add_or_refine(self, bucket: str, c_new: Conjecture) -> None:
#         """
#         Insert c_new into the bank for the given bucket ("lowers", "uppers", "equals"),
#         replacing dominated conjectures in the same (H, target, relation) context and
#         discarding c_new if an existing one dominates it.
#         """
#         assert bucket in ("lowers", "uppers", "equals")
#         cfg = self.config

#         existing = self.bank[bucket]
#         m_new = self._metrics_for(c_new)

#         kept: List[Conjecture] = []
#         dominated_new = False

#         for c_old in existing:
#             if not self._same_context(c_old, c_new):
#                 kept.append(c_old)
#                 continue

#             m_old = self._metrics_for(c_old)

#             if self._dominates(c_old, c_new, m_old, m_new):
#                 # existing conjecture is strictly better; drop new
#                 dominated_new = True
#                 kept.append(c_old)
#             elif self._dominates(c_new, c_old, m_new, m_old):
#                 # new conjecture strictly better; replace old (skip keeping old)
#                 continue
#             else:
#                 # incomparable; keep both
#                 kept.append(c_old)

#         if not dominated_new:
#             kept.append(c_new)

#         # optional: small dedup by string to avoid clutter
#         seen, deduped = set(), []
#         for c in kept:
#             s = str(c)
#             if s in seen:
#                 continue
#             seen.add(s)
#             deduped.append(c)

#         self.bank[bucket] = deduped

#     def add_conjectures(self, conjs: Iterable[Conjecture], *, stage: str = "") -> None:
#         """
#         Add a batch of conjectures from any generator, using dominance-based refinement.
#         Optionally record the originating stage in self.history.
#         """
#         n_before = {k: len(v) for k, v in self.bank.items()}
#         added = 0

#         for c in conjs:
#             if isinstance(c.relation, Ge):
#                 self._add_or_refine("lowers", c)
#                 added += 1
#             elif isinstance(c.relation, Le):
#                 self._add_or_refine("uppers", c)
#                 added += 1
#             elif isinstance(c.relation, Eq):
#                 self._add_or_refine("equals", c)
#                 added += 1
#             else:
#                 # Unknown relation type; ignore or log as needed.
#                 continue

#         n_after = {k: len(v) for k, v in self.bank.items()}
#         self.history.append(
#             {
#                 "stage": stage,
#                 "proposed": added,
#                 "sizes_before": n_before,
#                 "sizes_after": n_after,
#             }
#         )

#     # ───────────────────────── stages: LP & intricate ───────────────────────── #

#     def run_lp_affine(
#         self,
#         *,
#         features: Optional[List[str]] = None,
#         direction: str = "both",
#         max_denom: int = 20,
#         min_support: int = 3,
#     ) -> None:
#         """
#         Stage 1: simple affine LP fit with all (or selected) features.

#         Parameters
#         ----------
#         features : list of str, optional
#             If None, use all invariants except target.
#         direction : {"both", "lower", "upper"}
#             Which inequalities to fit.
#         """
#         if features is None:
#             feats = [c for c in self.lp.invariants if repr(c) != self.target]
#         else:
#             feats = [self.lp.exprs.get(f, None) for f in features]
#             feats = [f for f in feats if f is not None]

#         cfg = LPFitConfig(
#             target=self.target,
#             features=feats,
#             direction=direction,
#             max_denom=max_denom,
#             min_support=min_support,
#             tol=1e-9,
#             touch_atol=self.config.touch_atol,
#             touch_rtol=self.config.touch_rtol,
#         )
#         lowers, uppers, equals = self.lp.fit_affine(cfg)
#         self.add_conjectures(lowers, stage="lp_affine_lower")
#         self.add_conjectures(uppers, stage="lp_affine_upper")
#         self.add_conjectures(equals, stage="lp_affine_equal")

#     def run_k_affine(
#         self,
#         *,
#         k_values: Iterable[int] = (1, 2, 3),
#         hypotheses_limit: Optional[int] = 20,
#         min_touch: int = 3,
#         max_denom: int = 20,
#         top_m_by_variance: Optional[int] = 10,
#     ) -> None:
#         """
#         Stage 2: generalized k-feature affine bounds (k ∈ k_values).

#         Uses GraffitiLP.generate_k_affine_bounds and refines into the bank.
#         """
#         res = self.lp.generate_k_affine_bounds(
#             target=self.target,
#             k_values=k_values,
#             hypotheses_limit=hypotheses_limit,
#             min_touch=min_touch,
#             max_denom=max_denom,
#             tol=1e-9,
#             touch_atol=self.config.touch_atol,
#             touch_rtol=self.config.touch_rtol,
#             top_m_by_variance=top_m_by_variance,
#         )
#         self.add_conjectures(res["lowers"], stage="k_affine_lower")
#         self.add_conjectures(res["uppers"], stage="k_affine_upper")
#         self.add_conjectures(res["equals"], stage="k_affine_equal")

#     def run_integer_lift(self) -> None:
#         """
#         Stage 3: integer-aware lifting of all current Ge/Le conjectures.

#         The lifted conjectures are reintroduced into the bank and only replace
#         existing ones when they meaningfully dominate under the metrics.
#         """
#         ge_le: List[Conjecture] = []
#         ge_le.extend(self.bank["lowers"])
#         ge_le.extend(self.bank["uppers"])

#         if not ge_le:
#             return

#         lifted = lift_integer_aware(
#             df=self.df,
#             gcr=self.gcr,
#             conjectures=ge_le,
#             touch_atol=self.config.touch_atol,
#             touch_rtol=self.config.touch_rtol,
#         )
#         self.add_conjectures(lifted, stage="integer_lift")

#     def run_intricate(
#         self,
#         *,
#         weight: float = 0.5,
#         min_touch: int = 3,
#     ) -> None:
#         """
#         Stage 4: intricate mixed inequalities (sqrt / square mix).

#         Uses GraffitiLPIntricate.run_intricate_mixed_pipeline and refines results.
#         """
#         res = self.intricate.run_intricate_mixed_pipeline(
#             target_col=self.target,
#             weight=weight,
#             min_touch=min_touch,
#         )
#         self.add_conjectures(res["lowers"], stage="intricate_lower")
#         self.add_conjectures(res["uppers"], stage="intricate_upper")
#         self.add_conjectures(res["equals"], stage="intricate_equal")

#     # ───────────────────────── finalization & reporting ───────────────────────── #

#     def finalize_bank(self) -> Dict[str, List[Conjecture]]:
#         """
#         Apply a global dedup + metric-based sort (+ optional Morgan filter)
#         to the current bank, and optionally truncate to top_k_per_bucket.

#         Returns a new dict {"lowers": [...], "uppers": [...], "equals": [...]}.
#         """
#         cfg = self.config

#         out: Dict[str, List[Conjecture]] = {}
#         for bucket in ("lowers", "uppers", "equals"):
#             conjs = list(self.bank[bucket])

#             if not conjs:
#                 out[bucket] = []
#                 continue

#             # ensure metrics are attached (and recomputed consistently)
#             for c in conjs:
#                 self._metrics_for(c)

#             # dedup by string
#             seen, deduped = set(), []
#             for c in conjs:
#                 s = str(c)
#                 if s in seen:
#                     continue
#                 seen.add(s)
#                 deduped.append(c)

#             # optional Morgan filter for structural dominance/pruning
#             if morgan_filter is not None:
#                 deduped = list(morgan_filter(self.df, deduped).kept)

#             # final sort
#             def _key(c: Conjecture):
#                 return (
#                     getattr(c, "touch_count", 0),
#                     getattr(c, "support_n", 0),
#                     getattr(c, "touch_rate", 0.0),
#                 )

#             deduped.sort(key=_key, reverse=True)

#             if cfg.top_k_per_bucket is not None and cfg.top_k_per_bucket > 0:
#                 deduped = deduped[: int(cfg.top_k_per_bucket)]

#             out[bucket] = deduped

#         return out

#     def print_bank(self, bank: Optional[Dict[str, List[Conjecture]]] = None, k_per_bucket: int = 10) -> None:
#         """
#         Pretty-print a bank (default: the finalized one) with metrics.
#         """
#         if bank is None:
#             bank = self.finalize_bank()

#         def _section(title: str):
#             print("\n" + "-" * 80)
#             print(title)
#             print("-" * 80 + "\n")

#         _section("MATHDISCOVERY CONJECTURE BANK")

#         for bucket in ("lowers", "uppers", "equals"):
#             lst = bank.get(bucket, [])
#             print(f"[{bucket.upper()}] total={len(lst)}\n")
#             for c in lst[:k_per_bucket]:
#                 try:
#                     s = c.pretty(show_tol=False)
#                 except Exception:
#                     s = str(c)
#                 tc = getattr(c, "touch_count", "?")
#                 sn = getattr(c, "support_n", "?")
#                 tr = getattr(c, "touch_rate", "?")
#                 print("•", s)
#                 print(f"    touches={tc}, support={sn}, touch_rate={tr:.3f}" if isinstance(tr, float)
#                       else f"    touches={tc}, support={sn}, touch_rate={tr}")
#             print()

#     # ───────────────────────── high-level pipeline ───────────────────────── #

#     def run_full_pipeline(
#         self,
#         *,
#         lp_direction: str = "both",
#         lp_max_denom: int = 20,
#         lp_min_support: int = 3,
#         k_values: Iterable[int] = (1, 2, 3),
#         k_hypotheses_limit: Optional[int] = 20,
#         k_min_touch: int = 3,
#         k_max_denom: int = 20,
#         k_top_m_by_variance: Optional[int] = 10,
#         intricate_weight: float = 0.5,
#         intricate_min_touch: int = 3,
#         enable_integer_lift: bool = True,
#     ) -> Dict[str, List[Conjecture]]:
#         """
#         Convenience wrapper: run a reasonable default sequence:

#         1) LP affine fit (global).
#         2) k-affine bounds (multi-feature).
#         3) optional integer-aware lifting over all Ge/Le conjectures.
#         4) intricate mixed inequalities.

#         At each step, conjectures are fed into the bank with dominance-based refinement.
#         Finally returns the finalized bank (deduped, sorted, optionally Morgan-filtered).
#         """
#         # 1) base LP affine
#         self.run_lp_affine(
#             direction=lp_direction,
#             max_denom=lp_max_denom,
#             min_support=lp_min_support,
#         )

#         # 2) k-affine (multi-feature)
#         self.run_k_affine(
#             k_values=k_values,
#             hypotheses_limit=k_hypotheses_limit,
#             min_touch=k_min_touch,
#             max_denom=k_max_denom,
#             top_m_by_variance=k_top_m_by_variance,
#         )

#         # 3) integer-aware lift
#         if enable_integer_lift:
#             self.run_integer_lift()

#         # 4) intricate mixed
#         self.run_intricate(
#             weight=intricate_weight,
#             min_touch=intricate_min_touch,
#         )

#         # finalized view
#         return self.finalize_bank()

# src/txgraffiti2025/math_discovery.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from txgraffiti2025.graffiti_relations import GraffitiClassRelations
from txgraffiti2025.graffiti_lp import GraffitiLP, LPFitConfig
from txgraffiti2025.graffiti_intricate_mixed import GraffitiLPIntricate
from txgraffiti2025.graffiti_lp_lift_integer_aware import lift_integer_aware
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE

# Optional global pruning
try:
    from txgraffiti2025.processing.post import morgan_filter
except Exception:  # pragma: no cover
    morgan_filter = None  # type: ignore


# ───────────────────────── configs & metrics ───────────────────────── #

@dataclass
class MathDiscoveryConfig:
    """
    Tunables controlling metric computation and dominance checks.
    """
    touch_atol: float = 0.0
    touch_rtol: float = 0.0

    # how strict we are about "meaningful improvement"
    improvement_margin_eps: float = 1e-9      # margin tolerance
    improvement_touch_eps: int = 0           # allow equal or ≥ this more touches
    improvement_rate_eps: float = 1e-9       # min improvement in touch_rate

    # final bank trimming
    top_k_per_bucket: Optional[int] = None   # None ⇒ keep all


@dataclass
class ConjectureMetrics:
    """
    Summary statistics used for refinement dominance.
    """
    support_n: int
    touch_count: int
    touch_rate: float
    avg_margin: float
    complexity: float


# ───────────────────────── main class ───────────────────────── #

class MathDiscovery:
    r"""
    High-level orchestrator for conjectures of the form

        IF hypothesis H THEN (target relation RHS)

    where relation ∈ { ≥, ≤, = }.

    Responsibilities
    ----------------
    • Wrap GraffitiClassRelations, GraffitiLP, GraffitiLPIntricate.
    • Run LP-based and intricate mixed generators.
    • Integrate integer-aware lifting.
    • Maintain a conjecture bank with *dominance-based refinement*:
      a new conjecture only replaces an old one in the same context
      if it is meaningfully better (support, touches, rate, tightness, simplicity).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        config: Optional[MathDiscoveryConfig] = None,
    ) -> None:
        self.df = df
        self.target = target
        self.config = config or MathDiscoveryConfig()

        # core engines
        self.gcr = GraffitiClassRelations(df)
        self.lp = GraffitiLP(self.gcr)
        self.intricate = GraffitiLPIntricate(df)

        # main conjecture bank; categorized by direction
        self.bank: Dict[str, List[Conjecture]] = {
            "lowers": [],
            "uppers": [],
            "equals": [],
        }

        # lightweight log of what we did
        self.history: List[Dict[str, Any]] = []

    # ───────────────────────── helpers: masks, metrics, keys ───────────────────────── #

    def _condition_mask(self, cond) -> np.ndarray:
        """
        Return mask for a condition (Predicate or None) on self.df.
        Uses public GCR methods if available, otherwise legacy shim.
        """
        if cond is None or cond is TRUE:
            return np.ones(len(self.df), dtype=bool)
        if hasattr(self.gcr, "mask_for"):
            return self.gcr.mask_for(cond)
        if hasattr(self.gcr, "condition_mask_for"):
            return self.gcr.condition_mask_for(cond)
        if hasattr(self.gcr, "_mask_cached"):
            return self.gcr._mask_cached(cond)  # legacy
        # final fallback if Predicate exposes .mask(df)
        if hasattr(cond, "mask"):
            s = cond.mask(self.df).reindex(self.df.index, fill_value=False)
            if s.dtype is not bool:
                s = s.fillna(False).astype(bool, copy=False)
            return s.to_numpy(dtype=bool, copy=False)
        raise AttributeError("Cannot compute mask for condition")

    def _metrics_for(self, c: Conjecture) -> ConjectureMetrics:
        """
        Compute & attach summary metrics (support_n, touch_count, touch_rate,
        avg_margin, complexity) for a conjecture on its hypothesis support.
        """
        cfg = self.config
        cond = getattr(c, "condition", None)
        mask = self._condition_mask(cond)

        lhs = c.relation.left.eval(self.df).to_numpy(dtype=float, copy=False)
        rhs = c.relation.right.eval(self.df).to_numpy(dtype=float, copy=False)

        m = mask & np.isfinite(lhs) & np.isfinite(rhs)
        n = int(m.sum())
        if n == 0:
            metrics = ConjectureMetrics(
                support_n=0,
                touch_count=0,
                touch_rate=0.0,
                avg_margin=float("inf"),
                complexity=float("inf"),
            )
        else:
            diff = lhs[m] - rhs[m]
            tol_arr = cfg.touch_atol + cfg.touch_rtol * np.abs(rhs[m])
            touches = np.abs(diff) <= tol_arr
            touch_count = int(touches.sum())
            touch_rate = float(touch_count / n)

            # slack in the "correct" direction
            if isinstance(c.relation, Ge):
                margin = np.maximum(0.0, diff)        # y - rhs (≥0 ideally)
            elif isinstance(c.relation, Le):
                margin = np.maximum(0.0, -diff)       # rhs - y
            else:  # Eq
                margin = np.abs(diff)                 # closeness

            avg_margin = float(np.mean(margin))
            complexity = float(len(str(c.relation.right)))

            metrics = ConjectureMetrics(
                support_n=n,
                touch_count=touch_count,
                touch_rate=touch_rate,
                avg_margin=avg_margin,
                complexity=complexity,
            )

        # Attach for downstream tools & printing
        setattr(c, "support_n", metrics.support_n)
        setattr(c, "touch_count", metrics.touch_count)
        setattr(c, "touch_rate", metrics.touch_rate)
        setattr(c, "avg_margin", metrics.avg_margin)
        setattr(c, "complexity", metrics.complexity)

        return metrics

    def _conjecture_sort_key(self, c: Conjecture) -> Tuple[int, int, float]:
        """
        Canonical ordering key for conjectures in the bank:
        primary: touch_count (descending)
        secondary: support_n (descending)
        tertiary: touch_rate (descending).
        """
        return (
            getattr(c, "touch_count", 0),
            getattr(c, "support_n", 0),
            getattr(c, "touch_rate", 0.0),
        )

    # ───────────────────────── RHS target-usage guard ───────────────────────── #

    def _uses_target_on_rhs(self, c: Conjecture) -> bool:
        """
        Return True if the RHS expression of the conjecture depends on the
        target column. We reject such conjectures as trivial/self-referential.

        This is intentionally conservative: we check for a structural list of
        columns/variables if present, and fall back to a string-based check.
        """
        rhs = getattr(c.relation, "right", None)
        if rhs is None:
            return False

        # Try a few likely attributes that Expr/ColumnTerm-style objects expose
        for attr in ("columns", "column_names", "vars", "variables"):
            cols = getattr(rhs, attr, None)
            if cols is not None:
                # Normalize to strings for comparison
                if any(str(col) == self.target for col in cols):
                    return True

        # Fallback: string representation contains the exact target name
        rhs_str = str(rhs)
        return self.target in rhs_str

    # ───────────────────────── dominance & refinement ───────────────────────── #

    def _same_context(self, c1: Conjecture, c2: Conjecture) -> bool:
        """
        Two conjectures are in the *same context* if:
          • they have the same relation type (Ge, Le, Eq),
          • the same LHS (target Expr),
          • the same condition/hypothesis (up to string repr).
        """
        if type(c1.relation) is not type(c2.relation):
            return False
        if str(c1.relation.left) != str(c2.relation.left):
            return False

        cond1 = getattr(c1, "condition", None)
        cond2 = getattr(c2, "condition", None)
        return str(cond1) == str(cond2)

    def _dominates(
        self,
        old: Conjecture,
        new: Conjecture,
        m_old: ConjectureMetrics,
        m_new: ConjectureMetrics,
    ) -> bool:
        """
        Return True if `new` is a *meaningful* refinement of `old`,
        i.e., at least as good on all primary axes, and strictly
        better on at least one axis (support, touches, rate, tightness, simplicity).
        """
        cfg = self.config

        # Not allowed to be strictly worse on any primary axis
        if m_new.support_n < m_old.support_n:
            return False
        if m_new.touch_count < m_old.touch_count - cfg.improvement_touch_eps:
            return False
        if m_new.touch_rate < m_old.touch_rate - cfg.improvement_rate_eps:
            return False

        # Must be strictly better somehow
        better_support = m_new.support_n > m_old.support_n
        better_touches = m_new.touch_count > m_old.touch_count
        better_rate = m_new.touch_rate > m_old.touch_rate + cfg.improvement_rate_eps
        tighter = m_new.avg_margin + cfg.improvement_margin_eps < m_old.avg_margin
        simpler = (
            m_new.avg_margin <= m_old.avg_margin + cfg.improvement_margin_eps
            and m_new.complexity < m_old.complexity
        )

        return bool(better_support or better_touches or better_rate or tighter or simpler)

    def _add_or_refine(self, bucket: str, c_new: Conjecture) -> None:
        """
        Insert c_new into the bank for the given bucket ("lowers", "uppers", "equals"),
        replacing dominated conjectures in the same (H, target, relation) context and
        discarding c_new if an existing one dominates it.

        Additionally:
        • Reject conjectures whose RHS uses the target column directly.
        • Maintain each bucket sorted by (_conjecture_sort_key), descending.
        """
        assert bucket in ("lowers", "uppers", "equals")

        # Guard: do not allow target to appear on the RHS
        if self._uses_target_on_rhs(c_new):
            return

        existing = self.bank[bucket]
        m_new = self._metrics_for(c_new)

        kept: List[Conjecture] = []
        dominated_new = False

        for c_old in existing:
            if not self._same_context(c_old, c_new):
                kept.append(c_old)
                continue

            m_old = self._metrics_for(c_old)

            if self._dominates(c_old, c_new, m_old, m_new):
                # existing conjecture is strictly better; drop new
                dominated_new = True
                kept.append(c_old)
            elif self._dominates(c_new, c_old, m_new, m_old):
                # new conjecture strictly better; replace old (skip keeping old)
                continue
            else:
                # incomparable; keep both
                kept.append(c_old)

        if not dominated_new:
            kept.append(c_new)

        # small dedup by string to avoid clutter
        seen, deduped = set(), []
        for c in kept:
            s = str(c)
            if s in seen:
                continue
            seen.add(s)
            deduped.append(c)

        # keep bucket internally sorted at all times
        deduped.sort(key=self._conjecture_sort_key, reverse=True)
        self.bank[bucket] = deduped

    def add_conjectures(self, conjs: Iterable[Conjecture], *, stage: str = "") -> None:
        """
        Add a batch of conjectures from any generator, using dominance-based refinement.
        Optionally record the originating stage in self.history.

        Conjectures whose RHS uses the target column are silently discarded.
        """
        n_before = {k: len(v) for k, v in self.bank.items()}
        proposed = 0
        accepted = 0

        for c in conjs:
            proposed += 1

            # Global guard: never admit conjectures with target on RHS
            if self._uses_target_on_rhs(c):
                continue

            if isinstance(c.relation, Ge):
                self._add_or_refine("lowers", c)
                accepted += 1
            elif isinstance(c.relation, Le):
                self._add_or_refine("uppers", c)
                accepted += 1
            elif isinstance(c.relation, Eq):
                self._add_or_refine("equals", c)
                accepted += 1
            else:
                # Unknown relation type; ignore or log as needed.
                continue

        n_after = {k: len(v) for k, v in self.bank.items()}
        self.history.append(
            {
                "stage": stage,
                "proposed": proposed,
                "accepted": accepted,
                "sizes_before": n_before,
                "sizes_after": n_after,
            }
        )

    # ───────────────────────── stages: LP & intricate ───────────────────────── #

    def run_lp_affine(
        self,
        *,
        features: Optional[List[str]] = None,
        direction: str = "both",
        max_denom: int = 20,
        min_support: int = 3,
    ) -> None:
        """
        Stage 1: simple affine LP fit with all (or selected) features.

        Parameters
        ----------
        features : list of str, optional
            If None, use all invariants except target.
        direction : {"both", "lower", "upper"}
            Which inequalities to fit.
        """
        if features is None:
            # filter out the target from LP features as well
            feats = [c for c in self.lp.invariants if repr(c) != self.target]
        else:
            feats = [self.lp.exprs.get(f, None) for f in features]
            feats = [f for f in feats if f is not None]

        cfg = LPFitConfig(
            target=self.target,
            features=feats,
            direction=direction,
            max_denom=max_denom,
            min_support=min_support,
            tol=1e-9,
            touch_atol=self.config.touch_atol,
            touch_rtol=self.config.touch_rtol,
        )
        lowers, uppers, equals = self.lp.fit_affine(cfg)
        self.add_conjectures(lowers, stage="lp_affine_lower")
        self.add_conjectures(uppers, stage="lp_affine_upper")
        self.add_conjectures(equals, stage="lp_affine_equal")

    def run_k_affine(
        self,
        *,
        k_values: Iterable[int] = (1, 2, 3),
        hypotheses_limit: Optional[int] = 20,
        min_touch: int = 3,
        max_denom: int = 20,
        top_m_by_variance: Optional[int] = 10,
    ) -> None:
        """
        Stage 2: generalized k-feature affine bounds (k ∈ k_values).

        Uses GraffitiLP.generate_k_affine_bounds and refines into the bank.
        """
        res = self.lp.generate_k_affine_bounds(
            target=self.target,
            k_values=k_values,
            hypotheses_limit=hypotheses_limit,
            min_touch=min_touch,
            max_denom=max_denom,
            tol=1e-9,
            touch_atol=self.config.touch_atol,
            touch_rtol=self.config.touch_rtol,
            top_m_by_variance=top_m_by_variance,
        )
        self.add_conjectures(res["lowers"], stage="k_affine_lower")
        self.add_conjectures(res["uppers"], stage="k_affine_upper")
        self.add_conjectures(res["equals"], stage="k_affine_equal")

    def run_integer_lift(self) -> None:
        """
        Stage 3: integer-aware lifting of all current Ge/Le conjectures.

        The lifted conjectures are reintroduced into the bank and only replace
        existing ones when they meaningfully dominate under the metrics.
        """
        ge_le: List[Conjecture] = []
        ge_le.extend(self.bank["lowers"])
        ge_le.extend(self.bank["uppers"])

        if not ge_le:
            return

        lifted = lift_integer_aware(
            df=self.df,
            gcr=self.gcr,
            conjectures=ge_le,
            touch_atol=self.config.touch_atol,
            touch_rtol=self.config.touch_rtol,
        )
        self.add_conjectures(lifted, stage="integer_lift")

    def run_intricate(
        self,
        *,
        weight: float = 0.5,
        min_touch: int = 3,
    ) -> None:
        """
        Stage 4: intricate mixed inequalities (sqrt / square mix).

        Uses GraffitiLPIntricate.run_intricate_mixed_pipeline and refines results.
        """
        res = self.intricate.run_intricate_mixed_pipeline(
            target_col=self.target,
            weight=weight,
            min_touch=min_touch,
        )
        self.add_conjectures(res["lowers"], stage="intricate_lower")
        self.add_conjectures(res["uppers"], stage="intricate_upper")
        self.add_conjectures(res["equals"], stage="intricate_equal")

    # ───────────────────────── finalization & reporting ───────────────────────── #

    def finalize_bank(self) -> Dict[str, List[Conjecture]]:
        """
        Apply a global dedup + metric-based sort (+ optional Morgan filter)
        to the current bank, and optionally truncate to top_k_per_bucket.

        Returns a new dict {"lowers": [...], "uppers": [...], "equals": [...]}.
        """
        cfg = self.config

        out: Dict[str, List[Conjecture]] = {}
        for bucket in ("lowers", "uppers", "equals"):
            conjs = list(self.bank[bucket])

            if not conjs:
                out[bucket] = []
                continue

            # ensure metrics are attached (and recomputed consistently)
            for c in conjs:
                self._metrics_for(c)

            # dedup by string
            seen, deduped = set(), []
            for c in conjs:
                s = str(c)
                if s in seen:
                    continue
                seen.add(s)
                deduped.append(c)

            # optional Morgan filter for structural dominance/pruning
            if morgan_filter is not None:
                deduped = list(morgan_filter(self.df, deduped).kept)

            # final sort with the same key as internal bank ordering
            deduped.sort(key=self._conjecture_sort_key, reverse=True)

            if cfg.top_k_per_bucket is not None and cfg.top_k_per_bucket > 0:
                deduped = deduped[: int(cfg.top_k_per_bucket)]

            out[bucket] = deduped

        return out

    def print_bank(
        self,
        bank: Optional[Dict[str, List[Conjecture]]] = None,
        k_per_bucket: int = 10,
    ) -> None:
        """
        Pretty-print a bank (default: the finalized one) with metrics.
        """
        if bank is None:
            bank = self.finalize_bank()

        def _section(title: str):
            print("\n" + "-" * 80)
            print(title)
            print("-" * 80 + "\n")

        _section("MATHDISCOVERY CONJECTURE BANK")

        for bucket in ("lowers", "uppers", "equals"):
            lst = bank.get(bucket, [])
            print(f"[{bucket.upper()}] total={len(lst)}\n")
            for c in lst[:k_per_bucket]:
                try:
                    s = c.pretty(show_tol=False)
                except Exception:
                    s = str(c)
                tc = getattr(c, "touch_count", "?")
                sn = getattr(c, "support_n", "?")
                tr = getattr(c, "touch_rate", "?")
                if isinstance(tr, float):
                    print("•", s)
                    print(f"    touches={tc}, support={sn}, touch_rate={tr:.3f}")
                else:
                    print("•", s)
                    print(f"    touches={tc}, support={sn}, touch_rate={tr}")
            print()

    # ───────────────────────── high-level pipeline ───────────────────────── #

    def run_full_pipeline(
        self,
        *,
        lp_direction: str = "both",
        lp_max_denom: int = 20,
        lp_min_support: int = 3,
        k_values: Iterable[int] = (1, 2, 3),
        k_hypotheses_limit: Optional[int] = 20,
        k_min_touch: int = 3,
        k_max_denom: int = 20,
        k_top_m_by_variance: Optional[int] = 10,
        intricate_weight: float = 0.5,
        intricate_min_touch: int = 3,
        enable_integer_lift: bool = True,
    ) -> Dict[str, List[Conjecture]]:
        """
        Convenience wrapper: run a reasonable default sequence:

        1) LP affine fit (global).
        2) k-affine bounds (multi-feature).
        3) optional integer-aware lifting over all Ge/Le conjectures.
        4) intricate mixed inequalities.

        At each step, conjectures are fed into the bank with dominance-based refinement.
        Finally returns the finalized bank (deduped, sorted, optionally Morgan-filtered).
        """
        # 1) base LP affine
        self.run_lp_affine(
            direction=lp_direction,
            max_denom=lp_max_denom,
            min_support=lp_min_support,
        )

        # 2) k-affine (multi-feature)
        self.run_k_affine(
            k_values=k_values,
            hypotheses_limit=k_hypotheses_limit,
            min_touch=k_min_touch,
            max_denom=k_max_denom,
            top_m_by_variance=k_top_m_by_variance,
        )

        # 3) integer-aware lift
        if enable_integer_lift:
            self.run_integer_lift()

        # 4) intricate mixed
        self.run_intricate(
            weight=intricate_weight,
            min_touch=intricate_min_touch,
        )

        # finalized view
        return self.finalize_bank()
