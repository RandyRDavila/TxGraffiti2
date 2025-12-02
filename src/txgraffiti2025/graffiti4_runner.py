# src/txgraffiti2025/graffiti4_runner.py

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import Expr, Const, to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.sophie import (
    SophieCondition,
    discover_sophie_from_inequalities,
    rank_sophie_conditions_global,
)

if TYPE_CHECKING:
    # Avoid circular imports at runtime
    from txgraffiti2025.graffiti4 import Graffiti4, HypothesisInfo


# ───────────────────────── generic helpers ───────────────────────── #


def _dedup_conjectures(conjs: Sequence[Conjecture]) -> List[Conjecture]:
    """
    Simple signature-based deduplication shared by all stages.

    Uses Conjecture.signature() which is already canonical-ish.
    """
    seen: set[str] = set()
    out: List[Conjecture] = []
    for c in conjs:
        sig = c.signature()
        if sig not in seen:
            seen.add(sig)
            out.append(c)
    return out


def _attach_touches(df: pd.DataFrame, conjs: List[Conjecture]) -> None:
    """
    Compute touch_count(df) for each conjecture and attach as attribute.

    Uses the classmethod-style call you've been using:
        Conjecture.touch_count(c, df)
    """
    from txgraffiti2025.forms.generic_conjecture import Conjecture as _C

    for c in conjs:
        try:
            val = _C.touch_count(c, df)
        except Exception:
            val = 0
        setattr(c, "touch_count", int(val))


# ───────────────────────── base stage runner ───────────────────────── #


@dataclass
class BaseStageRunner:
    """
    Base class for all Graffiti4 stages.

    Responsibilities
    ----------------
    - Provide a *single* way to:
        * build linear Exprs from numeric coefficients,
        * wrap inequalities into Conjecture objects with attached hypotheses.
    - Run Morgan + Dalmatian once per stage.
    - Deduplicate, attach touch counts, filter by min_touch, and sort.
    """
    stage_name: str
    min_touch: int = 0
    max_denom: int = 20
    coef_tol: float = 1e-8
    max_coef_abs: float = 10.0  # soft guard against insane coefficients

    # ---------- to be overridden by concrete stages ----------

    def generate_raw(
        self,
        g4: Graffiti4,
        target_col: str,
        target_expr: Expr,
        others: Dict[str, Expr],
    ) -> List[Conjecture]:
        """
        Subclasses implement this to actually *generate* conjectures.

        Important:
        - Do NOT run Morgan or Dalmatian here.
        - Do NOT attach touches or sort here.
        - Just return raw Conjecture objects for this stage.
        """
        raise NotImplementedError

    # ---------- shared coefficient / Expr helpers ----------

    def _to_const_fraction(self, x: float) -> Const:
        return Const(Fraction(x).limit_denominator(self.max_denom))

    def _canonical_mul(self, coef: float, expr: Expr) -> Expr | None:
        """
        Shared way to build coef * expr:

          - |coef| < coef_tol      -> None (drop term)
          - coef ~ 1               -> expr
          - coef ~ -1              -> -expr
          - |coef| > max_coef_abs  -> None (skip as insane)
          - else                   -> (bounded-denominator rational)*expr
        """
        c = float(coef)
        if abs(c) < self.coef_tol:
            return None
        if abs(c) > self.max_coef_abs:
            return None
        if abs(c - 1.0) < self.coef_tol:
            return expr
        if abs(c + 1.0) < self.coef_tol:
            return Const(-1) * expr
        return self._to_const_fraction(c) * expr

    def build_linear_expr(
        self,
        beta: np.ndarray,
        intercept: float,
        features: Sequence[Expr],
    ) -> Expr:
        """
        Canonical linear combination:

            beta · features + intercept

        All runners should use this so that representations match and
        deduplication behaves as expected.
        """
        assert len(beta) == len(features)
        terms: List[Expr] = []

        # feature terms
        for c, f in zip(beta, features):
            term = self._canonical_mul(float(c), f)
            if term is not None:
                terms.append(term)

        # intercept term
        if abs(intercept) >= self.coef_tol and abs(intercept) <= self.max_coef_abs:
            terms.append(self._to_const_fraction(float(intercept)))

        if not terms:
            return Const(0)

        expr = terms[0]
        for t in terms[1:]:
            expr = expr + t
        return expr

    # ---------- wrap into Conjecture ----------

    def _wrap_ge(
        self,
        target_col: str,
        target_expr: Expr,
        rhs_expr: Expr,
        hyp: "HypothesisInfo",
    ) -> Conjecture:
        from txgraffiti2025.forms.generic_conjecture import Ge
        rel = Ge(target_expr, rhs_expr)
        return Conjecture(
            relation=rel,
            condition=hyp.pred,
            name=f"[{self.stage_name}] {target_col} ≥ {rhs_expr!r} under {hyp.name}",
        )

    def _wrap_le(
        self,
        target_col: str,
        target_expr: Expr,
        rhs_expr: Expr,
        hyp: "HypothesisInfo",
    ) -> Conjecture:
        from txgraffiti2025.forms.generic_conjecture import Le
        rel = Le(target_expr, rhs_expr)
        return Conjecture(
            relation=rel,
            condition=hyp.pred,
            name=f"[{self.stage_name}] {target_col} ≤ {rhs_expr!r} under {hyp.name}",
        )

    # ---------- post-processing pipeline for a stage ----------

    def postprocess(
        self,
        g4: Graffiti4,
        conjs: List[Conjecture],
    ) -> List[Conjecture]:
        """
        Shared post-processing for a single stage:

          1. Morgan filter (if present)
          2. Dalmatian filter (if present)
          3. signature-based deduplication
          4. attach touch counts
          5. filter by min_touch
          6. sort by (touch_count, support) descending
        """
        df = g4.df

        # 1–2: Morgan + Dalmatian heuristics (if wired)
        out = list(conjs)
        if g4.morgan_filter is not None:
            out = g4.morgan_filter(df, out)
        if g4.dalmatian_filter is not None:
            out = g4.dalmatian_filter(df, out)

        # 3: dedup
        out = _dedup_conjectures(out)

        # 4: touches
        _attach_touches(df, out)

        # 5: min_touch
        if self.min_touch > 0:
            out = [c for c in out if getattr(c, "touch_count", 0) >= self.min_touch]

        # 6: sort
        out.sort(
            key=lambda c: (
                getattr(c, "touch_count", 0),
                getattr(c, "support_n", getattr(c, "support", 0)),
            ),
            reverse=True,
        )
        return out

    # ---------- main callable interface ----------

    def __call__(
        self,
        g4: Graffiti4,
        target_col: str,
        target_expr: Expr,
        others: Dict[str, Expr],
    ) -> List[Conjecture]:
        raw = self.generate_raw(g4, target_col, target_expr, others)
        if not raw:
            return []
        return self.postprocess(g4, raw)


# ───────────────────────── concrete stages ───────────────────────── #


class ConstantStageRunner(BaseStageRunner):
    """
    Stage 0: constant bounds for each hypothesis:

        H ⇒ t ≥ c_min(H)
        H ⇒ t ≤ c_max(H)
    """

    def __init__(self, min_touch: int = 0, max_denom: int = 20):
        super().__init__(
            stage_name="const",
            min_touch=min_touch,
            max_denom=max_denom,
        )

    def generate_raw(
        self,
        g4: Graffiti4,
        target_col: str,
        target_expr: Expr,
        others: Dict[str, Expr],
    ) -> List[Conjecture]:
        df = g4.df
        vals = df[target_col].to_numpy(dtype=float)
        conjs: List[Conjecture] = []

        for hyp in g4.hypotheses:
            mask = np.asarray(hyp.mask, dtype=bool)
            finite = mask & np.isfinite(vals)
            if finite.sum() == 0:
                continue

            v = vals[finite]
            c_min = float(np.min(v))
            c_max = float(np.max(v))

            rhs_min = self.build_linear_expr(
                beta=np.array([], dtype=float),
                intercept=c_min,
                features=[],
            )
            rhs_max = self.build_linear_expr(
                beta=np.array([], dtype=float),
                intercept=c_max,
                features=[],
            )

            conjs.append(self._wrap_ge(target_col, target_expr, rhs_min, hyp))
            conjs.append(self._wrap_le(target_col, target_expr, rhs_max, hyp))

        return conjs


class RatioStageRunner(BaseStageRunner):
    """
    Stage 1: ratio-based bounds for each hypothesis and other invariant x:

        r_i = t_i / x_i

        c_min = min r_i,  c_max = max r_i

        H ⇒ t ≥ c_min * x
        H ⇒ t ≤ c_max * x
    """

    def __init__(
        self,
        min_touch: int = 0,
        min_support: int = 5,
        max_denom: int = 20,
        max_coef_abs: float = 10.0,
    ):
        super().__init__(
            stage_name="ratio",
            min_touch=min_touch,
            max_denom=max_denom,
            max_coef_abs=max_coef_abs,
        )
        self.min_support = min_support

    def generate_raw(
        self,
        g4: Graffiti4,
        target_col: str,
        target_expr: Expr,
        others: Dict[str, Expr],
    ) -> List[Conjecture]:
        df = g4.df
        t_all = df[target_col].to_numpy(dtype=float)
        conjs: List[Conjecture] = []

        for hyp in g4.hypotheses:
            H = np.asarray(hyp.mask, dtype=bool)
            if not H.any():
                continue

            for other_name, other_expr in others.items():
                if other_name == target_col:
                    continue

                try:
                    x_all = other_expr.eval(df).to_numpy(dtype=float)
                except Exception:
                    continue

                valid = (
                    H
                    & np.isfinite(t_all)
                    & np.isfinite(x_all)
                    & (x_all != 0.0)
                )
                if valid.sum() < self.min_support:
                    continue

                t = t_all[valid]
                x = x_all[valid]
                r = t / x
                if not np.isfinite(r).any():
                    continue

                c_min = float(np.min(r[np.isfinite(r)]))
                c_max = float(np.max(r[np.isfinite(r)]))

                # lower: t ≥ c_min * x
                rhs_min = self._canonical_mul(c_min, other_expr)
                if rhs_min is not None:
                    conjs.append(
                        self._wrap_ge(
                            target_col=target_col,
                            target_expr=target_expr,
                            rhs_expr=rhs_min,
                            hyp=hyp,
                        )
                    )

                # upper: t ≤ c_max * x
                rhs_max = self._canonical_mul(c_max, other_expr)
                if rhs_max is not None:
                    conjs.append(
                        self._wrap_le(
                            target_col=target_col,
                            target_expr=target_expr,
                            rhs_expr=rhs_max,
                            hyp=hyp,
                        )
                    )

        return conjs


# ───────────────────────── runner config & orchestrator ───────────────────────── #


@dataclass
class Graffiti4RunnerConfig:
    """
    Config for the top-level Graffiti4Runner.

    Parameters
    ----------
    min_touch : int
        Discard conjectures with touch_count < min_touch at stage level.
    complexity : int
        Number of stages to use (1 = const, 2 = const+ratio, etc.).
    include_invariant_products, include_abs, include_min_max : bool
        How to build the "others" pool from Graffiti4.
    """
    min_touch: int = 0
    complexity: int = 2   # 1 => const only, 2 => const + ratio, etc.
    include_invariant_products: bool = False
    include_abs: bool = True
    include_min_max: bool = True
    ratio_min_support: int = 5
    max_denom: int = 20
    max_coef_abs: float = 10.0


@dataclass
class RunnerResult:
    """
    Simple container for runner outputs.
    """
    target: str
    conjectures: List[Conjecture]
    sophie_conditions: List[SophieCondition]
    stage_breakdown: Dict[str, Any]


class Graffiti4Runner:
    """
    Orchestrates multiple stage runners on top of a Graffiti4 workspace.

    Usage
    -----
    g4 = Graffiti4(df, max_boolean_arity=2, ...)
    cfg = Graffiti4RunnerConfig(min_touch=3, complexity=2)
    runner = Graffiti4Runner(g4, cfg)

    result = runner.run(target="independence_number")
    """

    def __init__(self, g4: Graffiti4, config: Graffiti4RunnerConfig):
        self.g4 = g4
        self.cfg = config

        # Stage list in order of increasing complexity
        self.stages: List[BaseStageRunner] = [
            ConstantStageRunner(
                min_touch=config.min_touch,
                max_denom=config.max_denom,
            ),
            RatioStageRunner(
                min_touch=config.min_touch,
                min_support=config.ratio_min_support,
                max_denom=config.max_denom,
                max_coef_abs=config.max_coef_abs,
            ),
            # Later: LPSingleStageRunner, LPStageRunner, MixedStageRunner, ...
        ]

    # --- internal helper to build others pool using Graffiti4 logic ---

    def _build_others_pool(self, target: str) -> Dict[str, Expr]:
        """
        Delegate to Graffiti4's _build_others_pool (or reimplement if needed).
        """
        return self.g4._build_others_pool(
            target=target,
            include_invariant_products=self.cfg.include_invariant_products,
            include_abs=self.cfg.include_abs,
            include_min_max=self.cfg.include_min_max,
        )

    # --- Sophie helper (global, after all stages) ---

    def _discover_sophie_global(
        self,
        all_conjectures: List[Conjecture],
    ) -> List[SophieCondition]:
        if not all_conjectures:
            return []

        df_num = self.g4.df
        bool_cols = list(self.g4.gcr.boolean_cols)
        bool_df = df_num[bool_cols].copy() if bool_cols else pd.DataFrame(index=df_num.index)

        base_mask = self.g4.base_mask
        base_name = self.g4.base_name

        sophie_by_prop = discover_sophie_from_inequalities(
            df_num=df_num,
            bool_df=bool_df,
            base_mask=base_mask,
            base_name=base_name,
            inequality_conjectures=all_conjectures,
            **(self.g4.sophie_cfg or {}),
        )

        flat: List[SophieCondition] = []
        for _prop, conds in sophie_by_prop.items():
            flat.extend(conds)

        # Global ranking / Morgan-on-Sophie can be handled here if desired.
        ranked = rank_sophie_conditions_global(
            {sc.property_name: [sc] for sc in flat}
        )
        return ranked

    # --- public API ---

    def run(self, target: str) -> RunnerResult:
        if target not in self.g4.df.columns:
            raise KeyError(f"Target column '{target}' not found in df.")

        target_expr = to_expr(target)
        others = self._build_others_pool(target)

        all_conjectures: List[Conjecture] = []
        stage_breakdown: Dict[str, Any] = {}

        # Run stages up to cfg.complexity (1 => only stage 0, etc.)
        max_stage_index = min(self.cfg.complexity, len(self.stages))
        for stage in self.stages[:max_stage_index]:
            conjs = stage(self.g4, target, target_expr, others)
            all_conjectures.extend(conjs)
            stage_breakdown[stage.stage_name] = len(conjs)

        # Global dedup + final sort by touches (touches already attached per stage)
        all_conjectures = _dedup_conjectures(all_conjectures)
        # If some conjectures lost their touch_count due to dedup, recompute:
        _attach_touches(self.g4.df, all_conjectures)
        all_conjectures.sort(
            key=lambda c: getattr(c, "touch_count", 0),
            reverse=True,
        )

        # Sophie discovery once over the final inequality set
        sophie_conds = self._discover_sophie_global(all_conjectures)

        return RunnerResult(
            target=target,
            conjectures=all_conjectures,
            sophie_conditions=sophie_conds,
            stage_breakdown=stage_breakdown,
        )
