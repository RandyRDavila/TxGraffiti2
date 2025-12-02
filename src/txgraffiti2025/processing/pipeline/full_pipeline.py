"""
Full conjecturing pipeline orchestrator.

Stages
------
(1) Hypothesis + feature context selection
(2) Mini-batch generation with *local* pruning per hypothesis
    - truth → local Dalmatian (significance) → micro-Morgan → local Hazel (drop) → beam cap
(3) Global consolidation across hypotheses
    - Dalmatian → Morgan → Hazel (soft drop, ranking)
(4) (Optional) Lifting waves and numeric refinement (disabled by default)
    - Coefficient lifts (constants / reciprocals), intercept lifts
    - Immediate refinement of each accepted lift
    - Local prune per wave, then global consolidate again

Returns
-------
- final ranked conjectures (Hazel-kept, sorted by touch desc)
- Hazel scoreboard (diagnostics)
- constants cache (optionally warmed and/or expanded)
- per-hypothesis seeds and a small provenance ledger

Notes
-----
- All mask operations align to df.index.
- The mini-batch stage keeps candidate counts small *before* costly lifts.
- Lifting & refinement are available behind config toggles for later runs.

Typical use
-----------
>>> from txgraffiti.example_data import graph_data as df
>>> from txgraffiti2025.processing.pipeline.full_pipeline import run_full_pipeline, PipelineConfig
>>> res = run_full_pipeline(df, target="domination_number",
...     config=PipelineConfig(generators={"ratios","lp"}, context_mode="one"))
>>> for c in res.final[:10]:
...     print(c.pretty(arrow="⇒"))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Hypotheses utilities
from txgraffiti2025.processing.pre.hypotheses import (
    detect_base_hypothesis,
    enumerate_boolean_hypotheses,
)
from txgraffiti2025.forms.predicates import Predicate, Where

# Generators (used via the mini-batch controller)
from txgraffiti2025.processing.pipeline.mini_batch import (
    generate_with_local_pruning,
    MiniBatchConfig,
)

# Post / global filters
from txgraffiti2025.processing.post.dalmatian import dalmatian_filter
from txgraffiti2025.processing.post.morgan import morgan_filter
from txgraffiti2025.processing.post.hazel import hazel_rank

# Optional: constants cache and generalizers (kept behind toggles)
from txgraffiti2025.processing.pre.constants_cache import (
    precompute_constant_ratios_pairs,
    ConstantsCache,
)
from txgraffiti2025.processing.post.generalize_from_constants import (
    propose_generalizations_from_constants,
)
from txgraffiti2025.processing.post.reciprocal_generalizer import (
    propose_generalizations_from_reciprocals,
)
from txgraffiti2025.processing.post.intercept_generalizer import (
    propose_generalizations_from_intercept,
)
from txgraffiti2025.processing.post.refine_numeric import (
    refine_numeric_bounds,
    RefinementConfig,
)

# Pretty (only for provenance messages; robust to absence)
try:
    from txgraffiti2025.forms.pretty import format_conjecture
except Exception:  # pragma: no cover
    def format_conjecture(c, **kwargs):  # fallback
        return repr(c)


# -----------------------------
# Config & Result structures
# -----------------------------

@dataclass
class GlobalPruneConfig:
    """Consolidation knobs after per-hypothesis mini-batches."""
    hazel_drop_frac: float = 0.25
    hazel_atol: float = 1e-9
    beam_k_global: Optional[int] = 200  # None = no global beam cap


@dataclass
class LiftingConfig:
    """Optional lifting stage (disabled by default for speed)."""
    enable_coefficient_from_constants: bool = False
    enable_coefficient_from_reciprocals: bool = False
    enable_intercept_lifts: bool = False
    require_superset_for_lifts: bool = True
    atol_match: float = 1e-9
    # Budgets
    max_lifts_per_seed: int = 5
    max_new_per_wave: int = 500
    waves: int = 1
    # Intercept extras
    user_intercept_candidates: Optional[Sequence] = None
    relaxers: Optional[Sequence] = None


@dataclass
class PipelineConfig:
    """
    Overall pipeline configuration. Generator knobs are delegated to MiniBatchConfig;
    refinement knobs to RefinementConfig; global consolidation via GlobalPruneConfig.
    """
    # Which generators and how to iterate contexts:
    generators: Set[str] = field(default_factory=lambda: {"ratios", "lp"})
    context_mode: Literal["one", "two"] = "one"

    # Per-hypothesis mini-batch pruning (local) and generation caps:
    drop_frac_local: float = 0.50
    hazel_atol_local: float = 1e-9
    beam_k_per_H: int = 50
    max_seeds_per_generator_per_pair: Optional[int] = 10

    # Direction for generators: "upper" | "lower" | "both"
    direction: Literal["upper", "lower", "both"] = "both"

    # Global consolidation knobs:
    global_prune: GlobalPruneConfig = field(default_factory=GlobalPruneConfig)

    # Optional lifting & refinement:
    lifting: LiftingConfig = field(default_factory=LiftingConfig)
    refinement: RefinementConfig = field(default_factory=lambda: RefinementConfig(require_tighter=True))

    # Hypothesis enumeration caps:
    max_hypotheses: Optional[int] = None
    max_context_pairs_per_H: Optional[int] = None  # reserved for future use

    # Constants cache warming:
    warm_constants_on_base: bool = True
    constants_shifts: Sequence[int] = (-2, -1, 0, 1, 2)
    constants_min_support: int = 8
    constants_max_denominator: int = 50


@dataclass
class PipelineResult:
    final: List  # list[Conjecture]
    scoreboard: pd.DataFrame
    cache: Optional[ConstantsCache]
    seeds_by_hypothesis: Dict[str, List]  # repr(H) -> list[Conjecture]
    frontier: List  # post-global-prune, pre-lifts
    provenance: List[Dict[str, object]]


# -----------------------------
# Helpers
# -----------------------------

def _numeric_columns(df: pd.DataFrame, *, exclude: Sequence[str] = ()) -> List[str]:
    """All numeric (non-bool) columns except those in `exclude`."""
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            cols.append(c)
    return cols


def _mask(df: pd.DataFrame, cond: Optional[Predicate]) -> pd.Series:
    if cond is None:
        return pd.Series(True, index=df.index)
    m = cond.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)


def _ensure_constants_cache(
    df: pd.DataFrame,
    cache: Optional[ConstantsCache],
    *,
    base_hypothesis: Optional[Predicate],
    cfg: PipelineConfig,
) -> Optional[ConstantsCache]:
    """Optionally warm the constants cache on the base hypothesis."""
    if not cfg.warm_constants_on_base:
        return cache
    try:
        warmed = precompute_constant_ratios_pairs(
            df,
            base_hypothesis,
            shifts=cfg.constants_shifts,
            min_support=cfg.constants_min_support,
            max_denominator=cfg.constants_max_denominator,
        )
        if cache is None:
            return warmed
        # merge maps in place
        cache.hyp_to_key.update(warmed.hyp_to_key)
        cache.key_to_constants.update(warmed.key_to_constants)
        return cache
    except Exception:
        # warming is optional — swallow failures
        return cache


def _global_consolidate(
    df: pd.DataFrame,
    conjs: List,
    cfg: GlobalPruneConfig,
) -> Tuple[List, pd.DataFrame]:
    """Dalmatian → Morgan → Hazel (soft) with optional global beam cap."""
    # Dalmatian: truth + significance + structural dedup
    d = dalmatian_filter(conjs, df)
    # Morgan: most-general hypotheses per canonical conclusion
    m = morgan_filter(df, d)
    kept = m.kept
    # Hazel: rank by touch & drop bottom frac (soft)
    hz = hazel_rank(df, kept, drop_frac=cfg.hazel_drop_frac, atol=cfg.hazel_atol)
    ranked = hz.kept_sorted
    if cfg.beam_k_global is not None and len(ranked) > cfg.beam_k_global:
        ranked = ranked[: cfg.beam_k_global]
    return ranked, hz.scoreboard


def _mk_minibatch_cfg(pipe_cfg: PipelineConfig) -> MiniBatchConfig:
    """Translate PipelineConfig → MiniBatchConfig for mini-batch stage."""
    return MiniBatchConfig(
        direction=pipe_cfg.direction,
        generators=pipe_cfg.generators,
        context_mode=pipe_cfg.context_mode,
        drop_frac_local=pipe_cfg.drop_frac_local,
        hazel_atol=pipe_cfg.hazel_atol_local,
        beam_k_per_H=pipe_cfg.beam_k_per_H,
        max_seeds_per_generator_per_pair=pipe_cfg.max_seeds_per_generator_per_pair,
        # keep the bundled generator niceties at their defaults; users may tweak later
    )


def _provenance_record(c, *, stage: str, hypothesis: Optional[Predicate]) -> Dict[str, object]:
    """Minimal provenance record; expandable later."""
    try:
        pretty = c.pretty(arrow="⇒")
    except Exception:
        pretty = format_conjecture(c, show_condition=True)
    return {
        "stage": stage,
        "name": getattr(c, "name", "Conjecture"),
        "hypothesis": repr(hypothesis),
        "relation_text": format_conjecture(c, show_condition=False),
        "condition_text": format_conjecture(c, show_condition=True).split("⇒")[0].strip(),
    }


# -----------------------------
# Main entry point
# -----------------------------

def run_full_pipeline(
    df: pd.DataFrame,
    *,
    target: str,
    numeric_columns: Optional[Sequence[str]] = None,
    hypotheses: Optional[Sequence[Optional[Predicate]]] = None,
    config: PipelineConfig = PipelineConfig(),
    constants_cache: Optional[ConstantsCache] = None,
) -> PipelineResult:
    """
    Execute the full conjecturing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset of invariants and boolean columns for classes.
    target : str
        Column to bound.
    numeric_columns : list[str], optional
        Numeric columns to consider as features/contexts (target will be excluded).
        If None, inferred from df.
    hypotheses : list[Predicate|None], optional
        Hypotheses to evaluate. If None: detect base and enumerate boolean hypotheses.
    config : PipelineConfig
        Full pipeline configuration knobs.
    constants_cache : ConstantsCache | None
        Optional, re-usable cache across runs. Returned (possibly updated).

    Returns
    -------
    PipelineResult
        final ranked conjectures, Hazel scoreboard, constants cache,
        seeds grouped by hypothesis, frontier set, and provenance ledger.
    """
    if target not in df.columns:
        raise KeyError(f"target '{target}' not found in DataFrame.")

    # Numeric contexts (exclude target)
    num_cols = list(numeric_columns) if numeric_columns is not None else _numeric_columns(df, exclude=(target,))

    # Hypotheses selection
    if hypotheses is None:
        base = detect_base_hypothesis(df)
        H_all = enumerate_boolean_hypotheses(df, include_base=True)
        if config.max_hypotheses is not None and len(H_all) > config.max_hypotheses:
            H_all = H_all[: config.max_hypotheses]
    else:
        H_all = list(hypotheses)
        base = H_all[0] if H_all else None

    # Optional: warm constants cache on base class
    constants_cache = _ensure_constants_cache(
        df, constants_cache, base_hypothesis=base, cfg=config
    )

    # Stage (2) Per-hypothesis mini-batches with local pruning
    seeds_by_H: Dict[str, List] = {}
    provenance: List[Dict[str, object]] = []
    mini_cfg = _mk_minibatch_cfg(config)

    for H in H_all:
        H_norm = H if H is not None else Where(lambda d: pd.Series(True, index=d.index))
        seeds = generate_with_local_pruning(
            df,
            target=target,
            hypothesis=H,
            numeric_columns=num_cols,
            config=mini_cfg,
        )
        seeds_by_H[repr(H_norm)] = list(seeds)
        provenance.extend([_provenance_record(c, stage="mini_batch", hypothesis=H) for c in seeds])

    # Flatten seeds
    all_seeds = [c for lst in seeds_by_H.values() for c in lst]

    # Stage (3) Global consolidation across hypotheses
    frontier, scoreboard = _global_consolidate(df, all_seeds, cfg=config.global_prune)

    # Early exit: no lifts / refinement requested
    if not (
        config.lifting.enable_coefficient_from_constants
        or config.lifting.enable_coefficient_from_reciprocals
        or config.lifting.enable_intercept_lifts
    ):
        return PipelineResult(
            final=frontier,
            scoreboard=scoreboard,
            cache=constants_cache,
            seeds_by_hypothesis=seeds_by_H,
            frontier=frontier,
            provenance=provenance,
        )

    # -----------------------------
    # (4) Optional lifting waves + immediate refinement
    # -----------------------------
    waves_left = max(0, int(config.lifting.waves))
    accepted_global: List = list(frontier)

    for wave in range(waves_left):
        new_props: List = []
        produced = 0

        for seed in list(accepted_global):
            if produced >= config.lifting.max_new_per_wave:
                break

            # ---- Coefficient from constants ----
            if config.lifting.enable_coefficient_from_constants and constants_cache is not None:
                try:
                    gconst = propose_generalizations_from_constants(
                        df, seed, constants_cache,
                        candidate_hypotheses=H_all,
                        atol=config.lifting.atol_match,
                    )
                    for g in gconst[: config.lifting.max_lifts_per_seed]:
                        new_props.append(g.new_conjecture)
                        produced += 1
                        if produced >= config.lifting.max_new_per_wave:
                            break
                except Exception:
                    pass
            if produced >= config.lifting.max_new_per_wave:
                break

            # ---- Coefficient from reciprocals ----
            if config.lifting.enable_coefficient_from_reciprocals:
                try:
                    grecip = propose_generalizations_from_reciprocals(
                        df, seed, candidate_hypotheses=H_all
                    )
                    # returns Conjecture list
                    for c in grecip[: config.lifting.max_lifts_per_seed]:
                        new_props.append(c)
                        produced += 1
                        if produced >= config.lifting.max_new_per_wave:
                            break
                except Exception:
                    pass
            if produced >= config.lifting.max_new_per_wave:
                break

            # ---- Intercept lifts ----
            if config.lifting.enable_intercept_lifts:
                try:
                    gint = propose_generalizations_from_intercept(
                        df, seed, constants_cache,
                        candidate_hypotheses=H_all,
                        candidate_intercepts=config.lifting.user_intercept_candidates,
                        relaxers_Z=config.lifting.relaxers,
                        require_superset=config.lifting.require_superset_for_lifts,
                    )
                    for g in gint[: config.lifting.max_lifts_per_seed]:
                        new_props.append(g.new_conjecture)
                        produced += 1
                        if produced >= config.lifting.max_new_per_wave:
                            break
                except Exception:
                    pass

        if not new_props:
            break

        # Immediate refinement of each proposal (keep tighter variants; fallback to original if none)
        refined_all: List = []
        for c in new_props:
            try:
                refined = refine_numeric_bounds(df, c, config=config.refinement)
                if refined:
                    refined_all.extend(refined)
                else:
                    refined_all.append(c)
            except Exception:
                # If refinement fails, keep the original proposal
                refined_all.append(c)

        # Local prune on this wave’s products to keep quality high
        wave_kept, _ = _global_consolidate(df, refined_all, cfg=config.global_prune)
        provenance.extend([_provenance_record(c, stage=f"lift_wave_{wave+1}", hypothesis=c.condition) for c in wave_kept])

        # Merge into accepted set and consolidate again
        accepted_global = list({(repr(c.condition), repr(c.relation)): c for c in (accepted_global + wave_kept)}.values())
        accepted_global, scoreboard = _global_consolidate(df, accepted_global, cfg=config.global_prune)

    return PipelineResult(
        final=accepted_global,
        scoreboard=scoreboard,
        cache=constants_cache,
        seeds_by_hypothesis=seeds_by_H,
        frontier=frontier,
        provenance=provenance,
    )

