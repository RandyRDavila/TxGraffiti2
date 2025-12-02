from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Literal, Sequence, Set, Tuple, Optional

import itertools
import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.processing.post.dalmatian import dalmatian_filter
from txgraffiti2025.processing.post.morgan import morgan_filter
from txgraffiti2025.processing.post.hazel import hazel_rank

# generators
from txgraffiti2025.generators.ratios import ratios
from txgraffiti2025.generators.lp import lp_bounds, LPConfig
from txgraffiti2025.generators.convex import convex_hull


@dataclass
class MiniBatchConfig:
    direction: Literal["upper", "lower", "both"] = "both"
    generators: Set[str] = frozenset({"ratios", "lp"})
    context_mode: Literal["one", "two"] = "one"   # “one”: single context feature, “two”: pairs for lp/convex
    # local pruning knobs
    drop_frac_local: float = 0.50
    hazel_atol: float = 1e-9
    beam_k_per_H: int = 50
    # generation caps
    max_seeds_per_generator_per_pair: Optional[int] = 10
    # LP / convex niceties
    lp_max_denominator: int = 50
    lp_tol: float = 1e-9
    lp_min_support: int = 3
    convex_max_denominator: int = 100
    convex_drop_side_facets: bool = True
    convex_tol: float = 1e-8
    convex_min_support: Optional[int] = None
    # ratios niceties
    ratios_max_denominator: int = 100
    ratios_q_clip: Optional[float] = None
    ratios_min_support: int = 2
    ratios_simplify_condition: bool = True


def _truthy(c: Conjecture, df: pd.DataFrame) -> bool:
    try:
        return bool(c.is_true(df))
    except Exception:
        return False


def _limit(seq: Iterable[Conjecture], k: Optional[int]) -> List[Conjecture]:
    if k is None:
        return list(seq)
    out: List[Conjecture] = []
    for i, c in enumerate(seq):
        if i >= k:
            break
        out.append(c)
    return out


def _pairs(items: Sequence[str], cap: Optional[int]) -> Iterable[Tuple[str, str]]:
    it = itertools.combinations(items, 2)
    if cap is None:
        return it
    def _bounded():
        for i, p in enumerate(it):
            if i >= cap:
                break
            yield p
    return _bounded()


def generate_with_local_pruning(
    df: pd.DataFrame,
    *,
    target: str,
    hypothesis: Optional[Predicate],
    numeric_columns: Sequence[str],
    config: MiniBatchConfig = MiniBatchConfig(),
) -> List[Conjecture]:
    """
    Generate a compact, high-quality seed set for a single hypothesis using:
      truth check → local Dalmatian → micro-Morgan → local Hazel (aggressive drop).
    Returns the locally pruned conjectures, capped by beam_k_per_H.
    """
    # Normalize None → universal predicate that returns a boolean Series aligned to df.index
    H = hypothesis if hypothesis is not None else Where(lambda d: pd.Series(True, index=d.index))
    local_pool: List[Conjecture] = []

    def _append_and_prune(batch: List[Conjecture]) -> None:
        nonlocal local_pool

        # 1) Truth filter
        batch = [c for c in batch if _truthy(c, df)]
        if not batch:
            return

        # 2) Local Dalmatian significance vs. already-kept (within same (target,direction))
        pre = list(local_pool)
        survivors = dalmatian_filter(pre + batch, df)
        survivor_set = set(map(id, survivors))
        batch = [c for c in batch if id(c) in survivor_set]
        if not batch:
            return

        # 3) Micro-Morgan over (pool ∪ batch)
        m = morgan_filter(df, pre + batch)
        merged = m.kept

        # Keep only merged conjectures (drop any that Morgan superseded)
        local_pool = merged

        # 4) Local Hazel (aggressive)
        hz = hazel_rank(df, local_pool, drop_frac=config.drop_frac_local, atol=config.hazel_atol)
        local_pool = hz.kept_sorted

        # 5) Beam cap
        if config.beam_k_per_H is not None and len(local_pool) > config.beam_k_per_H:
            local_pool = local_pool[: config.beam_k_per_H]

    # ----------------
    # generation loops
    # ----------------
    if config.context_mode == "one":
        for g in numeric_columns:
            # ratios
            if "ratios" in config.generators:
                r_batch = list(ratios(
                    df,
                    features=[g],
                    target=target,
                    hypothesis=H,
                    max_denominator=config.ratios_max_denominator,
                    direction=config.direction,
                    q_clip=config.ratios_q_clip,
                    min_support=config.ratios_min_support,
                    simplify_condition=config.ratios_simplify_condition,
                ))
                r_batch = _limit(r_batch, config.max_seeds_per_generator_per_pair)
                _append_and_prune(r_batch)

            # lp
            if "lp" in config.generators:
                lp_batch = list(lp_bounds(
                    df,
                    hypothesis=hypothesis,
                    config=LPConfig(
                        features=[g],
                        target=target,
                        direction=config.direction,
                        max_denominator=config.lp_max_denominator,
                        tol=config.lp_tol,
                        min_support=config.lp_min_support,
                    ),
                ))
                lp_batch = _limit(lp_batch, config.max_seeds_per_generator_per_pair)
                _append_and_prune(lp_batch)

            # convex
            if "convex" in config.generators:
                ch_batch = list(convex_hull(
                    df,
                    features=[g],
                    target=target,
                    hypothesis=H,
                    max_denominator=config.convex_max_denominator,
                    drop_side_facets=config.convex_drop_side_facets,
                    tol=config.convex_tol,
                    min_support=config.convex_min_support,
                ))
                ch_batch = _limit(ch_batch, config.max_seeds_per_generator_per_pair)
                _append_and_prune(ch_batch)

    elif config.context_mode == "two":
        for (g1, g2) in _pairs(numeric_columns, cap=None):
            if "lp" in config.generators:
                lp_batch = list(lp_bounds(
                    df,
                    hypothesis=hypothesis,
                    config=LPConfig(
                        features=[g1, g2],
                        target=target,
                        direction=config.direction,
                        max_denominator=config.lp_max_denominator,
                        tol=config.lp_tol,
                        min_support=config.lp_min_support,
                    ),
                ))
                lp_batch = _limit(lp_batch, config.max_seeds_per_generator_per_pair)
                _append_and_prune(lp_batch)

            if "convex" in config.generators:
                ch_batch = list(convex_hull(
                    df,
                    features=[g1, g2],
                    target=target,
                    hypothesis=H,
                    max_denominator=config.convex_max_denominator,
                    drop_side_facets=config.convex_drop_side_facets,
                    tol=config.convex_tol,
                    min_support=config.convex_min_support,
                ))
                ch_batch = _limit(ch_batch, config.max_seeds_per_generator_per_pair)
                _append_and_prune(ch_batch)

            if "ratios" in config.generators:
                for g in (g1, g2):
                    r_batch = list(ratios(
                        df,
                        features=[g],
                        target=target,
                        hypothesis=H,
                        max_denominator=config.ratios_max_denominator,
                        direction=config.direction,
                        q_clip=config.ratios_q_clip,
                        min_support=config.ratios_min_support,
                        simplify_condition=config.ratios_simplify_condition,
                    ))
                    r_batch = _limit(r_batch, config.max_seeds_per_generator_per_pair)
                    _append_and_prune(r_batch)

    else:
        raise ValueError("context_mode must be 'one' or 'two'")

    return local_pool
