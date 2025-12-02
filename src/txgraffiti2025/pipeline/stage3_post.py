# src/txgraffiti2025/pipeline/stage3_post.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.forms.tidy import tidy_conjecture
from txgraffiti2025.processing.post.morgan import morgan_filter, MorganResult
from txgraffiti2025.processing.post import find_constant_ratios_for_conjecture
from txgraffiti2025.processing.post.generalize_from_constants import (
    propose_generalizations_from_constants, Generalization as ConstGen
)
from txgraffiti2025.processing.post.reciprocal_generalizer import (
    propose_generalizations_from_reciprocals
)
from txgraffiti2025.processing.post.refine_numeric import (
    refine_numeric_bounds, RefinementConfig
)
from txgraffiti2025.processing.pre.constants_cache import ConstantsCache
from txgraffiti2025.forms.pretty import format_relation, format_pred


@dataclass
class PostItem:
    conj: Conjecture
    pretty: str
    support: int
    violations: int
    touches: int
    avg_slack: float

@dataclass
class Stage3Result:
    kept: List[PostItem]
    dropped: List[tuple[Conjecture, str]]
    generalized_from_constants: List[ConstGen]
    generalized_from_reciprocals: List[Conjecture]
    refined: List[Conjecture]
    groups: dict[str, List[Conjecture]]

def _stats(df: pd.DataFrame, c: Conjecture) -> tuple[int,int,int,float]:
    applicable, holds, _ = c.check(df, auto_base=False)
    s = c.relation.slack(df).reindex(df.index)
    supp = int(applicable.sum())
    viol = int((applicable & ~holds).sum())
    touch = int((applicable & (s == 0)).sum())
    avg = float(s.loc[applicable].mean()) if supp else float("nan")
    return supp, viol, touch, avg

def run_stage3_post(
    df: pd.DataFrame,
    admitted: Sequence[Conjecture],
    *,
    constants_cache: Optional[ConstantsCache] = None,
    candidate_hypotheses: Optional[Sequence[Optional["Predicate"]]] = None,
    min_support_frac: float = 0.05,
    refinement: RefinementConfig = RefinementConfig(),
) -> Stage3Result:
    # 1) Morgan filter (dominance by hypothesis, per-conclusion)
    mres: MorganResult = morgan_filter(df, admitted)
    survivors = mres.kept

    # 2) Optional generalizations from constants / reciprocals
    gens_from_consts: List[ConstGen] = []
    gens_from_recip: List[Conjecture] = []
    if constants_cache is not None and candidate_hypotheses:
        for c in survivors:
            gens_from_consts += propose_generalizations_from_constants(
                df, c, constants_cache, candidate_hypotheses=candidate_hypotheses
            )
            gens_from_recip += propose_generalizations_from_reciprocals(
                df, c, candidate_hypotheses=candidate_hypotheses
            )

    # 3) Numeric refinements on survivors
    refined: List[Conjecture] = []
    for c in survivors:
        refined += refine_numeric_bounds(df, c, config=refinement)

    # 4) Pack pretty items & rank (touch desc, support desc, shorter first)
    items: List[PostItem] = []
    min_support = max(1, int(min_support_frac * len(df)))
    for c in survivors:
        supp, viol, touch, avg = _stats(df, c)
        if supp >= min_support and viol == 0:
            # tidy text
            text = f"{format_pred(c.condition)} ⇒ {format_relation(c.relation, unicode_ops=True, strip_ones=True)}"
            items.append(PostItem(c, tidy_conjecture(text), supp, viol, touch, avg))

    items.sort(key=lambda it: (-it.touches, -it.support, len(it.pretty)))

    return Stage3Result(
        kept=items,
        dropped=mres.dropped,
        generalized_from_constants=gens_from_consts,
        generalized_from_reciprocals=gens_from_recip,
        refined=refined,
        groups=mres.groups,
    )
