"""
Use precomputed constants to propose generalized ratio conjectures:

If a conjecture is
    H0 ⇒ target (≤/≥) c * feature

and under H0 there exists a constant ratio expression R = (A+a)/(B+b)
with value ≈ c, we propose replacing c by R and testing on candidate
superset hypotheses.

Extras:
- Reject coefficients that reference the *target* column (avoid tautologies).
- Simplify (R * feature) if R is (num / feature) → `num`.
- Skip if the simplified RHS is structurally identical to the target.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import to_expr, Expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate

from txgraffiti2025.forms.expr_utils import (
    expr_depends_on,
    simplify_coeff_times_feature,
    structurally_equal,
)

from txgraffiti2025.processing.pre.constants_cache import (
    ConstantsCache,
    constants_matching_coeff,
)

# Reuse the pattern extractor from your earlier module
from txgraffiti2025.processing.post.constant_ratios import _extract_ratio_pattern


def _mask(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
    if pred is None:
        return pd.Series(True, index=df.index)
    return pred.mask(df).reindex(df.index, fill_value=False).astype(bool)

def _is_subset(df: pd.DataFrame, A: Optional[Predicate], B: Optional[Predicate]) -> bool:
    a = _mask(df, A)
    b = _mask(df, B)
    return not (a & ~b).any()


@dataclass
class Generalization:
    from_conjecture: Conjecture
    new_conjecture: Conjecture
    reason: str


def propose_generalizations_from_constants(
    df: pd.DataFrame,
    conj: Conjecture,
    cache: ConstantsCache,
    *,
    candidate_hypotheses: Sequence[Optional[Predicate]],
    atol: float = 1e-9,
) -> List[Generalization]:
    """
    Propose generalized conjectures by swapping constant `c` with cached ratio expressions.

    Safeguards:
    - Disallow coefficients that depend on the target column (prevent γ ≥ (γ/α)·α).
    - Simplify (coeff * feature) via division cancellation when possible.
    - Drop proposals that reduce to a tautology γ ≥ γ or γ ≤ γ structurally.
    """
    patt = _extract_ratio_pattern(conj)
    if patt is None:
        return []

    # constants under conj.condition matching the coefficient
    candidates = constants_matching_coeff(cache, conj.condition, patt.coefficient, atol=atol)
    if not candidates:
        return []

    tgt_expr = to_expr(patt.target)
    gens: List[Generalization] = []

    for cr in candidates:
        coeff_expr: Expr = cr.expr

        # 1) Reject if coefficient depends on the target column (avoid self-reference).
        if expr_depends_on(coeff_expr, patt.target):
            continue

        # 2) Build (and simplify) RHS = coeff * feature.
        rhs_expr = simplify_coeff_times_feature(coeff_expr, patt.feature)

        # 3) If RHS structurally equals the target, skip (tautology).
        if structurally_equal(rhs_expr, tgt_expr):
            continue

        # 4) Construct the relation with simplified RHS.
        if patt.kind == "Ge":
            rel = Ge(tgt_expr, rhs_expr)
        else:
            rel = Le(tgt_expr, rhs_expr)

        # 5) Try on each candidate superset hypothesis.
        for Hsup in candidate_hypotheses:
            if conj.condition is not None and not _is_subset(df, conj.condition, Hsup):
                continue
            candidate = Conjecture(
                relation=rel,
                condition=Hsup,
                name=f"gen_from_const_{patt.kind}_{patt.target}_vs_{patt.feature}",
            )
            if candidate.is_true(df):
                why = "replaced constant by structural ratio; simplified RHS"
                gens.append(Generalization(from_conjecture=conj, new_conjecture= candidate, reason=why))

    return gens
