"""
Hazel heuristic (post-processing)

Rule:
    Discard the bottom 25% of conjectures by touch count (closeness).
    Keep only inequalities (Le, Ge). Sort survivors by touch desc.

Touch definition:
    touches = # of rows where |lhs - rhs| <= eps, with finite values.

Edge handling:
    - Ignore equalities and non-inequalities entirely.
    - If a conjecture errors during eval, exclude it from scoring.
    - If all exclusions lead to no valid scores, return [].
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
from txgraffiti2025.processing.utils import log_event

def _safe_touch_count(conj: Conjecture, df: pd.DataFrame, eps: float) -> Optional[int]:
    """Return touch count or None if evaluation fails or relation is not inequality."""
    rel = conj.relation
    if not isinstance(rel, (Le, Ge)):
        return None
    try:
        lhs = rel.left.eval(df)
        rhs = rel.right.eval(df)
        lhs = np.asarray(lhs, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        good = np.isfinite(lhs) & np.isfinite(rhs)
        if not np.any(good):
            return 0
        diff = np.abs(lhs[good] - rhs[good])
        return int((diff <= eps).sum())
    except Exception:
        return None

def hazel_filter(conjectures: List[Conjecture], df: pd.DataFrame, eps: float = 1e-6) -> List[Conjecture]:
    """
    Keep only conjectures in the top 75% by touch count (i.e., drop bottom quartile).
    Only considers inequalities (Le, Ge). Survivors sorted by touch desc.
    """
    if not conjectures:
        return []

    # Filter to inequalities up front
    ineqs = [c for c in conjectures if isinstance(c.relation, (Le, Ge))]
    if not ineqs:
        return []

    scored: List[Tuple[Conjecture, int]] = []
    for c in ineqs:
        t = _safe_touch_count(c, df, eps)
        if t is not None:
            scored.append((c, t))

    if not scored:
        return []

    touches = np.array([t for _, t in scored], dtype=float)
    cutoff = np.percentile(touches, 25)  # bottom quartile threshold

    # Strictly drop bottom quartile (strict > to avoid keeping all on ties at cutoff)
    survivors = [(c, t) for (c, t) in scored if t > cutoff]

    # If strict cut drops everything (e.g., all equal touches), keep them all
    if not survivors:
        survivors = scored

    survivors.sort(key=lambda x: x[1], reverse=True)
    log_event(f"Hazel: kept {len(survivors)}/{len(ineqs)} (strictly above 25th percentile unless tied)")
    return [c for c, _ in survivors]
