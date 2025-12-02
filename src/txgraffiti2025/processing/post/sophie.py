from __future__ import annotations
from typing import Callable, Iterable, List, Optional, Tuple, Dict
import pandas as pd
import numpy as np

from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
from txgraffiti2025.forms.predicates import Predicate

__all__ = ["sophie_accept", "sophie_filter", "by_target_direction"]

# ---------------- helpers ----------------

def _mask(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
    """Boolean mask for a predicate aligned to df.index."""
    if pred is None:
        return pd.Series(True, index=df.index)
    m = pred.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)

def _target_and_direction(conj: Conjecture) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to infer (target_column_name, 'le'|'ge') from the relation.
    Returns (None, None) if not an inequality or target cannot be inferred.
    """
    rel = conj.relation
    if isinstance(rel, Le):
        direction = "le"
    elif isinstance(rel, Ge):
        direction = "ge"
    else:
        return (None, None)

    # Best-effort: extract the column name from the left side
    left = getattr(rel, "left", None)
    for attr in ("col", "column", "name"):
        if hasattr(left, attr):
            v = getattr(left, attr)
            if isinstance(v, str):
                return (v, direction)

    # fallback: repr parsing (very defensive — ok to return None/None)
    r = repr(left)
    if isinstance(r, str) and len(r) >= 2 and r[0] == r[-1] == "'":
        return (r[1:-1], direction)

    return (None, direction)

# -------------- grouping keys --------------

def by_target_direction(conj: Conjecture) -> Tuple[Optional[str], Optional[str]]:
    """
    Key function for grouping: (target_column, 'le' or 'ge').
    Non-inequalities group under (None, None).
    """
    return _target_and_direction(conj)

# -------------- core API -------------------

def sophie_accept(
    new_conj: Conjecture,
    accepted: List[Conjecture],
    df: pd.DataFrame,
) -> bool:
    """
    Accept iff the hypothesis of `new_conj` covers at least one row
    not already covered by the union of hypotheses of `accepted`.
    (Purely hypothesis-coverage based; makes no judgment about RHS tightness.)
    """
    new_cover = _mask(df, new_conj.condition)

    if accepted:
        old_union = pd.concat([_mask(df, c.condition) for c in accepted], axis=1).any(axis=1)
    else:
        old_union = pd.Series(False, index=df.index)

    return bool((new_cover & ~old_union).any())


def sophie_filter(
    df: pd.DataFrame,
    conjectures: Iterable[Conjecture],
    *,
    keyfunc: Optional[Callable[[Conjecture], object]] = None,
    min_support: int = 1,
) -> List[Conjecture]:
    """
    Greedy pass over `conjectures` in given order.
    Keep a conjecture iff its hypothesis adds new rows beyond the union of
    hypotheses of previously kept conjectures in the same group (per `keyfunc`).

    - keyfunc=None → one global coverage pool
    - keyfunc=by_target_direction → separate pools per (target, direction)

    No printing; returns the kept list.
    """
    kept: List[Conjecture] = []
    if keyfunc is None:
        pools: Dict[object, List[Conjecture]] = {None: []}
    else:
        pools = {}

    for conj in conjectures:
        # support guard
        supp = int(_mask(df, conj.condition).sum())
        if supp < min_support:
            continue

        k = keyfunc(conj) if keyfunc is not None else None
        if k not in pools:
            pools[k] = []

        if sophie_accept(conj, pools[k], df):
            pools[k].append(conj)
            kept.append(conj)

    return kept
