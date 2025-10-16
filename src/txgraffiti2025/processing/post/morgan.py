# """
# Morgan heuristic (post-processing):

# Group conjectures by *conclusion* (same Relation/expr structure),
# and within each group keep only the conjecture(s) whose hypothesis
# is most general, i.e., holds on the most rows in the dataset.
# Tie-breakers: smaller predicate complexity, then name.

# This is domain-agnostic and relies on txgraffiti2025.forms abstractions.
# """

# from __future__ import annotations
# from typing import Dict, List, Tuple, Optional
# import numpy as np
# import pandas as pd

# from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.forms.utils import (
#     Expr, Const, ColumnTerm, LinearForm, BinOp, UnaryOp,
# )
# from txgraffiti2025.processing.utils import log_event


# # ---------------------------
# # Signatures for conclusions
# # ---------------------------

# def _ufunc_name(fn) -> str:
#     # Try to identify common numpy functions; fallback to generic name
#     name = getattr(fn, "__name__", None)
#     if name:
#         return name
#     mod = getattr(fn, "__module__", "")
#     return f"{mod}.fn"

# def _expr_sig(e: Expr) -> str:
#     # Recursively stringify expressions to a stable signature
#     if isinstance(e, Const):
#         return f"Const({e.value})"
#     if isinstance(e, ColumnTerm):
#         return f"Col({e.col})"
#     if isinstance(e, LinearForm):
#         terms = ";".join(f"{a}:{c}" for (a,c) in e.terms)
#         return f"Linear(a0={e.intercept};{terms})"
#     if isinstance(e, BinOp):
#         # try to map ufunc to symbol (best-effort)
#         fn = _ufunc_name(e.fn)
#         return f"Bin({fn},{_expr_sig(e.left)},{_expr_sig(e.right)})"
#     if isinstance(e, UnaryOp):
#         fn = _ufunc_name(e.fn)
#         return f"Una({fn},{_expr_sig(e.arg)})"
#     # Fallback (unknown Expr subtype)
#     return repr(e)

# def _relation_sig(rel) -> Tuple[str, str, str, Optional[float]]:
#     if isinstance(rel, Le):
#         return ("Le", _expr_sig(rel.left), _expr_sig(rel.right), None)
#     if isinstance(rel, Ge):
#         return ("Ge", _expr_sig(rel.left), _expr_sig(rel.right), None)
#     if isinstance(rel, Eq):
#         return ("Eq", _expr_sig(rel.left), _expr_sig(rel.right), rel.tol)
#     # Unknown relation type → treat as its repr
#     return (rel.__class__.__name__, repr(rel), "", None)

# def _conclusion_key(conj: Conjecture) -> Tuple[str, str, str, Optional[float]]:
#     return _relation_sig(conj.relation)


# # ---------------------------
# # Hypothesis generality
# # ---------------------------

# def _coverage(conj: Conjecture, df: pd.DataFrame) -> int:
#     if conj.condition is None:
#         return int(len(df))
#     m = conj.condition.mask(df)
#     # Ensure alignment and boolean dtype
#     return int(pd.Series(m, index=df.index).astype(bool).sum())

# def _predicate_size(pred: Optional[Predicate]) -> int:
#     """
#     A coarse complexity metric used as tie-breaker:
#     - None (global) = 0
#     - Otherwise 1 by default; compound predicates can override if needed.
#     We don't introspect arbitrary user predicates; this is a best-effort.
#     """
#     if pred is None:
#         return 0
#     # Heuristic: try to walk known dataclasses for And/Or/Not pattern
#     # Else default to 1
#     name = getattr(pred, "name", "")
#     if name in ("C_and", "C_or"):
#         # try attributes a,b
#         a = getattr(pred, "a", None)
#         b = getattr(pred, "b", None)
#         return 1 + _predicate_size(a) + _predicate_size(b)
#     if name == "C_not":
#         a = getattr(pred, "a", None)
#         return 1 + _predicate_size(a)
#     return 1


# # ---------------------------
# # Morgan core
# # ---------------------------

# def morgan_generalize(conjectures: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
#     """
#     For each group of conjectures that share the *same conclusion*,
#     keep only the one(s) whose hypothesis is most general (max coverage),
#     tie-breaking by smaller predicate_size, then by name lexicographically.
#     """
#     if not conjectures:
#         return []

#     # Group by conclusion key
#     groups: Dict[Tuple[str, str, str, Optional[float]], List[Conjecture]] = {}
#     for c in conjectures:
#         k = _conclusion_key(c)
#         groups.setdefault(k, []).append(c)

#     kept: List[Conjecture] = []
#     for key, bucket in groups.items():
#         # compute coverage + complexity
#         scored = []
#         for c in bucket:
#             cov = _coverage(c, df)
#             sz  = _predicate_size(c.condition)
#             scored.append((cov, -sz, c.name, c))  # -size so higher sort wins on smaller size

#         # sort: max coverage, then smaller predicate, then name
#         scored.sort(reverse=True)
#         best_cov = scored[0][0]
#         best_size = -scored[0][1]

#         # Keep *all* with equal best coverage; tie-break by minimal size
#         candidates = [t[-1] for t in scored if t[0] == best_cov]
#         # Among equal coverage, keep those with minimal predicate size
#         min_size = min(_predicate_size(c.condition) for c in candidates)
#         winners = [c for c in candidates if _predicate_size(c.condition) == min_size]

#         if len(winners) < len(bucket):
#             log_event(f"Morgan: reduced group {key[0]} to {len(winners)} from {len(bucket)} (best coverage={best_cov})")

#         kept.extend(winners)

#     return kept


"""
Morgan post-processing filter for conjectures.

Rule
----
Group conjectures by *conclusion* (the Relation). Within each group, keep only
those whose hypothesis is *most general* on the provided DataFrame:
- If H1 ⊂ H2 (strict subset under df), drop (H1 -> conclusion).
- If H1 == H2 (same mask), keep one (prefer the simplest/shortest condition name).
- If neither is subset of the other (overlapping or disjoint), keep both.

We canonicalize conclusions using the pretty formatter (strip 1·, no quotes, ASCII ops)
to ensure cosmetically-different but semantically-equal relations land in the same group.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.pretty import format_relation, format_pred


# ---------- helpers ----------

def _relation_key(conj: Conjecture) -> str:
    """
    Canonical text key for a Relation:
      - ASCII ops (>=, <=)
      - coefficients with 1 suppressed
      - no quotes around names
    Example: "independence_number >= 1/2 * matching_number"
    """
    # unicode_ops=False => ASCII, strip_ones=True keeps x not 1*x
    return format_relation(conj.relation, unicode_ops=False, strip_ones=True)


def _cond_mask(df: pd.DataFrame, cond: Optional[Predicate]) -> pd.Series:
    if cond is None:
        return pd.Series(True, index=df.index)
    m = cond.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)


def _is_subset(a: pd.Series, b: pd.Series) -> bool:
    """a ⊆ b under df (boolean Series aligned to df.index)."""
    return not (a & ~b).any()


def _is_strict_subset(a: pd.Series, b: pd.Series) -> bool:
    """a ⊂ b under df."""
    return _is_subset(a, b) and (b & ~a).any()


def _complexity_of_condition(conj: Conjecture) -> Tuple[int, int, int]:
    """
    Heuristic 'simplicity' of a condition: shorter pretty name, fewer characters, fewer parens.
    Used to break ties when masks are identical.
    """
    s = format_pred(conj.condition, unicode_ops=True)
    return (len(s.split("∧")), len(s), s.count("(") + s.count(")"))


# ---------- main API ----------

@dataclass
class MorganResult:
    kept: List[Conjecture]
    dropped: List[Tuple[Conjecture, str]]  # (conj, reason)
    groups: Dict[str, List[Conjecture]]    # relation_key -> original group


def morgan_filter(
    df: pd.DataFrame,
    conjectures: Iterable[Conjecture],
) -> MorganResult:
    """
    Apply Morgan filtering:
      - Group by canonicalized relation (conclusion)
      - Within each group, keep only hypotheses that are not strictly subsumed by another.

    Parameters
    ----------
    df : pd.DataFrame
        Data against which to evaluate hypothesis masks.
    conjectures : Iterable[Conjecture]
        Conjectures to filter.

    Returns
    -------
    MorganResult
        kept   : surviving conjectures
        dropped: removed conjectures with reason
        groups : mapping from relation_key -> list of original conjectures (for introspection)
    """
    # 1) bucket by conclusion
    groups: Dict[str, List[Conjecture]] = {}
    for c in conjectures:
        key = _relation_key(c)
        groups.setdefault(key, []).append(c)

    kept: List[Conjecture] = []
    dropped: List[Tuple[Conjecture, str]] = []

    # 2) for each conclusion, remove strictly more specific hypotheses
    for key, grp in groups.items():
        if len(grp) == 1:
            kept.extend(grp)
            continue

        # precompute masks
        masks = [(_cond_mask(df, c.condition), c) for c in grp]

        # Find maximal elements under ⊆ (keep those; drop strict subsets)
        survivors: List[Tuple[pd.Series, Conjecture]] = []
        for i, (mi, ci) in enumerate(masks):
            dominated = False
            for j, (mj, cj) in enumerate(masks):
                if i == j:
                    continue
                if _is_strict_subset(mi, mj):
                    dominated = True
                    break
            if not dominated:
                survivors.append((mi, ci))

        # If masks are identical among survivors, deduplicate by simplicity
        # Build equivalence classes by mask equality
        used = [False] * len(survivors)
        for i, (mi, ci) in enumerate(survivors):
            if used[i]:
                continue
            same = [i]
            for j, (mj, cj) in enumerate(survivors):
                if i != j and not (mi ^ mj).any():  # identical
                    same.append(j)
            # choose simplest representative
            reps = [survivors[k][1] for k in same]
            reps.sort(key=_complexity_of_condition)
            kept.append(reps[0])
            # mark others as dropped (duplicate mask for same conclusion)
            for k in same[1:]:
                used[k] = True
                dropped.append((survivors[k][1], "duplicate hypothesis mask (same conclusion)"))

        # Also mark those dominated earlier as dropped
        dominated_set = set(id(ci) for (m, ci) in masks) - set(id(c) for _, c in survivors)
        for m, c in masks:
            if id(c) in dominated_set:
                dropped.append((c, "strictly more specific hypothesis (dominated)"))

    return MorganResult(kept=kept, dropped=dropped, groups=groups)
