# # src/txgraffiti2025/processing/post/intercept_generalizer.py

# """
# Intercept generalizer (post-processing).

# Propose generalizations for inequalities that have an additive *intercept* on the
# RHS, by swapping a numeric Const intercept with:
#   - an expression pulled from a precomputed constants cache (ratio-style or numeric),
#   - a user-provided candidate intercept expression,
#   - or a "relaxer" Z (we try both +Z and -Z), which vanishes on the base class.

# Only conjectures that are TRUE on their candidate hypothesis are returned.
# Optionally require that the new hypothesis is a superset of the base hypothesis.

# This module is defensive about cache shape: the cache adapter works whether
# key_to_constants maps to an iterable of items or to a container with a ".constants"
# (or similar) attribute.
# """

# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Iterable, List, Optional, Any

# import numpy as np
# import pandas as pd

# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
# from txgraffiti2025.forms.utils import Expr, Const, to_expr, BinOp
# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.processing.pre.constants_cache import ConstantsCache  # cache type
# from txgraffiti2025.processing.post.generalize_from_constants import Generalization


# # -----------------------------
# # Small helpers
# # -----------------------------

# def _mask(df: pd.DataFrame, cond: Optional[Predicate]) -> pd.Series:
#     """Boolean mask for a predicate, aligned to df.index."""
#     if cond is None:
#         return pd.Series(True, index=df.index)
#     return cond.mask(df).reindex(df.index, fill_value=False).astype(bool)

# def _subset(df: pd.DataFrame, A: Optional[Predicate], B: Optional[Predicate]) -> bool:
#     """A ⊆ B under df."""
#     if B is None:
#         return True
#     a = _mask(df, A)
#     b = _mask(df, B)
#     return not (a & ~b).any()

# def _extract_intercept(expr: Expr) -> Optional[float]:
#     """Return float value if expr is a Const; otherwise None."""
#     if isinstance(expr, Const):
#         try:
#             return float(expr.value)
#         except Exception:
#             return None
#     return None

# def _make_relation(kind: str, left: Expr, right: Expr):
#     return Ge(left, right) if kind == "Ge" else Le(left, right)

# def _relation_kind(conj: Conjecture) -> Optional[str]:
#     r = conj.relation
#     if isinstance(r, Ge):
#         return "Ge"
#     if isinstance(r, Le):
#         return "Le"
#     return None


# # -----------------------------
# # Cache adapter (robust)
# # -----------------------------

# def _constants_for(cache: ConstantsCache, hyp: Optional[Predicate]) -> list:
#     """
#     Return a *list* of constant items for a hypothesis, tolerant of multiple cache shapes.

#     Supports:
#       - key_to_constants[key] being an iterable (list/tuple/set) of items
#       - key_to_constants[key] being a container object with any of:
#             .constants, .ratios, .items, .all, .all_items
#       - otherwise returns []
#     """
#     if cache is None:
#         return []

#     key = cache.hyp_to_key.get(repr(hyp))
#     if key is None:
#         return []

#     entry = cache.key_to_constants.get(key)
#     if entry is None:
#         return []

#     if isinstance(entry, (list, tuple, set)):
#         return list(entry)

#     for attr in ("constants", "ratios", "items", "all", "all_items"):
#         if hasattr(entry, attr):
#             try:
#                 seq = getattr(entry, attr)
#                 return list(seq) if isinstance(seq, Iterable) else []
#             except Exception:
#                 pass

#     try:
#         return list(entry)  # may raise TypeError
#     except TypeError:
#         return []


# # -----------------------------
# # Core: propose generalizations
# # -----------------------------

# @dataclass
# class InterceptCandidate:
#     """Represents a candidate c0 to replace a numeric intercept."""
#     value: float
#     text: str = ""        # human-readable explanation (optional)
#     support: int = 0      # number of rows used to establish constancy (optional)


# def propose_generalizations_from_intercept(
#     df: pd.DataFrame,
#     conj: Conjecture,
#     cache: ConstantsCache | None,
#     *,
#     candidate_hypotheses: Iterable[Optional[Predicate]],
#     candidate_intercepts: Iterable[Expr] | None = None,
#     relaxers_Z: Iterable[Expr] | None = None,
#     require_superset: bool = True,
#     tol: float = 1e-9,
# ) -> List[Generalization]:
#     """
#     Try to generalize a bound y (≤/≥) [ ... ± c ] by replacing a numeric intercept c
#     on the RHS with:
#       - an Expr from the cache (preferred) or a numeric value from the cache,
#       - a user-provided intercept Expr,
#       - ±Z for each provided relaxer Z.

#     Only returns Generalization objects whose new_conjecture is TRUE on its
#     hypothesis (and, if require_superset, the hypothesis is a superset of the base).
#     """
#     kind = _relation_kind(conj)
#     if kind is None:
#         return []

#     # We only consider RHS that has an explicit numeric Const child via + or -.
#     rel = conj.relation
#     left, right = rel.left, rel.right

#     # locate a Const leaf to replace
#     intercept_val: Optional[float] = None
#     replace_left_child = False
#     if isinstance(right, BinOp) and right.fn in (np.add, np.subtract):
#         # prefer replacing the Const leaf if present on either side
#         if _extract_intercept(right.left) is not None:
#             intercept_val = _extract_intercept(right.left)
#             replace_left_child = True
#         elif _extract_intercept(right.right) is not None:
#             intercept_val = _extract_intercept(right.right)
#             replace_left_child = False
#     else:
#         # Entire RHS might be Const (rare); allow full replacement.
#         intercept_val = _extract_intercept(right)

#     if intercept_val is None:
#         # Nothing to replace → return empty (tests expect this behavior)
#         return []

#     proposals: List[Generalization] = []

#     # For each target hypothesis, build a bag of candidate intercept *expressions*
#     for Hnew in candidate_hypotheses:
#         if require_superset and conj.condition is not None and not _subset(df, conj.condition, Hnew):
#             continue

#         candidate_exprs: List[Expr] = []

#         # 1) Cache-backed: prefer an 'expr' field (ratio), else numeric 'value'
#         if cache is not None:
#             for item in _constants_for(cache, Hnew):
#                 expr_from_cache = None
#                 if hasattr(item, "expr"):
#                     expr_from_cache = getattr(item, "expr")
#                 elif isinstance(item, dict) and "expr" in item:
#                     expr_from_cache = item["expr"]

#                 if expr_from_cache is not None:
#                     candidate_exprs.append(expr_from_cache)
#                     continue

#                 # fallback numeric value
#                 value = None
#                 if isinstance(item, tuple) and len(item) == 2:
#                     value = item[1]
#                 elif isinstance(item, dict):
#                     value = item.get("value")
#                 else:
#                     for v_attr in ("value", "val", "const", "c"):
#                         if hasattr(item, v_attr):
#                             value = getattr(item, v_attr)
#                             break
#                 try:
#                     v_float = float(value) if value is not None else None
#                 except Exception:
#                     v_float = None
#                 if v_float is not None:
#                     candidate_exprs.append(Const(v_float))

#         # 2) User-provided intercept candidates (expressions)
#         if candidate_intercepts:
#             for ci in candidate_intercepts:
#                 if isinstance(ci, Expr):
#                     candidate_exprs.append(ci)

#         # 3) Relaxers Z: try both +Z and -Z
#         if relaxers_Z:
#             for Z in relaxers_Z:
#                 if isinstance(Z, Expr):
#                     candidate_exprs.append(Z)
#                     candidate_exprs.append(-Z)

#         # Deduplicate by repr to avoid producing identical candidates repeatedly
#         seen = set()
#         uniq_exprs = []
#         for e in candidate_exprs:
#             k = repr(e)
#             if k not in seen:
#                 seen.add(k)
#                 uniq_exprs.append(e)

#         # Build new RHS for each candidate expression and test truth on Hnew
#         for repl_expr in uniq_exprs:
#             if isinstance(right, BinOp) and right.fn in (np.add, np.subtract):
#                 if replace_left_child:
#                     new_right = BinOp(right.fn, repl_expr, right.right)
#                 else:
#                     new_right = BinOp(right.fn, right.left, repl_expr)
#             else:
#                 new_right = repl_expr  # full replacement

#             new_rel = _make_relation(kind, left, new_right)
#             new_conj = Conjecture(relation=new_rel, condition=Hnew, name=conj.name)

#             if new_conj.is_true(df):
#                 proposals.append(Generalization(
#                     from_conjecture=conj,
#                     new_conjecture=new_conj,
#                     reason="intercept swap (cache)"
#                 ))

#     return proposals


# src/txgraffiti2025/processing/post/intercept_generalizer.py

from __future__ import annotations

"""
Intercept generalizer (post-processing).

Propose generalizations for inequalities that have an additive *intercept* on the
RHS, by swapping a numeric Const intercept with:
  - an expression pulled from a precomputed constants cache (ratio-style or numeric),
  - a user-provided candidate intercept expression,
  - or a "relaxer" Z (we try both +Z and -Z), optionally requiring Z≈0 on the base class.

Only conjectures that are TRUE on their candidate hypothesis are returned.
You can require the new hypothesis to be a (strict) superset of the base hypothesis.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.utils import Expr, Const, to_expr, BinOp
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.pre.constants_cache import ConstantsCache  # type hint only
from txgraffiti2025.processing.post.generalize_from_constants import Generalization

__all__ = [
    "InterceptCandidate",
    "propose_generalizations_from_intercept",
]

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mask(df: pd.DataFrame, cond: Optional[Predicate]) -> pd.Series:
    if cond is None:
        return pd.Series(True, index=df.index)
    return cond.mask(df).reindex(df.index, fill_value=False).astype(bool)

def _subset(df: pd.DataFrame, A: Optional[Predicate], B: Optional[Predicate]) -> bool:
    """A ⊆ B."""
    if B is None:
        return True
    a = _mask(df, A)
    b = _mask(df, B)
    return not (a & ~b).any()

def _strict_superset(df: pd.DataFrame, A: Optional[Predicate], B: Optional[Predicate]) -> bool:
    """B ⊃ A (strict)."""
    if not _subset(df, A, B):
        return False
    a = _mask(df, A)
    b = _mask(df, B)
    return (b & ~a).any()

def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _expr_finite_on_slice(df: pd.DataFrame, expr: Expr, mask: pd.Series, *, min_support: int = 1) -> bool:
    try:
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            vals = _finite(expr.eval(df))
    except Exception:
        return False
    vals = vals[mask].dropna()
    return vals.size >= int(min_support)

def _vanishes_on_base(df: pd.DataFrame, expr: Expr, mask: pd.Series, *, atol: float = 1e-9) -> bool:
    try:
        vals = _finite(expr.eval(df))[mask].dropna()
    except Exception:
        return False
    if vals.empty:
        return False
    return bool(np.all(np.isclose(vals.values, 0.0, atol=atol)))

def _relation_kind(conj: Conjecture) -> Optional[str]:
    r = conj.relation
    if isinstance(r, Ge):
        return "Ge"
    if isinstance(r, Le):
        return "Le"
    return None

def _extract_intercept_from_rhs(rhs: Expr) -> tuple[Optional[float], Optional[str]]:
    """
    Look for a top-level + or - with a Const leaf: return (value, side)
    where side is "left" or "right" indicating which child is the Const.
    If rhs is itself Const, return (value, None).
    """
    if isinstance(rhs, Const):
        try:
            return float(rhs.value), None
        except Exception:
            return None, None

    if isinstance(rhs, BinOp) and rhs.fn in (np.add, np.subtract):
        if isinstance(rhs.left, Const):
            try:
                return float(rhs.left.value), "left"
            except Exception:
                return None, None
        if isinstance(rhs.right, Const):
            try:
                return float(rhs.right.value), "right"
            except Exception:
                return None, None
    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# Cache adapter (robust)
# ─────────────────────────────────────────────────────────────────────────────

def _constants_for(cache: Optional[ConstantsCache], hyp: Optional[Predicate]) -> list:
    """
    Return a list of cache items for the hypothesis; tolerant to several shapes.
    Each item may have `.expr` (preferred) or a numeric attribute ('value'/'val'/...).
    """
    if cache is None:
        return []
    key = cache.hyp_to_key.get(repr(hyp)) if hasattr(cache, "hyp_to_key") else None
    entry = cache.key_to_constants.get(key) if (key is not None and hasattr(cache, "key_to_constants")) else None
    if entry is None:
        return []

    # entry may be an iterable or a container with a common attribute
    if isinstance(entry, (list, tuple, set)):
        return list(entry)
    for attr in ("constants", "ratios", "items", "all", "all_items"):
        if hasattr(entry, attr):
            seq = getattr(entry, attr)
            try:
                return list(seq)
            except Exception:
                break
    try:
        return list(entry)  # as last resort
    except Exception:
        return []

def _item_to_expr(item: Any) -> Optional[Expr]:
    """Best-effort conversion of a cache item to an Expr (Const or provided expr)."""
    e = getattr(item, "expr", None)
    if e is not None and isinstance(e, Expr):
        return e
    if isinstance(item, dict) and isinstance(item.get("expr"), Expr):
        return item["expr"]

    # numeric fallback
    val = None
    if isinstance(item, dict):
        val = item.get("value", item.get("val"))
    else:
        for attr in ("value", "val", "const", "c"):
            if hasattr(item, attr):
                val = getattr(item, attr)
                break
    if val is None:
        return None
    try:
        return Const(float(val))
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InterceptCandidate:
    """Represents a candidate intercept expression c0 to swap into RHS."""
    expr: Expr
    text: str = ""
    support: int = 0

# ─────────────────────────────────────────────────────────────────────────────
# Core: propose generalizations
# ─────────────────────────────────────────────────────────────────────────────

def propose_generalizations_from_intercept(
    df: pd.DataFrame,
    conj: Conjecture,
    cache: Optional[ConstantsCache],
    *,
    candidate_hypotheses: Iterable[Optional[Predicate]],
    candidate_intercepts: Optional[Iterable[Expr]] = None,
    relaxers_Z: Optional[Iterable[Expr]] = None,
    require_superset: bool = True,
    require_strict_superset: bool = True,
    relaxers_must_vanish_on_base: bool = True,
    relaxer_zero_atol: float = 1e-9,
    min_finite_support: int = 4,
) -> List[Generalization]:
    """
    If RHS contains an explicit numeric constant at top-level (+/- or pure Const),
    try replacing that *intercept* with expressions pulled from:
      - cache items (expr or numeric),
      - user-provided intercept expressions,
      - relaxers Z (optionally requiring Z≈0 on the base; we test ±Z).

    A proposal is kept iff the resulting Conjecture is TRUE on the candidate
    hypothesis. If `require_superset=True`, we also require the new hypothesis
    to be a superset (and, if `require_strict_superset=True`, a strict superset)
    of the original conjecture’s condition.
    """
    kind = _relation_kind(conj)
    if kind is None:
        return []

    base = conj.condition
    lhs = conj.relation.left
    rhs = conj.relation.right

    # Detect a numeric intercept to replace
    c_val, side = _extract_intercept_from_rhs(rhs)
    if c_val is None:
        # No replaceable numeric intercept at top-level
        return []

    proposals: List[Generalization] = []

    # Assemble candidate intercept expressions
    def gather_candidates(Hnew: Optional[Predicate]) -> List[Expr]:
        exprs: List[Expr] = []

        # 1) Cache-backed
        for item in _constants_for(cache, Hnew):
            e = _item_to_expr(item)
            if isinstance(e, Expr):
                exprs.append(e)

        # 2) User-provided
        if candidate_intercepts:
            for e in candidate_intercepts:
                if isinstance(e, Expr):
                    exprs.append(e)

        # 3) Relaxers ±Z
        if relaxers_Z:
            for Z in relaxers_Z:
                if not isinstance(Z, Expr):
                    continue
                if relaxers_must_vanish_on_base and base is not None:
                    if not _vanishes_on_base(df, Z, _mask(df, base), atol=relaxer_zero_atol):
                        continue
                exprs.append(Z)
                exprs.append(-Z)

        # Dedup by textual key
        uniq, seen = [], set()
        for e in exprs:
            k = repr(e)
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        return uniq

    # Build a new RHS with the intercept swapped
    def build_rhs_with_intercept(repl: Expr) -> Expr:
        if side is None:
            # entire rhs was Const(c): replace fully
            return repl
        # rhs is BinOp(add/subtract) with a Const child at `side`
        if isinstance(rhs, BinOp) and rhs.fn in (np.add, np.subtract):
            if side == "left":
                return BinOp(rhs.fn, repl, rhs.right)
            else:
                return BinOp(rhs.fn, rhs.left, repl)
        # Fallback (should not happen due to extraction guard)
        return rhs

    # Try each candidate hypothesis + candidate intercept expr
    for Hnew in candidate_hypotheses:
        if require_superset and base is not None:
            ok = _strict_superset(df, base, Hnew) if require_strict_superset else _subset(df, base, Hnew)
            if not ok:
                continue

        cand_mask = _mask(df, Hnew)
        cands = gather_candidates(Hnew)

        for e in cands:
            # Must be finite on the slice
            if not _expr_finite_on_slice(df, e, cand_mask, min_support=min_finite_support):
                continue

            new_rhs = build_rhs_with_intercept(e)
            new_rel = Ge(lhs, new_rhs) if kind == "Ge" else Le(lhs, new_rhs)
            new_conj = Conjecture(relation=new_rel, condition=Hnew, name=conj.name)

            if new_conj.is_true(df):
                proposals.append(Generalization(
                    from_conjecture=conj,
                    new_conjecture=new_conj,
                    reason="intercept swap",
                    witness_superset=Hnew,
                    meta={"replaced_const": c_val, "candidate_repr": repr(e)}
                ))

    # Dedup by (repr(new_conj), hypothesis mask)
    uniq: List[Generalization] = []
    seen: set[tuple[str, tuple]] = set()
    for g in proposals:
        key = (repr(g.new_conjecture), tuple(_mask(df, g.witness_superset).values) if g.witness_superset is not None else ())
        if key not in seen:
            seen.add(key)
            uniq.append(g)

    # Light ranking: prefer simpler RHS (short repr), then hypothesis size (bigger first)
    uniq.sort(key=lambda G: (len(repr(G.new_conjecture.relation.right)), -_mask(df, G.witness_superset).sum()))
    return uniq
