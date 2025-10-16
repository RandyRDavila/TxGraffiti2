# src/txgraffiti2025/processing/post/intercept_generalizer.py

"""
Intercept generalizer (post-processing).

This module proposes generalizations for bounds with an additive constant by
reusing precomputed per-hypothesis constants from the constants cache. It is
defensive about the cache shape: it works whether key_to_constants maps to a
plain iterable or to a container object (e.g., HypothesisConstants).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Any

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.utils import Expr, Const, to_expr, BinOp
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.pre.constants_cache import ConstantsCache  # cache type
from txgraffiti2025.processing.post.generalize_from_constants import Generalization


# -----------------------------
# Cache adapter (robust)
# -----------------------------

def _constants_for(cache: ConstantsCache, hyp: Optional[Predicate]) -> list:
    """
    Return a *list* of constant items for a hypothesis, tolerant of multiple cache shapes.

    Supports:
      - key_to_constants[key] being an iterable (list/tuple/set) of items
      - key_to_constants[key] being a container object with any of:
            .constants, .ratios, .items, .all, .all_items
      - otherwise returns []
    """
    if cache is None:
        return []

    key = cache.hyp_to_key.get(repr(hyp))
    if key is None:
        return []

    entry = cache.key_to_constants.get(key)
    if entry is None:
        return []

    # Already an iterable of items
    if isinstance(entry, (list, tuple, set)):
        return list(entry)

    # Known container-like attributes (be generous)
    for attr in ("constants", "ratios", "items", "all", "all_items"):
        if hasattr(entry, attr):
            try:
                seq = getattr(entry, attr)
                return list(seq) if isinstance(seq, Iterable) else []
            except Exception:
                pass

    # Last resort: if entry itself is iterable
    try:
        return list(entry)  # may raise TypeError
    except TypeError:
        return []


# -----------------------------
# Core: propose generalizations
# -----------------------------

@dataclass
class InterceptCandidate:
    """Represents a candidate c0 to replace a numeric intercept."""
    value: float
    text: str = ""        # human-readable explanation (optional)
    support: int = 0      # number of rows used to establish constancy (optional)


def _extract_intercept(expr: Expr) -> Optional[float]:
    """If expr is a pure numeric Const, return its float value; else None."""
    if isinstance(expr, Const):
        try:
            return float(expr.value)
        except Exception:
            return None
    return None


def _make_relation(kind: str, left: Expr, right: Expr):
    return Ge(left, right) if kind == "Ge" else Le(left, right)


def _relation_kind(conj: Conjecture) -> Optional[str]:
    r = conj.relation
    if isinstance(r, Ge):
        return "Ge"
    if isinstance(r, Le):
        return "Le"
    return None


def propose_generalizations_from_intercept(
    df: pd.DataFrame,
    conj: Conjecture,
    cache: ConstantsCache,
    *,
    candidate_hypotheses: Iterable[Optional[Predicate]],
    tol: float = 1e-9,
) -> List[Generalization]:
    """
    Try to generalize a ratio-style inequality that includes an additive *constant* term
    by replacing a numeric intercept with a constant expression known to hold on a
    more general hypothesis (from the constants cache).

    Only returns Generalization objects that:
      - are STRICT superset hypotheses, and
      - are true on their (larger) hypothesis.

    If nothing applicable is found, returns [].
    """
    kind = _relation_kind(conj)
    if kind is None:
        return []

    # We only consider relations of the form target (≤/≥) [linear/ratio expr].
    # Extract RHS and look for a pure numeric intercept Const to replace.
    rel = conj.relation
    left = rel.left
    right = rel.right

    # Quick scan: if RHS is BinOp add/sub, try to find a Const child
    intercept_val: Optional[float] = None
    if isinstance(right, BinOp) and right.fn in (np.add, np.subtract):
        # Gather leaves
        leaves = [right.left, right.right]
        for leaf in leaves:
            val = _extract_intercept(leaf)
            if val is not None:
                intercept_val = val
                break
    else:
        # Entire RHS might be just a Const (rare in this pipeline, but safe)
        intercept_val = _extract_intercept(right)

    if intercept_val is None:
        return []  # nothing to generalize here

    # Build candidate intercept constants from cache for more general hypotheses
    out: List[Generalization] = []
    for Hnew in candidate_hypotheses:
        if Hnew is conj.condition:
            continue  # skip same hypothesis; caller will check superset elsewhere

        const_items = _constants_for(cache, Hnew)
        if not const_items:
            continue

        # const_items can be any shape; we look for items that expose either:
        #   .name/.text and .value (float-like), or are (name, value) tuples/dicts.
        for item in const_items:
            # extract (name, value)
            name, value = None, None
            if isinstance(item, tuple) and len(item) == 2:
                name, value = item[0], item[1]
            elif isinstance(item, dict):
                name = item.get("name") or item.get("text") or item.get("expr")
                value = item.get("value")
            else:
                # Try common attributes
                for nm_attr in ("name", "text", "expr_text"):
                    if hasattr(item, nm_attr):
                        name = getattr(item, nm_attr)
                        break
                for v_attr in ("value", "val", "const", "c"):
                    if hasattr(item, v_attr):
                        value = getattr(item, v_attr)
                        break

            # If value is still missing, skip
            try:
                v_float = float(value) if value is not None else None
            except Exception:
                v_float = None
            if v_float is None:
                continue

            # Form new RHS by replacing numeric intercept with this constant value
            if isinstance(right, BinOp) and right.fn in (np.add, np.subtract):
                # Replace whichever child was the numeric Const
                if _extract_intercept(right.left) is not None:
                    new_right = BinOp(right.fn, Const(v_float), right.right)
                elif _extract_intercept(right.right) is not None:
                    new_right = BinOp(right.fn, right.left, Const(v_float))
                else:
                    continue
            else:
                # Full replacement (rare)
                new_right = Const(v_float)

            new_rel = _make_relation(kind, left, new_right)
            new_conj = Conjecture(relation=new_rel, condition=Hnew, name=conj.name)

            # Defer superset/validity checks to the caller; just wrap result.
            out.append(Generalization(from_conjecture=conj, new_conjecture=new_conj, reason=f"intercept={name}"))

    return out
