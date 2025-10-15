"""
Morgan heuristic (post-processing):

Group conjectures by *conclusion* (same Relation/expr structure),
and within each group keep only the conjecture(s) whose hypothesis
is most general, i.e., holds on the most rows in the dataset.
Tie-breakers: smaller predicate complexity, then name.

This is domain-agnostic and relies on txgraffiti2025.forms abstractions.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.utils import (
    Expr, Const, ColumnTerm, LinearForm, BinOp, UnaryOp,
)
from txgraffiti2025.processing.utils import log_event


# ---------------------------
# Signatures for conclusions
# ---------------------------

def _ufunc_name(fn) -> str:
    # Try to identify common numpy functions; fallback to generic name
    name = getattr(fn, "__name__", None)
    if name:
        return name
    mod = getattr(fn, "__module__", "")
    return f"{mod}.fn"

def _expr_sig(e: Expr) -> str:
    # Recursively stringify expressions to a stable signature
    if isinstance(e, Const):
        return f"Const({e.value})"
    if isinstance(e, ColumnTerm):
        return f"Col({e.col})"
    if isinstance(e, LinearForm):
        terms = ";".join(f"{a}:{c}" for (a,c) in e.terms)
        return f"Linear(a0={e.intercept};{terms})"
    if isinstance(e, BinOp):
        # try to map ufunc to symbol (best-effort)
        fn = _ufunc_name(e.fn)
        return f"Bin({fn},{_expr_sig(e.left)},{_expr_sig(e.right)})"
    if isinstance(e, UnaryOp):
        fn = _ufunc_name(e.fn)
        return f"Una({fn},{_expr_sig(e.arg)})"
    # Fallback (unknown Expr subtype)
    return repr(e)

def _relation_sig(rel) -> Tuple[str, str, str, Optional[float]]:
    if isinstance(rel, Le):
        return ("Le", _expr_sig(rel.left), _expr_sig(rel.right), None)
    if isinstance(rel, Ge):
        return ("Ge", _expr_sig(rel.left), _expr_sig(rel.right), None)
    if isinstance(rel, Eq):
        return ("Eq", _expr_sig(rel.left), _expr_sig(rel.right), rel.tol)
    # Unknown relation type → treat as its repr
    return (rel.__class__.__name__, repr(rel), "", None)

def _conclusion_key(conj: Conjecture) -> Tuple[str, str, str, Optional[float]]:
    return _relation_sig(conj.relation)


# ---------------------------
# Hypothesis generality
# ---------------------------

def _coverage(conj: Conjecture, df: pd.DataFrame) -> int:
    if conj.condition is None:
        return int(len(df))
    m = conj.condition.mask(df)
    # Ensure alignment and boolean dtype
    return int(pd.Series(m, index=df.index).astype(bool).sum())

def _predicate_size(pred: Optional[Predicate]) -> int:
    """
    A coarse complexity metric used as tie-breaker:
    - None (global) = 0
    - Otherwise 1 by default; compound predicates can override if needed.
    We don't introspect arbitrary user predicates; this is a best-effort.
    """
    if pred is None:
        return 0
    # Heuristic: try to walk known dataclasses for And/Or/Not pattern
    # Else default to 1
    name = getattr(pred, "name", "")
    if name in ("C_and", "C_or"):
        # try attributes a,b
        a = getattr(pred, "a", None)
        b = getattr(pred, "b", None)
        return 1 + _predicate_size(a) + _predicate_size(b)
    if name == "C_not":
        a = getattr(pred, "a", None)
        return 1 + _predicate_size(a)
    return 1


# ---------------------------
# Morgan core
# ---------------------------

def morgan_generalize(conjectures: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
    """
    For each group of conjectures that share the *same conclusion*,
    keep only the one(s) whose hypothesis is most general (max coverage),
    tie-breaking by smaller predicate_size, then by name lexicographically.
    """
    if not conjectures:
        return []

    # Group by conclusion key
    groups: Dict[Tuple[str, str, str, Optional[float]], List[Conjecture]] = {}
    for c in conjectures:
        k = _conclusion_key(c)
        groups.setdefault(k, []).append(c)

    kept: List[Conjecture] = []
    for key, bucket in groups.items():
        # compute coverage + complexity
        scored = []
        for c in bucket:
            cov = _coverage(c, df)
            sz  = _predicate_size(c.condition)
            scored.append((cov, -sz, c.name, c))  # -size so higher sort wins on smaller size

        # sort: max coverage, then smaller predicate, then name
        scored.sort(reverse=True)
        best_cov = scored[0][0]
        best_size = -scored[0][1]

        # Keep *all* with equal best coverage; tie-break by minimal size
        candidates = [t[-1] for t in scored if t[0] == best_cov]
        # Among equal coverage, keep those with minimal predicate size
        min_size = min(_predicate_size(c.condition) for c in candidates)
        winners = [c for c in candidates if _predicate_size(c.condition) == min_size]

        if len(winners) < len(bucket):
            log_event(f"Morgan: reduced group {key[0]} to {len(winners)} from {len(bucket)} (best coverage={best_cov})")

        kept.extend(winners)

    return kept
