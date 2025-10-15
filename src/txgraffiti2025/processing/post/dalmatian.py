"""
Dalmatian heuristic: truth and significance filtering for conjectures.

Used both as:
1. A pre-acceptance filter (during conjecture generation)
2. A post-processing refinement (after conjecture generation)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
from txgraffiti2025.processing.utils import truth_mask, touch_count, slack_summary, hash_conjecture, log_event


# ============================================================
# Helper: determine target and direction from relation
# ============================================================

def _target_and_direction(conj: Conjecture) -> tuple[str, str] | None:
    """Infer the target variable and inequality direction from the relation."""
    rel = conj.relation
    if isinstance(rel, Le):
        direction = "le"
    elif isinstance(rel, Ge):
        direction = "ge"
    else:
        return None
    # Try to guess target column name (lhs if ColumnTerm)
    left = getattr(rel.left, "col", None)
    if isinstance(left, str):
        return (left, direction)
    return None


# ============================================================
# Truth and Significance Tests
# ============================================================

def dalmatian_score(conj: Conjecture, df: pd.DataFrame) -> Dict[str, Any]:
    """Return a diagnostic score dictionary for the Dalmatian heuristic."""
    applicable, holds, fails = conj.check(df)
    truth_ok = len(fails) == 0
    tcount = touch_count(conj, df)
    min_slack, mean_slack = slack_summary(conj, df)
    return {
        "name": getattr(conj, "name", "Conjecture"),
        "truth_ok": truth_ok,
        "touch_count": tcount,
        "min_slack": min_slack,
        "mean_slack": mean_slack,
    }



def _is_significant(conj: Conjecture, df: pd.DataFrame, accepted: List[Conjecture]) -> bool:
    """
    True if this conjecture gives a strictly tighter bound for at least one instance,
    compared to already-accepted conjectures with the same (target, direction).
    """
    if not accepted:
        return True

    rel = conj.relation
    td = _target_and_direction(conj)
    if td is None:
        # If we can't infer (target,direction), don't block on significance
        return True
    tname, direction = td

    # Only compare against already-accepted conjectures with the same (target, direction)
    candidates = [c for c in accepted if _target_and_direction(c) == td]
    if not candidates:
        return True

    # Compare RHS expressions directly (avoid slack sign ambiguity)
    rhs_new = rel.right.eval(df)
    rhs_prev_list = [c.relation.right.eval(df) for c in candidates]

    eps = 1e-9
    if direction == "le":
        # Upper bounds: tighter if RHS_new < best previous RHS somewhere
        rhs_best = rhs_prev_list[0]
        for arr in rhs_prev_list[1:]:
            rhs_best = np.minimum(rhs_best, arr)
        return bool((rhs_new < rhs_best - eps).any())
    else:
        # Lower bounds: tighter if RHS_new > best previous RHS somewhere
        rhs_best = rhs_prev_list[0]
        for arr in rhs_prev_list[1:]:
            rhs_best = np.maximum(rhs_best, arr)
        return bool((rhs_new > rhs_best + eps).any())


# ============================================================
# Main filter
# ============================================================

def dalmatian_filter(conjectures: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
    """
    Apply the Dalmatian heuristic:
    - Remove conjectures failing the truth test.
    - Keep only those that improve at least one instance's bound (significance).
    - Deduplicate by structural hash.
    """
    kept: List[Conjecture] = []
    seen_hashes: set[str] = set()
    for conj in conjectures:
        try:
            score = dalmatian_score(conj, df)
            if not score["truth_ok"]:
                log_event(f"Rejected (fails truth): {conj.name}")
                continue
            if not _is_significant(conj, df, kept):
                log_event(f"Rejected (insignificant): {conj.name}")
                continue
            h = hash_conjecture(conj)
            if h in seen_hashes:
                log_event(f"Rejected (duplicate): {conj.name}")
                continue
            kept.append(conj)
            seen_hashes.add(h)
        except Exception as e:
            log_event(f"Error evaluating conjecture {getattr(conj,'name','?')}: {e}")
            continue
    return kept