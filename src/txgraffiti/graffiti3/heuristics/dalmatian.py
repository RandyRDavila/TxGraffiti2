# src/txgraffiti/graffiti3/heuristics/dalmatian.py

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.relations import (
    Conjecture,
    Le,
    Ge,
    TRUE,
    TRUE_Predicate,
)


def _direction_and_target(c: Conjecture) -> Tuple[str, str]:
    """
    Infer (direction, target_name) from the relation and metadata.

    direction ∈ {"upper", "lower", "other"}.
    """
    r = c.relation

    if isinstance(r, Le):
        direction = "upper"
    elif isinstance(r, Ge):
        direction = "lower"
    else:
        return "other", "?"

    t = getattr(c, "target_name", None)
    if t is None:
        # Fallback: use left-hand repr as an identifier
        try:
            t = repr(r.left)
        except Exception:
            t = "<?>"

    return direction, t


def _dalmatian_group_streaming(
    df: pd.DataFrame,
    group: Sequence[Conjecture],
    direction: str,
    eq_tol: float,
) -> List[Conjecture]:
    """
    Streaming Dalmatian for a single (target, direction, hypothesis) group.

    We maintain a running best bound u_*(·) over the stored conjectures u_j(·).
    A candidate c with RHS u(·) is kept iff ∃ row i such that:

      - u(i) < u_*(i) - eq_tol   (for upper bounds), or
      - u(i) > u_*(i) + eq_tol   (for lower bounds),

    mirroring the original significance test “strictly better somewhere”.
    """
    survivors: List[Conjecture] = []
    if not group:
        return survivors

    cond = group[0].condition or TRUE
    if isinstance(cond, TRUE_Predicate):
        base_mask = np.ones(len(df), dtype=bool)
    else:
        base_mask = cond.mask(df).to_numpy(dtype=bool)

    if not base_mask.any():
        # Hypothesis never holds; nothing to test for significance.
        return survivors

    idx = np.where(base_mask)[0]
    n_rows = len(idx)

    # Initialize best bounds: +∞ for upper bounds, -∞ for lower bounds.
    if direction == "upper":
        best_vals = np.full(n_rows, np.inf, dtype=float)
    else:  # "lower"
        best_vals = np.full(n_rows, -np.inf, dtype=float)

    # Process candidates in the given order (streaming semantics)
    for c in group:
        try:
            vals_full = c.relation.right.eval(df).to_numpy(dtype=float)
        except Exception:
            # If RHS evaluation fails, treat as unusable
            continue

        vals = vals_full[idx]
        finite_mask = np.isfinite(vals)
        if not finite_mask.any():
            # No usable entries under the hypothesis
            continue

        # Significance test: strictly better than current best somewhere.
        if direction == "upper":
            # smaller is better
            better_mask = np.zeros_like(vals, dtype=bool)
            better_mask[finite_mask] = vals[finite_mask] < best_vals[finite_mask] - eq_tol
        else:  # "lower", larger is better
            better_mask = np.zeros_like(vals, dtype=bool)
            better_mask[finite_mask] = vals[finite_mask] > best_vals[finite_mask] + eq_tol

        if better_mask.any():
            # Candidate is significant: keep it and update best_vals
            survivors.append(c)
            if direction == "upper":
                best_vals[finite_mask] = np.minimum(best_vals[finite_mask], vals[finite_mask])
            else:
                best_vals[finite_mask] = np.maximum(best_vals[finite_mask], vals[finite_mask])
        else:
            # Never strictly better than the current Dalmatian envelope ⇒ discard
            pass

    return survivors


def dalmatian_filter(
    df: pd.DataFrame,
    conjectures: Sequence[Conjecture],
    *,
    eq_tol: float = 1e-9,
) -> List[Conjecture]:
    """
    Dalmatian significance heuristic (Calloway et al. style).

    For each (target, direction, hypothesis) triple, we process conjectures
    in order, maintaining a running best bound over the *stored* conjectures.

    For upper bounds (Le):

        keep u iff ∃ graph G with H(G) true such that
            u(G) < min_j u_j(G)  (up to `eq_tol`),

    where u_j are the previously *stored* upper bounds for this group.

    For lower bounds (Ge), we flip the inequality:

        keep u iff ∃ G with H(G) true such that
            u(G) > max_j u_j(G).

    Already–kept conjectures are never removed, matching the “only added if…”
    behavior in the original Dalmatian definition.
    """
    if not conjectures:
        return []

    # Group by (target, direction, hypothesis)
    grouped: Dict[Tuple[str, str, str], List[Conjecture]] = defaultdict(list)
    for c in conjectures:
        direction, target_name = _direction_and_target(c)
        cond = c.condition or TRUE
        cond_key = repr(cond)
        grouped[(target_name, direction, cond_key)].append(c)

    survivors: List[Conjecture] = []

    for (target_name, direction, cond_key), group in grouped.items():
        if direction not in {"upper", "lower"}:
            # Non-inequality relations: pass through unchanged
            survivors.extend(group)
            continue

        # Preserve order within each group (streaming semantics)
        group_survivors = _dalmatian_group_streaming(
            df=df,
            group=group,
            direction=direction,
            eq_tol=eq_tol,
        )
        survivors.extend(group_survivors)

    return survivors
