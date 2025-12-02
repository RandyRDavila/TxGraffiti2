# txgraffiti2025/graffiti4_dalmatian.py

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import (
    Conjecture,
    Le,
    Ge,
    TRUE,
    TRUE_Predicate,
)


def _direction_and_target(c: Conjecture) -> Tuple[str, str]:
    """
    Infer (direction, target_name) from the relation and metadata.

    direction ∈ {"upper", "lower", "other"}
    """
    r = c.relation

    if isinstance(r, Le):
        direction = "upper"
    elif isinstance(r, Ge):
        direction = "lower"
    else:
        return "other", "?"

    # Prefer explicit attribute if runners set it
    t = getattr(c, "target_name", None)
    if t is None:
        # Fallback: use repr(left) as target id
        try:
            t = repr(r.left)
        except Exception:
            t = "<?>"

    return direction, t


def dalmatian_filter(
    df: pd.DataFrame,
    conjectures: Sequence[Conjecture],
    *,
    eq_tol: float = 1e-9,
) -> List[Conjecture]:
    """
    Dalmatian significance heuristic.

    For each group of conjectures with the same
        (target, direction, hypothesis)

    we keep only those inequalities that are *uniquely best* on at least
    one row where the hypothesis holds:

      - For upper bounds (Le): RHS is strictly *smallest* somewhere.
      - For lower bounds (Ge): RHS is strictly *largest* somewhere.

    Concretely, c is kept iff there exists a graph G such that:

      - G satisfies the conjecture’s hypothesis, and
      - for that G, its RHS is strictly better than all competing RHS
        values in the same group.

    This kills things like
        α ≤ n − Slater
    when we already have
        α ≤ n − γ
    and Slater ≤ γ on the dataset.
    """
    if not conjectures:
        return []

    # ---- group conjectures by (target, direction, hypothesis) ----
    grouped: Dict[Tuple[str, str, str], List[Conjecture]] = defaultdict(list)

    for c in conjectures:
        direction, target_name = _direction_and_target(c)
        cond = c.condition or TRUE
        cond_key = repr(cond)  # OK: purely internal grouping label
        grouped[(target_name, direction, cond_key)].append(c)

    survivors: List[Conjecture] = []

    # ---- process each group independently ----
    for (target_name, direction, cond_key), group in grouped.items():
        # We only know how to handle upper/lower inequalities
        if direction not in {"upper", "lower"}:
            survivors.extend(group)
            continue

        if not group:
            continue
        if len(group) == 1:
            # Single candidate: automatically significant
            survivors.extend(group)
            continue

        cond = group[0].condition or TRUE
        if isinstance(cond, TRUE_Predicate):
            base_mask = np.ones(len(df), dtype=bool)
        else:
            base_mask = cond.mask(df).to_numpy(dtype=bool)
        if not base_mask.any():
            # Hypothesis never holds; nothing to compare
            continue

        idx = np.where(base_mask)[0]
        n_rows = len(idx)
        n_conj = len(group)

        # Build RHS matrix: shape (n_rows, n_conj)
        rhs = np.full((n_rows, n_conj), np.nan, dtype=float)

        for j, c in enumerate(group):
            try:
                vals = c.relation.right.eval(df).to_numpy(dtype=float)[idx]
            except Exception:
                # If evaluation fails, mark as NaN and let it die
                continue
            rhs[:, j] = vals

        # Restrict to rows where all RHS are finite for fair comparison
        finite_mask = np.isfinite(rhs).all(axis=1)
        if not finite_mask.any():
            continue
        rhs = rhs[finite_mask, :]
        if rhs.shape[0] == 0:
            continue

        # ---- Dalmatian significance: unique best row for each inequality ----
        # For upper bounds, smaller RHS is better; for lower, larger is better.
        if direction == "upper":
            best_vals = rhs.min(axis=1)                            # shape (n_rows,)
            diff = rhs - best_vals[:, None]                        # ≥ 0 for all
            is_best = diff <= eq_tol                               # within tolerance of min
        else:  # direction == "lower"
            best_vals = rhs.max(axis=1)
            diff = best_vals[:, None] - rhs                        # ≥ 0
            is_best = diff <= eq_tol                               # within tolerance of max

        # How many inequalities tie as "best" on each row?
        n_best = is_best.sum(axis=1)                               # shape (n_rows,)

        # Unique best: best AND not tied with any other conjecture on that row.
        unique_best = is_best & (n_best[:, None] == 1)

        # Keep conjectures that are uniquely best on at least one row.
        for j, c in enumerate(group):
            if unique_best[:, j].any():
                survivors.append(c)

    return survivors
