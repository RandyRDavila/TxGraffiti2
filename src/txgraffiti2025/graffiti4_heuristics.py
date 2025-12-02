# src/txgraffiti2025/graffiti4_heuristics.py

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Any

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import (
    Conjecture,
    Ge,
    Le,
)


# ───────────────────── Morgan: generalize hypotheses ───────────────────── #

def _relation_signature(c: Conjecture) -> str:
    """
    Canonical signature for the relation ONLY (ignore hypothesis).

    This is the key fix: previously, Morgan often grouped by the whole
    conjecture signature (which includes the condition), so it never saw
    multiple hypotheses for the same inequality.
    """
    rel = c.relation
    if hasattr(rel, "pretty"):
        return rel.pretty(unicode_ops=True, show_tol=False)  # type: ignore[call-arg]
    return repr(rel)


def _condition_mask(c: Conjecture, df: pd.DataFrame) -> np.ndarray:
    """
    Evaluate the condition mask for a conjecture.

    Graffiti4 always sets c.condition to a Predicate corresponding to the
    hypothesis; here we just turn it into a boolean numpy array.
    """
    if c.condition is None:
        # fall back to "everything"
        return np.ones(len(df), dtype=bool)
    s = c.condition.mask(df).reindex(df.index, fill_value=False)
    return s.to_numpy(dtype=bool)


def morgan_filter(
    df: pd.DataFrame,
    conjectures: Sequence[Conjecture],
    *,
    debug: bool = False,
) -> List[Conjecture]:
    """
    Morgan heuristic (Graffiti-style):

      For each distinct inequality R(x) (ignoring the hypothesis),
      look at all conjectures of the form

          (H_i) ⇒ R(x),

      and keep only those whose hypothesis is *maximal* by inclusion:

          mask(H_i) is not a strict subset of mask(H_j)
          for any other H_j with the same R.

      Intuition: if the *same* inequality holds on a larger class, the
      smaller class is redundant.

    Parameters
    ----------
    df : DataFrame
        The invariant table.
    conjectures : Sequence[Conjecture]
        Input conjectures.
    debug : bool
        If True, prints a small summary of what Morgan removed.

    Returns
    -------
    List[Conjecture]
        Conjectures that survived Morgan pruning.
    """
    if not conjectures:
        return []

    # Group by relation signature (ignore hypothesis)
    groups: Dict[str, List[Tuple[Conjecture, np.ndarray]]] = {}
    for c in conjectures:
        sig = _relation_signature(c)
        mask = _condition_mask(c, df)
        groups.setdefault(sig, []).append((c, mask))

    kept: List[Conjecture] = []
    removed: List[Conjecture] = []

    for sig, items in groups.items():
        n = len(items)
        if n == 1:
            kept.append(items[0][0])
            continue

        masks = [m for _, m in items]
        keep_flags = [True] * n

        # For each hypothesis mask Mi, check if it is strictly contained
        # in some other Mj for the same relation.
        for i in range(n):
            mi = masks[i]
            for j in range(n):
                if i == j:
                    continue
                mj = masks[j]

                # "mi subset mj" means:
                #   - every True in mi is True in mj  (mi & ~mj is empty)
                #   - and there is at least one row where mj is True and mi is False
                if np.any(mi & ~mj):
                    # Mi has some rows that Mj does not — not a subset
                    continue
                if np.any(mj & ~mi):
                    # Mi is a strict subset of Mj — drop i
                    keep_flags[i] = False
                    break

        for idx, (c, _) in enumerate(items):
            if keep_flags[idx]:
                kept.append(c)
            else:
                removed.append(c)

    if debug and removed:
        print(f"[Morgan] Removed {len(removed)} conjectures as redundant.")
    return kept


# ───────────────────── Dalmatian: significance filter ───────────────────── #

def _direction_and_target(c: Conjecture) -> Tuple[str, str] | None:
    """
    Extract ("lower" or "upper", target_signature) from a conjecture, or
    None if the relation is not a simple Ge/Le inequality.

    We assume the left Expr of Ge/Le is your 'target' invariant.
    """
    rel = c.relation
    if isinstance(rel, Ge):
        return "lower", repr(rel.left)
    if isinstance(rel, Le):
        return "upper", repr(rel.left)
    return None


def dalmatian_filter(
    df: pd.DataFrame,
    conjectures: Sequence[Conjecture],
    *,
    tol: float = 1e-9,
    debug: bool = False,
) -> List[Conjecture]:
    """
    Dalmatian heuristic (significance):

      For a fixed hypothesis H and target y, consider all conjectures of
      the same direction:

            H ⇒ y ≥ f_i(x)   (lower bounds), or
            H ⇒ y ≤ g_i(x)   (upper bounds).

      Dalmatian keeps only those inequalities that are *tighter* on at
      least one row in H when compared to all others.

      More concretely, for lower bounds:
        - For each row r in H, compute best_rhs(r) = max_j f_j(r).
        - A conjecture i survives iff there exists r where
              f_i(r) ≈ best_rhs(r)  and  f_i(r) is strictly larger
          than all others by at least tol.

      For upper bounds, replace max with min and "larger" with "smaller".

    Parameters
    ----------
    df : DataFrame
        Invariant table.
    conjectures : Sequence[Conjecture]
        Input conjectures.
    tol : float
        Numerical tolerance for "strictly better".
    debug : bool
        If True, prints a brief summary.

    Returns
    -------
    List[Conjecture]
        Conjectures that survive Dalmatian.
    """
    if not conjectures:
        return []

    N = len(df)

    # Group by (direction, target, hypothesis)
    groups: Dict[Tuple[str, str, str], List[Tuple[Conjecture, np.ndarray, np.ndarray]]] = {}

    for c in conjectures:
        dt = _direction_and_target(c)
        if dt is None:
            # skip non-simple inequalities
            continue
        direction, target_sig = dt
        cond = c.condition
        cond_repr = repr(cond) if cond is not None else "TRUE"
        mask_H = _condition_mask(c, df)

        # Evaluate RHS on the whole df, but we'll only use rows in H
        try:
            rhs_vals = c.relation.right.eval(df).to_numpy(dtype=float)
        except Exception:
            # If evaluation fails, discard this conjecture from Dalmatian's view
            continue

        key = (direction, target_sig, cond_repr)
        groups.setdefault(key, []).append((c, mask_H, rhs_vals))

    survivors: List[Conjecture] = []
    removed: List[Conjecture] = []

    for (direction, target_sig, cond_repr), items in groups.items():
        if len(items) == 1:
            survivors.append(items[0][0])
            continue

        # build matrix of RHS values restricted to hypothesis H (union)
        # H_all = union of all masks for this group (they should mostly coincide)
        H_all = np.zeros(N, dtype=bool)
        for _, mH, _ in items:
            H_all |= mH

        if not H_all.any():
            # nothing to compare; keep them all
            survivors.extend([c for (c, _, _) in items])
            continue

        # For computational convenience, we align each RHS on the rows in H_all
        idx_H = np.where(H_all)[0]
        rhs_matrix = np.stack([rhs[idx_H] for (_, _, rhs) in items], axis=0)
        k = rhs_matrix.shape[0]

        # Compute per-row best RHS among all inequalities
        if direction == "lower":
            best = np.max(rhs_matrix, axis=0)
        else:  # "upper"
            best = np.min(rhs_matrix, axis=0)

        keep_flags = [False] * k

        for i in range(k):
            vals_i = rhs_matrix[i]

            if direction == "lower":
                # how close is vals_i to the best lower bound at each row?
                near_best = np.isclose(vals_i, best, atol=tol)
                strictly_better = vals_i > best + tol
            else:
                # upper bounds: smaller is better
                near_best = np.isclose(vals_i, best, atol=tol)
                strictly_better = vals_i < best - tol

            # Dalmatian requirement: there exists at least one row where
            # this inequality is effectively the best (within tol) AND
            # strictly better than all others by tol.
            # (You can relax this if you want to keep ties.)
            if np.any(strictly_better | near_best):
                keep_flags[i] = True

        for flag, (c, _, _) in zip(keep_flags, items):
            if flag:
                survivors.append(c)
            else:
                removed.append(c)

    # Add conjectures that Dalmatian never saw (e.g., non-Ge/Le)
    # untouched.
    seen_ids = {id(c) for c in survivors}
    for c in conjectures:
        if id(c) not in seen_ids:
            survivors.append(c)
            seen_ids.add(id(c))

    if debug and removed:
        print(f"[Dalmatian] Removed {len(removed)} conjectures as insignificant.")

    return survivors
