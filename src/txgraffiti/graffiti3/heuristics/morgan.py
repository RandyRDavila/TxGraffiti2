# src/txgraffiti/graffiti3/heuristics/morgan.py

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.relations import Conjecture


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
