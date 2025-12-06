# src/txgraffiti/graffiti3/utils.py

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.relations import Conjecture

def _filter_by_touch(
    df: pd.DataFrame,
    conjectures: Sequence[Conjecture],
    min_touches: int,
) -> List[Conjecture]:
    """
    Compute touch_count for each conjecture and discard those with
    touch_count < min_touches.

    This is intended to be used *before* expensive heuristics
    (Morgan, Dalmatian), so that they only see reasonably “tight”
    candidates.
    """
    if min_touches <= 0:
        # Nothing to filter; still ensure touch_count is computed once.
        out: List[Conjecture] = []
        for c in conjectures:
            try:
                c.touch_count(df)
            except Exception:
                setattr(c, "touch_count", 0)
            out.append(c)
        return out

    kept: List[Conjecture] = []
    for c in conjectures:
        try:
            t = c.touch_count(df)  # sets c.touch and c.touch_count
        except Exception:
            t = 0
            setattr(c, "touch_count", 0)
        if t >= min_touches:
            kept.append(c)
    return kept


def _dedup_conjectures(conjs: Sequence[Conjecture]) -> List[Conjecture]:
    """
    Stable dedup by conjecture.signature().

    Keeps the first occurrence of each signature and drops later duplicates.
    """
    seen: set[str] = set()
    out: List[Conjecture] = []
    for c in conjs:
        sig = c.signature()
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


def _nice_fraction(
    x: float,
    *,
    max_denom: int = 50,
    max_numer: int = 200,
) -> Optional[Fraction]:
    """
    Approximate x by a "nice" rational p/q with small numerator/denominator.

    Returns None if:
      - x is not finite, or
      - |p| > max_numer or q > max_denom.

    This is what prevents coefficients like 4740631186705785/8 from appearing.
    """
    if not np.isfinite(x):
        return None

    frac = Fraction(x).limit_denominator(max_denom)
    if abs(frac.numerator) > max_numer or abs(frac.denominator) > max_denom:
        return None
    return frac


def _annotate_and_sort_conjectures(
    df: pd.DataFrame,
    conjs: Sequence[Conjecture],
) -> List[Conjecture]:
    """
    Compute touch_count and support_n for each conjecture, deduplicate by
    signature, and sort by (touch_count, support_n) descending.
    """
    unique: List[Conjecture] = []
    seen: set[str] = set()

    for c in conjs:
        sig = c.signature()
        if sig in seen:
            continue
        seen.add(sig)

        # Compute touch_count once; Conjecture.touch_count mutates itself
        touch_attr = getattr(c, "touch_count", None)
        if callable(touch_attr):
            try:
                val = c.touch_count(df, auto_base=False)
            except TypeError:
                # Fallback if signature differs
                val = c.touch_count(df)
        else:
            # Already materialized as an int
            val = touch_attr if isinstance(touch_attr, int) else 0

        setattr(c, "touch_count", int(val))
        setattr(c, "touch", int(val))  # for backward compatibility

        # Compute support_n: how many rows are in the hypothesis class
        try:
            applicable, _, _ = c.check(df, auto_base=False)
            support = int(applicable.sum())
        except Exception:
            support = 0

        setattr(c, "support_n", support)
        setattr(c, "support", support)

        unique.append(c)

    unique.sort(
        key=lambda cc: (
            int(getattr(cc, "touch_count", 0)),
            int(getattr(cc, "support_n", 0)),
        ),
        reverse=True,
    )
    return unique

# def _annotate_and_sort_conjectures(
#     df: pd.DataFrame,
#     conjs: Sequence[Conjecture],
# ) -> List[Conjecture]:
#     """
#     Compute touch_count and support_n for each conjecture, deduplicate by
#     signature, and sort by a "purity-like" score rather than raw touch.

#     New annotations per conjecture:
#       - touch_count : int      (number of equality touches)
#       - touch       : int      (alias for backward compatibility)
#       - support_n   : int      (size of hypothesis class)
#       - support     : int      (alias)
#       - touch_density : float  (touch_count / support_n, in [0,1] when support_n>0)
#       - support_frac  : float  (support_n / len(df), coverage of the dataset)

#     Sorting order (descending):
#       1. touch_density   (fraction of the class where equality holds)
#       2. touch_count     (absolute number of equality instances)
#       3. support_n       (size of the hypothesis class)

#     This favors conjectures that are both "pure" (high density) and
#     non-trivial (decent touch/support), instead of just those with large
#     raw touch on huge classes.
#     """
#     unique: List[Conjecture] = []
#     seen: set[str] = set()
#     n_total = int(len(df))

#     for c in conjs:
#         sig = c.signature()
#         if sig in seen:
#             continue
#         seen.add(sig)

#         # --- touch_count (and touch) ---
#         touch_attr = getattr(c, "touch_count", None)
#         if callable(touch_attr):
#             try:
#                 touch_val = c.touch_count(df, auto_base=False)
#             except TypeError:
#                 # Fallback if signature differs
#                 touch_val = c.touch_count(df)
#         else:
#             # Already materialized as an int
#             touch_val = touch_attr if isinstance(touch_attr, int) else 0

#         touch = int(touch_val)
#         setattr(c, "touch_count", touch)
#         setattr(c, "touch", touch)  # backward compatibility

#         # --- support_n (and support) ---
#         try:
#             applicable, _, _ = c.check(df, auto_base=False)
#             support = int(applicable.sum())
#         except Exception:
#             support = 0

#         setattr(c, "support_n", support)
#         setattr(c, "support", support)

#         # --- densities / fractions ---
#         if support > 0:
#             density = float(touch) / float(support)
#             density = density*n_total
#         else:
#             density = 0.0

#         if n_total > 0:
#             support_frac = float(support) / float(n_total)
#             density = density*n_total
#         else:
#             support_frac = 0.0

#         setattr(c, "touch_density", density)
#         setattr(c, "support_frac", support_frac)

#         unique.append(c)

#     # Sort by (touch_density, touch_count, support_n) descending
#     unique.sort(
#         key=lambda cc: (
#             float(getattr(cc, "touch_density", 0.0)),
#             int(getattr(cc, "touch_count", 0)),
#             int(getattr(cc, "support_n", 0)),
#         ),
#         reverse=True,
#     )
#     return unique
