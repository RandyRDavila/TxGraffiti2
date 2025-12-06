from __future__ import annotations

from typing import List, Sequence

import pandas as pd

from txgraffiti.graffiti3.relations import Conjecture
from txgraffiti.graffiti3.heuristics.morgan import morgan_filter
from txgraffiti.graffiti3.heuristics.dalmatian import dalmatian_filter
from txgraffiti.graffiti3.utils import _annotate_and_sort_conjectures

def heuristic_runner(
    conjectures: Sequence[Conjecture],
    *,
    df: pd.DataFrame,
    morgan_filter=None,
    dalmatian_filter=None,
) -> List[Conjecture]:
    """
    Apply Morgan and Dalmatian-style filters in sequence, if provided.
    Expected signatures:

        morgan_filter(df, conjectures) -> list[Conjecture]
        dalmatian_filter(df, conjectures) -> list[Conjecture]

    (You can plug your existing implementations here; for now the default
    is to just return the input list sorted by touch/support.)
    """
    out = list(conjectures)

    if morgan_filter is not None:
        out = morgan_filter(df, out)

    if dalmatian_filter is not None:
        out = dalmatian_filter(df, out)

    # As a fallback, we lightly annotate and sort here as well,
    # though Graffiti4.conjecture will do a full annotation later.
    out = _annotate_and_sort_conjectures(df, out)
    return out
