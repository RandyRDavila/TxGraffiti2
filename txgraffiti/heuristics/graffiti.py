
from typing import List
from txgraffiti.logic import *

__all__ = [
    'dalmatian'
]

def dalmatian(
    new_conj: Conjecture,
    existing: List[Conjecture],
    df:       pd.DataFrame
) -> bool:
    """
    Return True iff
      1) new_conj holds on ALL rows of df, and
      2) new_conj is strictly tighter on at least one row
         than every existing upper‐bound conjecture
         with the same hypothesis.
    """
    # 1) validity check
    if not new_conj.is_true(df):
        return False

    # extract the hypothesis name
    H = new_conj.hypothesis

    # pick out all existing upper‐bounds with the same hypothesis
    # (we assume upper‐bounds are stored as target ≤ rhs)
    old_bounds = [
        c for c in existing
        if c.hypothesis == H
        and isinstance(c.conclusion, Inequality)
        and c.conclusion.lhs.name == new_conj.conclusion.lhs.name
        and c.conclusion.op in ("<=", "≤")
    ]

    # if there are no old upper‐bounds, accept immediately
    if not old_bounds:
        return True

    # compute new and old rhs‐values on all rows
    rhs_new = new_conj.conclusion.rhs(df)

    # build DataFrame of all old‐rhs
    old_rhs_df = pd.concat(
        [c.conclusion.rhs(df)
         for i, c in enumerate(old_bounds)],
        axis=1
    )
    min_old = old_rhs_df.min(axis=1)

    # 2) strict improvement: any row where rhs_new < min_old
    return bool((rhs_new < min_old).any())
