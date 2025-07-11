import pandas as pd

from txgraffiti.logic import Conjecture

__all__ = [
    'sophie'
]

def sophie(
    new_conj: Conjecture,
    accepted: list[Conjecture],
    df:       pd.DataFrame
) -> bool:
    """
    Accept `new_conj` only if its cover set adds at least one new row
    beyond the union of all previously accepted conjectures' cover sets.
    """
    # cover set of the new conjecture
    new_cover = new_conj.hypothesis(df)

    # union of old covers
    if accepted:
        old_union = pd.concat(
            [c.hypothesis(df) for c in accepted], axis=1
        ).any(axis=1)
    else:
        old_union = pd.Series(False, index=df.index)

    # must add at least one row not already covered
    delta = new_cover & ~old_union
    return bool(delta.any())
