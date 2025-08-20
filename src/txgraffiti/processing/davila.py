from txgraffiti.logic import *
from txgraffiti.heuristics import same_conclusion, is_strict_subset
from typing import List, Union
import pandas as pd

__all__ = [
    'filter_with_morgan'
]

def filter_with_morgan(
    candidates: list[Conjecture],
    df: pd.DataFrame,
) -> list[Conjecture]:
    """
    Apply the Morgan heuristic to a list of conjectures, removing those
    that are strictly less general than another with the same conclusion.
    """
    accepted: list[Conjecture] = []

    for new_conj in candidates:
        keep = True
        new_mask = new_conj.hypothesis(df)
        for old_conj in accepted[:]:
            if same_conclusion(old_conj, new_conj):
                old_mask = old_conj.hypothesis(df)
                if is_strict_subset(new_mask, old_mask):
                    # new is more specific → reject new
                    keep = False
                    break
                elif is_strict_subset(old_mask, new_mask):
                    # old is more specific → remove old
                    accepted.remove(old_conj)
        if keep:
            accepted.append(new_conj)

    return accepted
