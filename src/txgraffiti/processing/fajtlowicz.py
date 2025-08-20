from txgraffiti.logic import *
from txgraffiti.heuristics import dalmatian_accept
from typing import List, Union
import pandas as pd

__all__ = [
    'filter_with_dalmatian',
]

def filter_with_dalmatian(conjectures, df):
    """
    Apply the Dalmatian heuristic to a list of conjectures, retaining only those
    that are strictly tighter than all other matching conjectures on at least one row.

    Parameters
    ----------
    conjectures : list of Conjecture
        Candidate conjectures to evaluate.
    df : pd.DataFrame or KnowledgeTable
        Data used for truth and tightness checks.

    Returns
    -------
    list of Conjecture
        The accepted conjectures.
    """
    accepted = []

    for new_conj in conjectures:
        if not isinstance(new_conj.conclusion, Inequality):
            continue  # skip non-inequality conclusions

        # Compare against all other conjectures (not itself)
        others = [c for c in conjectures if c != new_conj]

        if dalmatian_accept(new_conj, others, df):
            accepted.append(new_conj)

    return accepted

