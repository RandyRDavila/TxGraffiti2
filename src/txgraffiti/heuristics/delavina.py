import pandas as pd
from typing import List, Union
from txgraffiti.logic import Conjecture, KnowledgeTable

__all__ = [
    'sophie_accept',
]

def sophie_accept(
    new_conj: Conjecture,
    accepted: List[Conjecture],
    df: Union[pd.DataFrame, KnowledgeTable],
) -> bool:
    """
    Decide whether to accept a new conjecture based on its cover set.

    A conjecture’s *cover set* is the set of rows where its hypothesis
    holds.  Under the Sophie heuristic, we accept `new_conj` only if
    its cover set includes at least one row *not* already covered by
    the union of cover sets of all `accepted` conjectures.

    Parameters
    ----------
    new_conj : Conjecture
        The candidate conjecture whose hypothesis cover set is tested.
    accepted : list of Conjecture
        Previously accepted conjectures.  Their hypothesis masks are
        unioned to form the existing coverage.
    df : pandas.DataFrame or KnowledgeTable
        The data on which hypotheses are evaluated.

    Returns
    -------
    bool
        True if `new_conj` covers at least one additional row beyond
        the union of all `accepted` cover sets, False otherwise.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti.logic import Property, Predicate, Conjecture
    >>> from txgraffiti.heuristics.delavina import sophie_accept
    >>> df = pd.DataFrame({
    ...     'alpha':     [1, 2, 3, 4],
    ...     'connected': [True, False, True, False],
    ... })
    >>> A = Property('alpha', lambda df: df['alpha'])
    >>> P = Predicate('connected', lambda df: df['connected'])
    >>> # conj1 covers rows 0 and 2
    >>> conj1 = P >> (A <= 10)
    >>> # conj2 covers the same rows → no new coverage
    >>> conj2 = P >> (A >= 0)
    >>> sophie_accept(conj2, [conj1], df)
    False
    >>> # conj3 covers row 0,2, plus row 1 (connected=False so hypothesis False)
    >>> # so still no new coverage
    >>> sophie_accept(conj3:= (P | ~P) >> (A >= 0), [conj1], df)
    True
    """
    # cover set of the new conjecture
    new_cover = new_conj.hypothesis(df)

    # union of old covers
    if accepted:
        old_union = pd.concat(
            [c.hypothesis(df) for c in accepted],
            axis=1
        ).any(axis=1)
    else:
        old_union = pd.Series(False, index=df.index)

    # must add at least one row not already covered
    delta = new_cover & ~old_union
    return bool(delta.any())
