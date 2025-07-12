import pandas as pd
from typing import List, Union
from txgraffiti.logic import Conjecture, Inequality, KnowledgeTable

__all__ = [
    'dalmatian_accept',
]


def dalmatian_accept(
    new_conj: Conjecture,
    existing: List[Conjecture],
    df: Union[pd.DataFrame, KnowledgeTable],
) -> bool:
    """
    Determine whether to accept a new upper‐bound conjecture based on
    the Dalmatian heuristic.

    The Dalmatian heuristic accepts `new_conj` if and only if:

        1. It is valid on *all* rows under its hypothesis.
        2. Its right‐hand side (RHS) is strictly smaller on at least one
           row compared to the minimum RHS of *all* existing upper‐bounds
           with the same hypothesis and same conclusion LHS.

    Parameters
    ----------
    new_conj : Conjecture
        Candidate conjecture of the form ``H → (lhs ≤ rhs)``.
    existing : list of Conjecture
        Previously accepted conjectures to compare against.
    df : pandas.DataFrame or KnowledgeTable
        Tabular data on which to evaluate hypotheses and RHS values.

    Returns
    -------
    bool
        True if `new_conj` is globally valid and strictly tighter than
        every matching existing bound; False otherwise.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti.logic import Property, Predicate, Conjecture, Inequality
    >>> from txgraffiti.heuristics.fajtlowicz import dalmatian_accept
    >>> df = pd.DataFrame({
    ...     'alpha':     [1, 2, 3],
    ...     'beta':      [3, 2, 1],
    ...     'connected': [True, True, True],
    ... })
    >>> P = Predicate('connected', lambda df: df['connected'])
    >>> A = Property('alpha', lambda df: df['alpha'])
    >>> B = Property('beta',  lambda df: df['beta'])
    >>> weak   = P >> (A <= B + 2)
    >>> strong = P >> (A <= B + 1)
    >>> best   = P >> (A <= B)
    >>> # No existing bounds ⇒ accept
    >>> dalmatian_accept(best, [], df)
    True
    >>> # Reject the weaker conjecture
    >>> dalmatian_accept(weak, [strong], df)
    False
    >>> # Accept the strictly tighter one
    >>> dalmatian_accept(best, [strong], df)
    True
    """
    # 1) validity check
    if not new_conj.is_true(df):
        return False

    H = new_conj.hypothesis

    # 2) collect all existing upper-bounds matching hypothesis & LHS
    old_bounds = [
        c for c in existing
        if c.hypothesis == H
        and isinstance(c.conclusion, Inequality)
        and c.conclusion.lhs.name == new_conj.conclusion.lhs.name
        and c.conclusion.op in ("<=", "≤")
    ]

    # If none, accept immediately
    if not old_bounds:
        return True

    # 3) compute new RHS values and the minimum old RHS per row
    rhs_new   = new_conj.conclusion.rhs(df)
    old_rhs_df = pd.concat(
        [c.conclusion.rhs(df) for c in old_bounds],
        axis=1
    )
    min_old   = old_rhs_df.min(axis=1)

    # 4) accept if there is any row where new_rhs < min_old
    return bool((rhs_new < min_old).any())
