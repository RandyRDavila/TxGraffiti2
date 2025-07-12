import pandas as pd
from typing import List, Tuple, Union

from txgraffiti.logic import Inequality, Conjecture, KnowledgeTable

__all__ = [
    'normalize_inequality_key',
    'same_conclusion',
    'is_strict_subset',
    'morgan_accept',
]


def normalize_inequality_key(ineq: Inequality) -> Tuple[str, str, str]:
    """
    Produce a canonical key for an inequality so that it is always
    represented in “lhs <= rhs” form.

    Parameters
    ----------
    ineq : Inequality
        An inequality between two Properties, e.g. `P >= Q` or `P < Q`.

    Returns
    -------
    key : tuple of str
        A 3‐tuple `(lhs_name, "<=", rhs_name)` such that the returned
        key always uses the `<=` operator, flipping `>=` or `>` by
        swapping operands if necessary.

    Examples
    --------
    >>> from txgraffiti.logic import Property, Inequality
    >>> P = Property('alpha', lambda df: df['alpha'])
    >>> Q = Property('beta',  lambda df: df['beta'])
    >>> ineq1 = Inequality(P, '>=', Q)
    >>> normalize_inequality_key(ineq1)
    ('beta', '<=', 'alpha')
    >>> ineq2 = Inequality(P, '<=', Q)
    >>> normalize_inequality_key(ineq2)
    ('alpha', '<=', 'beta')
    """
    lhs, op, rhs = ineq.lhs, ineq.op, ineq.rhs

    if op in (">=", ">", "≥"):
        # flip to "rhs <= lhs"
        return (rhs.name, "<=", lhs.name)
    else:
        # ≤, <, or "≤"
        return (lhs.name, "<=", rhs.name)


def same_conclusion(a: Conjecture, b: Conjecture) -> bool:
    """
    Determine whether two conjectures share the same logical conclusion,
    up to flipping reversed inequalities.

    Parameters
    ----------
    a : Conjecture
        First conjecture whose conclusion is an Inequality.
    b : Conjecture
        Second conjecture whose conclusion is an Inequality.

    Returns
    -------
    bool
        True if their conclusions map to the same canonical key via
        `normalize_inequality_key`, i.e. they assert the same bound.

    Examples
    --------
    >>> from txgraffiti.logic import Predicate, Property, Conjecture, Inequality
    >>> P = Predicate('connected', lambda df: df['connected'])
    >>> A = Property('alpha', lambda df: df['alpha'])
    >>> B = Property('beta',  lambda df: df['beta'])
    >>> c1 = Conjecture(P, Inequality(A, '<=', B))
    >>> c2 = Conjecture(P, Inequality(B, '>=', A))
    >>> same_conclusion(c1, c2)
    True
    """
    return normalize_inequality_key(a.conclusion) == normalize_inequality_key(b.conclusion)


def is_strict_subset(m1: pd.Series, m2: pd.Series) -> bool:
    """
    Check whether boolean mask `m1` is a strict subset of mask `m2`.

    That is, every True in `m1` is also True in `m2`, and `m2`
    has strictly more True entries than `m1`.

    Parameters
    ----------
    m1 : pandas.Series of bool
        Candidate subset mask.
    m2 : pandas.Series of bool
        Candidate superset mask.

    Returns
    -------
    bool
        True if `m1 & ~m2` has no True values (so `m1 ⊆ m2`) and
        `m2.sum() > m1.sum()`.

    Examples
    --------
    >>> import pandas as pd
    >>> m1 = pd.Series([True, False, True])
    >>> m2 = pd.Series([True, True, True])
    >>> is_strict_subset(m1, m2)
    True
    >>> # not a strict subset if sums equal
    >>> is_strict_subset(m1, m1)
    False
    """
    return ((m1 & ~m2).sum() == 0) and (m2.sum() > m1.sum())


def morgan_accept(
    new_conj: Conjecture,
    existing: List[Conjecture],
    df: Union[pd.DataFrame, KnowledgeTable],
) -> bool:
    """
    Accept `new_conj` only if no existing conjecture with the same
    logical conclusion has a hypothesis mask that strictly contains
    `new_conj`’s mask.

    In other words, we reject `new_conj` if there is already a strictly
    more general conjecture (same bound but wider hypothesis coverage).

    Parameters
    ----------
    new_conj : Conjecture
        The candidate conjecture to test.
    existing : list of Conjecture
        Previously accepted conjectures to compare against.
    df : pandas.DataFrame or KnowledgeTable
        The data table on which to evaluate hypothesis masks.

    Returns
    -------
    bool
        True if no strictly more general existing conjecture was found,
        False otherwise.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti.logic import Predicate, Property, Conjecture, Inequality
    >>> from txgraffiti.heuristics.davila import morgan_accept
    >>> df = pd.DataFrame({
    ...     'alpha':     [1, 2, 3],
    ...     'beta':      [3, 2, 1],
    ...     'connected': [True, True, True],
    ...     'tree':      [False, True, False],
    ... })
    >>> P_gen  = Predicate('connected', lambda df: df['connected'])
    >>> P_sub  = Predicate('tree',      lambda df: df['tree'])
    >>> A = Property('alpha', lambda df: df['alpha'])
    >>> B = Property('beta',  lambda df: df['beta'])
    >>> # less general hypothesis on tree
    >>> c1 = P_sub >> (A <= B)
    >>> # more general hypothesis on connected
    >>> c2 = P_gen >> (A <= B)
    >>> # c2 covers strictly more rows → accept c2 but not c1 if c2 exists
    >>> morgan_accept(c1, [c2], df)
    False
    >>> morgan_accept(c2, [c1], df)
    True
    """
    new_mask = new_conj.hypothesis(df)
    for old in existing:
        if same_conclusion(old, new_conj):
            old_mask = old.hypothesis(df)
            if is_strict_subset(new_mask, old_mask):
                return False
    return True
