"""
Convex-hull-based conjecture generator.

This module defines a generator that builds linear inequality conjectures
of the form `target ≥ RHS` or `target ≤ RHS` by computing the convex hull
of feature-target vectors restricted to a logical hypothesis.
"""


import numpy as np
from scipy.spatial import ConvexHull, QhullError
import pandas as pd
from typing import List, Iterator
from fractions import Fraction

from txgraffiti.logic import Constant, Property, Predicate, Conjecture, Inequality
from txgraffiti.generators.registry import register_gen

__all__ = [
    'convex_hull',
]

@register_gen
def convex_hull(
    df: pd.DataFrame,
    *,
    features:   List[Property],
    target:     Property,
    hypothesis: Predicate,
    drop_side_facets: bool = True,
    tol:              float = 1e-8
) -> Iterator[Conjecture]:
    """
    Generate linear inequality conjectures using the convex hull of invariant vectors.

    This function constructs the convex hull of points in `R^{k+1}` formed by appending
    the `target` value to the values of each feature in `features`, restricted to rows
    satisfying the given `hypothesis`. It interprets each facet of the convex hull as
    a linear inequality between `target` and a linear combination of the features.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing invariant values of mathematical objects.

    features : List[Property]
        A list of numeric-valued properties (functions on `df`) to appear on the RHS
        of the inequality.

    target : Property
        The property to appear alone on the LHS of each inequality.

    hypothesis : Predicate
        A Boolean predicate restricting the rows (objects) used in convex hull generation.

    drop_side_facets : bool, optional
        If True (default), discard facets where the target coefficient is nearly 0, i.e.,
        the inequality does not bound the target directly.

    tol : float, optional
        Numerical tolerance for filtering small coefficients. Default is `1e-8`.

    Yields
    ------
    Conjecture
        A conjecture of the form `hypothesis → target ≤ RHS` or `hypothesis → target ≥ RHS`,
        where RHS is a linear combination of the features with rational coefficients.

    Notes
    -----
    - Uses `scipy.spatial.ConvexHull` to derive inequalities from geometric facets.
    - Coefficients are approximated by rational numbers using `Fraction.limit_denominator()`.
    - If the convex hull cannot be constructed due to degeneracies, it is recomputed
      with `qhull_options="QJ"` to jog input points slightly.

    Examples
    --------
    >>> from txgraffiti.logic import Property, TRUE
    >>> from txgraffiti.generators.convex_hull import convex_hull
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 4, 8], 't': [3, 6, 11]})
    >>> a = Property('a', lambda df: df['a'])
    >>> b = Property('b', lambda df: df['b'])
    >>> t = Property('t', lambda df: df['t'])
    >>> list(convex_hull(df, features=[a, b], target=t, hypothesis=TRUE))
    [Conjecture(TRUE → t >= 1*a + 1*b), Conjecture(TRUE → t <= 2*a + 3*b)]
    """

    # … same body as before …
    mask, subdf = hypothesis(df), df[hypothesis(df)]
    k = len(features)
    if subdf.shape[0] < k+2:
        return
    pts = np.column_stack([p(subdf).values for p in features] + [target(subdf).values])
    try:
        hull = ConvexHull(pts)
    except QhullError:
        hull = ConvexHull(pts, qhull_options="QJ")

    for eq in hull.equations:
        a_all, b0 = eq[:-1], eq[-1]
        a_feats, a_y = a_all[:-1], a_all[-1]

        if drop_side_facets and abs(a_y) < tol:
            continue

        coeffs    = -a_feats / a_y
        intercept = Fraction(-b0    / a_y).limit_denominator()

        rhs: Property = Constant(intercept)
        for coef, feat in zip(coeffs, features):
            if abs(coef) < tol:
                continue

            coef = Fraction(coef).limit_denominator()
            rhs = rhs + (Constant(coef) * feat)

        if a_y > 0:
            ineq = Inequality(target, "<=", rhs)
        else:
            ineq = Inequality(target, ">=", rhs)

        yield Conjecture(hypothesis, ineq)
