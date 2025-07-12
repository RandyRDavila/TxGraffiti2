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
from txgraffiti.utils.safe_generator import safe_generator

__all__ = [
    'convex_hull',
]

@safe_generator
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
    - If the convex hull cannot be constructed due to degeneracies, it is recomputed with `qhull_options="QJ"` to jog input points slightly.

    Examples
    --------
    >>> from txgraffiti import KnowledgeTable
    >>> from txgraffiti.generators.convex_hull import convex_hull
    >>> df = KnowledgeTable({
    ...     'alpha': [1, 2, 3],
    ...     'beta': [3, 1, 1],
    ...     'connected': [True, True, True],
    ...     'tree': [False, False, True],
    ... })
    >>> target = df.alpha
    >>> features = [df.beta]
    >>> hypothesis = df.connected
    >>> for conj in convex_hull(df, features=features, target=target, hypothesis=hypothesis):
    ...     print(conj)
    <Conj (connected) → (alpha >= ((-1/2 * beta) + 5/2))>
    <Conj (connected) → (alpha <= ((-1 * beta) + 4))>
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
