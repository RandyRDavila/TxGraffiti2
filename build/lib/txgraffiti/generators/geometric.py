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
