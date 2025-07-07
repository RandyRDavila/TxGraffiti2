import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from typing import List
from fractions import Fraction

from txgraffiti.logic.conjecture_logic import Property, Predicate, Inequality, Conjecture

def generate_hull_conjectures(
    df: pd.DataFrame,
    features: List[Property],
    target: Property,
    hypothesis: Predicate,
    drop_side_facets: bool = True
) -> List[Conjecture]:
    """
    For the rows where `hypothesis` holds, compute the convex hull
    of the (features + target) points, then for each facet of that hull
    emit a Conjecture(hypothesis, InequalityPredicate) of the form
        target ≤ rhs(x)   or   target ≥ rhs(x)
    depending on the sign of the target‐coefficient in the facet equation.
    """
    # 1) restrict to hypothesis‐true rows
    mask  = hypothesis(df)
    subdf = df[mask]
    k     = len(features)

    # need at least k+2 points in R^{k+1}
    if subdf.shape[0] < (k + 2):
        return []

    # 2) build the point‐cloud array of shape (n_points, k+1)
    try:
        pts = np.column_stack([p(subdf).values for p in features] +
                              [target(subdf).values])
    except Exception as e:
        raise ValueError(f"Error evaluating features/target on subdf: {e}")

    # 3) compute the convex hull (with fallback for degenerate cases)
    try:
        hull = ConvexHull(pts)
    except QhullError:
        hull = ConvexHull(pts, qhull_options="QJ")

    conjectures: List[Conjecture] = []

    # 4) each hull.equations row is [a1,…,a_k, a_y, b] s.t.  a·x + a_y*y + b = 0
    for eq in hull.equations:
        a_all   = eq[:-1]   # length k+1
        b_const = eq[-1]
        a_feats = a_all[:-1]
        a_y     = a_all[-1]

        # optionally drop "side facets" that don't involve the target
        if drop_side_facets and abs(a_y) < 1e-8:
            continue

        # rewrite  a_feats·x + a_y*y + b ≤ 0  ⇒  y ≤/≥  (−a_feats·x − b)/a_y
        coeffs = -a_feats / a_y
        const  = -b_const / a_y

        # build pretty terms, skipping zeros and dropping 1*
        terms    = []
        tol      = 1e-8
        for c, feat in zip(coeffs, features):
            if abs(c) < tol:
                continue
            frac = Fraction(str(c)).limit_denominator()
            if frac == 1:
                terms.append(f"{feat.name}")
            else:
                terms.append(f"{frac}*{feat.name}")

        # append constant only if nonzero
        if abs(const) >= tol:
            const_frac = Fraction(str(const)).limit_denominator()
            terms.append(str(const_frac))

        # if nothing survived, it's just zero
        if not terms:
            rhs_name = "0"
        else:
            rhs_name = " + ".join(terms)

        # wrap in parentheses for clarity
        rhs_name = f"({rhs_name})"

        def rhs_func(
            df,
            coeffs=coeffs,
            const=const,
            feats=features
        ) -> pd.Series:
            s = const
            for c, p in zip(coeffs, feats):
                s = s + c * p(df)
            return s

        rhs_prop = Property(rhs_name, rhs_func)

        # choose inequality direction so the "inside" of the hull is satisfied
        if a_y > 0:
            ineq = Inequality(target, "≤", rhs_prop)
        else:
            ineq = Inequality(target, "≥", rhs_prop)

        conjectures.append(Conjecture(hypothesis, ineq))

    return list(set(conjectures)) if conjectures else []

__all__ = [
    "generate_hull_conjectures",
]
