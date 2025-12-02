# src/txgraffiti2025/generators/convex.py

"""
Convex-hull-based conjecture generator (R2 family).

Build linear inequality conjectures of the form

    H ⇒ target ≥ RHS      or      H ⇒ target ≤ RHS

by computing the convex hull of feature–target points restricted to a
hypothesis mask. Each facet (supporting hyperplane) of the hull induces
a linear bound on the target in terms of the features.

Details
-------
Given points in R^{k+1} with coordinates (x_1, …, x_k, y), each hull facet
is represented by a normal vector a = (a_1, …, a_k, a_y) and offset b0,
meaning:

    a_1 x_1 + … + a_k x_k + a_y y + b0 <= 0        (facet halfspace)

If |a_y| is not tiny, rearrange to:

    y  <=  (-a_1/a_y) x_1 + … + (-a_k/a_y) x_k + (-b0/a_y)   if a_y > 0
    y  >=  (-a_1/a_y) x_1 + … + (-a_k/a_y) x_k + (-b0/a_y)   if a_y < 0

Facets with |a_y| ~ 0 do not bound y directly and can be dropped.

Parameters
----------
df : pd.DataFrame
features : list[str]
target : str
hypothesis : Predicate
max_denominator : int, default 100
    Rationalize coefficients via Fraction.limit_denominator.
drop_side_facets : bool, default True
    Drop facets with |a_y| < tol.
tol : float, default 1e-8
    Numerical tolerance for filtering tiny coefficients.
min_support : int, default k+2
    Minimum rows (after mask & numeric filtering) needed for a (k+1)-D hull.

Notes
-----
- Uses scipy.spatial.ConvexHull; when Qhull fails (degeneracy), retries with
  qhull_options="QJ" (joggling) for robustness.
- Coefficients are converted to pretty rationals for readable output; actual
  evaluation remains numeric.

Example
-------
>>> import pandas as pd
>>> from txgraffiti2025.forms.predicates import Where
>>> from txgraffiti2025.generators.convex import convex_hull
>>> df = pd.DataFrame({"H":[True]*5, "x":[1,2,3,4,5], "y":[2,4,6,8,10]})
>>> H = Where(lambda d: d["H"])
>>> list(convex_hull(df, features=["x"], target="y", hypothesis=H))[:2]  # doctest: +ELLIPSIS
[Conjecture(...), Conjecture(...)]
"""

from __future__ import annotations
from fractions import Fraction
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from txgraffiti2025.forms.utils import Const, to_expr, Expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.utils.safe_generator import safe_generator

__all__ = ["convex_hull"]


def _numify(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _to_const(val: float, max_denominator: Optional[int]) -> Const:
    if max_denominator is None:
        return Const(float(val))
    return Const(Fraction(float(val)).limit_denominator(max_denominator))

def _rhs_expr_from_facet(
    a_feats: np.ndarray,
    a_y: float,
    b0: float,
    used_feats: List[str],
    *,
    max_denominator: Optional[int],
    tol: float,
) -> Expr:
    """
    Build RHS for the facet inequality:
        if a_y > 0:  y <= Σ (-a_i/a_y) x_i  + (-b0/a_y)
        if a_y < 0:  y >= Σ (-a_i/a_y) x_i  + (-b0/a_y)
    This returns only the RHS Expr (coefficients assembled). Caller decides Ge/Le.
    """
    # intercept term
    intercept = -b0 / a_y
    rhs: Expr = _to_const(intercept, max_denominator)

    # feature terms
    for coef, col in zip(-a_feats / a_y, used_feats):
        if abs(coef) < tol:
            continue
        rhs = rhs + (_to_const(coef, max_denominator) * to_expr(col))

    return rhs

def _collect_points(
    df: pd.DataFrame,
    *,
    features: List[str],
    target: str,
    mask: pd.Series,
) -> Tuple[np.ndarray, pd.Index, List[str]]:
    """
    Build the (k+1)-dim matrix of points [X | y] for rows where all
    selected columns are numeric and present.
    """
    sub = df.loc[mask]
    # keep only existing features
    cols = [c for c in features if c in sub.columns]
    if target not in sub.columns or len(cols) == 0:
        return np.empty((0, 0)), sub.index[:0], cols

    X_df = pd.concat([_numify(sub[c]) for c in cols], axis=1)
    y_s = _numify(sub[target])

    valid = (~X_df.isna().any(axis=1)) & y_s.notna()
    if valid.sum() == 0:
        return np.empty((0, 0)), sub.index[:0], cols

    pts = np.column_stack([X_df.loc[valid].to_numpy(dtype=float), y_s.loc[valid].to_numpy(dtype=float)])
    return pts, sub.index[valid], cols

@safe_generator
def convex_hull(
    df: pd.DataFrame,
    *,
    features: List[str],
    target: str,
    hypothesis: Predicate,
    max_denominator: int = 100,
    drop_side_facets: bool = True,
    tol: float = 1e-8,
    min_support: Optional[int] = None,  # defaults to k+2
) -> Iterator[Conjecture]:
    """
    Generate linear inequality conjectures from hull facets under a hypothesis.
    """
    # Applicability mask from hypothesis
    Hmask = hypothesis.mask(df).reindex(df.index, fill_value=False).astype(bool)
    if not Hmask.any():
        return

    pts, valid_idx, used_feats = _collect_points(df, features=features, target=target, mask=Hmask)
    k = len(used_feats)
    if k == 0 or pts.shape[0] == 0:
        return

    # Minimal rows for a (k+1)-dimensional hull: k+2 points
    needed = (k + 2) if min_support is None else int(min_support)
    if pts.shape[0] < needed:
        return

    # Compute hull; jog if degenerate
    try:
        hull = ConvexHull(pts)
    except QhullError:
        hull = ConvexHull(pts, qhull_options="QJ")

    # Each row: a_1..a_k, a_y, b0  (for halfspace a·z + b0 <= 0)
    for eq in hull.equations:
        a_all, b0 = eq[:-1], eq[-1]
        a_feats, a_y = a_all[:-1], a_all[-1]

        if drop_side_facets and abs(a_y) < tol:
            continue

        # Orientation: a_y > 0 → y <= RHS ; a_y < 0 → y >= RHS
        if abs(a_y) < tol:
            # Skip numerically vertical/side facets (don’t bound y)
            continue

        rhs_expr = _rhs_expr_from_facet(
            a_feats, a_y, b0, used_feats,
            max_denominator=max_denominator, tol=tol
        )
        lhs_expr = to_expr(target)

        if a_y > 0:
            rel = Le(lhs_expr, rhs_expr)
            name = f"convex_upper_{target}_vs_{'_'.join(used_feats)}"
        else:
            rel = Ge(lhs_expr, rhs_expr)
            name = f"convex_lower_{target}_vs_{'_'.join(used_feats)}"

        yield Conjecture(relation=rel, condition=hypothesis, name=name)
