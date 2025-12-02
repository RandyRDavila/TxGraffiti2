# tests/unit/txgraffiti2025/generators/test_convex.py

import pandas as pd
import numpy as np

from txgraffiti2025.generators.convex import convex_hull
from txgraffiti2025.forms.predicates import Where, Predicate
from txgraffiti2025.forms.generic_conjecture import Ge, Le


def _H_true(df):
    return Where(lambda d: pd.Series(True, index=d.index))


def test_convex_basic_bounds_and_holds():
    # y between ~2x and ~3x under H
    rng = np.arange(1.0, 6.0)
    df = pd.DataFrame({
        "H": [True] * len(rng),
        "x": rng,
        "y": np.array([2.0, 2.2, 2.8, 2.5, 2.9]) * rng,  # inside [2x, 3x]
    })
    H = Predicate.from_column("H")

    conjs = list(convex_hull(
        df, features=["x"], target="y", hypothesis=H,
        max_denominator=None,          # <-- use raw float facets to avoid tightening
        drop_side_facets=True
    ))

    assert len(conjs) >= 1
    assert all(isinstance(c.relation, (Ge, Le)) for c in conjs)
    assert all(c.condition is H for c in conjs)

    # Each emitted conjecture should hold on applicable rows (allow tiny FP slack)
    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert applicable.equals(H.mask(df))
        # If any numerical blip remains, ensure it's within 1e-9 slack
        if not holds[applicable].all():
            sl = c.relation.slack(df).loc[applicable]
            assert float(sl.min()) >= -1e-9

def test_convex_drops_side_facets_vertical_edges():
    # A rectangle in (x,y): hull facets include x=0 and x=1 (vertical) → drop_side_facets removes them
    df = pd.DataFrame({
        "H": [True]*5,
        "x": [0.0, 0.0, 1.0, 1.0, 0.5],
        "y": [0.0, 1.0, 0.0, 1.0, 0.5],
    })
    H = Predicate.from_column("H")

    conjs = list(convex_hull(
        df, features=["x"], target="y", hypothesis=H,
        max_denominator=100, drop_side_facets=True
    ))

    # We should not get conjectures that correspond to vertical sides (a_y ~ 0)
    # Expect bounds only in y (upper/lower-ish).
    assert len(conjs) >= 1
    assert all(isinstance(c.relation, (Ge, Le)) for c in conjs)

    # Names should reflect 'upper'/'lower' on y, not extra facet types
    names = " ".join(c.name for c in conjs)
    assert ("upper" in names) or ("lower" in names)

    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert holds[applicable].all()
        assert failures.empty


def test_convex_handles_colinear_with_qj():
    # Colinear points (degenerate hull) → ConvexHull may need QJ joggling; ensure it doesn't crash
    x = np.array([0, 1, 2, 3, 4], dtype=float)
    y = 2.0 * x
    df = pd.DataFrame({"H": [True]*len(x), "x": x, "y": y})
    H = Predicate.from_column("H")

    conjs = list(convex_hull(
        df, features=["x"], target="y", hypothesis=H,
        max_denominator=100, drop_side_facets=True
    ))

    # Should emit at least one inequality (depending on joggling, could be 1 or 2)
    assert len(conjs) >= 1
    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert holds[applicable].all()
        assert failures.empty


def test_convex_min_support_and_missing_feature():
    # Only 2 valid points with one feature → min_support defaults to k+2 = 3 → emits nothing
    df_small = pd.DataFrame({"H": [True, True], "x": [0.0, 1.0], "y": [0.0, 2.0]})
    H = Predicate.from_column("H")

    no_conjs = list(convex_hull(df_small, features=["x"], target="y", hypothesis=H))
    assert no_conjs == []

    # Missing feature list → gracefully skip
    df = pd.DataFrame({"H": [True, True, True], "x": [1, 2, 3], "y": [2, 4, 6]})
    none_feats = list(convex_hull(df, features=["missing"], target="y", hypothesis=H))
    assert none_feats == []


def test_convex_respects_mask_and_nan_filtering():
    # Half rows masked out; also introduce NaNs that should be filtered
    df = pd.DataFrame({
        "H": [False, True, True, False, True, True],
        "x": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
        "y": [0.0, 2.1, 4.2, 6.0, np.nan, 10.5],
    })
    H = Predicate.from_column("H")

    conjs = list(convex_hull(
        df, features=["x"], target="y", hypothesis=H,
        max_denominator=50, drop_side_facets=True
    ))

    # Either we get a bound (if enough valid points remain) or nothing; in either case, no crash.
    if conjs:
        for c in conjs:
            applicable, holds, failures = c.check(df, auto_base=False)
            # Only applicable where H is True (NaNs already filtered internally for facet fit)
            assert holds[applicable].all()
            assert failures.empty


def test_convex_no_support_when_hypothesis_false_everywhere():
    df = pd.DataFrame({"H": [False, False], "x": [1.0, 2.0], "y": [2.0, 4.0]})
    H = Predicate.from_column("H")
    conjs = list(convex_hull(df, features=["x"], target="y", hypothesis=H))
    assert conjs == []
