import pytest
import pandas as pd

from txgraffiti2025.generators.ratios import ratios
from txgraffiti2025.forms.predicates import Where, Predicate
from txgraffiti2025.forms.generic_conjecture import Ge, Le


# -----------------------
# Fixtures & helpers
# -----------------------

@pytest.fixture
def df_example():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 4, 8, 16],
        "c": [1, 1, 2, 2],
        "connected": [True, True, False, True],
    })


def connected_pred():
    return Where(lambda df: df["connected"].astype(bool))


def _H_true(df):
    return Predicate.from_column("H")


# -----------------------
# Core behavior
# -----------------------

def test_ratios_basic_bounds(df_example):
    hy = connected_pred()
    conjs = list(ratios(df_example, features=["b"], target="a", hypothesis=hy))
    assert len(conjs) == 2
    for conj in conjs:
        # condition is simplified to H (no sign slice needed since b>0 and no zeros)
        assert conj.condition is hy
        assert isinstance(conj.relation, (Ge, Le))
        # relation can be evaluated and aligns to df
        out = conj.relation.evaluate(df_example)
        assert out.index.equals(df_example.index)


def test_ratios_skips_zero_divisions():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 0, 0], "connected": [True, True, True]})
    hy = Where(lambda df: df["connected"])
    result = list(ratios(df, features=["y"], target="x", hypothesis=hy))
    # y is zero ⇒ no pos/neg slices with support ⇒ nothing emitted
    assert result == []


def test_ratios_directionality(df_example):
    hy = connected_pred()
    both = list(ratios(df_example, features=["b"], target="a", hypothesis=hy, direction="both"))
    lower = list(ratios(df_example, features=["b"], target="a", hypothesis=hy, direction="lower"))
    upper = list(ratios(df_example, features=["b"], target="a", hypothesis=hy, direction="upper"))
    assert len(both) == 2
    assert len(lower) == 1 and isinstance(lower[0].relation, Ge)
    assert len(upper) == 1 and isinstance(upper[0].relation, Le)


def test_ratios_handles_missing_column(df_example):
    hy = connected_pred()
    res = list(ratios(df_example, features=["nonexistent"], target="a", hypothesis=hy))
    assert res == []  # gracefully skip missing feature


def test_ratios_empty_hypothesis(df_example):
    df = df_example.copy()
    df["connected"] = False
    hy = connected_pred()
    res = list(ratios(df, features=["b"], target="a", hypothesis=hy))
    assert res == []


def test_ratios_value_shape(df_example):
    hy = connected_pred()
    res = list(ratios(df_example, features=["b"], target="a", hypothesis=hy))
    assert any("ratio_lower" in c.name for c in res)
    assert any("ratio_upper" in c.name for c in res)
    for conj in res:
        out = conj.relation.evaluate(df_example)
        assert out.index.equals(df_example.index)


# -----------------------
# Condition simplification & sign handling
# -----------------------

def test_positive_only_with_simplification():
    # y between 2x and 3.1x; x>0 and no zeros → condition should simplify to H
    df = pd.DataFrame({
        "H": [True, True, True, True],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 4.2, 9.0, 12.4],  # ratios: 2.0, 2.1, 3.0, 3.1
    })
    H = _H_true(df)
    conjs = list(ratios(
        df, features=["x"], target="y", hypothesis=H,
        max_denominator=50, direction="both", simplify_condition=True
    ))
    # Expect two conjectures: lower and upper, and applicable rows == H (no GT0 extra)
    assert len(conjs) == 2
    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert applicable.equals(H.mask(df))  # simplified condition
        assert holds.all()
        assert failures.empty


def test_negative_only_orientation():
    # x < 0 only; y = k x with k in {2, 3} → ratios {2,3}
    df = pd.DataFrame({
        "H": [True, True, True, True],
        "x": [-1.0, -2.0, -3.0, -4.0],
        "y": [-2.0, -6.0, -6.0, -12.0],  # k: 2, 3, 2, 3
    })
    H = _H_true(df)
    conjs = list(ratios(df, features=["x"], target="y", hypothesis=H, max_denominator=100))
    # Two conjectures again (lower uses max r, upper uses min r)
    assert len(conjs) == 2
    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert holds.all()
        assert failures.empty


def test_mixed_sign_splits_into_pos_neg_slices():
    # Mixed sign x with some zeros → expect separate pos/neg conjectures, no simplification
    df = pd.DataFrame({
        "H": [True]*8,
        "x": [-3, -2, -1, 0, 0, 1, 2, 3],
        # keep positive slice just inside the upper bound to avoid FP equality edges
        "y": [-9, -6, -2, 0, 0, 2.1, 4.2, 6.29],
    })
    H = _H_true(df)
    conjs = list(ratios(
        df, features=["x"], target="y", hypothesis=H,
        max_denominator=100, direction="both", simplify_condition=True
    ))
    # Expect up to 4 conjectures: {lower,upper} × {pos,neg}
    assert 2 <= len(conjs) <= 4
    # All emitted conjectures should hold on their applicable slice
    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert holds[applicable].all()
        assert failures.empty


def test_direction_filtering_and_min_support():
    df = pd.DataFrame({
        "H": [True, True, True],
        "x": [1.0, 2.0, 3.0],
        "y": [2.0, 4.0, 6.0],
    })
    H = _H_true(df)

    # lower only → 1 conjecture (Ge)
    conjs_lower = list(ratios(df, features=["x"], target="y", hypothesis=H, direction="lower"))
    assert len(conjs_lower) == 1 and isinstance(conjs_lower[0].relation, Ge)

    # upper only → 1 conjecture (Le)
    conjs_upper = list(ratios(df, features=["x"], target="y", hypothesis=H, direction="upper"))
    assert len(conjs_upper) == 1 and isinstance(conjs_upper[0].relation, Le)

    # Raise min_support so that slices are too small → none
    conjs_none = list(ratios(df, features=["x"], target="y", hypothesis=H, min_support=10))
    assert len(conjs_none) == 0


def test_q_clip_robustness_and_missing_feature_skipped():
    # Outlier at end; q_clip should ignore extremes for bounds
    df = pd.DataFrame({
        "H": [True]*6,
        "x": [1, 2, 3, 4, 5, 6],
        "y": [2.0, 4.1, 6.1, 8.0, 10.0, 1000.0],  # heavy outlier on last row
    })
    H = _H_true(df)
    # Include a missing feature name to ensure it's skipped gracefully
    # Focus on lower bounds; upper bounds can still be invalidated by a large high-end outlier.
    conjs = list(ratios(
        df, features=["x", "z_missing"], target="y", hypothesis=H,
        q_clip=0.1, max_denominator=100, direction="lower"
    ))
    assert len(conjs) >= 1
    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert holds[applicable].all()
        assert failures.empty


def test_hypothesis_with_no_support_emits_nothing():
    df = pd.DataFrame({
        "H": [False, False, False],
        "x": [1.0, 2.0, 3.0],
        "y": [2.0, 4.0, 6.0],
    })
    H = _H_true(df)
    conjs = list(ratios(df, features=["x"], target="y", hypothesis=H))
    assert conjs == []


def test_condition_equivalent_to_H_when_uniform_sign_no_zeros():
    # Verify simplification outcome by comparing applicable mask to H
    df = pd.DataFrame({
        "H": [True, True, True, True],
        "x": [1.0, 2.0, 3.0, 4.0],   # positive and no zeros
        "y": [2.0, 4.0, 6.0, 8.0],
    })
    H = _H_true(df)
    conjs = list(ratios(df, features=["x"], target="y", hypothesis=H, simplify_condition=True))
    assert len(conjs) == 2
    for c in conjs:
        applicable, holds, failures = c.check(df, auto_base=False)
        assert applicable.equals(H.mask(df))
        assert holds.all() and failures.empty
