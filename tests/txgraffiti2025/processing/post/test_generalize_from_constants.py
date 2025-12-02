import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, TRUE
from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.processing.pre.constants_cache import (
    precompute_constant_ratios,
    precompute_constant_ratios_pairs,
)
from txgraffiti2025.processing.post.generalize_from_constants import (
    propose_generalizations_from_constants,
)

def _H_col(name: str) -> Predicate:
    return Predicate.from_column(name)


def test_basic_generalization_replaces_constant_with_ratio():
    # Under H, y ≈ 2*x, and there exists a ratio u/v ≈ 2 (independent of y,x)
    df = pd.DataFrame({
        "H": [True, True, True, True],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.0, 4.0, 6.0, 8.0],
        "u": [2.0, 4.0, 6.0, 8.0],
        "v": [1.0, 2.0, 3.0, 4.0],  # u/v == 2 on all rows
    })
    H = _H_col("H")

    # Build a cache with H so that (u/v) is recorded as a constant ~ 2
    cache = precompute_constant_ratios(
        df, hypotheses=[H], numeric_cols=["u", "v"], shifts=[0], min_support=2, max_denominator=100
    )

    # Original conjecture: H ⇒ y ≥ 2 * x
    base = Conjecture(relation=Ge("y", 2 * to_expr("x")), condition=H, name="base")
    gens = propose_generalizations_from_constants(
        df, base, cache, candidate_hypotheses=[H, TRUE], atol=1e-9
    )

    assert len(gens) >= 1
    for g in gens:
        # New conjecture should still hold under the candidate condition
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        assert holds[applicable].all()
        assert failures.empty
        # Should be same direction and same variables (y vs x)
        assert isinstance(g.new_conjecture.relation, Ge)


def test_reject_ratio_that_depends_on_target():
    # y = 3*x and the "obvious" candidate ratio would be (y/x) == 3, but this depends on target y
    df = pd.DataFrame({
        "H": [True, True, True, True],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [3.0, 6.0, 9.0, 12.0],
    })
    H = _H_col("H")

    # Cache built from (y, x) will include (y/x) ~ 3
    cache = precompute_constant_ratios(
        df, hypotheses=[H], numeric_cols=["y", "x"], shifts=[0], min_support=2, max_denominator=100
    )

    base = Conjecture(relation=Ge("y", 3 * to_expr("x")), condition=H, name="base")
    gens = propose_generalizations_from_constants(
        df, base, cache, candidate_hypotheses=[H, TRUE], atol=1e-9
    )
    # Should reject (y/x) because it depends on the target column 'y'
    assert gens == []


def test_simplify_coeff_times_feature_cancels_feature():
    # Construct a constant ratio (u/x) == 2 under H; original conjecture uses c=2
    # The generalized RHS becomes (u/x) * x -> simplifies to u
    df = pd.DataFrame({
        "H": [True, True, True, True],
        "x": [1.0, 2.0, 3.0, 4.0],
        "u": [2.0, 4.0, 6.0, 8.0],     # u/x == 2
        "y": [2.0, 4.0, 6.0, 8.0],     # y == u, so y >= 2*x holds with equality
    })
    H = _H_col("H")

    cache = precompute_constant_ratios(
        df, hypotheses=[H], numeric_cols=["u", "x"], shifts=[0], min_support=2, max_denominator=100
    )

    base = Conjecture(relation=Ge("y", 2 * to_expr("x")), condition=H, name="base")
    gens = propose_generalizations_from_constants(
        df, base, cache, candidate_hypotheses=[H], atol=1e-9
    )

    # We expect at least one generalization via (u/x)
    assert len(gens) >= 1
    for g in gens:
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        # Since (u/x)*x simplifies to u and y == u, this is tight: y ≥ u holds
        assert holds[applicable].all()
        assert failures.empty


def test_subset_filter_only_supersets():
    # H0 ⊂ H1, and constants computed under H0 allow a generalization we’ll test on H1.
    df = pd.DataFrame({
        "H0": [False, True, True, False, True, True],
        "H1": [True, True, True, True, True, True],   # superset
        "x":  [1, 2, 3, 4, 5, 6],
        "y":  [2, 4, 6, 8, 10, 12],
        "u":  [2, 4, 6, 8, 10, 12],
        "v":  [1, 2, 3, 4, 5, 6],                      # u/v == 2
    })
    H0 = _H_col("H0")
    H1 = _H_col("H1")

    cache = precompute_constant_ratios(
        df, hypotheses=[H0], numeric_cols=["u", "v"], shifts=[0], min_support=2, max_denominator=100
    )

    base = Conjecture(relation=Ge("y", 2 * to_expr("x")), condition=H0, name="base")
    # Provide candidates including a disjoint one that is *not* a superset
    H_disjoint = Where(lambda d: pd.Series([False]*len(d), index=d.index))
    gens = propose_generalizations_from_constants(
        df, base, cache, candidate_hypotheses=[H1, H_disjoint], atol=1e-9
    )

    # We should get proposals only for H1 (the superset), not the disjoint mask
    assert len(gens) >= 1
    for g in gens:
        assert g.new_conjecture.condition is H1
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        assert holds[applicable].all()
        assert failures.empty


def test_no_matching_constants_returns_empty():
    df = pd.DataFrame({
        "H": [True, True, True, True],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [10.0, 10.0, 10.0, 10.0],  # not proportional to x with any small rational ratio present
        "u": [1.1, 2.2, 3.3, 4.4],
        "v": [1.0, 2.0, 3.0, 5.0],      # u/v is not constant
    })
    H = _H_col("H")

    cache = precompute_constant_ratios(
        df, hypotheses=[H], numeric_cols=["u", "v"], shifts=[0], min_support=3, max_denominator=50
    )

    base = Conjecture(relation=Le("y", 2 * to_expr("x")), condition=H, name="base")
    gens = propose_generalizations_from_constants(
        df, base, cache, candidate_hypotheses=[H], atol=1e-9
    )
    assert gens == []
