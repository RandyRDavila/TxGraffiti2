import types
import numpy as np
import pandas as pd
import pytest

# Core expression/predicate utilities
from txgraffiti2025.forms.utils import (
    to_expr,
    abs_,
    log,
    floor,
    ColumnTerm,
    Const,
)
from txgraffiti2025.forms.predicates import (
    EQ0,
    GE,
    LE,
    InSet,
    Between,
    Predicate,
)

# Pre-processing: hypotheses + simplification
from txgraffiti2025.processing.pre.hypotheses import (
    list_boolean_columns,
    detect_base_hypothesis,
    enumerate_boolean_hypotheses,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)

# Post-processing: ratios & generalizers
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
from txgraffiti2025.processing.post.constant_ratios import (
    find_constant_ratios_for_conjecture,
    _extract_ratio_pattern,
)
from txgraffiti2025.processing.post.generalize_from_constants import (
    propose_generalizations_from_constants,
)
from txgraffiti2025.processing.post.reciprocal_generalizer import (
    find_reciprocal_matches_for_conjecture,
    propose_generalizations_from_reciprocals,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

class TruePredicate(Predicate):
    name = "(TRUE)"
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)


def _mask(df: pd.DataFrame, pred: Predicate | None) -> pd.Series:
    if pred is None:
        return pd.Series(True, index=df.index)
    return pred.mask(df).reindex(df.index, fill_value=False).astype(bool)


def _strict_superset(df: pd.DataFrame, A: Predicate | None, B: Predicate | None) -> bool:
    """rows(A) ⊂ rows(B)"""
    a = _mask(df, A)
    b = _mask(df, B)
    return not (a & ~b).any() and (b & ~a).any()


def _df_simple(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = rng.integers(low=1, high=10, size=n)
    # t = 2 * f is an exact 2x relation
    t = 2 * f
    return pd.DataFrame({"f": f, "t": t})


def _df_for_hypotheses(n: int = 20) -> pd.DataFrame:
    """
    Build a DF with:
      - t = 2 * f (so Le(t, 2*f) is true)
      - base (boolean) true on first n//3 rows
      - broader (boolean) true on first n//2 rows (strict superset of base)
    Also include Z so that max(1/Z) == 2 on the broader slice.
    """
    f = np.arange(1, n + 1, dtype=float)
    t = 2 * f
    base = np.zeros(n, dtype=bool)
    broader = np.zeros(n, dtype=bool)
    base[: n // 3] = True
    broader[: n // 2] = True  # strict superset of base

    # Z chosen so that on the prefix, max(1/Z) = 2 (set a 0.5 once)
    Z = np.ones(n, dtype=float)
    Z[0] = 0.5  # 1/Z max = 2

    return pd.DataFrame({"f": f, "t": t, "base": base, "broader": broader, "Z": Z})


# -----------------------------------------------------------------------------
# forms/utils + predicates: smoke
# -----------------------------------------------------------------------------

def test__as_series_and_columnterm_numeric():
    df = pd.DataFrame({"x": [1, 2, 3], "b": [True, False, True]})
    # arithmetic: should coerce to float Series
    e = to_expr("x") + 1
    s = e.eval(df)
    assert s.dtype.kind == "f"
    # boolean column preserved when used directly (predicates)
    assert df["b"].dtype == bool


def test_EQ0_exact_and_tolerant():
    df = pd.DataFrame({"x": [0.0, 1e-10, 1e-6, 1.0]})
    m0 = EQ0("x").mask(df)                # exact zero
    assert m0.tolist() == [True, False, False, False]
    m1 = EQ0("x", tol=1e-9).mask(df)      # tolerant
    assert m1.tolist() == [True, True, False, False]


def test_between_and_inset_and_compare():
    df = pd.DataFrame({"k": [1, 2, 3, 4], "x": [0, 1, 2, 3]})
    assert Between("x", 1, 2).mask(df).tolist() == [False, True, True, False]
    assert InSet("k", {2, 4}).mask(df).tolist() == [False, True, False, True]
    assert GE("x", 1).mask(df).tolist() == [False, True, True, True]
    assert LE("x", 2).mask(df).tolist() == [True, True, True, False]


def test_logop_base_validation():
    df = pd.DataFrame({"y": [1, 10, 100]})
    assert np.allclose(log("y", base=10).eval(df).values, [0, 1, 2])


# -----------------------------------------------------------------------------
# pre/hypotheses + simplification: smoke
# -----------------------------------------------------------------------------

def test_nullable_boolean_and_ints():
    df = pd.DataFrame({
        "connected": pd.Series([True, True, None, True], dtype="boolean"),
        "bipartite": pd.Series([1, 0, 1, None], dtype="Int64"),
        "x": [1, 2, 3, 4],
    })
    cols = list_boolean_columns(df)
    assert set(cols) == {"connected", "bipartite"}

    H0 = detect_base_hypothesis(df)  # not all True, so TRUE
    assert isinstance(H0, Predicate)

    Hs = enumerate_boolean_hypotheses(df)
    for H in Hs:
        m = H.mask(df)
        assert m.index.equals(df.index) and m.dtype == bool


def test_pairs_strict_subset_and_names():
    df = pd.DataFrame({
        "A": [1, 1, 0, 0],  # int binary
        "B": [1, 0, 1, 0],
        "C": [1, 1, 1, 1],
    })
    Hs = enumerate_boolean_hypotheses(df, treat_binary_ints=True)
    names = [getattr(h, "name", repr(h)) for h in Hs]
    # Expect singles for A,B and the pair (in some order); C is base if include_base=True
    assert any("((A) ∧ (B))" in n or "((B) ∧ (A))" in n for n in names)


def test_simplify_handles_nullable_and_ints():
    df = pd.DataFrame({
        "A": pd.Series([1, 1, 0, None], dtype="Int64"),
        "B": pd.Series([True, None, True, False], dtype="boolean"),
        "x": [0, 1, 2, 3],
    })
    Hs = enumerate_boolean_hypotheses(df, treat_binary_ints=True)
    kept, eqs = simplify_and_dedup_hypotheses(df, Hs, min_support=1, treat_binary_ints=True)

    for H in kept:
        m = H.mask(df)
        assert m.index.equals(df.index) and m.dtype == bool

    # equivalences list should not duplicate symmetric pairs
    seen = set()
    for e in eqs:
        k = tuple(sorted([e.A.name, e.B.name]))
        assert k not in seen
        seen.add(k)


# -----------------------------------------------------------------------------
# post/constant_ratios: smoke
# -----------------------------------------------------------------------------

def test_extract_ratio_pattern_positive_and_skip_negative():
    conj_pos = Conjecture(relation=Le(to_expr("t"), Const(2) * to_expr("f")), condition=None)
    patt = _extract_ratio_pattern(conj_pos)
    assert patt is not None
    assert patt.kind == "Le"
    assert patt.target == "t"
    assert patt.feature == "f"
    assert pytest.approx(patt.coefficient) == 2.0

    # negative coefficient: skip
    conj_neg = Conjecture(relation=Le(to_expr("t"), Const(-3) * to_expr("f")), condition=None)
    patt2 = _extract_ratio_pattern(conj_neg)
    assert patt2 is None


def test_find_constant_ratios_matches_and_formats():
    # Build a DF where (A+1)/(B) is ~constant (≈2)
    df = pd.DataFrame({
        "A": [1, 3, 5, 7, 9, 11, 13, 15],   # A+1 = [2,4,6,8,10,12,14,16]
        "B": [1, 2, 3, 4, 5, 6, 7, 8],      # ratio ~ 2
        "t": [0] * 8,
        "f": [0] * 8,
    })
    conj = Conjecture(relation=Le(to_expr("t"), Const(2) * to_expr("f")), condition=TruePredicate())

    crs = find_constant_ratios_for_conjecture(
        df,
        conj,
        numeric_cols=["A", "B"],
        shifts=(0, 1),
        atol=1e-9,
        rtol=1e-9,
        min_support=6,
        max_denominator=10,
    )
    assert len(crs) > 0
    # Expect at least one with (A+1)/B ≈ 2
    ok = [
        cr for cr in crs
        if cr.numerator == "A" and cr.denominator == "B" and cr.shift_num == 1 and cr.shift_den == 0
    ]
    assert len(ok) >= 1
    # display should be "2" (exact rational)
    assert any(cr.value_display == "2" for cr in ok)


# -----------------------------------------------------------------------------
# post/generalize_from_constants: strict-superset behavior
# -----------------------------------------------------------------------------

def test_propose_generalizations_from_constants_accepts_identical_coeff(monkeypatch):
    """
    constants_matching_coeff returns an expression equal to the learned coefficient (2).
    With the strict-superset rule AND original hypothesis TRUE, we expect **no** generalizations.
    """
    df = _df_simple(n=12)
    conj = Conjecture(relation=Le(to_expr("t"), Const(2) * to_expr("f")), condition=TruePredicate())

    # Fake cache + monkeypatch constants_matching_coeff to return Const(2)
    class DummyCache: ...
    cache = DummyCache()

    def fake_constants_matching_coeff(cache_, cond, coeff, atol=1e-9):
        # mimic objects with .expr
        return [types.SimpleNamespace(expr=Const(2))]

    monkeypatch.setattr(
        "txgraffiti2025.processing.pre.constants_cache.constants_matching_coeff",
        fake_constants_matching_coeff,
        raising=True,
    )

    gens = propose_generalizations_from_constants(
        df, conj, cache, candidate_hypotheses=[conj.condition], atol=1e-9
    )
    # Original hypothesis is TRUE; no strict superset exists → expect none
    assert len(gens) == 0


# -----------------------------------------------------------------------------
# post/reciprocal_generalizer: strict-superset behavior
# -----------------------------------------------------------------------------

def test_find_reciprocal_matches_and_propose_generalizations():
    """
    Use column Z so that max(1/Z) == 2 on the slice.
    We should find raw reciprocal matches, but with hypothesis TRUE we **do not**
    accept generalized conjectures (strict-superset rule).
    """
    df = pd.DataFrame({
        "f": [1, 2, 3, 4, 5],
        "t": [2, 4, 6, 8, 10],          # t = 2*f holds exactly
        "Z": [0.5, 1.0, 4.0, 2.0, 10.0], # 1/Z max = 2.0 (at 0.5)
    })
    conj = Conjecture(relation=Le(to_expr("t"), Const(2) * to_expr("f")), condition=TruePredicate())

    # Raw matches should appear
    matches = find_reciprocal_matches_for_conjecture(
        df, conj, candidate_cols=["Z"], shifts=(0,), min_support=3, atol=1e-9, rtol=1e-9
    )
    assert len(matches) >= 1
    assert any(abs(m.extremum_value - 2.0) < 1e-9 for m in matches)

    # Proposals should be empty (no strict superset of TRUE)
    gens = propose_generalizations_from_reciprocals(
        df,
        conj,
        candidate_hypotheses=[conj.condition],
        candidate_cols=["Z"],
        shifts=(0,),
        min_support=3,
        atol=1e-9,
        rtol=1e-9,
    )
    assert len(gens) == 0


def test_constants_generalization_requires_strict_superset(monkeypatch):
    df = _df_for_hypotheses(n=24)
    conj = Conjecture(
        relation=Le(to_expr("t"), Const(2) * to_expr("f")),
        condition=Predicate.from_column("base"),
    )

    # monkeypatch constants_matching_coeff to return Const(2) as a structural constant
    class DummyCache: ...
    cache = DummyCache()

    def fake_constants_matching_coeff(cache_, cond, coeff, *, atol=1e-9, rtol=1e-9):
        return [types.SimpleNamespace(expr=Const(2))]

    monkeypatch.setattr(
        "txgraffiti2025.processing.pre.constants_cache.constants_matching_coeff",
        fake_constants_matching_coeff,
        raising=True,
    )

    H_base = Predicate.from_column("base")
    H_broader = Predicate.from_column("broader")

    # Only same hypothesis -> expect zero generalizations
    gens_same_only = propose_generalizations_from_constants(
        df, conj, cache, candidate_hypotheses=[H_base], atol=1e-9
    )
    assert len(gens_same_only) == 0

    # Include a strict superset -> expect at least one generalization
    gens_with_broader = propose_generalizations_from_constants(
        df, conj, cache, candidate_hypotheses=[H_base, H_broader], atol=1e-9
    )
    assert len(gens_with_broader) >= 1
    for g in gens_with_broader:
        assert _strict_superset(df, conj.condition, g.new_conjecture.condition)
        assert g.new_conjecture.is_true(df)


def test_reciprocal_generalization_requires_strict_superset():
    df = _df_for_hypotheses(n=24)
    conj = Conjecture(
        relation=Le(to_expr("t"), Const(2) * to_expr("f")),
        condition=Predicate.from_column("base"),
    )

    H_base = Predicate.from_column("base")
    H_broader = Predicate.from_column("broader")

    # Only same hypothesis -> expect zero generalizations
    gs_same = propose_generalizations_from_reciprocals(
        df,
        conj,
        candidate_hypotheses=[H_base],
        candidate_cols=["Z"],
        shifts=(0,),
        min_support=4,
        atol=1e-9,
        rtol=1e-9,
    )
    assert len(gs_same) == 0

    # Include strict superset -> expect some generalizations
    gs_broader = propose_generalizations_from_reciprocals(
        df,
        conj,
        candidate_hypotheses=[H_base, H_broader],
        candidate_cols=["Z"],
        shifts=(0,),
        min_support=4,
        atol=1e-9,
        rtol=1e-9,
    )
    assert len(gs_broader) >= 1
    for c in gs_broader:
        assert _strict_superset(df, conj.condition, c.condition)
        assert c.is_true(df)


def test_reciprocal_generalization_from_TRUE_emits_none():
    df = _df_for_hypotheses(n=24)
    conj = Conjecture(
        relation=Le(to_expr("t"), Const(2) * to_expr("f")),
        condition=None,  # TRUE
    )
    H_broader = Predicate.from_column("broader")
    gs = propose_generalizations_from_reciprocals(
        df,
        conj,
        candidate_hypotheses=[H_broader],
        candidate_cols=["Z"],
        shifts=(0,),
        min_support=4,
        atol=1e-9,
        rtol=1e-9,
    )
    assert len(gs) == 0
