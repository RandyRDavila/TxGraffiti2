import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.processing.pre.constants_cache import (
    precompute_constant_ratios,
    constants_matching_coeff,
    build_pairwise_hypotheses,
    precompute_constant_ratios_pairs,
    _mask_for, _mask_key,  # internal helpers used for a light sanity check
)

# -----------------------------
# Helpers / fixtures
# -----------------------------

def _H_col(name="H"):
    return Predicate.from_column(name)

def _df_basic():
    # b = 2*a exactly; c = a + 1; d = 3*c - 2
    a = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    b = 2.0 * a
    c = a + 1.0
    d = 3.0 * c - 2.0
    return pd.DataFrame({
        "H": True,
        "a": a, "b": b, "c": c, "d": d,
        "flag1": [1, 0, 1, 1, 0, 1],  # boolean-like numeric
        "flag2": [0, 1, 1, 0, 1, 0],
        "cat":   ["x", "y", "x", "y", "x", "y"],  # non-numeric
    })

# -----------------------------
# Core cache tests
# -----------------------------

def test_precompute_detects_simple_ratio_and_formats_value():
    df = _df_basic()
    H = _H_col("H")
    cache = precompute_constant_ratios(
        df,
        hypotheses=[H],
        numeric_cols=["a", "b", "c", "d"],
        shifts=(0,),           # pure ratio (no shifts)
        min_support=3,
        max_denominator=50,    # expect rational "2"
    )
    # There should be an entry for H
    key = _mask_key(_mask_for(df, H))
    assert key in cache.key_to_constants

    consts = cache.constants_for(H)
    assert isinstance(consts, list) and len(consts) >= 1

    # Find b/a == 2
    hits = [cr for cr in consts
            if cr.numerator == "b" and cr.denominator == "a"
            and cr.shift_num == 0 and cr.shift_den == 0
            and np.isclose(cr.value_float, 2.0)]
    assert hits, "Expected to find the constant ratio (b/a)=2"
    cr = hits[0]
    assert cr.value_display in ("2", "2.0")  # rationalized to "2" is preferred

    # Expr should evaluate to b/a on the applicable mask
    m = H.mask(df)
    out = cr.expr.eval(df.loc[m])
    np.testing.assert_allclose(out.to_numpy(), (df.loc[m, "b"] / df.loc[m, "a"]).to_numpy())

def test_constants_matching_coeff_finds_expected_constant():
    df = _df_basic()
    H = _H_col("H")
    cache = precompute_constant_ratios(
        df,
        hypotheses=[H],
        numeric_cols=["a", "b"],
        shifts=(0,),
        min_support=3,
        max_denominator=100,
    )
    matches = constants_matching_coeff(cache, H, coeff=2.0, atol=1e-12, rtol=1e-12)
    assert any(cr.numerator == "b" and cr.denominator == "a" for cr in matches)

def test_precompute_ignores_non_numeric_and_skips_zero_denominators():
    # y has zeros → y as denominator with shift 0 should be dropped
    df = pd.DataFrame({
        "H": [True, True, True, True],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 0.0, 0.0, 0.0],  # all zeros
        "z": [1.0, 1.0, 1.0, 1.0],
        "cat": ["a", "b", "a", "b"],  # non-numeric
    })
    H = _H_col("H")
    cache = precompute_constant_ratios(
        df,
        hypotheses=[H],
        numeric_cols=None,  # let it auto-detect numeric (non-bool) cols
        shifts=(0,),        # no shift → zero denom stays zero -> NaN -> no constant
        min_support=3,
    )
    consts = cache.constants_for(H)
    # No ratio with denominator 'y' at shift 0 should appear
    assert all(not (cr.denominator == "y" and cr.shift_den == 0) for cr in consts)

# -----------------------------
# Pairwise hypothesis builder
# -----------------------------

def test_build_pairwise_hypotheses_respects_base_and_filters_empty():
    df = _df_basic()
    base = _H_col("H")  # all True
    Hs = build_pairwise_hypotheses(df, base, boolean_cols=["flag1", "flag2"])
    # Should include base and both base∧flag predicates
    assert len(Hs) >= 3
    # First should be base (mask-identical)
    base_mask = base.mask(df)
    assert base_mask.equals(_mask_for(df, Hs[0]))
    # Others produce strictly smaller applicable masks
    for H in Hs[1:]:
        m = _mask_for(df, H)
        assert m.any() and (m <= base_mask).all()

def test_precompute_constant_ratios_pairs_runs_and_builds_cache():
    df = _df_basic()
    base = _H_col("H")
    cache = precompute_constant_ratios_pairs(
        df,
        base,
        boolean_cols=["flag1", "flag2"],
        numeric_cols=["a", "b", "c"],
        shifts=(0,),
        min_support=3,
        max_denominator=20,
    )
    # Should at least have an entry for base
    key = _mask_key(_mask_for(df, base))
    assert key in cache.key_to_constants
    # Querying constants_for(base) returns a list (possibly empty, but present)
    assert isinstance(cache.constants_for(base), list)

def test_rationalization_none_keeps_decimal_string():
    df = _df_basic()
    H = _H_col("H")
    # Create a non-nice constant via shifts: (a+1)/(a) ~ 1 + 1/a (not constant),
    # instead use b/c = (2a)/(a+1) which varies, so we craft a controlled one:
    # Let u = 1.25 * a  and v = a  -> u/v = 1.25
    df2 = df.copy()
    df2["u"] = 1.25 * df2["a"]
    df2["v"] = df2["a"]
    cache = precompute_constant_ratios(
        df2,
        hypotheses=[H],
        numeric_cols=["u", "v"],
        shifts=(0,),
        min_support=3,
        max_denominator=None,   # keep as float
    )
    consts = cache.constants_for(H)
    hit = [cr for cr in consts if cr.numerator == "u" and cr.denominator == "v"]
    assert hit
    # value_display should be a decimal-like "1.25", not a fraction
    assert any("." in cr.value_display for cr in hit)
