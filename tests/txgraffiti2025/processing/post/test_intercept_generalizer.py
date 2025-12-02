import numpy as np
import pandas as pd
import pytest

from txgraffiti2025.forms.utils import to_expr, Const, BinOp
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Where, Predicate
from txgraffiti2025.processing.post.intercept_generalizer import (
    propose_generalizations_from_intercept,
)
from txgraffiti2025.processing.pre.constants_cache import (
    ConstantsCache,
    HypothesisConstants,
    ConstantRatio,
)


def _mk_cache_for(
    df: pd.DataFrame,
    hyp: Predicate,
    items,
) -> ConstantsCache:
    """
    Build a minimal ConstantsCache mapping repr(hyp) -> items.

    `items` should be either:
      - a list of ConstantRatio (preferred; has .expr),
      - or any iterable of objects exposing .expr or .value (float-like).
    """
    key = "K"
    return ConstantsCache(
        df_index_fingerprint=tuple(range(len(df))),
        hyp_to_key={repr(hyp): key},
        key_to_constants={key: HypothesisConstants(hypothesis=hyp, mask_key=key, constants=list(items))},
    )


# -------------------------------------------------------------------------
# 1) cache ratio Expr replaces + intercept and proposal holds on superset
# -------------------------------------------------------------------------
def test_cache_ratio_expr_addition_and_superset_holds():
    # H1 ⊂ H2
    df = pd.DataFrame({
        "H1": [True, True, False, False],
        "H2": [True, True, True,  False],
        "x":  [1.0, 2.0, 3.0, 4.0],
        # make all H2 rows satisfy y <= 3x + 2
        "y":  [4.9, 7.9, 10.9, 17.0],
        "num": [1, 1, 1, 1],
        "den": [0, 0, 0, 0],
    })
    H1 = Where(lambda d: d["H1"])
    H2 = Where(lambda d: d["H2"])

    # base conjecture: y <= 3x + 2 on H1
    base = Conjecture(
        relation=Le("y", 3.0 * to_expr("x") + Const(2.0)),
        condition=H1,
        name="base_le_add2",
    )

    # cache for H2: (num+1)/(den+1) == 2
    ratio_expr = (to_expr("num") + Const(1)) / (to_expr("den") + Const(1))
    cr = ConstantRatio(
        numerator="num", denominator="den",
        shift_num=1, shift_den=1,
        value_float=2.0, value_display="2", support=3,
        expr=ratio_expr,
    )
    cache = _mk_cache_for(df, H2, [cr])

    # Try to generalize to H2 using cache; require H2 ⊇ H1
    props = propose_generalizations_from_intercept(
        df, base, cache,
        candidate_hypotheses=[H2],
        require_superset=True,
    )

    # Expect at least one (should swap +2 -> +((num+1)/(den+1)))
    assert len(props) >= 1
    # And it should be true on H2
    for g in props:
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        assert applicable.equals(H2.mask(df))
        assert holds[applicable].all()
        assert failures.empty
        # sanity: RHS contains the ratio structure
        text = repr(g.new_conjecture.relation)
        assert "num" in text and "den" in text


# -------------------------------------------------------------------------
# 2) subtraction: ensure we replace the correct child and preserve the minus
# -------------------------------------------------------------------------
def test_cache_numeric_value_subtraction_preserved():
    df = pd.DataFrame({
        "H1": [True, True, False],
        "H2": [True, True, True],
        "x":  [1.0, 2.0, 3.0],
        # We will target y >= 2x - 0.5 on H2 after replacement.
        "y":  [1.6, 3.6, 5.9],
    })
    H1 = Where(lambda d: d["H1"])
    H2 = Where(lambda d: d["H2"])

    # base: y >= 2x - 1 on H1  (intercept is the right child of subtraction)
    base = Conjecture(
        relation=Ge("y", 2.0 * to_expr("x") - Const(1.0)),
        condition=H1,
        name="base_ge_minus1",
    )

    # supply cache numeric 0.5 for H2 (no Expr, numeric-only path)
    class _NumOnly:
        def __init__(self, value): self.value = value
        support = 3
        text = "0.5"

    cache = _mk_cache_for(df, H2, [_NumOnly(0.5)])

    props = propose_generalizations_from_intercept(
        df, base, cache, candidate_hypotheses=[H2], require_superset=True
    )
    assert len(props) >= 1

    # Verify form is y >= 2x - 0.5 and it holds
    for g in props:
        rel = g.new_conjecture.relation
        # minus preserved (cheap check)
        assert "-" in repr(rel)
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        assert applicable.equals(H2.mask(df))
        assert holds[applicable].all()
        assert failures.empty


# -------------------------------------------------------------------------
# 3) user-provided candidate Exprs + superset filter
# -------------------------------------------------------------------------
def test_user_candidates_and_superset_filter():
    df = pd.DataFrame({
        "H1": [True, True,  False, False],
        "H2": [True, True,  True,  False],   # H1 ⊂ H2
        "x":  [1.0, 2.0,   3.0,   4.0],
        "D":  [3,   3,     4,     5],       # Δ
        # We want y <= 5x + (Δ-3) to hold on H2
        "y":  [5.0*1 + 0,  5.0*2 + 0,  5.0*3 + 1, 5.0*4 + 2],
    })
    H1 = Where(lambda d: d["H1"])
    H2 = Where(lambda d: d["H2"])

    base = Conjecture(
        relation=Le("y", 5.0 * to_expr("x") + Const(0.0)),
        condition=H1,
        name="base_le_5x",
    )

    # User-provided candidate (Δ - 3)
    cand_intercepts = [to_expr("D") - Const(3)]

    props = propose_generalizations_from_intercept(
        df, base, cache=None,
        candidate_hypotheses=[H2],
        candidate_intercepts=cand_intercepts,
        require_superset=True,
    )
    assert len(props) >= 1
    for g in props:
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        assert applicable.equals(H2.mask(df))
        assert holds[applicable].all()
        assert failures.empty


# -------------------------------------------------------------------------
# 4) relaxers Z that vanish on base; keep only true proposals
# -------------------------------------------------------------------------
def test_relaxers_vanish_on_base_and_keep_only_true():
    # Build H1 ⊂ H2 with delta==3 on H1, >=3 elsewhere.
    df = pd.DataFrame({
        "H1":   [True, True, False, False],
        "H2":   [True, True, True,  False],
        "x":    [2.0, 3.0, 4.0, 5.0],
        "delta":[3.0, 3.0, 4.0, 5.0],
        # We want y <= x + |delta-3| to be true on H2
        "y":    [2.0 + 0.0, 3.0 + 0.0, 4.0 + 1.0, 5.0 + 2.0],
    })
    H1 = Where(lambda d: d["H1"])
    H2 = Where(lambda d: d["H2"])

    # base: y <= x + 0 on H1
    base = Conjecture(
        relation=Le("y", to_expr("x") + Const(0.0)),
        condition=H1,
        name="base_le_x",
    )

    # Relaxer Z = |delta - 3| (vanishes on H1)
    Z = (to_expr("delta") - Const(3))
    # we'll try both +Z and -Z; only +Z should plausibly be true on H2
    props = propose_generalizations_from_intercept(
        df, base, cache=None,
        candidate_hypotheses=[H2],
        relaxers_Z=[Z.abs_() if hasattr(Z, "abs_") else (Z * Z) ** Const(0.5)],  # be robust: |Z|
        require_superset=True,
    )
    # At least one true proposal expected (the +Z one)
    assert len(props) >= 1
    for g in props:
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        assert applicable.equals(H2.mask(df))
        assert holds[applicable].all()
        assert failures.empty


# -------------------------------------------------------------------------
# 5) keep only true — bogus candidate must not survive
# -------------------------------------------------------------------------
def test_only_true_kept_bogus_candidate_dropped():
    df = pd.DataFrame({
        "H1": [True, True, False],
        "H2": [True, True, True],
        "x":  [1.0, 2.0, 3.0],
        "y":  [3.0, 6.0, 9.0],
        "B":  [100, 100, 100],
    })
    H1 = Where(lambda d: d["H1"])
    H2 = Where(lambda d: d["H2"])

    base = Conjecture(
        relation=Le("y", 3.0 * to_expr("x") + Const(0.0)),
        condition=H1,
        name="base_le_3x",
    )

    # Candidate intercept that would make the bound too tight to hold
    bad = to_expr("B") - Const(2000)
    props = propose_generalizations_from_intercept(
        df, base, cache=None,
        candidate_hypotheses=[H2],
        candidate_intercepts=[bad],
        require_superset=True,
    )
    assert props == []


# -------------------------------------------------------------------------
# 6) cache adapter with real ConstantRatio/HypothesisConstants shapes
# -------------------------------------------------------------------------
def test_cache_adapter_real_shapes():
    df = pd.DataFrame({
        "H": [True, True, True],
        "x": [1.0, 2.0, 3.0],
        "y": [3.0*1 + 1.0, 3.0*2 + 1.0, 3.0*3 + 1.0],  # y <= 3x + 1
        "num": [0, 0, 0],    # (num+2)/(den+1) = 1 if num=0, den=0
        "den": [0, 0, 0],
    })
    H = Where(lambda d: d["H"])

    base = Conjecture(
        relation=Le("y", 3.0 * to_expr("x") + Const(1.0)),
        condition=H,
        name="base_le_3x_plus1",
    )

    expr = (to_expr("num") + Const(2)) / (to_expr("den") + Const(1))  # == 2/1 == 2 (won't be used)
    # also include a numeric-only "value" entry to ensure adapter can read it
    cr1 = ConstantRatio("num", "den", 2, 1, 1.0, "1", 3,  # this one matches intercept=1
                        expr=(to_expr("num") + Const(1)) / (to_expr("den") + Const(1)))
    cr2 = ConstantRatio("num", "den", 2, 1, 2.0, "2", 3, expr=expr)

    cache = _mk_cache_for(df, H, [cr1, cr2])

    props = propose_generalizations_from_intercept(
        df, base, cache, candidate_hypotheses=[H], require_superset=False
    )
    # Should propose at least one; they’re all tested for truth.
    assert len(props) >= 1
    for g in props:
        applicable, holds, failures = g.new_conjecture.check(df, auto_base=False)
        assert applicable.equals(H.mask(df))
        assert holds[applicable].all()
        assert failures.empty
