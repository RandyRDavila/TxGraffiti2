# tests/unit/txgraffiti2025/processing/post/test_joint_generalize.py

import pandas as pd
import pytest

from txgraffiti2025.forms.utils import to_expr, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le
from txgraffiti2025.forms.predicates import Where, Predicate
from txgraffiti2025.processing.pre.constants_cache import (
    ConstantsCache, HypothesisConstants, ConstantRatio
)
from txgraffiti2025.processing.post.generalize import propose_joint_generalizations


def _mk_cache(df, H, items):
    key = "K"
    return ConstantsCache(
        df_index_fingerprint=tuple(range(len(df))),
        hyp_to_key={repr(H): key},
        key_to_constants={key: HypothesisConstants(hypothesis=H, mask_key=key, constants=items)}
    )


def test_joint_generalization_ratio_plus_intercept():
    # H1 ⊂ H2
    df = pd.DataFrame({
        "H1": [True, True, False, False],
        "H2": [True, True, True,  False],
        "x":  [1.0, 2.0, 3.0, 4.0],
        # Target bound to hold on H2 after joint lift:
        # y <= (num/den)*x + ((u+1)/(v+1)) where num/den=2 and (u+1)/(v+1)=3
        "y":  [2*1 + 3 - 0.1, 2*2 + 3 - 0.1, 2*3 + 3 - 0.1, 2*4 + 3 - 0.1],
        "num": [2, 2, 2, 2], "den": [1, 1, 1, 1],    # num/den == 2
        "u": [2, 2, 2, 2],   "v": [0, 0, 0, 0],     # (u+1)/(v+1) == 3/1 == 3
    })

    H1 = Where(lambda d: d["H1"])
    H2 = Where(lambda d: d["H2"])

    # Base on H1: y <= 1*x + 0  (so we must lift both slope (to 2) and intercept (to 3))
    base = Conjecture(Le("y", 1.0 * to_expr("x") + Const(0.0)), condition=H1)

    # Cache on H2 for slope and intercept ratios
    slope_expr = to_expr("num") / to_expr("den")                  # == 2
    icpt_expr  = (to_expr("u") + Const(1)) / (to_expr("v") + Const(1))  # == 3

    slope_cr = ConstantRatio("num","den",0,0,2.0,"2",3, expr=slope_expr)
    icpt_cr  = ConstantRatio("u","v",1,1,3.0,"3",3, expr=icpt_expr)
    cache = _mk_cache(df, H2, [slope_cr, icpt_cr])

    # Ask for joint proposals to H2
    props = propose_joint_generalizations(
        df, base, cache=cache, candidate_hypotheses=[H2], require_superset=True
    )

    assert len(props) >= 1
    s = repr(props[0].relation)
    # Should include both coefficient num/den and intercept (u+1)/(v+1)
    assert "num" in s and "den" in s and "u" in s and "v" in s
