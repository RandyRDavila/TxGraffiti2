# src/txgraffiti/graffiti3/runners/constant.py

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


from txgraffiti.graffiti3.exprs import Expr, to_expr
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le
from txgraffiti.graffiti3.types import HypothesisInfo

def constant_runner(
    *,
    target_col: str,
    target_expr: Expr,
    hypotheses: Sequence[HypothesisInfo],
    df: pd.DataFrame,
) -> List[Conjecture]:
    """
    Stage-0 generator: for each hypothesis h, produce constant bounds

        h ⇒ target ≥ c_min
        h ⇒ target ≤ c_max

    where c_min, c_max are taken over rows satisfying h & base.
    """
    conjs: List[Conjecture] = []
    vals = df[target_col].to_numpy(dtype=float)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        finite = mask & np.isfinite(vals)

        if finite.sum() == 0:
            continue

        v = vals[finite]
        # For constants, we expect integers; enforce that explicitly.
        c_min = int(np.min(v))
        c_max = int(np.max(v))

        rhs_min = to_expr(c_min)
        rhs_max = to_expr(c_max)

        rel_ge = Ge(left=target_expr, right=rhs_min)
        rel_le = Le(left=target_expr, right=rhs_max)

        c_ge = Conjecture(
            relation=rel_ge,
            condition=hyp.pred,
            name=f"[const-min] {target_col} under {hyp.name}",
        )

        c_le = Conjecture(
            relation=rel_le,
            condition=hyp.pred,
            name=f"[const-max] {target_col} under {hyp.name}",
        )
        c_ge.target_name = target_col
        c_le.target_name = target_col
        conjs.extend([c_ge, c_le])

    return conjs
