# src/txgraffiti2025/processing/post/reciprocal_generalizer.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from txgraffiti2025.forms.utils import to_expr, Const, Expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate
# reuse the ratio-pattern extractor
from txgraffiti2025.processing.post.constant_ratios import _extract_ratio_pattern

def _mask(df: pd.DataFrame, cond: Optional[Predicate]) -> pd.Series:
    if cond is None:
        return pd.Series(True, index=df.index)
    return cond.mask(df).reindex(df.index, fill_value=False).astype(bool)

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]

@dataclass
class ReciprocalMatch:
    column: str
    shift: int
    extremum_value: float     # min or max of 1/(col+shift) over hypothesis
    support: int
    expr: Expr                # 1 / (col + shift)

def find_reciprocal_matches_for_conjecture(
    df: pd.DataFrame,
    conj: Conjecture,
    *,
    candidate_cols: Optional[Sequence[str]] = None,
    shifts: Sequence[int] = (0, 1),   # try Δ, Δ+1 by default
    atol: float = 1e-9,
    rtol: float = 1e-9,
    min_support: int = 8,
) -> List[ReciprocalMatch]:
    """
    Look for 1/(B + s) whose extremum on conj.condition equals the learned coefficient c.

    For target ≥ c*feature: use min(1/(B+s)) ≈ c.
    For target ≤ c*feature: use max(1/(B+s)) ≈ c.
    """
    patt = _extract_ratio_pattern(conj)
    if patt is None:
        return []  # only for ratio-style conjectures

    cols = list(candidate_cols) if candidate_cols is not None else _numeric_cols(df)
    m = _mask(df, conj.condition)
    D = df.loc[m]

    out: List[ReciprocalMatch] = []
    for col in cols:
        if col not in D.columns:
            continue
        s_all = pd.to_numeric(D[col], errors="coerce")
        if s_all.notna().sum() < min_support:
            continue

        for s in shifts:
            denom = s_all + float(s)
            denom = denom.replace(0.0, np.nan)
            inv = 1.0 / denom
            inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
            if inv.size < min_support:
                continue

            val = float(inv.min()) if isinstance(conj.relation, Ge) else float(inv.max())
            if np.isclose(val, float(patt.coefficient), atol=atol, rtol=rtol):
                expr = Const(1) / (to_expr(col) + Const(s)) if s != 0 else Const(1) / to_expr(col)
                out.append(ReciprocalMatch(
                    column=col, shift=int(s), extremum_value=val,
                    support=int(inv.size), expr=expr
                ))
    # sort: more support first, then small |shift|
    out.sort(key=lambda r: (-r.support, abs(r.shift), r.column))
    return out

def propose_generalizations_from_reciprocals(
    df: pd.DataFrame,
    conj: Conjecture,
    *,
    candidate_hypotheses: Sequence[Optional[Predicate]],
    candidate_cols: Optional[Sequence[str]] = None,
    shifts: Sequence[int] = (0, 1),
    min_support: int = 8,
    atol: float = 1e-9,
) -> List[Conjecture]:
    """
    Replace c by 1/(B+s) and try on more general hypotheses; keep those that are true.
    """
    from txgraffiti2025.processing.post.constant_ratios import _extract_ratio_pattern  # ensure available
    patt = _extract_ratio_pattern(conj)
    if patt is None:
        return []

    matches = find_reciprocal_matches_for_conjecture(
        df, conj, candidate_cols=candidate_cols, shifts=shifts, min_support=min_support, atol=atol, rtol=atol
    )
    if not matches:
        return []

    # helper: A ⊆ B?
    def _subset(A: Optional[Predicate], B: Optional[Predicate]) -> bool:
        a = _mask(df, A); b = _mask(df, B)
        return not (a & ~b).any()

    new_conjs: List[Conjecture] = []
    for rm in matches:
        coeff_expr = rm.expr  # 1/(B+s)
        # build relation with Expr coefficient
        if patt.kind == "Ge":
            rel = Ge(to_expr(patt.target), coeff_expr * to_expr(patt.feature))
        else:
            rel = Le(to_expr(patt.target), coeff_expr * to_expr(patt.feature))

        for Hsup in candidate_hypotheses:
            # require original condition ⊆ superset
            if conj.condition is not None and not _subset(conj.condition, Hsup):
                continue
            cand = Conjecture(relation=rel, condition=Hsup,
                              name=f"gen_recip_{patt.kind}_{patt.target}_vs_{patt.feature}")
            if cand.is_true(df):
                new_conjs.append(cand)

    return new_conjs
