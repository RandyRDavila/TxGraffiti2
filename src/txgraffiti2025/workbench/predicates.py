# src/txgraffiti2025/workbench/predicates.py

from __future__ import annotations
from typing import List
from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from .arrays import mask_from_pred, as_series
import numpy as np

def _mk_name(kind: str, conj: Conjecture) -> str:
    Hname = conj.condition.pretty() if hasattr(conj.condition, "pretty") else repr(conj.condition)
    lhs   = conj.relation.left.pretty()
    rhs   = conj.relation.right.pretty()
    tag = {"onbound":"==", "strict_lt":"<", "strict_gt":">"}.get(kind, "?")
    return f"[{Hname}] :: {lhs} {tag} {rhs}"

def _touch_mask(conj: Conjecture, df, rtol=1e-8, atol=1e-8) -> np.ndarray:
    lhs = as_series(conj.relation.left.eval(df), df.index).astype(float)
    rhs = as_series(conj.relation.right.eval(df), df.index).astype(float)
    Hm  = mask_from_pred(df, conj.condition)
    eq  = np.isclose(lhs.values, rhs.values, rtol=rtol, atol=atol)
    return Hm & eq

def _strict_mask(conj: Conjecture, df, side: str, rtol=1e-8, atol=1e-8) -> np.ndarray:
    lhs = as_series(conj.relation.left.eval(df), df.index).astype(float)
    rhs = as_series(conj.relation.right.eval(df), df.index).astype(float)
    Hm  = mask_from_pred(df, conj.condition)
    tol = np.maximum(atol, rtol * np.maximum(1.0, np.abs(rhs.values)))
    if isinstance(conj.relation, Le):
        strict = (lhs.values < rhs.values - tol) if side == "lt" else (lhs.values > rhs.values + tol)
    elif isinstance(conj.relation, Ge):
        strict = (lhs.values > rhs.values + tol) if side == "gt" else (lhs.values < rhs.values - tol)
    else:
        raise ValueError("Conjecture relation must be Le or Ge")
    return Hm & strict

def predicates_from_conjecture(
    conj: Conjecture,
    *,
    make_eq: bool = True,
    make_strict: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> List[Predicate]:
    preds: List[Predicate] = []
    if make_eq:
        def fn_eq(df, _c=conj): return _touch_mask(_c, df, rtol=rtol, atol=atol)
        p = Where(fn=fn_eq, name=_mk_name("onbound", conj))
        p._derived_hypothesis = conj.condition
        preds.append(p)
    if make_strict:
        if isinstance(conj.relation, Le):
            def fn_lt(df, _c=conj): return _strict_mask(_c, df, side="lt", rtol=rtol, atol=atol)
            p = Where(fn=fn_lt, name=_mk_name("strict_lt", conj))
            p._derived_hypothesis = conj.condition
            preds.append(p)
        elif isinstance(conj.relation, Ge):
            def fn_gt(df, _c=conj): return _strict_mask(_c, df, side="gt", rtol=rtol, atol=atol)
            p = Where(fn=fn_gt, name=_mk_name("strict_gt", conj))
            p._derived_hypothesis = conj.condition
            preds.append(p)
    return preds
