# src/txgraffiti2025/processing/post/refine_numeric.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import Expr, Const, to_expr, BinOp
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate

__all__ = ["refine_numeric_bounds", "RefinementConfig"]


# --------------------- config ---------------------

@dataclass
class RefinementConfig:
    """
    Controls which candidates we try. We only construct Exprs that are supported
    by our expression system; floor/ceil/log are attempted *iff* the Expr exposes
    methods floor_/ceil_/log_ (we check with hasattr).
    """
    try_whole_rhs_floor: bool = True
    try_whole_rhs_ceil: bool = False   # useful for lower bounds; depends on Expr support
    try_prod_floor: bool = False       # only if (C*G).floor_() exists; else skipped
    try_coeff_round_const: bool = True # if the coefficient C is numeric Const, floor/ceil it
    try_intercept_round_const: bool = True  # if the intercept B is numeric Const, floor/ceil it
    try_sqrt_intercept: bool = True    # uses ** Const(0.5) (domain-checked)
    try_log_intercept: bool = False    # only if intercept has .log_()
    # Keep only candidates that are pointwise tighter than baseline on the mask.
    require_tighter: bool = True


# --------------------- helpers ---------------------

def _mask(df: pd.DataFrame, cond: Optional[Predicate]) -> pd.Series:
    if cond is None:
        return pd.Series(True, index=df.index)
    m = cond.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)

def _kind(conj: Conjecture) -> Optional[str]:
    r = conj.relation
    if isinstance(r, Ge): return "Ge"
    if isinstance(r, Le): return "Le"
    return None

def _mk(kind: str, left: Expr, right: Expr):
    return Ge(left, right) if kind == "Ge" else Le(left, right)

def _is_numeric_const(e: Expr) -> bool:
    try:
        return isinstance(e, Const) and np.isfinite(float(e.value))
    except Exception:
        return False

def _rhs_decompose_mul_add(rhs: Expr) -> Tuple[Optional[Expr], Optional[Expr]]:
    """
    Try to see RHS as (product) + (intercept). Returns (product, intercept) or (None, None).
    We only handle one top-level + or -; for subtraction, we treat as prod + (-inter).
    """
    if isinstance(rhs, BinOp) and rhs.fn in (np.add, np.subtract):
        prod, inter = rhs.left, rhs.right
        if rhs.fn is np.subtract:
            inter = -inter
        return prod, inter
    return None, None

def _expr_has(e: Expr, name: str) -> bool:
    return hasattr(e, name) and callable(getattr(e, name))

def _eval_series(e, df: pd.DataFrame) -> pd.Series:
    """
    Evaluate `e` to a float Series aligned to df.index.

    Supports:
      - pandas Series (reindexed/coerced)
      - scalars (float/int)
      - numpy arrays (length == len(df) or scalar)
      - column name strings
      - Expr trees (Const/ColumnTerm/BinOp/UnaryOp/LogOp/LinearForm/etc.)

    Values may be NaN where domain is invalid (e.g., log<=0, sqrt<0).
    """
    # Already a Series
    if isinstance(e, pd.Series):
        return pd.to_numeric(e.reindex(df.index), errors="coerce")

    # Scalars
    if isinstance(e, (int, float)):
        return pd.Series(float(e), index=df.index, dtype=float)

    # Numpy array
    if isinstance(e, np.ndarray):
        if e.ndim == 0:
            return pd.Series(float(e), index=df.index, dtype=float)
        if e.shape[0] != len(df):
            raise ValueError("Array length does not match DataFrame length.")
        return pd.Series(e.astype(float), index=df.index)

    # Column name string
    if isinstance(e, str):
        if e not in df.columns:
            raise KeyError(f"Required column '{e}' not found in DataFrame.")
        return pd.to_numeric(df[e], errors="coerce")

    # Expr tree (covers Const / ColumnTerm / BinOp / UnaryOp / LogOp / LinearForm / etc.)
    if isinstance(e, Expr):
        # const fast-path
        if isinstance(e, Const):
            try:
                val = float(e.value)
            except Exception:
                val = np.nan
            return pd.Series(val, index=df.index, dtype=float)

        # generic Expr evaluation
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            out = e.eval(df)  # expected to yield a Series aligned to df.index
        return pd.to_numeric(out.reindex(df.index), errors="coerce")

    # Objects that expose .eval(df) (very last resort)
    if hasattr(e, "eval"):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            out = e.eval(df)
        return pd.to_numeric(pd.Series(out, index=df.index), errors="coerce")

    raise AttributeError(f"Cannot evaluate expression node to Series: {e!r}")

def _finite(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def _domain_ok_sqrt(x: pd.Series) -> bool:
    return (x.dropna() >= 0).all()

def _domain_ok_log(x: pd.Series) -> bool:
    return (x.dropna() > 0).all()

def _floor_for_Le_ceil_for_Ge(kind: str, val: float) -> float:
    """Upper bound (Le) tightens by flooring; lower bound (Ge) tightens by ceiling."""
    return math.floor(val) if kind == "Le" else math.ceil(val)

def _round_all_numeric_consts(expr: Expr, kind: str) -> Optional[Expr]:
    """
    Return a copy of expr with every numeric Const rounded in the safe tightening
    direction (floor for Le, ceil for Ge). If nothing changes, return None.
    """
    changed = False

    def _round_val(v: float) -> float:
        return math.floor(v) if kind == "Le" else math.ceil(v)

    def _visit(e: Expr) -> Expr:
        nonlocal changed
        if isinstance(e, Const) and np.isfinite(float(e.value)):
            v = float(e.value)
            nv = _round_val(v)
            if nv != v:
                changed = True
                return Const(nv)
            return e
        if isinstance(e, BinOp):
            l2 = _visit(e.left)
            r2 = _visit(e.right)
            if (l2 is not e.left) or (r2 is not e.right):
                return BinOp(e.fn, l2, r2)
            return e
        return e

    out = _visit(expr)
    return out if changed else None


# --------------------- main API ---------------------

def refine_numeric_bounds(
    df: pd.DataFrame,
    conj: Conjecture,
    *,
    config: RefinementConfig = RefinementConfig(),
) -> List[Conjecture]:
    """
    Generate variants of `conj` by applying numeric-tightening transforms to the RHS.
    Only returns conjectures that remain TRUE on the same hypothesis, and (by default)
    are pointwise strictly tighter than the original RHS on the mask.

    We handle composite RHS of the form  (C*G) + B  where C could itself be a ratio Expr
    and B could be another Expr (e.g., sqrt(inv4)). Any transform that cannot be expressed
    by our Expr system is skipped.
    """
    kind = _kind(conj)
    if kind is None:
        return []

    mask = _mask(df, conj.condition)
    if not mask.any():
        return []

    left = conj.relation.left
    rhs0 = conj.relation.right

    # baseline numeric RHS on the mask
    baseline = _finite(_eval_series(rhs0, df)).loc[mask]
    if baseline.isna().any():
        # undefined RHS on mask → nothing to refine safely
        return []

    prod_expr, inter_expr = _rhs_decompose_mul_add(rhs0)

    candidates: List[Expr] = []

    # ---- Whole RHS floor/ceil if methods exist on the expression ----
    if config.try_whole_rhs_floor and _expr_has(rhs0, "floor_"):
        candidates.append(rhs0.floor_())
    if config.try_whole_rhs_ceil and _expr_has(rhs0, "ceil_"):
        candidates.append(rhs0.ceil_())

    # ---- Piecewise transforms when we can see product + intercept ----
    if prod_expr is not None and inter_expr is not None:
        # product floor if supported
        if config.try_prod_floor and _expr_has(prod_expr, "floor_"):
            candidates.append(prod_expr.floor_() + inter_expr)

        # If product looks like C * G, we can round C and/or B (intercept)
        if isinstance(prod_expr, BinOp) and prod_expr.fn is np.multiply:
            C, G = prod_expr.left, prod_expr.right

            # Round C alone
            if config.try_coeff_round_const and _is_numeric_const(C):
                cval = float(C.value)
                c_new = _floor_for_Le_ceil_for_Ge(kind, cval)
                if np.isfinite(c_new):
                    candidates.append(Const(c_new) * G + inter_expr)

            # Round B alone
            if config.try_intercept_round_const and _is_numeric_const(inter_expr):
                bval = float(inter_expr.value)
                b_new = _floor_for_Le_ceil_for_Ge(kind, bval)
                if np.isfinite(b_new):
                    candidates.append(C * G + Const(b_new))

            # Round both C and B together (often gives the cleanest integer form)
            if (config.try_coeff_round_const and _is_numeric_const(C)
                and config.try_intercept_round_const and _is_numeric_const(inter_expr)):
                cval = float(C.value)
                bval = float(inter_expr.value)
                c_new = _floor_for_Le_ceil_for_Ge(kind, cval)
                b_new = _floor_for_Le_ceil_for_Ge(kind, bval)
                if np.isfinite(c_new) and np.isfinite(b_new):
                    candidates.append(Const(c_new) * G + Const(b_new))

        # sqrt/log transforms on intercept (expressible)
        if config.try_sqrt_intercept:
            try:
                inter_num = _finite(_eval_series(inter_expr, df)).loc[mask]
                if inter_num.notna().all() and _domain_ok_sqrt(inter_num):
                    candidates.append(prod_expr + (inter_expr ** Const(0.5)))
            except Exception:
                pass

        if config.try_log_intercept and _expr_has(inter_expr, "log_"):
            inter_num = _finite(_eval_series(inter_expr, df)).loc[mask]
            if inter_num.notna().all() and _domain_ok_log(inter_num):
                candidates.append(prod_expr + inter_expr.log_())

    # --- fallback: round all numeric constants anywhere in RHS (structure-agnostic) ---
    fallback = _round_all_numeric_consts(rhs0, kind)
    if fallback is not None:
        candidates.append(fallback)

    # Deduplicate structurally
    uniq: List[Expr] = []
    seen = set()
    for e in candidates:
        key = repr(e)
        if key not in seen:
            seen.add(key)
            uniq.append(e)

    # Evaluate and keep only true (and optionally tighter) variants
    out: List[Conjecture] = []
    y = _finite(_eval_series(left, df)).loc[mask]

    eps = 1e-12  # strictness slack
    for rhs in uniq:
        try:
            s = _finite(_eval_series(rhs, df)).loc[mask]
        except Exception:
            continue
        if s.isna().any():
            continue

        if kind == "Le":
            # must still be true
            truth = (y <= s).all()
            # non-worse everywhere (<= baseline + eps), strict somewhere (< baseline - eps)
            nonworse = (s <= baseline + eps).all()
            strict_somewhere = (s < baseline - eps).any()
            tighter_ok = nonworse and (strict_somewhere if config.require_tighter else True)
        else:  # Ge
            truth = (y >= s).all()
            # non-worse everywhere (>= baseline - eps), strict somewhere (> baseline + eps)
            nonworse = (s >= baseline - eps).all()
            strict_somewhere = (s > baseline + eps).any()
            tighter_ok = nonworse and (strict_somewhere if config.require_tighter else True)

        if truth and (tighter_ok or not config.require_tighter):
            out.append(Conjecture(
                relation=_mk(kind, left, rhs),
                condition=conj.condition,
                name=(conj.name or "refined") + "_refined"
            ))

    return out
