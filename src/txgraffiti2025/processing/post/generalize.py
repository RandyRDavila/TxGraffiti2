"""
Generalize post-processor for affine upper bounds:

    (H) -> target <= m*other + b

We try to replace constant m with a data-driven ratio (X+δ1)/(Y+δ2) (δ ∈ {-2,-1,0,1,2}),
chosen so that on the seed rows (where H holds) the ratio equals m (≈ within tol).
Then we weaken H to the most general hypothesis that still makes the inequality true.

Design:
- Numeric columns only; exclude booleans.
- Candidate ratios allow same column if shifts differ; skip trivial (X,δ)==(Y,δ).
- Among all ratios that match m on the seed, pick the one whose best weakening has
  the largest support (row count); tie-break by largest global variance of the ratio.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Relation
from txgraffiti2025.forms.utils import Expr, Const, ColumnTerm, BinOp, to_expr
from txgraffiti2025.forms.predicates import Predicate, AndPred
from txgraffiti2025.processing.utils import log_event


# -------------------------
# Pattern matching: target <= m*other + b
# -------------------------

@dataclass
class AffinePattern:
    target: Expr
    other: Expr
    m_const: float
    b_const: float

def _is_const(e: Expr) -> Optional[float]:
    return e.value if isinstance(e, Const) else None

def _is_col(e: Expr) -> Optional[str]:
    return e.col if isinstance(e, ColumnTerm) else None

def _is_binop(e: Expr, ufunc_name: str) -> bool:
    return isinstance(e, BinOp) and getattr(e.fn, "__name__", "") == ufunc_name

def _match_affine_le(rel: Relation) -> Optional[AffinePattern]:
    if not isinstance(rel, Le):
        return None
    left = rel.left
    right = rel.right

    # right ≡ add(mul(m, other), b) with m,b optional (default 1,0)
    m_val = 1.0
    b_val = 0.0
    mult = right

    if _is_binop(right, "add"):
        A, B = right.left, right.right
        a_const = _is_const(A)
        b_const = _is_const(B)
        if a_const is not None and b_const is None:
            b_val = a_const; mult = B
        elif b_const is not None and a_const is None:
            b_val = b_const; mult = A
        else:
            # ambiguous sum; give up
            return None

    if _is_binop(mult, "multiply"):
        A, B = mult.left, mult.right
        a_const = _is_const(A)
        b_const = _is_const(B)
        if a_const is not None and b_const is None:
            m_val = a_const; other = B
        elif b_const is not None and a_const is None:
            m_val = b_const; other = A
        else:
            return None
    else:
        # right ≡ other + b (i.e., m=1)
        other = mult

    return AffinePattern(target=left, other=other, m_const=float(m_val), b_const=float(b_val))


# -------------------------
# Hypothesis handling
# -------------------------

def _flatten_and(p: Predicate) -> List[Predicate]:
    if isinstance(p, AndPred):
        return _flatten_and(p.a) + _flatten_and(p.b)
    return [p]

def _generate_weakenings(cond: Optional[Predicate]) -> List[Optional[Predicate]]:
    """
    Order: drop exactly one conjunct, then drop 2..n-1, then None, then original.
    """
    if cond is None:
        return [None]
    parts = _flatten_and(cond)
    n = len(parts)
    out: List[Optional[Predicate]] = []
    from itertools import combinations

    # drop exactly one
    if n >= 2:
        for drop in combinations(range(n), 1):
            keep = [parts[i] for i in range(n) if i not in drop]
            c = keep[0]
            for p in keep[1:]:
                c = c & p
            out.append(c)
    # drop 2..n-1
    if n >= 3:
        for k in range(2, n):
            for kept_idx in combinations(range(n), n - k):
                keep = [parts[i] for i in kept_idx]
                if not keep:
                    continue
                c = keep[0]
                for p in keep[1:]:
                    c = c & p
                out.append(c)
    # None then original
    out.append(None)
    c = parts[0]
    for p in parts[1:]:
        c = c & p
    out.append(c)
    return out

def _mask(cond: Optional[Predicate], df: pd.DataFrame) -> pd.Series:
    return cond.mask(df) if cond is not None else pd.Series(True, index=df.index)


# -------------------------
# Ratio candidates
# -------------------------

def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]

def _shifted_col(col: str, k: int) -> Expr:
    return BinOp(np.add, ColumnTerm(col), Const(float(k)))

def _ratio_expr(num: Expr, den: Expr) -> Expr:
    return BinOp(np.divide, num, den)

def _candidate_ratios(df: pd.DataFrame) -> List[Tuple[str,int,str,int,Expr]]:
    cols = _numeric_columns(df)
    deltas = [-2, -1, 0, 1, 2]
    cands: List[Tuple[str,int,str,int,Expr]] = []
    for X in cols:
        for Y in cols:
            for d1 in deltas:
                for d2 in deltas:
                    # skip identical (X+δ)/(X+δ)
                    if (X == Y) and (d1 == d2):
                        continue
                    num = _shifted_col(X, d1) if d1 != 0 else ColumnTerm(X)
                    den = _shifted_col(Y, d2) if d2 != 0 else ColumnTerm(Y)
                    cands.append((X, d1, Y, d2, _ratio_expr(num, den)))
    return cands


# -------------------------
# Matching / selection
# -------------------------

def _approx_equal_all(a: np.ndarray, b: float, tol: float = 1e-9) -> bool:
    a = np.asarray(a, dtype=float)
    good = np.isfinite(a)
    if not np.any(good):
        return False
    return np.all(np.isclose(a[good], b, atol=tol, rtol=0.0))



def generalize_one(conj: Conjecture, df: pd.DataFrame, *, tol_match: float = 1e-9) -> Conjecture:
    """
    Generalize a linear conjecture to ratio form, preferring non-None weakenings and stable tie-breaks.
    """
    pat = _match_affine_le(conj.relation)
    if pat is None:
        return conj

    seed_mask = _mask(conj.condition, df)
    if seed_mask.sum() == 0:
        return conj

    m = pat.m_const
    cands = _candidate_ratios(df)
    weakenings = _generate_weakenings(conj.condition)

    best_nonnone = None
    best_none = None

    # simple per-column stats for tie-breaking
    col_stats = {}
    for c in _numeric_columns(df):
        arr = df[c].to_numpy(dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            col_stats[c] = (0.0, 0.0, 0.0)
        else:
            col_stats[c] = (
                float(np.nanmax(finite) - np.nanmin(finite)),  # range
                float(np.nanvar(finite)),                      # variance
                float(np.nanmean(finite)),                     # mean
            )

    for (X, d1, Y, d2, expr) in cands:
        try:
            vals = np.asarray(expr.eval(df), dtype=float)
            if not np.any(np.isfinite(vals)):
                continue
            seed_vals = vals[seed_mask.to_numpy()]
            if not np.any(np.isfinite(seed_vals)) or not _approx_equal_all(seed_vals, m, tol_match):
                continue

            new_right = BinOp(np.add, BinOp(np.multiply, expr, pat.other), Const(pat.b_const))
            new_rel = Le(pat.target, new_right)
            ok_mask = new_rel.evaluate(df)

            local_best = None
            local_best_stats = None  # will hold (median_R_minus_m)

            for H in weakenings:
                Hmask = _mask(H, df)
                supp = int(Hmask.sum())
                if supp == 0:
                    continue
                if bool((ok_mask[Hmask]).all()):
                    is_non_none = H is not None
                    # compute margin on this weakening (upper bound: prefer median(R) - m >= 0)
                    vals_H = vals[Hmask.to_numpy()]
                    med = float(np.nanmedian(vals_H)) if np.any(np.isfinite(vals_H)) else float("-inf")
                    margin = med - float(m)
                    cand = (supp, is_non_none)
                    if (local_best is None) or (cand > (local_best[0], local_best[1])) or \
                       (cand == (local_best[0], local_best[1]) and (local_best_stats is None or margin > local_best_stats)):
                        local_best = (supp, is_non_none, H)
                        local_best_stats = margin
            if local_best is None:
                continue

            support, is_non_none, H = local_best
            same_col = int(X == Y and d1 != d2)
            shift_mag = abs(int(d1)) + abs(int(d2))
            rngX, varX, meanX = col_stats.get(X, (0,0,0))
            ratio_var = float(np.nanvar(vals))
            name_bias = (("max" in X) - ("min" in X))

            # margin (for the chosen H) used as a global tie-break too
            Hmask = _mask(H, df)
            vals_H = vals[Hmask.to_numpy()]
            med = float(np.nanmedian(vals_H)) if np.any(np.isfinite(vals_H)) else float("-inf")
            margin = med - float(m)

            # Primary: support, then margin (prefer >=0), then sameCol, -shift, num_strength, ratio_var, name_bias
            key = (support, margin, same_col, -shift_mag, (rngX, varX, meanX), ratio_var, name_bias)

            desc = f"({X}{d1:+d})/({Y}{d2:+d})" if d1 or d2 else f"({X})/({Y})"
            bucket = best_nonnone if is_non_none else best_none
            if bucket is None or key > bucket[0]:
                if is_non_none:
                    best_nonnone = (key, desc, expr, H)
                else:
                    best_none = (key, desc, expr, H)
        except Exception:
            continue

    pick = best_nonnone or best_none
    if pick is None:
        return conj

    _, desc, expr, H = pick
    new_right = BinOp(np.add, BinOp(np.multiply, expr, pat.other), Const(pat.b_const))
    new_rel = Le(pat.target, new_right)
    new_name = getattr(conj, "name", "conj") + f" | generalized {desc}"
    log_event(f"Generalize: {conj.name} -> {new_name}")
    return Conjecture(new_rel, H, name=new_name)


def generalize(conjectures: Iterable[Conjecture], df: pd.DataFrame, *, tol_match: float = 1e-9) -> List[Conjecture]:
    out: List[Conjecture] = []
    for c in conjectures:
        try:
            out.append(generalize_one(c, df, tol_match=tol_match))
        except Exception as e:
            log_event(f"Generalize error on {getattr(c,'name','?')}: {e}")
            out.append(c)
    return out
