"""
Constant-ratio discovery under a conjecture's hypothesis.

Given a Conjecture and a DataFrame, find column ratios (and small shifted ratios)
that are constant on the hypothesis rows, e.g.
    (max_degree / min_degree) == 1
or
    (min_degree - 1) / radius == 2/3

We also try small integer shifts on numerator/denominator:
    (A + a) / (B + b)  where a, b ∈ { -2, -1, 0, 1, 2 }  (configurable)

Additionally, if the conjecture is of the ratio type
    target ≤/≥ c * feature
we flag all discovered constants that are numerically equal to c (within tol).

This is a *post* tool: no core classes changed; you can call it from a notebook
or pipeline step to surface structural constants suggested by the current class.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
from fractions import Fraction

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from txgraffiti2025.forms.utils import Expr, Const, ColumnTerm, BinOp, UnaryOp, to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.pretty import format_expr  # for readable formulas


# --------------------------
# Helpers
# --------------------------

def _is_numeric_series(s: pd.Series) -> bool:
    # numeric but not boolean
    return is_numeric_dtype(s) and not is_bool_dtype(s)

def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _is_constant(series: pd.Series, *, atol: float, rtol: float, min_support: int) -> Tuple[bool, float]:
    """
    Check if `series` is (approximately) constant over finite values.
    Returns (is_const, const_value).
    """
    v = _finite(series).dropna()
    if len(v) < min_support:
        return (False, np.nan)
    vmax = v.max()
    vmin = v.min()
    span = float(vmax - vmin)
    tol = float(atol + rtol * max(1.0, np.abs(v).max()))
    if span <= tol:
        # representative value: median for stability
        return (True, float(v.median()))
    return (False, np.nan)

def _ratio_series(num: pd.Series, den: pd.Series) -> pd.Series:
    x = _finite(num)
    y = _finite(den)
    # exclude zeros in denominator
    y = y.replace(0.0, np.nan)
    return x / y

def _rationalize(x: float, max_den: Optional[int]) -> Fraction | float:
    if max_den is None:
        return float(x)
    return Fraction(x).limit_denominator(max_den)

# --------------------------
# Extract (target, coeff, feature) from ratio-style conjectures
# --------------------------

@dataclass
class RatioPattern:
    kind: str            # "Ge" or "Le"
    target: str
    feature: str
    coefficient: float   # numeric for matching; keep your pretty via Const(Fraction) in the relation

def _extract_ratio_pattern(conj: Conjecture) -> Optional[RatioPattern]:
    """
    Recognize relations of the shape:
        target <= coeff * feature
        target >= coeff * feature
    where `target` and `feature` are column names and `coeff` is Const(...).
    """
    rel = conj.relation
    kind = rel.__class__.__name__  # "Ge" or "Le" (or "Eq" -> we bail)
    if isinstance(rel, Eq):
        return None

    left = getattr(rel, "left", None)
    right = getattr(rel, "right", None)

    # We expect left = ColumnTerm(target), right = BinOp(mul, Const(k), ColumnTerm(feature)) or flipped
    def _match_mul(expr: Expr) -> Optional[Tuple[float, str]]:
        if isinstance(expr, BinOp) and getattr(expr, "fn", None) is np.multiply:  # type: ignore
            L, R = expr.left, expr.right
            # (Const * Col) or (Col * Const)
            if isinstance(L, Const) and isinstance(R, ColumnTerm):
                return (float(L.value), R.col)
            if isinstance(R, Const) and isinstance(L, ColumnTerm):
                return (float(R.value), L.col)
        return None

    # try pattern
    if isinstance(left, ColumnTerm):
        m = _match_mul(right)
        if m is not None:
            coeff, feat = m
            return RatioPattern(kind=kind, target=left.col, feature=feat, coefficient=float(coeff))

    # also allow flipped: (coeff*feat) on the left and ColumnTerm(target) on the right
    if isinstance(right, ColumnTerm):
        m = _match_mul(left)
        if m is not None:
            coeff, feat = m
            # inequality flips sides; normalize to target on left
            # e.g., (k*feat) <= target  -> target >= k*feat  (Ge)
            #       (k*feat) >= target  -> target <= k*feat  (Le)
            if kind == "Le":
                new_kind = "Ge"
            elif kind == "Ge":
                new_kind = "Le"
            else:
                new_kind = kind
            return RatioPattern(kind=new_kind, target=right.col, feature=feat, coefficient=float(coeff))

    return None

# --------------------------
# Main API
# --------------------------

@dataclass
class ConstantRatio:
    numerator: str
    denominator: str
    shift_num: int
    shift_den: int
    value_float: float
    value_display: str      # e.g. "1/2" or "2/3" or "1"
    support: int
    formula_expr: Expr      # Expr for (num+shift_num)/(den+shift_den)
    matches_conj_coeff: bool

def find_constant_ratios_for_conjecture(
    df: pd.DataFrame,
    conj: Conjecture,
    *,
    numeric_cols: Optional[Sequence[str]] = None,
    shifts: Sequence[int] = (-2, -1, 0, 1, 2),
    atol: float = 1e-9,
    rtol: float = 1e-9,
    min_support: int = 8,
    max_denominator: Optional[int] = 50,
) -> List[ConstantRatio]:
    """
    Discover (A + a)/(B + b) that are constant under conj.condition.

    Parameters
    ----------
    df : DataFrame
    conj : Conjecture
        The conjecture whose hypothesis (condition) defines the class.
    numeric_cols : list[str], optional
        Which columns to consider as numeric invariants. Default: all numeric (non-bool) columns.
    shifts : iterable of int, default (-2,-1,0,1,2)
        Small integer offsets to try on numerator and denominator.
    atol, rtol : float
        Absolute/relative tolerances for "constant" test.
    min_support : int
        Minimum number of usable rows under the hypothesis to accept a constant.
    max_denominator : int or None
        Rationalize constants via Fraction.limit_denominator for display.

    Returns
    -------
    list[ConstantRatio]
    """
    mask = (conj.condition.mask(df) if conj.condition is not None else pd.Series(True, index=df.index))
    mask = mask.reindex(df.index, fill_value=False).astype(bool)
    if not mask.any():
        return []

    data = df.loc[mask]

    # decide numeric columns
    if numeric_cols is None:
        numeric_cols = [c for c in data.columns if _is_numeric_series(data[c])]

    patt = _extract_ratio_pattern(conj)

    out: List[ConstantRatio] = []
    for i, num in enumerate(numeric_cols):
        for j, den in enumerate(numeric_cols):
            if i == j:
                continue
            sN_all = pd.to_numeric(data[num], errors="coerce")
            sD_all = pd.to_numeric(data[den], errors="coerce")

            for a in shifts:
                for b in shifts:
                    sN = sN_all + float(a)
                    sD = sD_all + float(b)
                    rr = _ratio_series(sN, sD)
                    is_const, val = _is_constant(rr, atol=atol, rtol=rtol, min_support=min_support)
                    if not is_const:
                        continue

                    # rationalize for display
                    disp = _rationalize(val, max_denominator)
                    if isinstance(disp, Fraction):
                        if disp.denominator == 1:
                            value_display = f"{disp.numerator}"
                        else:
                            value_display = f"{disp.numerator}/{disp.denominator}"
                    else:
                        # float
                        value_display = str(int(disp)) if float(disp).is_integer() else f"{disp}"

                    # build Expr for (num+a)/(den+b)
                    num_expr = to_expr(num) if a == 0 else (to_expr(num) + Const(a))
                    den_expr = to_expr(den) if b == 0 else (to_expr(den) + Const(b))
                    formula_expr = num_expr / den_expr

                    matches = False
                    if patt is not None:
                        matches = np.isclose(val, patt.coefficient, atol=atol, rtol=rtol)

                    out.append(ConstantRatio(
                        numerator=num,
                        denominator=den,
                        shift_num=int(a),
                        shift_den=int(b),
                        value_float=float(val),
                        value_display=value_display,
                        support=int(rr.dropna().shape[0]),
                        formula_expr=formula_expr,
                        matches_conj_coeff=bool(matches),
                    ))

    # sort: first those that match the conjecture's coeff, then by support desc, then shorter shifts
    out.sort(key=lambda cr: (
        not cr.matches_conj_coeff,
        -cr.support,
        abs(cr.shift_num) + abs(cr.shift_den),
        cr.numerator, cr.denominator, cr.value_float
    ))
    return out
