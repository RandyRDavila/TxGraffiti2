# src/txgraffiti2025/generators/ratios_exprs.py
"""
Ratio-based conjecture generator (R2) with Expr features.

Emits, under a hypothesis H:
    c_min <= y / other <= c_max   ⇒
    H ⇒ y ≥ c_min * other
    H ⇒ y ≤ c_max * other

- Features can be column names or Exprs (sqrt("x"), log("x"), ("x")**2, ...).
- If multiple features are provided, they are multiplied to form "other".
- Finite-only guard: drops rows with NaN or ±inf in y or other.
- Mixed-sign handling:
    * If "other" is strictly >0 (or strictly <0) under H, emit bounds under H.
    * If "other" is mixed sign:
        - If allow_mixed_sign_splits=True, split with extra masks
          (H ∧ other>0) and (H ∧ other<0) using a MaskPredicate (works for composite Exprs).
        - Else skip safely.

Naming:
    {name_prefix}_{lower|upper}_{target}_vs_{pretty_other}_{pos|neg}
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.utils import Expr, Const, to_expr
from txgraffiti2025.utils.safe_generator import safe_generator

__all__ = ["ratios_bounds", "RatiosConfig"]


# ───────────────────────── Pred: mask wrapper for composite splits ───────────────────────── #

class MaskPredicate(Predicate):
    """
    A simple predicate backed by a boolean Series mask (over df.index).
    Useful to express (other>0) or (other<0) when 'other' is a composite Expr.
    """
    def __init__(self, mask: pd.Series, name: str = "mask"):
        self._mask = mask.astype(bool)
        self._name = name

    def mask(self, df: pd.DataFrame) -> pd.Series:
        return self._mask.reindex(df.index, fill_value=False)

    def __repr__(self) -> str:
        return f"Mask({self._name})"


# ───────────────────────── Helpers ───────────────────────── #

def _finite(s: pd.Series) -> pd.Series:
    return s.notna() & np.isfinite(s.to_numpy(dtype=float))

def _const_from_rational(x: float) -> Const:
    try:
        frac = Fraction(x).limit_denominator()
        if float(frac) == float(x):
            return Const(frac)
    except Exception:
        pass
    return Const(float(x))

def _approx_one_sided(x: float, max_den: Optional[int], *, direction: str) -> float:
    """
    One-sided rational approximation with denominator <= max_den.
      direction="down": return r <= x (tight from below)
      direction="up":   return r >= x (tight from above)
    """
    if max_den is None:
        return float(x)
    from math import floor, ceil
    if direction == "down":
        best = -float("inf")
        for q in range(1, max_den + 1):
            p = floor(x * q + 1e-15)
            cand = p / q
            if cand <= x and cand > best:
                best = cand
        return float(x) if best == -float("inf") else float(best)
    if direction == "up":
        best = float("inf")
        for q in range(1, max_den + 1):
            p = ceil(x * q - 1e-15)
            cand = p / q
            if cand >= x and cand < best:
                best = cand
        return float(x) if best == float("inf") else float(best)
    raise ValueError("direction must be 'down' or 'up'")

def _ratio_min_max(t: pd.Series, f: pd.Series, q_clip: Optional[float]) -> Tuple[float, float]:
    """Compute min/max (or quantile-clipped) of t/f on finite entries."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        r = pd.to_numeric(t, errors="coerce") / pd.to_numeric(f, errors="coerce")
    r = r[_finite(r)]
    if len(r) == 0:
        raise ValueError("No finite ratios available.")
    if q_clip is None:
        return float(r.min()), float(r.max())
    q = float(q_clip)
    return float(r.quantile(q)), float(r.quantile(1.0 - q))

def _product_expr(exprs: Sequence[Union[str, Expr]]) -> Tuple[Expr, List[str]]:
    """Multiply features to build a single Expr; return (Expr, pretty-names)."""
    xs: List[Expr] = [to_expr(e) for e in exprs]
    pretty: List[str] = [repr(e) for e in xs]
    out = xs[0]
    for e in xs[1:]:
        out = out * e
    return out, pretty


# ───────────────────────── Config ───────────────────────── #

@dataclass
class RatiosConfig:
    features: Sequence[Union[str, Expr]]
    target: str
    direction: str = "both"            # "both" | "upper" | "lower"
    max_denominator: int = 100
    q_clip: Optional[float] = None     # e.g., 0.01 for robust bounds
    min_support: int = 2
    min_valid_frac: float = 0.20
    allow_mixed_sign_splits: bool = True   # now supports composite Expr using MaskPredicate
    warn: bool = False
    name_prefix: Optional[str] = "ratiox"


# ───────────────────────── Generator ───────────────────────── #

@safe_generator
def ratios_bounds(
    df: pd.DataFrame,
    *,
    hypothesis: Predicate,
    config: RatiosConfig,
) -> Iterator[Conjecture]:
    """
    Generate ratio bounds with Expr features (no DataFrame mutation).

    Yields Conjectures:
        H ⇒ target ≥ c_min * other
        H ⇒ target ≤ c_max * other
    """
    Hmask = hypothesis.mask(df).reindex(df.index, fill_value=False)
    if not Hmask.any():
        return

    other_expr, parts = _product_expr(config.features)
    other_name = "*".join(parts)

    sub = df.loc[Hmask]
    if config.target not in df.columns:
        if config.warn:
            print(f"[ratiox] target '{config.target}' not in DataFrame; skip.")
        return

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        t = pd.to_numeric(sub[config.target], errors="coerce")
        f = pd.to_numeric(other_expr.eval(sub), errors="coerce")

    # finite-only
    valid_global = _finite(t) & _finite(f)
    masked_count = int(len(sub))
    valid_count = int(valid_global.sum())
    valid_frac = (valid_count / masked_count) if masked_count > 0 else 0.0

    if valid_count < config.min_support or valid_frac < config.min_valid_frac:
        if config.warn:
            print(f"[ratiox] insufficient global support (valid={valid_count}, masked={masked_count}, frac={valid_frac:.2f}); skip.")
        return

    # sign slices
    pos_mask = valid_global & f.gt(0)
    neg_mask = valid_global & f.lt(0)
    zer_mask = valid_global & f.eq(0)

    pos_n = int(pos_mask.sum())
    neg_n = int(neg_mask.sum())
    zer_n = int(zer_mask.sum())

    pos_only = pos_n >= config.min_support and neg_n == 0 and zer_n == 0
    neg_only = neg_n >= config.min_support and pos_n == 0 and zer_n == 0
    mixed_sign = (pos_n >= config.min_support) and (neg_n >= config.min_support)

    def _emit(slice_mask: pd.Series, suffix: str, extra_pred: Optional[Predicate] = None):
        try:
            rmin, rmax = _ratio_min_max(t[slice_mask], f[slice_mask], config.q_clip)
        except ValueError:
            if config.warn:
                print(f"[ratiox] no finite ratios for slice '{suffix}'; skip.")
            return

        if suffix == "pos":
            cmin = _const_from_rational(_approx_one_sided(rmin, config.max_denominator, direction="down"))
            cmax = _const_from_rational(_approx_one_sided(rmax, config.max_denominator, direction="up"))
            cond = (hypothesis if extra_pred is None else (hypothesis & extra_pred))
            if config.direction in ("both", "lower"):
                yield Conjecture(
                    relation=Ge(to_expr(config.target), cmin * other_expr),
                    condition=cond,
                    name=f"{config.name_prefix}_lower_{config.target}_vs_{other_name}_pos",
                )
            if config.direction in ("both", "upper"):
                yield Conjecture(
                    relation=Le(to_expr(config.target), cmax * other_expr),
                    condition=cond,
                    name=f"{config.name_prefix}_upper_{config.target}_vs_{other_name}_pos",
                )
        elif suffix == "neg":
            c_for_lower = _const_from_rational(_approx_one_sided(rmax, config.max_denominator, direction="up"))
            c_for_upper = _const_from_rational(_approx_one_sided(rmin, config.max_denominator, direction="down"))
            cond = (hypothesis if extra_pred is None else (hypothesis & extra_pred))
            if config.direction in ("both", "lower"):
                yield Conjecture(
                    relation=Ge(to_expr(config.target), c_for_lower * other_expr),
                    condition=cond,
                    name=f"{config.name_prefix}_lower_{config.target}_vs_{other_name}_neg",
                )
            if config.direction in ("both", "upper"):
                yield Conjecture(
                    relation=Le(to_expr(config.target), c_for_upper * other_expr),
                    condition=cond,
                    name=f"{config.name_prefix}_upper_{config.target}_vs_{other_name}_neg",
                )

    # Uniform positive / negative
    if pos_only:
        yield from _emit(pos_mask, "pos")
        return
    if neg_only:
        yield from _emit(neg_mask, "neg")
        return

    # Mixed sign
    if mixed_sign:
        if config.allow_mixed_sign_splits:
            # Use MaskPredicate so this works for composite Exprs
            if pos_n >= config.min_support:
                pos_pred = MaskPredicate(pos_mask.reindex(df.index, fill_value=False), name=f"{other_name}>0")
                yield from _emit(pos_mask, "pos", extra_pred=pos_pred)
            if neg_n >= config.min_support:
                neg_pred = MaskPredicate(neg_mask.reindex(df.index, fill_value=False), name=f"{other_name}<0")
                yield from _emit(neg_mask, "neg", extra_pred=neg_pred)
        else:
            if config.warn:
                print(f"[ratiox] mixed-sign '{other_name}' and splits disabled; skip.")
        return

    # Not enough per-sign support
    if config.warn:
        print(f"[ratiox] insufficient per-sign support for '{other_name}'; skip.")
    return
