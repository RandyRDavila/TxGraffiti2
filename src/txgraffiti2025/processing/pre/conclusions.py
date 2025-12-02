# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Iterable, List, Dict, Tuple, Optional, Callable, Sequence

# import numpy as np
# import pandas as pd
# from pandas.api.types import is_bool_dtype, is_numeric_dtype

# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
# from txgraffiti2025.forms.linear import linear_le, linear_ge, linear_eq, linear_expr
# from txgraffiti2025.forms.logexp import log_base, exp_e, sqrt as expr_sqrt
# from txgraffiti2025.forms.nonlinear import power
# from txgraffiti2025.forms.utils import to_expr, Expr


# __all__ = [
#     "list_numeric_targets",
#     "feature_transforms",
#     "generate_conclusions_for",
#     "generate_all_conclusions",
# ]

# # ──────────────────────────────────────────────────────────────────────────────
# # Column discovery
# # ──────────────────────────────────────────────────────────────────────────────

# def list_numeric_targets(df: pd.DataFrame, *, exclude: Sequence[str] = ()) -> List[str]:
#     """
#     Return numeric (non-boolean) columns that can act as 'target' variables.
#     """
#     out: List[str] = []
#     for c in df.columns:
#         if c in exclude:
#             continue
#         s = df[c]
#         if is_bool_dtype(s):
#             continue
#         if is_numeric_dtype(s):
#             out.append(c)
#     return out

# # ──────────────────────────────────────────────────────────────────────────────
# # Feature transforms φ(other)
# # ──────────────────────────────────────────────────────────────────────────────

# @dataclass
# class Feature:
#     name: str          # pretty label, e.g. "other", "sqrt(other)", "log(other)"
#     expr: Expr         # expression φ(other)
#     domain_mask: pd.Series  # rows where φ(other) is well-defined (w.r.t. H)

# TransformBuilder = Callable[[str, pd.DataFrame, pd.Series], Optional[Feature]]

# def _build_identity(other: str, df: pd.DataFrame, H: pd.Series) -> Optional[Feature]:
#     m = H & pd.to_numeric(df[other], errors="coerce").notna()
#     if not m.any():
#         return None
#     return Feature(other, to_expr(other), m)

# def _build_sqrt(other: str, df: pd.DataFrame, H: pd.Series) -> Optional[Feature]:
#     s = pd.to_numeric(df[other], errors="coerce")
#     m = H & s.notna() & (s >= 0)
#     if not m.any():
#         return None
#     return Feature(f"√({other})", expr_sqrt(other), m)

# def _build_log(other: str, df: pd.DataFrame, H: pd.Series, base: Optional[float] = None) -> Optional[Feature]:
#     s = pd.to_numeric(df[other], errors="coerce")
#     m = H & s.notna() & (s > 0)
#     if not m.any():
#         return None
#     base_tag = "" if base is None else (f"_{int(base)}" if float(base).is_integer() else f"_{base:g}")
#     return Feature(f"ln({other})" if base is None else f"log{base_tag}({other})", log_base(other, base=base), m)

# def _build_square(other: str, df: pd.DataFrame, H: pd.Series) -> Optional[Feature]:
#     s = pd.to_numeric(df[other], errors="coerce")
#     m = H & s.notna()
#     if not m.any():
#         return None
#     return Feature(f"({other})²", power(other, 2.0), m)

# def feature_transforms(
#     df: pd.DataFrame,
#     other: str,
#     H_mask: pd.Series,
#     *,
#     include: Sequence[str] = ("id", "sqrt", "log", "square"),
#     log_bases: Sequence[Optional[float]] = (None,),  # None→ln, or e.g. (10,) for log10
# ) -> List[Feature]:
#     """
#     Build available φ(other) features and the masks where they are well-defined under H.
#     """
#     builders: Dict[str, TransformBuilder] = {
#         "id": _build_identity,
#         "sqrt": _build_sqrt,
#         "square": _build_square,
#     }
#     feats: List[Feature] = []
#     for key in include:
#         if key == "log":
#             for b in log_bases:
#                 f = _build_log(other, df, H_mask, base=b)
#                 if f is not None:
#                     feats.append(f)
#             continue
#         bld = builders.get(key)
#         if bld is None:
#             continue
#         f = bld(other, df, H_mask)
#         if f is not None:
#             feats.append(f)
#     return feats

# # ──────────────────────────────────────────────────────────────────────────────
# # Ratio bounds and “LP-like” affine tightening
# # ──────────────────────────────────────────────────────────────────────────────

# @dataclass
# class Bound:
#     kind: str            # "<=", ">=", "=="
#     a0: float            # intercept (0.0 for pure ratio)
#     coef: float          # coefficient on ϕ(other)
#     feature: Feature     # which ϕ(other)
#     support: int         # number of rows used
#     touches: int         # # rows with equality (tight within tol)
#     rho: float           # mean slack margin (signed appropriately)
#     tol: float = 1e-9

# def _ratio_bounds(y: np.ndarray, x: np.ndarray, *, tol: float = 1e-9) -> List[Tuple[str, float, int, int, float]]:
#     """
#     Tight through-origin bounds y <= c x and y >= c x for x>0 rows.
#     Returns list of (kind, c, support, touches, mean_margin).
#     """
#     # keep rows with finite numbers and x>0
#     valid = np.isfinite(y) & np.isfinite(x) & (x > 0)
#     yv = y[valid]; xv = x[valid]
#     n = yv.size
#     out: List[Tuple[str, float, int, int, float]] = []
#     if n == 0:
#         return out

#     r = yv / xv

#     # y <= c x  with c = max r
#     c_le = float(np.max(r))
#     slack_le = c_le * xv - yv
#     touches_le = int(np.sum(np.isclose(slack_le, 0.0, atol=tol)))
#     out.append(("<=", c_le, n, touches_le, float(np.mean(slack_le))))

#     # y >= c x  with c = min r
#     c_ge = float(np.min(r))
#     slack_ge = yv - c_ge * xv
#     touches_ge = int(np.sum(np.isclose(slack_ge, 0.0, atol=tol)))
#     out.append((">=", c_ge, n, touches_ge, float(np.mean(slack_ge))))

#     # If both sides essentially same c (degenerate), we can suggest equality
#     if np.isclose(c_le, c_ge, atol=tol):
#         out.append(("==", c_le, n, max(touches_le, touches_ge), float(np.mean(np.abs(yv - c_le * xv)))))

#     return out

# def _affine_tight_bounds(
#     y: np.ndarray,
#     x: np.ndarray,
#     *,
#     tol: float = 1e-9,
#     candidate_b: Optional[np.ndarray] = None,
# ) -> List[Tuple[str, float, float, int, int, float]]:
#     """
#     Approximate LP: pick slope b from a handful of ratio candidates, then compute
#     the smallest intercept a so that all constraints hold (≤ or ≥).
#     Return list of (kind, a, b, support, touches, mean_margin).
#     """
#     valid = np.isfinite(y) & np.isfinite(x)
#     yv = y[valid]; xv = x[valid]
#     n = yv.size
#     out: List[Tuple[str, float, float, int, int, float]] = []
#     if n == 0:
#         return out

#     # candidate slopes from robust set of ratios (avoid overflow/div by zero)
#     if candidate_b is None:
#         r = yv / np.where(np.abs(xv) > 0, xv, np.nan)
#         r = r[np.isfinite(r)]
#         if r.size == 0:
#             return out
#         # choose quantiles + extremes for coverage
#         qs = np.quantile(r, [0.0, 0.25, 0.5, 0.75, 1.0])
#         candidate_b = np.unique(qs)

#     for b in candidate_b:
#         # For ≤: minimal a so that y ≤ a + b x  is a = max_i (y_i - b x_i)
#         a_le = float(np.max(yv - b * xv))
#         slack_le = (a_le + b * xv) - yv
#         touches_le = int(np.sum(np.isclose(slack_le, 0.0, atol=tol)))
#         out.append(("<=", a_le, b, n, touches_le, float(np.mean(slack_le))))

#         # For ≥: minimal a so that y ≥ -a + b x  → equivalently y + a ≥ b x
#         # Reparameterize as y ≥ a + b x with negative a? Simpler dual:
#         # For ≥: we want y ≥ a + b x ⇒ choose a_ge = max_i ( (a + b x_i) - y_i )??? Not symmetrical.
#         # Let's use: y ≥ a + b x  ⇒  a must be ≤ min_i (y_i - b x_i). Tight upper a:
#         a_ge = float(np.min(yv - b * xv))
#         slack_ge = yv - (a_ge + b * xv)
#         touches_ge = int(np.sum(np.isclose(slack_ge, 0.0, atol=tol)))
#         out.append((">=", a_ge, b, n, touches_ge, float(np.mean(slack_ge))))

#         # If model collapses to equality (both near-tight both ways), offer ==
#         # We'll announce equality only when all residuals are small both sides.
#         if np.all(np.isclose(yv, a_le + b * xv, atol=tol)):
#             out.append(("==", a_le, b, n, n, 0.0))

#     return out

# # ──────────────────────────────────────────────────────────────────────────────
# # Assembly: Relations + Conjectures
# # ──────────────────────────────────────────────────────────────────────────────

# def _make_relation(kind: str, target: str, a0: float, b: float, feat: Feature, tol: float) -> Le | Ge | Eq:
#     """
#     Build a Relation using our linear helpers for consistent pretty/repr.
#     """
#     if kind == "<=":
#         return linear_le(a0, [(b, repr(feat.expr))], target)
#     if kind == ">=":
#         return linear_ge(a0, [(b, repr(feat.expr))], target)
#     # equality
#     return linear_eq(a0, [(b, repr(feat.expr))], target, tol=tol)

# def _eval_feature(df: pd.DataFrame, feat: Feature) -> np.ndarray:
#     return np.asarray(pd.to_numeric(feat.expr.eval(df), errors="coerce"), dtype=float)

# def generate_conclusions_for(
#     df: pd.DataFrame,
#     hyp: Predicate,
#     *,
#     target: str,
#     others: Iterable[str],
#     transforms: Sequence[str] = ("id", "sqrt", "log", "square"),
#     log_bases: Sequence[Optional[float]] = (None, 10),
#     tol: float = 1e-9,
# ) -> List[Conjecture]:
#     """
#     For a single hypothesis mask H and a chosen target column:
#       - Build φ(other) features for each `other != target`
#       - Produce ratio and affine-tight bounds (≤, ≥, maybe ==)
#       - Return Conjectures conditioned on `hyp`
#     """
#     # H mask (NA-safe)
#     H_mask = hyp.mask(df).reindex(df.index, fill_value=False).astype(bool)
#     y = pd.to_numeric(df[target], errors="coerce")
#     H_mask &= y.notna()

#     out: List[Conjecture] = []
#     # Dedup relations by (kind, a0, coef, feature_name) signature to avoid repeats
#     seen: set[Tuple[str, float, float, str]] = set()

#     for other in others:
#         if other == target:
#             continue

#         # Build features for this other under H
#         feats = feature_transforms(df, other, H_mask, include=transforms, log_bases=log_bases)
#         if not feats:
#             continue

#         for feat in feats:
#             dom = feat.domain_mask
#             if not dom.any():
#                 continue
#             yv = y[dom].to_numpy(dtype=float)
#             xv = _eval_feature(df.loc[dom], feat)

#             # 1) Through-origin ratio bounds
#             for kind, c, support, touches, mean_margin in _ratio_bounds(yv, xv, tol=tol):
#                 a0 = 0.0
#                 b = float(c)
#                 sig = (kind, round(a0, 12), round(b, 12), feat.name)
#                 if sig in seen:
#                     continue
#                 seen.add(sig)
#                 R = _make_relation(kind, target, a0, b, feat, tol)
#                 out.append(Conjecture(R, condition=hyp))

#             # 2) Affine tightening (approx LP): scan b from a few candidates; optimize a
#             ratio_candidates = None  # let helper compute from data
#             for kind, a0, b, support, touches, mean_margin in _affine_tight_bounds(yv, xv, tol=tol, candidate_b=ratio_candidates):
#                 sig = (kind, round(a0, 12), round(b, 12), feat.name)
#                 if sig in seen:
#                     continue
#                 seen.add(sig)
#                 R = _make_relation(kind, target, a0, b, feat, tol)
#                 out.append(Conjecture(R, condition=hyp))

#     return out

# def generate_all_conclusions(
#     df: pd.DataFrame,
#     hyps: Iterable[Predicate],
#     *,
#     targets: Optional[Sequence[str]] = None,
#     transforms: Sequence[str] = ("id", "sqrt", "log", "square"),
#     log_bases: Sequence[Optional[float]] = (None, 10),
#     tol: float = 1e-9,
# ) -> Dict[Tuple[str, str], List[Conjecture]]:
#     """
#     For each hypothesis H and each target numeric column:
#       emit a list of Conjectures H ⇒ (relation on target vs φ(other)).
#     Returns dict keyed by (H_repr, target).
#     """
#     if targets is None:
#         targets = list_numeric_targets(df)

#     results: Dict[Tuple[str, str], List[Conjecture]] = {}
#     numeric_cols = list_numeric_targets(df)  # pool for 'other'
#     for H in hyps:
#         H_repr = repr(H)
#         for tgt in targets:
#             others = [c for c in numeric_cols if c != tgt]
#             conjs = generate_conclusions_for(
#                 df,
#                 H,
#                 target=tgt,
#                 others=others,
#                 transforms=transforms,
#                 log_bases=log_bases,
#                 tol=tol,
#             )
#             results[(H_repr, tgt)] = conjs
#     return results

# src/txgraffiti2025/processing/pre/conclusions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional, Callable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
from txgraffiti2025.forms.logexp import log_base, exp_e, sqrt as expr_sqrt
from txgraffiti2025.forms.nonlinear import power
from txgraffiti2025.forms.utils import to_expr, Expr

__all__ = [
    "list_numeric_targets",
    "feature_transforms",
    "generate_conclusions_for",
    "generate_all_conclusions",
]

# ──────────────────────────────────────────────────────────────────────────────
# Column discovery
# ──────────────────────────────────────────────────────────────────────────────

def list_numeric_targets(df: pd.DataFrame, *, exclude: Sequence[str] = ()) -> List[str]:
    """Return numeric (non-boolean) columns that can act as target variables."""
    out: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        if is_bool_dtype(s):
            continue
        if is_numeric_dtype(s):
            out.append(c)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Feature transforms φ(other)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Feature:
    name: str               # pretty label, e.g. "other", "√(other)", "ln(other)"
    expr: Expr              # expression φ(other)
    domain_mask: pd.Series  # rows where φ(other) is well-defined under H

TransformBuilder = Callable[[str, pd.DataFrame, pd.Series], Optional[Feature]]

def _build_identity(other: str, df: pd.DataFrame, H: pd.Series) -> Optional[Feature]:
    s = pd.to_numeric(df[other], errors="coerce")
    m = H & s.notna()
    if not m.any():
        return None
    return Feature(other, to_expr(other), m)

def _build_sqrt(other: str, df: pd.DataFrame, H: pd.Series) -> Optional[Feature]:
    s = pd.to_numeric(df[other], errors="coerce")
    m = H & s.notna() & (s >= 0)
    if not m.any():
        return None
    return Feature(f"√({other})", expr_sqrt(other), m)

def _build_log(other: str, df: pd.DataFrame, H: pd.Series, base: Optional[float] = None) -> Optional[Feature]:
    s = pd.to_numeric(df[other], errors="coerce")
    m = H & s.notna() & (s > 0)
    if not m.any():
        return None
    base_tag = "" if base is None else (f"_{int(base)}" if float(base).is_integer() else f"_{base:g}")
    return Feature(f"ln({other})" if base is None else f"log{base_tag}({other})", log_base(other, base=base), m)

def _build_square(other: str, df: pd.DataFrame, H: pd.Series) -> Optional[Feature]:
    s = pd.to_numeric(df[other], errors="coerce")
    m = H & s.notna()
    if not m.any():
        return None
    return Feature(f"({other})²", power(other, 2.0), m)

def feature_transforms(
    df: pd.DataFrame,
    other: str,
    H_mask: pd.Series,
    *,
    include: Sequence[str] = ("id", "sqrt", "log", "square"),
    log_bases: Sequence[Optional[float]] = (None,),  # None→ln; e.g. (10,) for log10
) -> List[Feature]:
    """Build available φ(other) features and their well-defined masks under H."""
    builders: Dict[str, TransformBuilder] = {
        "id": _build_identity,
        "sqrt": _build_sqrt,
        "square": _build_square,
    }
    feats: List[Feature] = []
    for key in include:
        if key == "log":
            for b in log_bases:
                f = _build_log(other, df, H_mask, base=b)
                if f is not None:
                    feats.append(f)
            continue
        bld = builders.get(key)
        if bld is None:
            continue
        f = bld(other, df, H_mask)
        if f is not None:
            feats.append(f)
    return feats

# ──────────────────────────────────────────────────────────────────────────────
# Ratio bounds and affine tightening
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Bound:
    kind: str            # "<=", ">=", "=="
    a0: float            # intercept (0.0 for pure ratio)
    coef: float          # coefficient on ϕ(other)
    feature: Feature
    support: int
    touches: int
    rho: float           # mean slack margin (signed appropriately)
    tol: float = 1e-9

def _ratio_bounds(y: np.ndarray, x: np.ndarray, *, tol: float = 1e-9) -> List[Tuple[str, float, int, int, float]]:
    """
    Tight through-origin bounds y <= c x and y >= c x using rows with x>0.
    Returns (kind, c, support, touches, mean_margin).
    """
    valid = np.isfinite(y) & np.isfinite(x) & (x > 0)
    yv = y[valid]; xv = x[valid]
    n = yv.size
    out: List[Tuple[str, float, int, int, float]] = []
    if n == 0:
        return out

    r = yv / xv

    # y <= c x  with c = max r
    c_le = float(np.max(r))
    slack_le = c_le * xv - yv
    touches_le = int(np.sum(np.isclose(slack_le, 0.0, atol=tol)))
    out.append(("<=", c_le, n, touches_le, float(np.mean(slack_le))))

    # y >= c x  with c = min r
    c_ge = float(np.min(r)))
    slack_ge = yv - c_ge * xv
    touches_ge = int(np.sum(np.isclose(slack_ge, 0.0, atol=tol)))
    out.append((">=", c_ge, n, touches_ge, float(np.mean(slack_ge))))

    # Degenerate case ≈ single c both sides → suggest equality
    if np.isclose(c_le, c_ge, atol=tol):
        out.append(("==", c_le, n, max(touches_le, touches_ge), float(np.mean(np.abs(yv - c_le * xv)))))

    return out

def _affine_tight_bounds(
    y: np.ndarray,
    x: np.ndarray,
    *,
    tol: float = 1e-9,
    candidate_b: Optional[np.ndarray] = None,
) -> List[Tuple[str, float, float, int, int, float]]:
    """
    Approximate LP:
      pick slopes b (quantiles of y/x), then set the tightest intercepts:
        - For y <= a + b x:     a_le = max_i (y_i - b x_i); slack = a_le + b x_i - y_i
        - For y >= a + b x:     a_ge = min_i (y_i - b x_i); slack = y_i - (a_ge + b x_i)
    Returns (kind, a, b, support, touches, mean_margin).
    """
    valid = np.isfinite(y) & np.isfinite(x)
    yv = y[valid]; xv = x[valid]
    n = yv.size
    out: List[Tuple[str, float, float, int, int, float]] = []
    if n == 0:
        return out

    if candidate_b is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = yv / np.where(np.abs(xv) > 0, xv, np.nan)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return out
        qs = np.quantile(r, [0.0, 0.25, 0.5, 0.75, 1.0])
        candidate_b = np.unique(qs)

    for b in candidate_b:
        # ≤ branch
        a_le = float(np.max(yv - b * xv))
        slack_le = (a_le + b * xv) - yv
        touches_le = int(np.sum(np.isclose(slack_le, 0.0, atol=tol)))
        out.append(("<=", a_le, b, n, touches_le, float(np.mean(slack_le))))

        # ≥ branch
        a_ge = float(np.min(yv - b * xv))
        slack_ge = yv - (a_ge + b * xv)
        touches_ge = int(np.sum(np.isclose(slack_ge, 0.0, atol=tol)))
        out.append((">=", a_ge, b, n, touches_ge, float(np.mean(slack_ge))))

        # Equality when the ≤ model is everywhere tight
        if np.all(np.isclose(yv, a_le + b * xv, atol=tol)):
            out.append(("==", a_le, b, n, n, 0.0))

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Assembly: Relations + Conjectures
# ──────────────────────────────────────────────────────────────────────────────

def _make_relation(kind: str, target: str, a0: float, b: float, feat_expr: Expr, tol: float) -> Le | Ge | Eq:
    """
    Build Relation directly with Expr (avoid stringifying expressions).
    """
    left = (to_expr(a0) + b * feat_expr) if a0 != 0.0 else (b * feat_expr)
    if kind == "<=":
        return Le(left, target)
    if kind == ">=":
        return Ge(left, target)
    return Eq(left, target, tol=tol)

def _eval_feature(df: pd.DataFrame, feat: Feature) -> np.ndarray:
    return np.asarray(pd.to_numeric(feat.expr.eval(df), errors="coerce"), dtype=float)

def generate_conclusions_for(
    df: pd.DataFrame,
    hyp: Predicate,
    *,
    target: str,
    others: Iterable[str],
    transforms: Sequence[str] = ("id", "sqrt", "log", "square"),
    log_bases: Sequence[Optional[float]] = (None, 10),
    tol: float = 1e-9,
) -> List[Conjecture]:
    """
    For a single hypothesis mask H and chosen target y:
      - Build φ(other) for each other ≠ target
      - Fit ratio and affine-tight bounds
      - Return Conjectures conditioned on hyp
    """
    H_mask = hyp.mask(df).reindex(df.index, fill_value=False).astype(bool)
    y = pd.to_numeric(df[target], errors="coerce")
    H_mask &= y.notna()

    out: List[Conjecture] = []
    seen: set[Tuple[str, float, float, str]] = set()  # (kind, a0, b, feature_name)

    for other in others:
        if other == target:
            continue

        feats = feature_transforms(df, other, H_mask, include=transforms, log_bases=log_bases)
        if not feats:
            continue

        for feat in feats:
            dom = feat.domain_mask
            if not dom.any():
                continue
            yv = y[dom].to_numpy(dtype=float)
            xv = _eval_feature(df.loc[dom], feat)

            # 1) Ratio bounds (x>0)
            for kind, c, support, touches, mean_margin in _ratio_bounds(yv, xv, tol=tol):
                a0, b = 0.0, float(c)
                sig = (kind, round(a0, 12), round(b, 12), feat.name)
                if sig in seen:
                    continue
                seen.add(sig)
                R = _make_relation(kind, target, a0, b, feat.expr, tol)
                out.append(Conjecture(R, condition=hyp))

            # 2) Affine tightening
            for kind, a0, b, support, touches, mean_margin in _affine_tight_bounds(yv, xv, tol=tol):
                sig = (kind, round(a0, 12), round(b, 12), feat.name)
                if sig in seen:
                    continue
                seen.add(sig)
                R = _make_relation(kind, target, a0, b, feat.expr, tol)
                out.append(Conjecture(R, condition=hyp))

    return out

def generate_all_conclusions(
    df: pd.DataFrame,
    hyps: Iterable[Predicate],
    *,
    targets: Optional[Sequence[str]] = None,
    transforms: Sequence[str] = ("id", "sqrt", "log", "square"),
    log_bases: Sequence[Optional[float]] = (None, 10),
    tol: float = 1e-9,
) -> Dict[Tuple[str, str], List[Conjecture]]:
    """
    For each hypothesis H and each target numeric column y:
      emit a list of Conjectures H ⇒ (relation on y vs φ(other)).
    Returns dict keyed by (repr(H), target).
    """
    if targets is None:
        targets = list_numeric_targets(df)

    results: Dict[Tuple[str, str], List[Conjecture]] = {}
    numeric_cols = list_numeric_targets(df)

    for H in hyps:
        H_repr = repr(H)
        for tgt in targets:
            others = [c for c in numeric_cols if c != tgt]
            conjs = generate_conclusions_for(
                df, H, target=tgt, others=others,
                transforms=transforms, log_bases=log_bases, tol=tol,
            )
            results[(H_repr, tgt)] = conjs
    return results
