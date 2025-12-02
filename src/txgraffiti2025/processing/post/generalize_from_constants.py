# # src/txgraffiti2025/processing/post/generalize_from_constants.py
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Iterable, List, Optional, Sequence, Tuple, Any

# import numpy as np
# import pandas as pd

# from txgraffiti2025.forms.utils import to_expr, Expr, Const, BinOp
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.processing.post.constant_ratios import extract_ratio_pattern  # public
# from txgraffiti2025.forms.expr_utils import expr_depends_on  # strong dep-check

# # Optional, only for type hints; we don't require this module at runtime
# try:
#     from txgraffiti2025.processing.pre.constants_cache import ConstantsCache  # type: ignore
# except Exception:  # pragma: no cover
#     ConstantsCache = Any  # type: ignore


# # -------------------------------------------------------------------------
# # Mask / set utils
# # -------------------------------------------------------------------------

# def _mask(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
#     if pred is None:
#         return pd.Series(True, index=df.index)
#     return pred.mask(df).reindex(df.index, fill_value=False).astype(bool)

# def _is_strict_superset(df: pd.DataFrame, A: Optional[Predicate], B: Optional[Predicate]) -> bool:
#     """
#     Is rows(A) a strict subset of rows(B)? (Equivalently, B ⊃ A.)
#     None means TRUE over df.
#     """
#     a = _mask(df, A); b = _mask(df, B)
#     return not (a & ~b).any() and (b & ~a).any()

# def _conj_is_true(conj: Conjecture, df: pd.DataFrame) -> bool:
#     applicable, holds, _ = conj.check(df)
#     return bool((applicable & holds).all())


# # -------------------------------------------------------------------------
# # Robust stats helpers
# # -------------------------------------------------------------------------

# def _finite(series: pd.Series) -> pd.Series:
#     return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

# def _robust_cv(series: pd.Series) -> float:
#     v = _finite(series).dropna()
#     if v.empty:
#         return float("inf")
#     med = float(v.median())
#     if med == 0.0:
#         return float("inf")
#     mad = float((v - med).abs().median())
#     return abs(mad / med)

# def _is_near_one_on_slice(
#     df: pd.DataFrame,
#     K: Expr,
#     H: Optional[Predicate],
#     *,
#     med_tol: float = 1e-2,
#     cv_tol: float = 2e-2,
# ) -> bool:
#     """
#     Return True iff K ≈ 1 on the H-slice (median close to 1 and low dispersion).
#     """
#     m = _mask(df, H)
#     try:
#         with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
#             kvals = _finite(K.eval(df))
#     except Exception:
#         return False
#     kvals = kvals[m].dropna()
#     if kvals.empty:
#         return False
#     med = float(kvals.median())
#     cv = _robust_cv(kvals)
#     return (abs(med - 1.0) <= med_tol) and (cv <= cv_tol)


# # -------------------------------------------------------------------------
# # Inputs & outputs
# # -------------------------------------------------------------------------

# @dataclass
# class ConstantCandidate:
#     """
#     Generic constant-coefficient candidate:
#       expr : an Expr like (N+a)/(D+b)
#       hyp  : a Predicate on which the expression was mined/validated
#       info : any metadata (support, cv, qspan, etc.)
#     """
#     expr: Expr
#     hyp: Optional[Predicate]
#     info: dict

# @dataclass
# class Generalization:
#     from_conjecture: Conjecture
#     new_conjecture: Conjecture
#     reason: str
#     witness_superset: Optional[Predicate]
#     meta: dict


# # -------------------------------------------------------------------------
# # Adapters
# # -------------------------------------------------------------------------

# def constants_from_ratios_hits(
#     hits: Iterable[Any],  # objects with .formula_expr and .hypothesis at minimum
# ) -> List[ConstantCandidate]:
#     """
#     Build ConstantCandidate list from the constant-ratio miner results.
#     We read: .formula_expr (Expr), .hypothesis (Predicate | None), and optional
#     fields like .support/.cv/.qspan for metadata if present.
#     """
#     out: List[ConstantCandidate] = []
#     for h in hits:
#         expr = getattr(h, "formula_expr", None)
#         hyp = getattr(h, "hypothesis", None)
#         if not isinstance(expr, Expr):
#             continue
#         meta = {}
#         for key in (
#             "support", "cv", "qspan", "value_float", "value_display",
#             "numerator", "denominator", "shift_num", "shift_den",
#             "matches_conj_coeff"
#         ):
#             if hasattr(h, key):
#                 meta[key] = getattr(h, key)
#         out.append(ConstantCandidate(expr=expr, hyp=hyp, info=meta))
#     return out


# def constants_from_cache(
#     cache: ConstantsCache,
#     hypotheses: Sequence[Optional[Predicate]],
# ) -> List[ConstantCandidate]:
#     """
#     Tolerant adapter for various cache shapes.
#     Accepts items with .expr or with a numeric .value (wrapped as Const(value)).
#     """
#     if cache is None:
#         return []
#     out: List[ConstantCandidate] = []

#     # Heuristic accessor, handles hyp_to_key/key_to_constants if present.
#     if hasattr(cache, "hyp_to_key") and hasattr(cache, "key_to_constants"):
#         def get_for(h):
#             k = cache.hyp_to_key.get(repr(h))
#             return cache.key_to_constants.get(k) if k is not None else None
#     else:
#         def get_for(h):  # type: ignore
#             if hasattr(cache, "get"):
#                 try:
#                     return cache.get(repr(h))
#                 except Exception:
#                     return getattr(cache, "all", None)
#             return getattr(cache, "all", None)

#     for H in hypotheses:
#         bag = get_for(H)
#         if bag is None:
#             continue

#         # Normalize to iterable
#         if isinstance(bag, (list, tuple, set)):
#             seq: Iterable[Any] = bag
#         else:
#             for attr in ("constants", "ratios", "items", "all", "all_items"):
#                 if hasattr(bag, attr):
#                     seq = getattr(bag, attr)
#                     break
#             else:
#                 try:
#                     seq = list(bag)  # type: ignore
#                 except Exception:
#                     continue

#         for item in seq:
#             expr = getattr(item, "expr", None)
#             if expr is None and isinstance(item, dict):
#                 expr = item.get("expr")
#             if expr is None:
#                 # fallback numeric
#                 val = None
#                 if isinstance(item, dict):
#                     val = item.get("value")
#                 else:
#                     for v_attr in ("value", "val", "const", "c"):
#                         if hasattr(item, v_attr):
#                             val = getattr(item, v_attr)
#                             break
#                 if val is not None:
#                     try:
#                         expr = Const(float(val))
#                     except Exception:
#                         expr = None
#             if isinstance(expr, Expr):
#                 out.append(ConstantCandidate(expr=expr, hyp=H, info={"source": "cache"}))
#     return out


# # -------------------------------------------------------------------------
# # Core generalizer
# # -------------------------------------------------------------------------

# def propose_generalizations_from_constants(
#     df: pd.DataFrame,
#     conj: Conjecture,
#     *,
#     candidate_hypotheses: Sequence[Optional[Predicate]],
#     # You may pass either/both of these:
#     constant_candidates: Optional[Sequence[ConstantCandidate]] = None,
#     constants_cache: Optional[ConstantsCache] = None,
#     # safety/config
#     require_strict_superset: bool = True,
#     # tolerances
#     atol: float = 1e-6,
#     rtol: float = 5e-2,
#     # filters
#     drop_near_one_coeffs: bool = True,
#     near_one_med_tol: float = 1e-2,
#     near_one_cv_tol: float = 2e-2,
# ) -> List[Generalization]:
#     """
#     Replace numeric coefficient c in `target (≤/≥) c·feature` with structural Expr K
#     (e.g., (N+a)/(D+b)) and try hypotheses H ⊃ base where the new conjecture
#     remains TRUE.

#     Filters:
#       - K must NOT depend on the target OR the feature.
#       - If K ≈ 1 on the candidate superset and does NOT strictly improve over 1·feature,
#         it is dropped (optional, enabled by default).

#     Returns a list of Generalization records for each accepted candidate.
#     """
#     patt = extract_ratio_pattern(conj)
#     if patt is None:
#         return []

#     tgt = to_expr(patt.target)
#     feat = to_expr(patt.feature)

#     # Collect candidates
#     cands: List[ConstantCandidate] = []
#     if constant_candidates:
#         cands.extend(constant_candidates)
#     if constants_cache is not None:
#         cands.extend(constants_from_cache(constants_cache, candidate_hypotheses))

#     out: List[Generalization] = []

#     # Precompute trivial 1·feature RHS constructor
#     def _trivial_conj_for(Hsup: Optional[Predicate]) -> Conjecture:
#         rhs1 = BinOp(np.multiply, Const(1.0), feat)
#         rel1 = Ge(tgt, rhs1) if patt.kind == "Ge" else Le(tgt, rhs1)
#         return Conjecture(relation=rel1, condition=Hsup)

#     for Hsup in candidate_hypotheses:
#         # Enforce strict generalization if requested
#         if require_strict_superset:
#             if conj.condition is None:
#                 # TRUE has no strict superset
#                 continue
#             if not _is_strict_superset(df, conj.condition, Hsup):
#                 continue

#         trivial_conj = _trivial_conj_for(Hsup)

#         for cc in cands:
#             K = cc.expr

#             # Strong dependency checks: forbid K mentioning target OR feature
#             if expr_depends_on(K, patt.target):
#                 continue
#             if expr_depends_on(K, patt.feature):
#                 continue

#             # Build the generalized inequality with K
#             rhs = BinOp(np.multiply, K, feat)
#             rel = Ge(tgt, rhs) if patt.kind == "Ge" else Le(tgt, rhs)
#             new_conj = Conjecture(relation=rel, condition=Hsup, name=(conj.name or "conj") + "_genK")

#             # Accept only if TRUE on Hsup
#             if not _conj_is_true(new_conj, df):
#                 continue

#             # Drop near-1 coefficients unless they strictly tighten vs 1·feature
#             if drop_near_one_coeffs and _is_near_one_on_slice(
#                 df, K, Hsup, med_tol=near_one_med_tol, cv_tol=near_one_cv_tol
#             ):
#                 # Compare RHS_K vs RHS_1 on applicable rows
#                 appK, _, _ = new_conj.check(df)
#                 app1, _, _ = trivial_conj.check(df)
#                 app = (appK & app1)

#                 # FIX: use Expr.eval(df), not .evaluate(df)
#                 K_rhs = _finite(new_conj.relation.right.eval(df)).reindex(df.index)[app]
#                 one_rhs = _finite(trivial_conj.relation.right.eval(df)).reindex(df.index)[app]

#                 if isinstance(new_conj.relation, Le):
#                     # Keep only if RHS_K < RHS_1 somewhere (strict improvement)
#                     if not (K_rhs < one_rhs - 1e-12).any():
#                         continue
#                 else:
#                     # Ge: Keep only if RHS_K > RHS_1 somewhere
#                     if not (K_rhs > one_rhs + 1e-12).any():
#                         continue

#             # Optional: alignment with original c on the base slice
#             align = None
#             if conj.condition is not None:
#                 m0 = _mask(df, conj.condition)
#                 try:
#                     with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
#                         kvals0 = _finite(K.eval(df))
#                 except Exception:
#                     kvals0 = pd.Series(np.nan, index=df.index)
#                 kvals0 = kvals0[m0].dropna()
#                 if kvals0.size:
#                     med = float(kvals0.median())
#                     align = {"K@base_median": med, "abs_err_to_c": abs(med - float(patt.coefficient))}

#             out.append(Generalization(
#                 from_conjecture=conj,
#                 new_conjecture=new_conj,
#                 reason="replace numeric coefficient with structural constant",
#                 witness_superset=Hsup,
#                 meta={"candidate_info": cc.info, "align": align},
#             ))

#     # Deduplicate by (repr, hypothesis mask)
#     seen = set()
#     uniq: List[Generalization] = []
#     for g in out:
#         key = (repr(g.new_conjecture), tuple(_mask(df, g.new_conjecture.condition).values))
#         if key not in seen:
#             seen.add(key)
#             uniq.append(g)

#     return uniq


# src/txgraffiti2025/processing/post/generalize_from_constants.py
from __future__ import annotations

"""
Generalize ratio-style conjectures by replacing a numeric coefficient c with a
structural constant expression K (e.g., (N+a)/(D+b)) mined from data.

Given a conjecture like:
    H ⇒  target (≤/≥) c · feature

we propose:
    H_sup ⇒ target (≤/≥) K · feature

where H_sup is a (strict) superset of H (optional), and K is a candidate Expr
that (a) does not depend on target or feature, and (b) keeps the conjecture true.

Inputs for K can come directly from the constant-ratio miner
(see constant_ratios.py) or from a pre-cache.

Public API:
    - constants_from_ratios_hits(hits)
    - constants_from_cache(cache, hypotheses)
    - propose_generalizations_from_constants(...)

This module is conservative: it filters K≈1 unless using K strictly tightens the
bound relative to 1·feature on the applicable rows.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import to_expr, Expr, Const, BinOp
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.post.constant_ratios import extract_ratio_pattern  # public
from txgraffiti2025.forms.expr_utils import expr_depends_on  # strong dep-check

try:
    # Optional only for typing; not required at runtime
    from txgraffiti2025.processing.pre.constants_cache import ConstantsCache  # type: ignore
except Exception:  # pragma: no cover
    ConstantsCache = Any  # type: ignore


__all__ = [
    "ConstantCandidate",
    "Generalization",
    "constants_from_ratios_hits",
    "constants_from_cache",
    "propose_generalizations_from_constants",
]


# -------------------------------------------------------------------------
# Mask / set utils
# -------------------------------------------------------------------------

def _mask(df: pd.DataFrame, pred: Optional[Predicate]) -> pd.Series:
    """Aligned boolean mask; None means TRUE."""
    if pred is None:
        return pd.Series(True, index=df.index, dtype=bool)
    m = pred.mask(df).reindex(df.index, fill_value=False)
    return m.astype(bool, copy=False)

def _is_strict_superset(df: pd.DataFrame, A: Optional[Predicate], B: Optional[Predicate]) -> bool:
    """
    Is rows(A) a strict subset of rows(B)? (Equivalently, B ⊃ A.)
    None means TRUE (all rows).
    """
    a, b = _mask(df, A), _mask(df, B)
    return not (a & ~b).any() and (b & ~a).any()

def _conj_is_true(conj: Conjecture, df: pd.DataFrame) -> bool:
    applicable, holds, _ = conj.check(df)
    return bool((applicable & holds).all())


# -------------------------------------------------------------------------
# Robust stats helpers
# -------------------------------------------------------------------------

def _finite(series: pd.Series) -> pd.Series:
    """Coerce to numeric and replace ±inf with NaN."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _robust_cv(series: pd.Series) -> float:
    """
    Robust coefficient of variation via MAD/median.
    Returns +inf if median≈0 or empty.
    """
    v = _finite(series).dropna()
    if v.empty:
        return float("inf")
    med = float(v.median())
    if med == 0.0:
        return float("inf")
    mad = float((v - med).abs().median())
    return abs(mad / med)

def _is_near_one_on_slice(
    df: pd.DataFrame,
    K: Expr,
    H: Optional[Predicate],
    *,
    med_tol: float = 1e-2,
    cv_tol: float = 2e-2,
) -> bool:
    """
    True iff K ≈ 1 on the H-slice (median close to 1 and low dispersion).
    """
    m = _mask(df, H)
    try:
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            kvals = _finite(K.eval(df))
    except Exception:
        return False
    kvals = kvals[m].dropna()
    if kvals.empty:
        return False
    med = float(kvals.median())
    cv = _robust_cv(kvals)
    return (abs(med - 1.0) <= med_tol) and (cv <= cv_tol)


# -------------------------------------------------------------------------
# Inputs & outputs
# -------------------------------------------------------------------------

@dataclass
class ConstantCandidate:
    """
    Generic constant-coefficient candidate:
      expr : an Expr like (N+a)/(D+b)
      hyp  : a Predicate on which the expression was mined/validated (or None)
      info : free-form metadata (support, cv, qspan, etc.)
    """
    expr: Expr
    hyp: Optional[Predicate]
    info: dict

@dataclass
class Generalization:
    """
    A successful generalization:
      from_conjecture : original conjecture
      new_conjecture  : generalized conjecture using K
      reason          : short textual reason
      witness_superset: the H_sup hypothesis on which the generalized statement holds
      meta            : extra debugging/diagnostics (candidate info, align stats)
    """
    from_conjecture: Conjecture
    new_conjecture: Conjecture
    reason: str
    witness_superset: Optional[Predicate]
    meta: dict


# -------------------------------------------------------------------------
# Adapters
# -------------------------------------------------------------------------

def constants_from_ratios_hits(
    hits: Iterable[Any],  # must expose .formula_expr and .hypothesis at minimum
) -> List[ConstantCandidate]:
    """
    Build ConstantCandidate list from the constant-ratio miner results.
    Reads: .formula_expr (Expr), .hypothesis (Predicate | None). Copies common
    metadata if present (support/cv/qspan/value_*/numerator/denominator/shifts).
    """
    out: List[ConstantCandidate] = []
    for h in hits:
        expr = getattr(h, "formula_expr", None)
        hyp = getattr(h, "hypothesis", None)
        if not isinstance(expr, Expr):
            continue
        meta: Dict[str, Any] = {}
        for key in (
            "support", "cv", "qspan", "value_float", "value_display",
            "numerator", "denominator", "shift_num", "shift_den",
            "matches_conj_coeff",
        ):
            if hasattr(h, key):
                meta[key] = getattr(h, key)
        out.append(ConstantCandidate(expr=expr, hyp=hyp, info=meta))
    return out


def constants_from_cache(
    cache: ConstantsCache,
    hypotheses: Sequence[Optional[Predicate]],
) -> List[ConstantCandidate]:
    """
    Tolerant adapter for various cache shapes.
    Accepts entries with .expr or with a numeric .value (wrapped as Const(value)).
    """
    if cache is None:
        return []
    out: List[ConstantCandidate] = []

    # Flexible accessor supporting the pre.constants_cache layout
    if hasattr(cache, "hyp_to_key") and hasattr(cache, "key_to_constants"):
        def get_for(h):
            k = cache.hyp_to_key.get(repr(h))
            return cache.key_to_constants.get(k) if k is not None else None
    else:
        def get_for(h):  # type: ignore
            if hasattr(cache, "get"):
                try:
                    return cache.get(repr(h))
                except Exception:
                    return getattr(cache, "all", None)
            return getattr(cache, "all", None)

    for H in hypotheses:
        bag = get_for(H)
        if bag is None:
            continue

        # Normalize to an iterable of items
        if isinstance(bag, (list, tuple, set)):
            seq: Iterable[Any] = bag
        else:
            for attr in ("constants", "ratios", "items", "all", "all_items"):
                if hasattr(bag, attr):
                    seq = getattr(bag, attr)
                    break
            else:
                try:
                    seq = list(bag)  # type: ignore
                except Exception:
                    continue

        for item in seq:
            expr = getattr(item, "expr", None)
            if expr is None and isinstance(item, dict):
                expr = item.get("expr")
            if expr is None:
                # fallback numeric
                val = None
                if isinstance(item, dict):
                    val = item.get("value")
                else:
                    for v_attr in ("value", "val", "const", "c"):
                        if hasattr(item, v_attr):
                            val = getattr(item, v_attr)
                            break
                if val is not None:
                    try:
                        expr = Const(float(val))
                    except Exception:
                        expr = None
            if isinstance(expr, Expr):
                out.append(ConstantCandidate(expr=expr, hyp=H, info={"source": "cache"}))
    return out


# -------------------------------------------------------------------------
# Core generalizer
# -------------------------------------------------------------------------

def propose_generalizations_from_constants(
    df: pd.DataFrame,
    conj: Conjecture,
    *,
    candidate_hypotheses: Sequence[Optional[Predicate]],
    # You may pass either/both of these:
    constant_candidates: Optional[Sequence[ConstantCandidate]] = None,
    constants_cache: Optional[ConstantsCache] = None,
    # configuration
    require_strict_superset: bool = True,
    # tolerances
    atol: float = 1e-6,
    rtol: float = 5e-2,
    # filters
    drop_near_one_coeffs: bool = True,
    near_one_med_tol: float = 1e-2,
    near_one_cv_tol: float = 2e-2,
) -> List[Generalization]:
    """
    Replace numeric coefficient c in `target (≤/≥) c·feature` with structural Expr K
    (e.g., (N+a)/(D+b)) and try hypotheses H_sup ⊃ base where the new conjecture
    remains TRUE.

    Filters:
      - K must NOT depend on the target OR the feature (no leakage).
      - If K ≈ 1 on H_sup and does NOT strictly improve 1·feature, drop it.

    Returns a list of Generalization records for each accepted candidate.
    """
    patt = extract_ratio_pattern(conj)
    if patt is None:
        return []

    tgt = to_expr(patt.target)
    feat = to_expr(patt.feature)

    # Gather candidates
    cands: List[ConstantCandidate] = []
    if constant_candidates:
        cands.extend(constant_candidates)
    if constants_cache is not None:
        cands.extend(constants_from_cache(constants_cache, candidate_hypotheses))

    out: List[Generalization] = []

    # RHS = 1·feature (reference)
    def _trivial_conj_for(Hsup: Optional[Predicate]) -> Conjecture:
        rhs1 = BinOp(np.multiply, Const(1.0), feat)
        rel1 = Ge(tgt, rhs1) if patt.kind == "Ge" else Le(tgt, rhs1)
        return Conjecture(relation=rel1, condition=Hsup)

    for Hsup in candidate_hypotheses:
        # Enforce strict generalization if requested
        if require_strict_superset:
            if conj.condition is None:  # TRUE has no strict superset
                continue
            if not _is_strict_superset(df, conj.condition, Hsup):
                continue

        trivial_conj = _trivial_conj_for(Hsup)

        for cc in cands:
            K = cc.expr

            # Strong dependency checks: forbid K mentioning target OR feature
            if expr_depends_on(K, patt.target):
                continue
            if expr_depends_on(K, patt.feature):
                continue

            # Build candidate: target (<=/>=) K * feature
            rhs = BinOp(np.multiply, K, feat)
            rel = Ge(tgt, rhs) if patt.kind == "Ge" else Le(tgt, rhs)
            new_conj = Conjecture(relation=rel, condition=Hsup, name=(conj.name or "conj") + "_genK")

            # Must hold on Hsup
            if not _conj_is_true(new_conj, df):
                continue

            # Optionally drop near-1 unless strictly tighter than 1·feature
            if drop_near_one_coeffs and _is_near_one_on_slice(
                df, K, Hsup, med_tol=near_one_med_tol, cv_tol=near_one_cv_tol
            ):
                appK, _, _ = new_conj.check(df)
                app1, _, _ = trivial_conj.check(df)
                app = (appK & app1)

                # Use Expr.eval(df) for RHS comparisons
                KR = _finite(new_conj.relation.right.eval(df)).reindex(df.index)[app]
                R1 = _finite(trivial_conj.relation.right.eval(df)).reindex(df.index)[app]

                if isinstance(new_conj.relation, Le):
                    # Keep only if RHS_K < RHS_1 somewhere (strict improvement)
                    if not (KR < R1 - 1e-12).any():
                        continue
                else:  # Ge
                    if not (KR > R1 + 1e-12).any():
                        continue

            # Optional alignment meta vs original numeric coefficient on base
            align = None
            if conj.condition is not None:
                m0 = _mask(df, conj.condition)
                try:
                    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                        kvals0 = _finite(K.eval(df))
                except Exception:
                    kvals0 = pd.Series(np.nan, index=df.index)
                kvals0 = kvals0[m0].dropna()
                if kvals0.size:
                    med = float(kvals0.median())
                    align = {"K@base_median": med, "abs_err_to_c": abs(med - float(patt.coefficient))}

            out.append(Generalization(
                from_conjecture=conj,
                new_conjecture=new_conj,
                reason="replace numeric coefficient with structural constant",
                witness_superset=Hsup,
                meta={"candidate_info": cc.info, "align": align},
            ))

    # Deduplicate by (repr(new_conjecture), mask bytes of its condition)
    seen: set[Tuple[str, bytes]] = set()
    uniq: List[Generalization] = []
    for g in out:
        mbytes = _mask(df, g.new_conjecture.condition).to_numpy(dtype=np.uint8).tobytes()
        key = (repr(g.new_conjecture), mbytes)
        if key not in seen:
            seen.add(key)
            uniq.append(g)

    return uniq
