# # """
# # Dalmatian heuristic: truth and significance filtering for conjectures.

# # Used both as:
# # 1. A pre-acceptance filter (during conjecture generation)
# # 2. A post-processing refinement (after conjecture generation)
# # """

# # from __future__ import annotations
# # import pandas as pd
# # import numpy as np
# # from typing import List, Dict, Any
# # from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
# # from txgraffiti2025.processing.utils import truth_mask, touch_count, slack_summary, hash_conjecture, log_event


# # # ============================================================
# # # Helper: determine target and direction from relation
# # # ============================================================

# # def _target_and_direction(conj: Conjecture) -> tuple[str, str] | None:
# #     """Infer the target variable and inequality direction from the relation."""
# #     rel = conj.relation
# #     if isinstance(rel, Le):
# #         direction = "le"
# #     elif isinstance(rel, Ge):
# #         direction = "ge"
# #     else:
# #         return None
# #     # Try to guess target column name (lhs if ColumnTerm)
# #     left = getattr(rel.left, "col", None)
# #     if isinstance(left, str):
# #         return (left, direction)
# #     return None


# # # ============================================================
# # # Truth and Significance Tests
# # # ============================================================

# # def dalmatian_score(conj: Conjecture, df: pd.DataFrame) -> Dict[str, Any]:
# #     """Return a diagnostic score dictionary for the Dalmatian heuristic."""
# #     applicable, holds, fails = conj.check(df)
# #     truth_ok = len(fails) == 0
# #     tcount = touch_count(conj, df)
# #     min_slack, mean_slack = slack_summary(conj, df)
# #     return {
# #         "name": getattr(conj, "name", "Conjecture"),
# #         "truth_ok": truth_ok,
# #         "touch_count": tcount,
# #         "min_slack": min_slack,
# #         "mean_slack": mean_slack,
# #     }



# # def _is_significant(conj: Conjecture, df: pd.DataFrame, accepted: List[Conjecture]) -> bool:
# #     """
# #     True if this conjecture gives a strictly tighter bound for at least one instance,
# #     compared to already-accepted conjectures with the same (target, direction).
# #     """
# #     if not accepted:
# #         return True

# #     rel = conj.relation
# #     td = _target_and_direction(conj)
# #     if td is None:
# #         # If we can't infer (target,direction), don't block on significance
# #         return True
# #     tname, direction = td

# #     # Only compare against already-accepted conjectures with the same (target, direction)
# #     candidates = [c for c in accepted if _target_and_direction(c) == td]
# #     if not candidates:
# #         return True

# #     # Compare RHS expressions directly (avoid slack sign ambiguity)
# #     rhs_new = rel.right.eval(df)
# #     rhs_prev_list = [c.relation.right.eval(df) for c in candidates]

# #     eps = 1e-9
# #     if direction == "le":
# #         # Upper bounds: tighter if RHS_new < best previous RHS somewhere
# #         rhs_best = rhs_prev_list[0]
# #         for arr in rhs_prev_list[1:]:
# #             rhs_best = np.minimum(rhs_best, arr)
# #         return bool((rhs_new < rhs_best - eps).any())
# #     else:
# #         # Lower bounds: tighter if RHS_new > best previous RHS somewhere
# #         rhs_best = rhs_prev_list[0]
# #         for arr in rhs_prev_list[1:]:
# #             rhs_best = np.maximum(rhs_best, arr)
# #         return bool((rhs_new > rhs_best + eps).any())


# # # ============================================================
# # # Main filter
# # # ============================================================

# # def dalmatian_filter(conjectures: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
# #     """
# #     Apply the Dalmatian heuristic:
# #     - Remove conjectures failing the truth test.
# #     - Keep only those that improve at least one instance's bound (significance).
# #     - Deduplicate by structural hash.
# #     """
# #     kept: List[Conjecture] = []
# #     seen_hashes: set[str] = set()
# #     for conj in conjectures:
# #         try:
# #             score = dalmatian_score(conj, df)
# #             if not score["truth_ok"]:
# #                 log_event(f"Rejected (fails truth): {conj.name}")
# #                 continue
# #             if not _is_significant(conj, df, kept):
# #                 log_event(f"Rejected (insignificant): {conj.name}")
# #                 continue
# #             h = hash_conjecture(conj)
# #             if h in seen_hashes:
# #                 log_event(f"Rejected (duplicate): {conj.name}")
# #                 continue
# #             kept.append(conj)
# #             seen_hashes.add(h)
# #         except Exception as e:
# #             log_event(f"Error evaluating conjecture {getattr(conj,'name','?')}: {e}")
# #             continue
# #     return kept

# """
# Dalmatian heuristic: truth and significance filtering for conjectures.

# Used both as:
# 1. A pre-acceptance filter (during conjecture generation)
# 2. A post-processing refinement (after conjecture generation)
# """

# from __future__ import annotations
# import pandas as pd
# import numpy as np
# from typing import List, Dict, Any, Optional

# from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge
# from txgraffiti2025.processing.utils import (
#     truth_mask, touch_count, slack_summary, hash_conjecture, log_event
# )

# # ============================================================
# # Helper: robustly extract target name and direction
# # ============================================================

# def _extract_col_name(e) -> Optional[str]:
#     """Try to recover a column/variable name from an Expr-like node."""
#     # Try common attributes used by different Expr impls
#     for attr in ("col", "column", "name"):
#         if hasattr(e, attr):
#             v = getattr(e, attr)
#             if isinstance(v, str):
#                 return v
#     # Fallback: repr like "'alpha'"
#     r = repr(e)
#     if isinstance(r, str) and len(r) >= 2 and r[0] == r[-1] == "'":
#         return r[1:-1]
#     # Raw string as a last resort
#     if isinstance(e, str):
#         return e
#     return None

# def _target_and_direction(conj: Conjecture) -> tuple[str, str] | None:
#     """Infer the (target column name, inequality direction) from the relation."""
#     rel = conj.relation
#     if isinstance(rel, Le):
#         direction = "le"
#     elif isinstance(rel, Ge):
#         direction = "ge"
#     else:
#         return None
#     tname = _extract_col_name(rel.left)
#     return (tname, direction) if tname else None

# def _applicable_mask(df: pd.DataFrame, cond) -> pd.Series:
#     """Return applicability mask aligned to df.index."""
#     if cond is None:
#         return pd.Series(True, index=df.index)
#     m = cond.mask(df)
#     return m.reindex(df.index, fill_value=False).astype(bool)

# # ============================================================
# # Truth and Significance Tests
# # ============================================================

# def dalmatian_score(conj: Conjecture, df: pd.DataFrame) -> Dict[str, Any]:
#     """Return a diagnostic score dictionary for the Dalmatian heuristic."""
#     applicable, holds, fails = conj.check(df)
#     truth_ok = len(fails) == 0
#     tcount = touch_count(conj, df)
#     min_slack, mean_slack = slack_summary(conj, df)
#     return {
#         "name": getattr(conj, "name", "Conjecture"),
#         "truth_ok": truth_ok,
#         "touch_count": tcount,
#         "min_slack": min_slack,
#         "mean_slack": mean_slack,
#     }

# def _is_significant(conj: Conjecture, df: pd.DataFrame, accepted: List[Conjecture]) -> bool:
#     """
#     True if this conjecture gives a strictly tighter bound somewhere on rows where
#     both it and an already-accepted bound apply, OR it increases coverage by applying
#     to rows where no prior bound (same target/direction) applied.
#     """
#     if not accepted:
#         return True

#     rel = conj.relation
#     td = _target_and_direction(conj)
#     if td is None:
#         # If we can't infer (target,direction), don't block on significance
#         return True
#     tname, direction = td

#     prevs = [c for c in accepted if _target_and_direction(c) == td]
#     if not prevs:
#         return True

#     # Coverage gain: new applies where no previous applies
#     new_app = _applicable_mask(df, conj.condition)
#     any_prev_app = None
#     for p in prevs:
#         p_app = _applicable_mask(df, p.condition)
#         any_prev_app = p_app if any_prev_app is None else (any_prev_app | p_app)
#     if any_prev_app is None:
#         any_prev_app = pd.Series(False, index=df.index)

#     if (new_app & ~any_prev_app).any():
#         return True

#     # Intersection-tightness
#     try:
#         rhs_new = np.asarray(rel.right.eval(df), dtype=float)
#     except Exception:
#         return False

#     eps = 1e-9
#     if direction == "le":
#         # Lower RHS is tighter
#         best_prev_rhs = None
#         best_prev_app = None
#         for p in prevs:
#             try:
#                 r = np.asarray(p.relation.right.eval(df), dtype=float)
#                 a = _applicable_mask(df, p.condition)
#             except Exception:
#                 continue
#             best_prev_rhs = r if best_prev_rhs is None else np.minimum(best_prev_rhs, r)
#             best_prev_app = a if best_prev_app is None else (best_prev_app | a)

#         if best_prev_rhs is None:
#             return True

#         inter = (new_app & best_prev_app).to_numpy()
#         if inter.any():
#             return bool((rhs_new[inter] < best_prev_rhs[inter] - eps).any())
#         return False

#     else:
#         # direction == "ge": higher RHS is tighter
#         best_prev_rhs = None
#         best_prev_app = None
#         for p in prevs:
#             try:
#                 r = np.asarray(p.relation.right.eval(df), dtype=float)
#                 a = _applicable_mask(df, p.condition)
#             except Exception:
#                 continue
#             best_prev_rhs = r if best_prev_rhs is None else np.maximum(best_prev_rhs, r)
#             best_prev_app = a if best_prev_app is None else (best_prev_app | a)

#         if best_prev_rhs is None:
#             return True

#         inter = (new_app & best_prev_app).to_numpy()
#         if inter.any():
#             return bool((rhs_new[inter] > best_prev_rhs[inter] + eps).any())
#         return False


# # ============================================================
# # Main filter
# # ============================================================

# def dalmatian_filter(conjectures: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
#     """
#     Apply the Dalmatian heuristic:
#     - Remove conjectures failing the truth test.
#     - Keep only those that improve at least one instance's bound (significance).
#     - Deduplicate by structural hash.
#     """
#     kept: List[Conjecture] = []
#     seen_hashes: set[str] = set()
#     for conj in conjectures:
#         try:
#             score = dalmatian_score(conj, df)
#             if not score["truth_ok"]:
#                 log_event(f"Rejected (fails truth): {conj.name}")
#                 continue
#             if not _is_significant(conj, df, kept):
#                 log_event(f"Rejected (insignificant): {conj.name}")
#                 continue
#             h = hash_conjecture(conj)
#             if h in seen_hashes:
#                 log_event(f"Rejected (duplicate): {conj.name}")
#                 continue
#             kept.append(conj)
#             seen_hashes.add(h)
#         except Exception as e:
#             log_event(f"Error evaluating conjecture {getattr(conj,'name','?')}: {e}")
#             continue
#     return kept

# txgraffiti2025/processing/post/dalmatian.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge

# ============================================================
# Minimal, local helpers (no external utils dependency)
# ============================================================

def _applicable_mask(df: pd.DataFrame, cond) -> pd.Series:
    if cond is None:
        return pd.Series(True, index=df.index)
    m = cond.mask(df)
    return pd.Series(m, index=df.index).astype(bool)

def _mask_key(df: pd.DataFrame, cond) -> str:
    """
    Compact fingerprint of the hypothesis mask (aligned to df.index).
    Ensures significance comparisons stay within the SAME hypothesis only.
    """
    m = _applicable_mask(df, cond).to_numpy(dtype=np.uint8, copy=False)
    return m.tobytes().hex()

def _truth_ok(conj: Conjecture, df: pd.DataFrame) -> bool:
    applicable, holds, _ = conj.check(df, auto_base=True)
    applicable = pd.Series(applicable, index=df.index).astype(bool)
    holds = pd.Series(holds, index=df.index).astype(bool)
    return bool(holds[applicable].all())

def _safe_eval_rhs(df: pd.DataFrame, conj: Conjecture) -> Optional[np.ndarray]:
    rel = conj.relation
    if isinstance(rel, Eq):
        return None
    try:
        rhs = rel.right.eval(df)
        rhs = pd.to_numeric(rhs, errors="coerce").to_numpy(dtype=float)
        return rhs
    except Exception:
        return None

def _extract_col_name(e) -> Optional[str]:
    for attr in ("col", "column", "name"):
        v = getattr(e, attr, None)
        if isinstance(v, str):
            return v
    if isinstance(e, str):
        return e
    r = repr(e)
    if isinstance(r, str) and len(r) >= 2 and r[0] == r[-1] == "'":
        return r[1:-1]
    return None

def _target_and_direction(conj: Conjecture) -> Optional[Tuple[str, str]]:
    rel = conj.relation
    if isinstance(rel, Le):
        direction = "le"
    elif isinstance(rel, Ge):
        direction = "ge"
    else:
        return None
    tname = _extract_col_name(getattr(rel, "left", None))
    return (tname, direction) if tname else None

def _mask_intersection_any(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (a & b).to_numpy()

def _structural_hash(conj: Conjecture) -> str:
    return repr((conj.relation, conj.condition))

# ============================================================
# Significance test (within a single (target,dir,hypothesis) bucket)
# ============================================================

def _is_significant_within_bucket(
    conj: Conjecture,
    df: pd.DataFrame,
    bucket_accepteds: List[Conjecture],
) -> bool:
    """
    Keep conj if, within THIS bucket (same target, direction, and hypothesis mask),
      (a) it’s the first; or
      (b) it’s strictly tighter on some applicable row.
    Coverage-gain is irrelevant inside the same mask (identical coverage by construction).
    """
    if not bucket_accepteds:
        return True

    td = _target_and_direction(conj)
    if td is None:
        return True  # can’t assess; allow

    direction = td[1]
    rhs_new = _safe_eval_rhs(df, conj)
    if rhs_new is None:
        # Equalities or non-evaluable RHS: treat as not significant vs same-mask inequalities.
        return False

    new_app = _applicable_mask(df, conj.condition)

    # Best previous RHS among already accepted in this SAME bucket
    best_prev_rhs = None
    for p in bucket_accepteds:
        r = _safe_eval_rhs(df, p)
        if r is None:
            # Previous equality with same target/direction and same mask → effectively unbeatable
            continue
        if best_prev_rhs is None:
            best_prev_rhs = r.copy()
        else:
            if direction == "le":
                best_prev_rhs = np.fmin(best_prev_rhs, r)
            else:
                best_prev_rhs = np.fmax(best_prev_rhs, r)

    if best_prev_rhs is None:
        return True

    inter = new_app.to_numpy()
    if not inter.any():
        # Same mask bucket guarantees identical coverage, but be defensive
        return False

    eps = 1e-9
    rhs_new_i = rhs_new[inter]
    best_prev_rhs_i = best_prev_rhs[inter]
    if direction == "le":
        return bool(np.any(rhs_new_i < best_prev_rhs_i - eps))
    else:
        return bool(np.any(rhs_new_i > best_prev_rhs_i + eps))

# ============================================================
# Public API
# ============================================================

def dalmatian_score(conj: Conjecture, df: pd.DataFrame) -> Dict[str, Any]:
    try:
        ok = _truth_ok(conj, df)
    except Exception:
        ok = False

    touch = 0
    min_slack = float("inf")
    mean_slack = float("inf")
    try:
        applicable, holds, _ = conj.check(df, auto_base=True)
        applicable = pd.Series(applicable, index=df.index).astype(bool)
        holds = pd.Series(holds, index=df.index).astype(bool)
        ok_mask = applicable & holds
        rel = conj.relation
        if isinstance(rel, Eq):
            l = rel.left.eval(df); r = rel.right.eval(df)
            touch = int((pd.to_numeric(l, errors="coerce") - pd.to_numeric(r, errors="coerce")).abs()
                        .le(float(rel.tol)).where(ok_mask, False).sum())
            s = - (pd.to_numeric(l, errors="coerce") - pd.to_numeric(r, errors="coerce")).abs()
        else:
            s = rel.slack(df)
            s = pd.to_numeric(s, errors="coerce")
            touch = int(s.where(ok_mask, np.nan).abs().le(1e-9).sum())
        s = s.where(ok_mask)
        if s.notna().any():
            min_slack = float(s.min())
            mean_slack = float(s.mean())
    except Exception:
        pass

    return {
        "name": getattr(conj, "name", "Conjecture"),
        "truth_ok": bool(ok),
        "touch_count": int(touch),
        "min_slack": float(min_slack),
        "mean_slack": float(mean_slack),
    }

def dalmatian_filter(conjectures: List[Conjecture], df: pd.DataFrame) -> List[Conjecture]:
    """
    Truth + significance + dedup (silent).
    Significance comparisons are done ONLY within the same bucket:
        bucket = (target_name, direction, hypothesis_mask_key)
    so different hypotheses (even if overlapping) are never compared.
    """
    kept_by_bucket: Dict[Tuple[str, str, str], List[Conjecture]] = {}
    seen: set[str] = set()
    out: List[Conjecture] = []

    for conj in conjectures:
        # Dedup first
        h = _structural_hash(conj)
        if h in seen:
            continue

        # Truth check
        try:
            if not _truth_ok(conj, df):
                continue
        except Exception:
            continue

        # Bucket key: (target, dir, mask_key)
        td = _target_and_direction(conj)
        if td is None:
            # If not an inequality, skip significance and accept into its own bucket keyed by None + mask
            mk = _mask_key(df, conj.condition)
            bucket = (None, None, mk)  # type: ignore
        else:
            mk = _mask_key(df, conj.condition)
            bucket = (td[0], td[1], mk)

        bucket_list = kept_by_bucket.setdefault(bucket, [])

        # Significance only within the same bucket
        try:
            if td is not None and not _is_significant_within_bucket(conj, df, bucket_list):
                continue
        except Exception:
            continue

        bucket_list.append(conj)
        out.append(conj)
        seen.add(h)

    return out
