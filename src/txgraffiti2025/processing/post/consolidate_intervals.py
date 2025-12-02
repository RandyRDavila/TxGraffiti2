# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Iterable, List, Optional, Tuple, Dict, DefaultDict
# from collections import defaultdict

# import numpy as np
# import pandas as pd

# from txgraffiti2025.forms.utils import Expr, Const, to_expr, BinOp
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
# from txgraffiti2025.forms.predicates import Predicate


# # ---------------------------
# # Utilities
# # ---------------------------

# def _mask(df: pd.DataFrame, H: Optional[Predicate]) -> pd.Series:
#     if H is None:
#         return pd.Series(True, index=df.index)
#     return H.mask(df).reindex(df.index, fill_value=False).astype(bool)

# def _same_mask(df: pd.DataFrame, A: Optional[Predicate], B: Optional[Predicate]) -> bool:
#     a = _mask(df, A); b = _mask(df, B)
#     return bool((a == b).all())

# def _finite_series(s: pd.Series) -> pd.Series:
#     return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

# def _eval(expr: Expr, df: pd.DataFrame) -> pd.Series:
#     with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
#         return _finite_series(expr.eval(df))

# def _is_le_pointwise(
#     df: pd.DataFrame, e1: Expr, e2: Expr, mask: pd.Series, tol: float
# ) -> bool:
#     s1 = _eval(e1, df)[mask]
#     s2 = _eval(e2, df)[mask]
#     ok = s1.notna() & s2.notna()
#     if not ok.any():
#         return False
#     return bool((s1[ok] <= s2[ok] + tol).all())

# def _struct_key(e: Expr) -> str:
#     return repr(e)

# def _colterm_name(e: Expr) -> Optional[str]:
#     # best-effort: find ColumnTerm('name') at the top-level left of relation
#     r = repr(e)
#     # cheap parse:
#     marker = "ColumnTerm('"
#     i = r.find(marker)
#     if i == -1:
#         return None
#     j = r.find("')", i + len(marker))
#     if j == -1:
#         return None
#     return r[i+len(marker):j]


# # ---------------------------
# # Output dataclasses
# # ---------------------------

# @dataclass
# class Chain:
#     hypothesis: Optional[Predicate]
#     target: str
#     lowers: List[Expr]      # ordered: L1 <= L2 <= ... <= target
#     uppers: List[Expr]      # ordered: target <= U1 <= U2 <= ...
#     pieces: List[Conjecture]  # the individual links we used to certify the chain

#     def pretty(self, unicode_ops: bool = True) -> str:
#         Htxt = repr(self.hypothesis) if self.hypothesis is not None else "TRUE"
#         parts: List[str] = []
#         # lower chain
#         for i, L in enumerate(self.lowers):
#             parts.append(L.pretty(unicode_ops=unicode_ops) if hasattr(L, "pretty") else repr(L))
#             parts.append(" ≤ ")
#         # target
#         parts.append(self.target)
#         # upper chain
#         for i, U in enumerate(self.uppers):
#             parts.append(" ≤ ")
#             parts.append(U.pretty(unicode_ops=unicode_ops) if hasattr(U, "pretty") else repr(U))
#         return f"({Htxt}) ⇒ " + "".join(parts)


# # ---------------------------
# # Core consolidation
# # ---------------------------

# def consolidate_interval_chains(
#     df: pd.DataFrame,
#     conjectures: Iterable[Conjecture],
#     *,
#     tol: float = 1e-9,
# ) -> List[Chain]:
#     """
#     Given a bag of conjectures, build interval chains per (hypothesis-mask, target).

#     We consider:
#       - upper bounds: target ≤ U
#       - lower bounds: target ≥ L

#     Steps:
#       1) group by identical hypothesis mask and by target column
#       2) within each group, collect all U (for Le) and all L (for Ge)
#       3) drop dominated U's (if U_a ≤ U_b pointwise => drop U_b); analogously for L's
#       4) attempt to totally order each remaining set; if total order succeeds, produce a chain
#          (if not totally orderable, we keep the minimal antichain – still useful)
#       5) if both lowers and uppers exist and max(L) ≤ min(U), assemble L_chain ≤ target ≤ U_chain
#     """
#     # 1) group by (mask signature, target)
#     # build a stable signature for hypothesis mask
#     groups: Dict[Tuple[Tuple[bool, ...], str], Dict[str, List[Tuple[Conjecture, Expr]]]] = defaultdict(lambda: {"uppers": [], "lowers": []})

#     for c in conjectures:
#         rel = c.relation
#         # only inequalities; skip Eq
#         if isinstance(rel, Eq):
#             continue

#         left = rel.left
#         right = rel.right
#         target_name = _colterm_name(left)
#         if target_name is None:
#             # try symmetric: maybe target on right for >= vs <= normalization
#             target_name = _colterm_name(right)
#             if target_name is None:
#                 continue

#         # normalize: we want target on the left for keying purposes
#         is_upper = isinstance(rel, Le)
#         is_lower = isinstance(rel, Ge)

#         if not (is_upper or is_lower):
#             continue

#         # hypothesis signature (mask)
#         m = _mask(df, c.condition)
#         key = (tuple(bool(x) for x in m.values), target_name)

#         # ensure target really sits on the left in our interpretation
#         # if target is on right, swap & flip orientation
#         if _colterm_name(left) == target_name:
#             tgt_expr = left
#             other = right
#             # orientation unchanged
#             if is_upper:
#                 groups[key]["uppers"].append((c, other))
#             else:
#                 groups[key]["lowers"].append((c, other))
#         else:
#             # target on right
#             tgt_expr = right
#             other = left
#             # flip orientation if needed
#             if is_upper:
#                 # (LHS ≤ RHS), and RHS=target ⇒ other ≤ target ⇒ lower bound
#                 groups[key]["lowers"].append((c, other))
#             elif is_lower:
#                 # (LHS ≥ RHS), and RHS=target ⇒ other ≥ target ⇒ upper bound
#                 groups[key]["uppers"].append((c, other))

#     # 2–5) per group, prune dominance, order, and assemble chain
#     out: List[Chain] = []

#     for (mask_sig, target_name), sides in groups.items():
#         H: Optional[Predicate] = None
#         # recover any representative hypothesis (all share mask); for display only
#         if sides["uppers"]:
#             H = sides["uppers"][0][0].condition
#         elif sides["lowers"]:
#             H = sides["lowers"][0][0].condition

#         m = pd.Series(list(mask_sig), index=df.index)

#         def prune_and_order(exprs: List[Expr]) -> Tuple[List[Expr], List[Conjecture]]:
#             """
#             Remove dominated expressions and order the rest if possible.
#             Returns ordered exprs and a list of conjecture 'pieces' used.
#             """
#             if not exprs:
#                 return [], []

#             # dedup structurally
#             uniq_exprs: List[Expr] = []
#             seen = set()
#             for _, e in exprs:
#                 k = _struct_key(e)
#                 if k not in seen:
#                     seen.add(k)
#                     uniq_exprs.append(e)

#             # dominance pruning
#             keep = [True] * len(uniq_exprs)
#             for i, ei in enumerate(uniq_exprs):
#                 if not keep[i]:
#                     continue
#                 for j, ej in enumerate(uniq_exprs):
#                     if i == j or not keep[j]:
#                         continue
#                     # For uppers, if ei ≤ ej everywhere, drop ej
#                     # For lowers, if ei ≥ ej everywhere, drop ej
#                     if current_side == "uppers":
#                         if _is_le_pointwise(df, ei, ej, m, tol):
#                             keep[j] = False
#                     else:  # lowers
#                         if _is_le_pointwise(df, ej, ei, m, tol):
#                             keep[j] = False

#             reduced = [e for kflag, e in zip(keep, uniq_exprs) if kflag]

#             # try to totally order
#             if len(reduced) <= 1:
#                 return reduced, []

#             # simple O(n^2) topological by pairwise ≤ tests
#             # build graph ei ≤ ej edges (for uppers); reversed for lowers
#             n = len(reduced)
#             adj = [[False]*n for _ in range(n)]
#             for i in range(n):
#                 for j in range(n):
#                     if i == j:
#                         continue
#                     if current_side == "uppers":
#                         le = _is_le_pointwise(df, reduced[i], reduced[j], m, tol)
#                         if le:
#                             adj[i][j] = True
#                     else:
#                         # lowers: we want ei ≤ ej for order on the chain L1 ≤ L2 ≤ …
#                         le = _is_le_pointwise(df, reduced[i], reduced[j], m, tol)
#                         if le:
#                             adj[i][j] = True

#             # compute a linear extension if possible by sorting with comparator
#             # use a safe sort: compare by the median value as a fallback, but only
#             # keep order if pairwise ≤ holds; else leave as-is
#             try:
#                 medvals = []
#                 for e in reduced:
#                     s = _eval(e, df)[m].dropna()
#                     medvals.append(float(s.median()) if s.size else np.inf)
#                 idxs = list(range(n))
#                 idxs.sort(key=lambda k: medvals[k])  # ascending
#                 # verify chain
#                 ok_chain = True
#                 for a, b in zip(idxs, idxs[1:]):
#                     if not _is_le_pointwise(df, reduced[a], reduced[b], m, tol):
#                         ok_chain = False
#                         break
#                 if ok_chain:
#                     ordered = [reduced[k] for k in idxs]
#                 else:
#                     ordered = reduced  # cannot guarantee comparability
#             except Exception:
#                 ordered = reduced

#             return ordered, []

#         # process uppers
#         current_side = "uppers"
#         uppers_expr = [(c, e) for (c, e) in sides["uppers"]]
#         uppers_ordered, pieces_u = prune_and_order(uppers_expr)

#         # process lowers
#         current_side = "lowers"
#         lowers_expr = [(c, e) for (c, e) in sides["lowers"]]
#         lowers_ordered, pieces_l = prune_and_order(lowers_expr)

#         # Ensure sandwich compatibility: max(lower) ≤ min(upper)
#         if lowers_ordered and uppers_ordered:
#             Lmax = lowers_ordered[-1]
#             Umin = uppers_ordered[0]
#             if not _is_le_pointwise(df, Lmax, Umin, m, tol):
#                 # incompatible; drop the side that breaks compatibility least
#                 # heuristic: keep the side with smaller span (by median)
#                 try:
#                     sL = _eval(Lmax, df)[m].dropna()
#                     sU = _eval(Umin, df)[m].dropna()
#                     # if Lmax > Umin somewhere, prefer dropping the side whose med is farther
#                     if sL.size and sU.size and float(sL.median()) <= float(sU.median()):
#                         # keep both anyway (best effort)
#                         pass
#                     else:
#                         # fallback: keep separate chains by side (still useful)
#                         pass
#                 except Exception:
#                     pass

#         chain = Chain(
#             hypothesis=H,
#             target=target_name,
#             lowers=lowers_ordered,
#             uppers=uppers_ordered,
#             pieces=[p for p, _ in sides["uppers"]] + [p for p, _ in sides["lowers"]],
#         )
#         out.append(chain)

#     return out


from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import Expr, Const, to_expr, BinOp, ColumnTerm
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, Eq
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.pretty import format_expr


# =========================
# Helpers
# =========================

def _mask(df: pd.DataFrame, H: Optional[Predicate]) -> pd.Series:
    if H is None:
        return pd.Series(True, index=df.index)
    return H.mask(df).reindex(df.index, fill_value=False).astype(bool, copy=False)

def _mask_signature(m: pd.Series) -> Tuple[bool, ...]:
    # Stable, index-aligned boolean signature
    return tuple(bool(x) for x in m.reindex(m.index).to_numpy())

def _finite_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)

def _eval(expr: Expr, df: pd.DataFrame) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = expr.eval(df)
    return _finite_series(pd.to_numeric(out, errors="coerce")).reindex(df.index)

def _as_colterm_name(e: Expr) -> Optional[str]:
    return e.col if isinstance(e, ColumnTerm) else None

def _struct_key(e: Expr) -> str:
    return repr(e)

def _le_pointwise(
    df: pd.DataFrame,
    a: Expr,
    b: Expr,
    mask: pd.Series,
    *,
    tol: float,
    min_valid: int = 1,
) -> bool:
    sA = _eval(a, df)[mask]
    sB = _eval(b, df)[mask]
    ok = sA.notna() & sB.notna()
    if ok.sum() < min_valid:
        return False
    return bool((sA[ok] <= sB[ok] + tol).all())


# =========================
# Output
# =========================

@dataclass
class Chain:
    hypothesis: Optional[Predicate]
    target: str
    lowers: List[Expr]                 # L1 <= L2 <= ... <= target
    uppers: List[Expr]                 # target <= U1 <= U2 <= ...
    pieces: List[Conjecture]           # all bounds used to certify this chain

    def pretty(self, unicode_ops: bool = True) -> str:
        Htxt = repr(self.hypothesis) if self.hypothesis is not None else "TRUE"

        def fmt(e):
            return format_expr(e) if isinstance(e, Expr) else str(e)

        parts: List[str] = []
        for L in self.lowers:
            parts.append(fmt(L))
            parts.append(" ≤ ")
        parts.append(self.target)
        for U in self.uppers:
            parts.append(" ≤ ")
            parts.append(fmt(U))
        return f"({Htxt}) ⇒ " + "".join(parts)



# =========================
# Core
# =========================

def consolidate_interval_chains(
    df: pd.DataFrame,
    conjectures: Iterable[Conjecture],
    *,
    tol: float = 1e-9,
    min_valid_pairs: int = 3,    # require at least a few comparable points
) -> List[Chain]:
    """
    Consolidate a bag of inequalities into interval-style chains per (hypothesis, target).

    Recognized bounds:
      - Upper:  target ≤ U
      - Lower:  target ≥ L
    If the target appears on the RHS, we flip the inequality orientation accordingly.

    Steps per group (same hypothesis mask, same target):
      1) Gather all candidate U and L.
      2) Deduplicate structurally (repr-based).
      3) Prune dominated bounds:
           - For uppers: drop U_j if exists U_i with U_i ≤ U_j pointwise.
           - For lowers: drop L_j if exists L_i with L_j ≤ L_i pointwise (i.e., keep larger lowers).
      4) Attempt to totally order each remaining set by verified pointwise ≤.
         If not fully comparable, use median values to propose an order—but
         only keep that order if each adjacent pair is pointwise ≤.
      5) Emit a Chain with the ordered lowers (ascending) and uppers (ascending).
         The chain is meaningful even if only one side exists.
         If both sides exist and max(L) ≤ min(U) fails, we still emit both sides
         (the consumer can decide whether to use only one side).
    """
    # Group by (mask signature, target name)
    grouped: Dict[Tuple[Tuple[bool, ...], str], Dict[str, List[Tuple[Conjecture, Expr]]]] = \
        defaultdict(lambda: {"uppers": [], "lowers": []})

    for conj in conjectures:
        rel = conj.relation
        if isinstance(rel, Eq):
            continue

        left, right = rel.left, rel.right
        left_name = _as_colterm_name(left)
        right_name = _as_colterm_name(right)

        # Need a recognizable target name on at least one side
        target = left_name or right_name
        if target is None:
            continue

        # Normalize orientation into "target on the left" semantics
        is_upper = isinstance(rel, Le)
        is_lower = isinstance(rel, Ge)

        if not (is_upper or is_lower):
            continue

        m = _mask(df, conj.condition)
        key = (_mask_signature(m), target)

        if left_name == target:
            # target on left: relational direction is literal
            if is_upper:
                grouped[key]["uppers"].append((conj, right))
            else:
                grouped[key]["lowers"].append((conj, right))
        else:
            # target on right: flip meaning
            if is_upper:
                # (lhs ≤ target) ⇒ lhs is a *lower* bound
                grouped[key]["lowers"].append((conj, left))
            else:
                # (lhs ≥ target) ⇒ lhs is an *upper* bound
                grouped[key]["uppers"].append((conj, left))

    out: List[Chain] = []

    for (mask_sig, target), sides in grouped.items():
        mask = pd.Series(mask_sig, index=df.index, dtype=bool)

        # pick a representative hypothesis for display (they share mask)
        hyp: Optional[Predicate] = None
        if sides["uppers"]:
            hyp = sides["uppers"][0][0].condition
        elif sides["lowers"]:
            hyp = sides["lowers"][0][0].condition

        def dedup(exprs: List[Tuple[Conjecture, Expr]]) -> Tuple[List[Expr], List[Conjecture]]:
            seen = set()
            keep_exprs: List[Expr] = []
            pieces: List[Conjecture] = []
            for c, e in exprs:
                k = _struct_key(e)
                if k in seen:
                    continue
                seen.add(k)
                keep_exprs.append(e)
                pieces.append(c)
            return keep_exprs, pieces

        uppers_all, pieces_u = dedup(sides["uppers"])
        lowers_all, pieces_l = dedup(sides["lowers"])

        # dominance pruning
        def prune_uppers(exprs: List[Expr]) -> List[Expr]:
            if len(exprs) <= 1:
                return exprs
            keep = [True] * len(exprs)
            for i, ei in enumerate(exprs):
                if not keep[i]:
                    continue
                for j, ej in enumerate(exprs):
                    if i == j or not keep[j]:
                        continue
                    # if ei ≤ ej pointwise, drop ej
                    if _le_pointwise(df, ei, ej, mask, tol=tol, min_valid=min_valid_pairs):
                        keep[j] = False
            return [e for e, k in zip(exprs, keep) if k]

        def prune_lowers(exprs: List[Expr]) -> List[Expr]:
            if len(exprs) <= 1:
                return exprs
            keep = [True] * len(exprs)
            for i, ei in enumerate(exprs):
                if not keep[i]:
                    continue
                for j, ej in enumerate(exprs):
                    if i == j or not keep[j]:
                        continue
                    # for lowers keep the larger ones:
                    # drop ej if ej ≤ ei (i.e., ei dominates/raises the floor)
                    if _le_pointwise(df, ej, ei, mask, tol=tol, min_valid=min_valid_pairs):
                        keep[j] = False
            return [e for e, k in zip(exprs, keep) if k]

        uppers = prune_uppers(uppers_all)
        lowers = prune_lowers(lowers_all)

        # try to order each side by verified ≤
        def order_by_le(exprs: List[Expr]) -> List[Expr]:
            if len(exprs) <= 1:
                return exprs
            # propose sort by median; validate adjacency
            meds: List[float] = []
            for e in exprs:
                s = _eval(e, df)[mask].dropna()
                meds.append(float(s.median()) if s.size else np.inf)
            idx = list(range(len(exprs)))
            idx.sort(key=lambda k: meds[k])
            ordered = [exprs[k] for k in idx]
            # validate chain
            ok = True
            for a, b in zip(ordered, ordered[1:]):
                if not _le_pointwise(df, a, b, mask, tol=tol, min_valid=min_valid_pairs):
                    ok = False
                    break
            return ordered if ok else exprs

        uppers = order_by_le(uppers)
        lowers = order_by_le(lowers)

        # (Optional) compatibility check between max(lower) and min(upper).
        # If incompatible, we still return both sides; the consumer can decide how to use them.
        if lowers and uppers:
            Lmax = lowers[-1]
            Umin = uppers[0]
            # No action if incompatible; still useful to surface both chains.

        out.append(Chain(
            hypothesis=hyp,
            target=target,
            lowers=lowers,
            uppers=uppers,
            pieces=[*pieces_l, *pieces_u],
        ))

    # stable order: by hypothesis repr, then target, then sizes
    out.sort(key=lambda ch: (repr(ch.hypothesis), ch.target, len(ch.lowers), len(ch.uppers)))
    return out
