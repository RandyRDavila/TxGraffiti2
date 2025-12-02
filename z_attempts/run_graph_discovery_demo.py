# # run_graph_numeric_discovery_demo.py
# from __future__ import annotations
# import math
# import numpy as np
# import pandas as pd
# from itertools import combinations, product

# from txgraffiti.example_data import graph_data as df

# from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge
# from txgraffiti2025.forms.predicates import Predicate, Where, AndPred, LT, GT
# from txgraffiti2025.forms.pretty import format_pred
# from txgraffiti2025.forms.utils import to_expr

# def _mask(df: pd.DataFrame, pred: Predicate | None) -> pd.Series:
#     if pred is None:
#         return pd.Series(True, index=df.index)
#     m = pred.mask(df)
#     return m.reindex(df.index, fill_value=False).astype(bool)

# def pretty_pred(pred: Predicate | None) -> str:
#     return format_pred(pred, unicode_ops=True)

# def _flatten_and_pred(p):
#     if p is None:
#         return []
#     if isinstance(p, AndPred):
#         return _flatten_and_pred(p.a) + _flatten_and_pred(p.b)
#     return [p]

# def _is_true_pred(p) -> bool:
#     return getattr(p, "name", "") == "TRUE"

# def _and_many(parts):
#     parts = [q for q in parts if q is not None and not _is_true_pred(q)]
#     if not parts:
#         return None
#     out = parts[0]
#     for q in parts[1:]:
#         out = AndPred(out, q)
#     return out

# def normalize_and(pred):
#     if pred is None or _is_true_pred(pred):
#         return None
#     parts = _flatten_and_pred(pred)
#     seen, uniq = set(), []
#     for q in parts:
#         k = repr(q)
#         if k not in seen:
#             seen.add(k)
#             uniq.append(q)
#     return _and_many(uniq)

# def _pred_key(df: pd.DataFrame, pred) -> tuple:
#     pn = normalize_and(pred)
#     key_repr = repr(pn)
#     m = _mask(df, pn)
#     supp = int(m.sum())
#     nz = tuple(map(int, np.flatnonzero(m.values[:min(len(m), 64)])))
#     return (key_repr, supp, nz)

# class TruePred(Predicate):
#     def __init__(self):
#         self.name = "TRUE"
#     def mask(self, d: pd.DataFrame) -> pd.Series:
#         return pd.Series(True, index=d.index, dtype=bool)

# def drop_boolean_columns(dfin: pd.DataFrame) -> pd.DataFrame:
#     bool_cols = [c for c in dfin.columns if pd.api.types.is_bool_dtype(dfin[c])]
#     bin_like = []
#     for c in dfin.columns:
#         if c in bool_cols:
#             continue
#         vals = dfin[c].dropna().unique()
#         if len(vals) <= 2 and set(vals).issubset({0, 1}):
#             bin_like.append(c)
#     drop_cols = list(set(bool_cols + bin_like))
#     out = dfin.drop(columns=drop_cols, errors="ignore").copy()
#     return out

# def weakest_true_lower_and_eq_mask(df: pd.DataFrame, col: str) -> tuple[float, pd.Series]:
#     y = pd.to_numeric(df[col], errors="coerce")
#     y = y.dropna()
#     if y.empty:
#         raise ValueError(f"No numeric data in column {col}.")
#     c = float(y.min())
#     eq_mask = (pd.to_numeric(df[col], errors="coerce") == c).fillna(False)
#     return c, eq_mask

# def atom_eq_col_const(col: str, c: float) -> Predicate:
#     if abs(c - int(round(c))) < 1e-12:
#         c_txt = str(int(round(c)))
#     else:
#         c_txt = f"{c:g}"
#     return Where(lambda d, col=col, c=c: (pd.to_numeric(d[col], errors="coerce") == c).fillna(False),
#                  name=f"({col}={c_txt})")

# def atom_ge_col_const(col: str, c: float) -> Predicate:
#     if abs(c - int(round(c))) < 1e-12:
#         c_txt = str(int(round(c)))
#     else:
#         c_txt = f"{c:g}"
#     return Where(lambda d, col=col, c=c: (pd.to_numeric(d[col], errors="coerce") >= c).fillna(False),
#                  name=f"({col}≥{c_txt})")

# def _tightness_preds(df: pd.DataFrame, conj: Conjecture, *, atol=1e-9):
#     rel = conj.relation
#     cond = conj.condition
#     def _eq_mask(d, L, R, tol):
#         Ls = pd.to_numeric(L.eval(d), errors="coerce")
#         Rs = pd.to_numeric(R.eval(d), errors="coerce")
#         return pd.Series(np.isclose(Ls - Rs, 0.0, atol=float(tol)), index=d.index)
#     if isinstance(rel, Le):
#         P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
#                            name=f"(({rel.left}) = ({rel.right}))")
#         P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
#         P_str = AndPred(cond, LT(rel.left, rel.right)) if cond else LT(rel.left, rel.right)
#         return [normalize_and(P_eq), normalize_and(P_str)]
#     if isinstance(rel, Ge):
#         P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
#                            name=f"(({rel.left}) = ({rel.right}))")
#         P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
#         P_str = AndPred(cond, GT(rel.left, rel.right)) if cond else GT(rel.left, rel.right)
#         return [normalize_and(P_eq), normalize_and(P_str)]
#     if isinstance(rel, Eq):
#         P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, rel.tol),
#                            name=f"(({rel.left}) = ({rel.right}))")
#         P_eq = AndPred(cond, P_eq_inner) if cond else P_eq_inner
#         return [normalize_and(P_eq)]
#     return []

# def _support(m: pd.Series) -> int:
#     return int(m.sum())

# def _prob(p: int, n: int) -> float:
#     return 0.0 if n <= 0 else p / float(n)

# def _entropy(p: float) -> float:
#     if p <= 0 or p >= 1:
#         return 0.0
#     return -p*math.log(p + 1e-15) - (1-p)*math.log(1-p + 1e-15)

# def lift(Hm: pd.Series, Cm: pd.Series) -> float:
#     N = len(Hm)
#     pH = _prob(_support(Hm), N)
#     pC = _prob(_support(Cm), N)
#     pHC = _prob(_support(Hm & Cm), N)
#     return 0.0 if pH == 0 or pC == 0 else pHC / (pH * pC)

# def info_gain(Hm: pd.Series, Cm: pd.Series) -> float:
#     N = len(Hm)
#     if N == 0:
#         return 0.0
#     pH = _prob(_support(Hm), N)
#     H_prior = _entropy(pH)
#     Nc = _support(Cm)
#     pH_c = _prob(_support(Hm & Cm), max(Nc,1))
#     pH_nc = _prob(_support(Hm & ~Cm), max(N-Nc,1))
#     H_post = (Nc/N) * _entropy(pH_c) + ((N-Nc)/N) * _entropy(pH_nc)
#     return max(0.0, H_prior - H_post)

# def _and_all(preds):
#     it = iter(preds)
#     acc = next(it)
#     for p in it:
#         acc = AndPred(acc, p)
#     parts = [format_pred(p, unicode_ops=True) for p in preds]
#     acc.name = "(" + ") ∧ (".join(parts) + ")"
#     return acc

# def _or_pred(a: Predicate, b: Predicate) -> Predicate:
#     name = f"({format_pred(a, unicode_ops=True)}) ∨ ({format_pred(b, unicode_ops=True)})"
#     return Where(lambda d, A=a, B=b: _mask(d, A) | _mask(d, B), name=name)

# def build_candidate_hypotheses_with_disjunctions(
#     df: pd.DataFrame,
#     base_hyps,
#     *,
#     include_or: bool = True,
#     min_support: int = 1,
# ):
#     cands = list(base_hyps)
#     base = list(base_hyps)
#     for A, B in combinations(base, 2):
#         try:
#             cand = _and_all([A, B])
#             m = _mask(df, cand)
#             if int(m.sum()) >= min_support:
#                 cands.append(cand)
#         except Exception:
#             pass
#     if include_or:
#         seen_keys = set()
#         for A, B in combinations(base, 2):
#             try:
#                 mA = _mask(df, A); mB = _mask(df, B)
#                 if not (mA & ~mB).any() or not (mB & ~mA).any():
#                     continue
#                 OR = _or_pred(A, B)
#                 mOR = _mask(df, OR)
#                 if int(mOR.sum()) < min_support:
#                     continue
#                 key = tuple(sorted([getattr(A, "name", repr(A)), getattr(B, "name", repr(B))]))
#                 if key in seen_keys:
#                     continue
#                 seen_keys.add(key)
#                 cands.append(OR)
#             except Exception:
#                 pass
#     uniq, seen = [], set()
#     for p in cands:
#         k = getattr(p, "name", repr(p))
#         if k not in seen:
#             seen.add(k)
#             uniq.append(p)
#     return uniq

# def mine_implications_basic(df, tight_preds, candidate_preds, *, min_support=1):
#     out = []
#     for H in tight_preds:
#         Hn = normalize_and(H)
#         Hm = _mask(df, Hn)
#         supH = int(Hm.sum())
#         if supH < min_support:
#             continue
#         for C in candidate_preds:
#             Cn = normalize_and(C)
#             Cm = _mask(df, Cn)
#             if not Cm.any():
#                 continue
#             implies = not (Hm & ~Cm).any()
#             if implies:
#                 equiv = not (Cm & ~Hm).any()
#                 out.append({
#                     "H": Hn, "C": Cn,
#                     "support_H": supH,
#                     "support_C": int(Cm.sum()),
#                     "equiv": bool(equiv),
#                     "IG": info_gain(Hm, Cm),
#                     "lift": lift(Hm, Cm),
#                     "pH_given_C": _prob(_support(Hm & Cm), max(_support(Cm),1)),
#                 })
#     return out

# def minimal_or_cover(df, Hmask: pd.Series, cand_preds, max_terms=3):
#     U = Hmask.copy()
#     cover = []
#     used = set()
#     for _ in range(max_terms):
#         bestC, best_gain = None, 0
#         for C in cand_preds:
#             if id(C) in used:
#                 continue
#             Cm = _mask(df, C)
#             gain = int((U & Cm).sum())
#             if gain > best_gain:
#                 best_gain, bestC = gain, C
#         if bestC is None or best_gain == 0:
#             break
#         cover.append(bestC)
#         used.add(id(bestC))
#         U = U & ~_mask(df, bestC)
#         if not U.any():
#             break
#     return cover if not U.any() else []

# def ratio_bounds_for_H(df: pd.DataFrame, H: Predicate, num_cols: list[str], *, robust_q=1.0):
#     Hm = _mask(df, H)
#     if not Hm.any():
#         return []
#     rows = df.loc[Hm]
#     out = []
#     for L, R in product(num_cols, num_cols):
#         if L == R:
#             continue
#         Rv = pd.to_numeric(rows[R], errors="coerce").to_numpy()
#         Lv = pd.to_numeric(rows[L], errors="coerce").to_numpy()
#         ok = np.isfinite(Rv) & np.isfinite(Lv) & (Rv > 0)
#         if not np.any(ok):
#             continue
#         ratios = Lv[ok] / Rv[ok]
#         c = float(np.quantile(ratios, robust_q))
#         C = Where(lambda d, L=L, R=R, c=c: (
#                   pd.to_numeric(d[L], errors="coerce") <= c * pd.to_numeric(d[R], errors="coerce")
#                  ).fillna(False),
#                  name=f"({L} ≤ {c:.6g}·{R})")
#         Cm = _mask(df, C)
#         out.append({
#             "H": H, "C": C,
#             "c": c, "L": L, "R": R,
#             "support_H": int(Hm.sum()),
#             "support_C": int(Cm.sum()),
#             "IG": info_gain(Hm, Cm),
#             "lift": lift(Hm, Cm),
#         })
#     out.sort(key=lambda r: (-r["IG"], -r["lift"], -r["support_H"], r["support_C"], r["L"], r["R"]))
#     return out

# def mk_ge(hyp: Predicate | None, left: str, right, name="") -> Conjecture:
#     return Conjecture(Ge(to_expr(left), to_expr(right)), hyp, name=name or f"{left}_ge_{right}")

# def conjectures_for_numeric_minima(df_num: pd.DataFrame, cols: list[str]) -> tuple[list[Conjecture], list[Predicate]]:
#     TRUE = TruePred()
#     conj = []
#     eq_preds = []
#     for col in cols:
#         try:
#             c, eq_mask = weakest_true_lower_and_eq_mask(df_num, col)
#         except Exception:
#             continue
#         cjt = mk_ge(TRUE, col, c, name=f"{col}_ge_min")
#         conj.append(cjt)
#         eq_preds.append(atom_eq_col_const(col, c))
#     return conj, eq_preds

# def main(max_cols: int = 10, ratio_top_per_H: int = 3, overall_ratio_top: int = 10):
#     dforig = df.copy()
#     dfn = drop_boolean_columns(dforig)

#     num_cols = [c for c in dfn.columns if pd.api.types.is_numeric_dtype(dfn[c])]
#     if not num_cols:
#         print("No numeric columns to work with after dropping booleans.")
#         return

#     var = dfn[num_cols].var(numeric_only=True).sort_values(ascending=False)
#     pick = [c for c in var.index.tolist() if dfn[c].notna().any()][:max_cols]

#     print("=== Using numeric columns (top by variance) ===")
#     for c in pick:
#         print(" -", c)

#     conj, eq_preds = conjectures_for_numeric_minima(dfn, pick)

#     print("\n=== Class A: Weakest-true lower-bound conjectures (TRUE ⇒ col ≥ min) ===")
#     for cjt in conj:
#         print(" -", cjt.pretty(arrow="⇒"))

#     print("\n=== Tightness classes induced by minima (H = (col=min)) ===")
#     for p in eq_preds:
#         supp = int(_mask(dfn, p).sum())
#         print(f" - {pretty_pred(p)}   support={supp}")

#     tight_preds = []
#     for cjt in conj:
#         tight_preds.extend(_tightness_preds(dfn, cjt, atol=1e-9))

#     ge_preds = []
#     for col in pick:
#         c, _ = weakest_true_lower_and_eq_mask(dfn, col)
#         ge_preds.append(atom_ge_col_const(col, c))

#     base_candidates = eq_preds + ge_preds
#     candidates = build_candidate_hypotheses_with_disjunctions(
#         dfn, base_candidates, include_or=False, min_support=1
#     )

#     mined = mine_implications_basic(dfn, tight_preds, candidates, min_support=1)
#     mined.sort(key=lambda r: (-r["equiv"], -r["IG"], -r["lift"], -r["support_H"], r["support_C"],
#                               format_pred(r["H"], unicode_ops=True), format_pred(r["C"], unicode_ops=True)))

#     print("\n=== Class B1: Best single-consequent implications from tightness classes (ranked by IG) ===")
#     if not mined:
#         print("(none)")
#     else:
#         top = mined[:20]
#         for r in top:
#             arrow = "≡" if r["equiv"] else "⇒"
#             print(f"{format_pred(r['H'], unicode_ops=True)} {arrow} {format_pred(r['C'], unicode_ops=True)}   "
#                   f"[support(H)={r['support_H']}, support(C)={r['support_C']}, "
#                   f"lift={r['lift']:.2f}, IG={r['IG']:.3f}, p(H|C)={r['pH_given_C']:.2f}]")

#     print("\n=== Class B2: Minimal disjunctions when no strong single consequent fits (up to 3 terms) ===")
#     printed_any = False
#     for H in tight_preds:
#         Hm = _mask(dfn, H)
#         if _support(Hm) == 0:
#             continue
#         singles_for_H = [r for r in mined if repr(normalize_and(r["H"])) == repr(normalize_and(H))]
#         if singles_for_H and singles_for_H[0]["IG"] >= 0.2:
#             continue
#         cover = minimal_or_cover(dfn, Hm, candidates, max_terms=3)
#         if not cover or len(cover) <= 1:
#             continue
#         union = pd.Series(False, index=dfn.index)
#         for C in cover:
#             union |= _mask(dfn, C)
#         IG = info_gain(Hm, union)
#         LIFT = lift(Hm, union)
#         if IG < 0.05:
#             continue
#         printed_any = True
#         rhs = " ∨ ".join(format_pred(C, unicode_ops=True) for C in cover)
#         print(f"{format_pred(H, unicode_ops=True)} ⇒ ({rhs})   "
#               f"[support(H)={_support(Hm)}, support(∨)={_support(union)}, lift={LIFT:.2f}, IG={IG:.3f}]")
#     if not printed_any:
#         print("(no disjunctions needed at this stage)")

#     print("\n=== Class C: Best ratio bounds under learned tightness classes (ranked by IG) ===")
#     all_ratios = []
#     for H in eq_preds:
#         ratios = ratio_bounds_for_H(dfn, H, pick, robust_q=1.0)
#         if ratios:
#             best_H = ratios[:ratio_top_per_H]
#             all_ratios.extend(best_H)
#             print(f"\n-- Under H={pretty_pred(H)} (support={_support(_mask(dfn,H))}) --")
#             for r in best_H:
#                 print(f"  {format_pred(H, unicode_ops=True)} ⇒ ({r['L']} ≤ {r['c']:.6g}·{r['R']})  "
#                       f"[support(H)={r['support_H']}, support(C)={r['support_C']}, "
#                       f"lift={r['lift']:.2f}, IG={r['IG']:.3f}]")
#     if all_ratios:
#         all_ratios.sort(key=lambda r: (-r["IG"], -r["lift"], -r["support_H"], r["support_C"]))
#         print(f"\n-- Overall top {min(overall_ratio_top, len(all_ratios))} ratio conjectures --")
#         for r in all_ratios[:overall_ratio_top]:
#             print(f"  {format_pred(r['H'], unicode_ops=True)} ⇒ ({r['L']} ≤ {r['c']:.6g}·{r['R']})  "
#                   f"[support(H)={r['support_H']}, support(C)={r['support_C']}, "
#                   f"lift={r['lift']:.2f}, IG={r['IG']:.3f}]")
#     else:
#         print("(no ratio bounds met the scoring criteria)")

# if __name__ == "__main__":
#     main(max_cols=10, ratio_top_per_H=3, overall_ratio_top=10)

# run_graph_numeric_discovery_demo.py
# Numeric-only TxGraffiti loop on graph_data:
#  - Drop all boolean columns
#  - Class A: TRUE ⇒ col ≥ min(col)
#  - Induce tightness classes H: (col=min)
#  - Class B: Mine implications H ⇒ C where C drawn from:
#       * (col=min), (col≥min) if NOT global-true, and percentile thresholds (≤/≥)
#       * small AND/OR compositions (pairwise)
#     Filter out trivial Cs (global-true or restating H)
#     Rank by Information Gain & Lift
#  - Class C: Ratio bounds under each H:  Y ≤ c · X (robust c; pretty fraction)
#
# Requires txgraffiti2025.* modules and txgraffiti.example_data.graph_data
# run_graph_discovery_demo.py
# Numeric-only TxGraffiti loop on graph_data:
#  A) TRUE => col >= min(col), induce tightness classes (col = min)
#  B) Mine implications from tightness classes with strong triviality filters
#  C) Propose & rank ratio bounds Y <= a * X under each tightness class
#  D) Promote best consequents as new hypotheses and re-mine once

from __future__ import annotations
import math
from fractions import Fraction
from itertools import combinations
import numpy as np
import pandas as pd

# --- data ---
from txgraffiti.example_data import graph_data as df

# --- core TxGraffiti pieces ---
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq, Le, Ge
from txgraffiti2025.forms.predicates import Predicate, Where, AndPred, LT, GT
from txgraffiti2025.forms.pretty import format_pred
from txgraffiti2025.forms.utils import to_expr


# =========================================================
# Pretty helpers / masks / normalize
# =========================================================

def _mask(df: pd.DataFrame, pred: Predicate | None) -> pd.Series:
    if pred is None:
        return pd.Series(True, index=df.index)
    m = pred.mask(df)
    return m.reindex(df.index, fill_value=False).astype(bool)

def pretty_pred(pred: Predicate | None) -> str:
    return format_pred(pred, unicode_ops=True)

def _flatten_and_pred(p):
    if p is None:
        return []
    if isinstance(p, AndPred):
        return _flatten_and_pred(p.a) + _flatten_and_pred(p.b)
    return [p]

def _is_true_pred(p) -> bool:
    return getattr(p, "name", "") == "TRUE"

def _and_many(parts):
    parts = [q for q in parts if q is not None and not _is_true_pred(q)]
    if not parts:
        return None
    out = parts[0]
    for q in parts[1:]:
        out = AndPred(out, q)
    return out

def normalize_and(pred):
    if pred is None or _is_true_pred(pred):
        return None
    parts = _flatten_and_pred(pred)
    seen, uniq = set(), []
    for q in parts:
        k = repr(q)
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return _and_many(uniq)

def _pred_key(df: pd.DataFrame, pred) -> tuple:
    pn = normalize_and(pred)
    key_repr = repr(pn)
    m = _mask(df, pn)
    supp = int(m.sum())
    nz = tuple(map(int, np.flatnonzero(m.values[:min(len(m), 64)])))
    return (key_repr, supp, nz)


# =========================================================
# TRUE predicate
# =========================================================

class TruePred(Predicate):
    def __init__(self):
        self.name = "TRUE"
        self._cols = set()
    def mask(self, d: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=d.index, dtype=bool)


# =========================================================
# Drop boolean columns (and binary 0/1 columns)
# =========================================================

def drop_boolean_columns(dfin: pd.DataFrame) -> pd.DataFrame:
    bool_cols = [c for c in dfin.columns if pd.api.types.is_bool_dtype(dfin[c])]
    bin_like = []
    for c in dfin.columns:
        if c in bool_cols:
            continue
        try:
            vals = pd.unique(dfin[c].dropna())
            if len(vals) <= 2 and set(vals).issubset({0, 1}):
                bin_like.append(c)
        except Exception:
            pass
    drop_cols = list(set(bool_cols + bin_like))
    return dfin.drop(columns=drop_cols, errors="ignore").copy()


# =========================================================
# Atom predicates with metadata (column usage)
# =========================================================

def atom_eq_col_const(col: str, c: float) -> Predicate:
    if abs(c - int(round(c))) < 1e-12:
        c_txt = str(int(round(c)))
    else:
        c_txt = str(c)
    P = Where(lambda d, col=col, c=c: (pd.to_numeric(d[col], errors="coerce") == c).fillna(False),
              name=f"({col}={c_txt})")
    P._cols = {col}
    P._kind = "eq_const"
    P._col = col
    P._const = c
    return P

def atom_ge_col_const(col: str, c: float) -> Predicate:
    if abs(c - int(round(c))) < 1e-12:
        c_txt = str(int(round(c)))
    else:
        c_txt = str(c)
    P = Where(lambda d, col=col, c=c: (pd.to_numeric(d[col], errors="coerce") >= c).fillna(False),
              name=f"({col}≥{c_txt})")
    P._cols = {col}
    P._kind = "ge_const"
    P._col = col
    P._const = c
    return P

class OrPred(Predicate):
    def __init__(self, a: Predicate, b: Predicate):
        self.a = a
        self.b = b
        self.name = f"({format_pred(a, unicode_ops=True)}) ∨ ({format_pred(b, unicode_ops=True)})"
        # track columns used
        cols_a = getattr(a, "_cols", set())
        cols_b = getattr(b, "_cols", set())
        self._cols = set(cols_a) | set(cols_b)
        self._kind = "or"
    def mask(self, d: pd.DataFrame) -> pd.Series:
        return _mask(d, self.a) | _mask(d, self.b)

def OR(a: Predicate, b: Predicate) -> Predicate:
    return OrPred(a, b)


# =========================================================
# Weakest-true lower bound (TRUE ⇒ col ≥ min), and equality mask
# =========================================================

def weakest_true_lower_and_eq_mask(df: pd.DataFrame, col: str) -> tuple[float, pd.Series]:
    y = pd.to_numeric(df[col], errors="coerce")
    y = y.dropna()
    if y.empty:
        raise ValueError(f"No numeric data in column {col}.")
    c = float(y.min())
    eq_mask = (df[col] == c).fillna(False)
    return c, eq_mask


# =========================================================
# Conjecture tightness → tightness predicates
# =========================================================

def _tightness_preds(df: pd.DataFrame, conj: Conjecture, *, atol=1e-9):
    rel = conj.relation
    cond = conj.condition

    def _eq_mask(d, L, R, tol):
        Ls = pd.to_numeric(L.eval(d), errors="coerce")
        Rs = pd.to_numeric(R.eval(d), errors="coerce")
        return pd.Series(np.isclose(Ls - Rs, 0.0, atol=float(tol)), index=d.index)

    if isinstance(rel, Le):
        P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
                           name=f"(({rel.left}) = ({rel.right}))")
        P_eq_inner._cols = set()
        P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
        P_str = AndPred(cond, LT(rel.left, rel.right)) if cond else LT(rel.left, rel.right)
        return [normalize_and(P_eq), normalize_and(P_str)]

    if isinstance(rel, Ge):
        P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, atol),
                           name=f"(({rel.left}) = ({rel.right}))")
        P_eq_inner._cols = set()
        P_eq  = AndPred(cond, P_eq_inner) if cond else P_eq_inner
        P_str = AndPred(cond, GT(rel.left, rel.right)) if cond else GT(rel.left, rel.right)
        return [normalize_and(P_eq), normalize_and(P_str)]

    if isinstance(rel, Eq):
        P_eq_inner = Where(lambda d, L=rel.left, R=rel.right: _eq_mask(d, L, R, rel.tol),
                           name=f"(({rel.left}) = ({rel.right}))")
        P_eq_inner._cols = set()
        P_eq = AndPred(cond, P_eq_inner) if cond else P_eq_inner
        return [normalize_and(P_eq)]
    return []


# =========================================================
# Candidate hypotheses: single atoms + pairwise AND/OR (cross-column only)
# =========================================================

def build_candidates(df: pd.DataFrame, base, *, include_or=True, min_support=1):
    cands = list(base)
    base = list(base)

    # Pairwise ANDs
    for A, B in combinations(base, 2):
        try:
            # keep AND even if columns overlap; ANDs are rarely vacuous
            cand = AndPred(A, B)
            cand.name = f"({format_pred(A, unicode_ops=True)}) ∧ ({format_pred(B, unicode_ops=True)})"
            cand._cols = set(getattr(A, "_cols", set())) | set(getattr(B, "_cols", set()))
            m = _mask(df, cand)
            if int(m.sum()) >= min_support:
                cands.append(cand)
        except Exception:
            pass

    # Pairwise ORs (cross-column only)
    if include_or:
        seen_keys = set()
        for A, B in combinations(base, 2):
            try:
                # require cross-column diversity to avoid vacuous monotone ORs
                if set(getattr(A, "_cols", set())) & set(getattr(B, "_cols", set())):
                    continue
                mA = _mask(df, A); mB = _mask(df, B)
                # skip subset relations (A⊆B or B⊆A) to avoid trivial OR
                if not (mA & ~mB).any() or not (mB & ~mA).any():
                    continue
                ORp = OR(A, B)
                mOR = _mask(df, ORp)
                if int(mOR.sum()) < min_support:
                    continue
                key = tuple(sorted([getattr(A, "name", repr(A)), getattr(B, "name", repr(B))]))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                cands.append(ORp)
            except Exception:
                pass

    # dedupe by name
    uniq, seen = [], set()
    for p in cands:
        k = getattr(p, "name", repr(p))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


# =========================================================
# Stats: implies, equiv, support, lift, IG
# =========================================================

def compute_stats(df: pd.DataFrame, H: Predicate, C: Predicate):
    Hm = _mask(df, H)
    Cm = _mask(df, C)
    n = len(df)
    a = int((Hm & Cm).sum())
    b = int((Hm & ~Cm).sum())
    c = int((~Hm & Cm).sum())
    d = int((~Hm & ~Cm).sum())
    supH = a + b
    supC = a + c
    if supH == 0 or supC == 0:
        return None
    implies = (b == 0)
    equiv = (b == 0 and c == 0)
    pH = supH / n
    pC = supC / n
    pH_given_C = a / supC if supC else 0.0
    lift = (pH_given_C / pH) if pH > 0 else 0.0
    # info gain: IG(H|C) = H(H) - H(H|C)
    def H2(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -p*math.log2(p) - (1-p)*math.log2(1-p)
    H_H = H2(pH)
    H_H_given_C = (supC/n)*H2(a/supC) + (1 - supC/n)*H2(b/(n - supC) if n - supC > 0 else 0)
    IG = H_H - H_H_given_C
    return {
        "implies": implies,
        "equiv": equiv,
        "support_H": supH,
        "support_C": supC,
        "lift": lift,
        "IG": IG,
        "pH": pH,
        "pC": pC,
        "pH_given_C": pH_given_C,
    }


# =========================================================
# Triviality filters
# =========================================================

def consequent_is_trivial_or_selfcolumn(df: pd.DataFrame, H: Predicate, C: Predicate) -> bool:
    Hm = _mask(df, H); Cm = _mask(df, C)
    # identical masks: trivial
    if Hm.equals(Cm):
        return True
    # any conjunct of H implies C? (Cn covers that part already)
    for part in _flatten_and_pred(H):
        pm = _mask(df, part)
        if not (pm & ~Cm).any():  # part ⊆ C
            return True
    # block same-column monotone consequents: if H and C use a common column and C is ge/eq
    colsH = set(getattr(H, "_cols", set()))
    colsC = set(getattr(C, "_cols", set()))
    if colsH & colsC:
        kindC = getattr(C, "_kind", "")
        if kindC in {"ge_const", "eq_const"}:
            return True
        # If C is OR but one disjunct shares columns with H, likely trivial
        if getattr(C, "_kind", "") == "or":
            if colsH & set(getattr(C.a, "_cols", set())):
                return True
            if colsH & set(getattr(C.b, "_cols", set())):
                return True
    return False


# =========================================================
# Ratio bounds: under H, propose Y <= a * X, tight on H
# =========================================================

def safe_ratio_max(y: np.ndarray, x: np.ndarray) -> float | None:
    # max(y/x) over nonzero, positive x; ignore NaNs
    mask = (~np.isnan(y)) & (~np.isnan(x)) & (x > 0)
    if not mask.any():
        return None
    vals = y[mask] / x[mask]
    if vals.size == 0:
        return None
    m = float(np.max(vals))
    if not np.isfinite(m):
        return None
    return m

def nice_rational_up(v: float, max_den: int = 24) -> Fraction:
    if not np.isfinite(v) or v <= 0:
        return Fraction(0, 1)
    # upward rounding to ensure y<=a*x holds
    f = Fraction(v).limit_denominator(max_den)
    if float(f) < v - 1e-12:
        # bump minimally upward
        f = Fraction(f.numerator + 1, f.denominator)
    return f

def ratio_conjectures(df: pd.DataFrame, H: Predicate, cols: list[str], top_k_per_H: int = 3):
    """Return list of (text, stats, (Y,X,a)) ranked by IG."""
    Hm = _mask(df, H)
    idx = df.index[Hm]
    if len(idx) == 0:
        return []
    out = []
    # restrict to positive-valued numeric cols for X; Y any numeric
    for Y in cols:
        y = pd.to_numeric(df.loc[idx, Y], errors="coerce").to_numpy()
        if np.all(~np.isfinite(y)):
            continue
        for X in cols:
            if X == Y:
                continue
            x = pd.to_numeric(df.loc[idx, X], errors="coerce").to_numpy()
            a_max = safe_ratio_max(y, x)
            if a_max is None or a_max <= 0:
                continue
            a_rat = nice_rational_up(a_max, max_den=24)
            # Build a Where predicate for Y <= a * X
            a_val = float(a_rat)
            def make_pred(Y=Y, X=X, a=a_val, a_rat=a_rat):
                P = Where(lambda d, Y=Y, X=X, a=a: (
                    pd.to_numeric(d[Y], errors="coerce") <= a * pd.to_numeric(d[X], errors="coerce")
                ).fillna(False),
                name=f"({Y} ≤ {a_rat.numerator}/{a_rat.denominator}·{X})" if a_rat.denominator != 1
                     else f"({Y} ≤ {a_rat.numerator}·{X})")
                P._cols = {Y, X}
                P._kind = "ratio"
                P._Y = Y; P._X = X; P._a = a_val
                return P
            C = make_pred()
            stats = compute_stats(df, H, C)
            if not stats:
                continue
            # discard zero-IG or near-global-true
            if stats["IG"] <= 0:
                continue
            out.append((C, stats))
    # rank by IG then lift, then support_H desc
    out.sort(key=lambda t: (-t[1]["IG"], -t[1]["lift"], -t[1]["support_H"], format_pred(H, unicode_ops=True), format_pred(t[0], unicode_ops=True)))
    return out[:top_k_per_H]


# =========================================================
# Implication mining with filters and ranking
# =========================================================

def mine_implications(df, tight_preds, candidate_preds, *, min_support_H=4, min_support_C=8):
    out = []
    N = len(df)
    for H in tight_preds:
        Hn = normalize_and(H)
        if Hn is None:
            continue
        Hm = _mask(df, Hn)
        supH = int(Hm.sum())
        if supH < min_support_H:
            continue
        for C in candidate_preds:
            Cn = normalize_and(C)
            if Cn is None:
                continue
            Cm = _mask(df, Cn)
            supC = int(Cm.sum())
            if supC < min_support_C:
                continue
            # triviality filters
            if consequent_is_trivial_or_selfcolumn(df, Hn, Cn):
                continue
            stats = compute_stats(df, Hn, Cn)
            if not stats:
                continue
            if stats["implies"]:
                out.append({
                    "H": Hn, "C": Cn,
                    **stats
                })
    # dedupe
    seen = set(); uniq = []
    for r in out:
        key = (_pred_key(df, r["H"]), _pred_key(df, r["C"]), r["equiv"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    # keep most-general H per consequent
    uniq2 = keep_most_general_per_consequent(df, uniq)
    # rank: equivalence first, then IG, lift, support_H desc
    uniq2.sort(key=lambda r: (
        not r["equiv"],
        -r["IG"],
        -r["lift"],
        -r["support_H"],
        r["support_C"],
        format_pred(r["H"], unicode_ops=True),
        format_pred(r["C"], unicode_ops=True),
    ))
    return uniq2

def keep_most_general_per_consequent(df: pd.DataFrame, mined: list[dict]) -> list[dict]:
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in mined:
        Ck = _pred_key(df, r["C"])
        buckets[Ck].append(r)
    kept_all = []
    for _, items in buckets.items():
        masks = []
        for r in items:
            Hn = normalize_and(r["H"])
            masks.append((_mask(df, Hn), r))
        n = len(masks)
        dominated = [False]*n
        for i in range(n):
            if dominated[i]:
                continue
            mi, ri = masks[i]
            for j in range(n):
                if i == j or dominated[i]:
                    continue
                mj, rj = masks[j]
                # if H_i ⊆ H_j prefer H_j (more general)
                if not (mi & ~mj).any():
                    si, sj = int(mi.sum()), int(mj.sum())
                    if sj > si:
                        dominated[i] = True
                    elif si == sj:
                        # tie-break lexicographically
                        hi = format_pred(ri["H"], unicode_ops=True)
                        hj = format_pred(rj["H"], unicode_ops=True)
                        if hj < hi:
                            dominated[i] = True
        for k, d in enumerate(dominated):
            if not d:
                kept_all.append(masks[k][1])
    return kept_all


# =========================================================
# Main
# =========================================================

def main(max_cols: int = 10, ratio_top_per_H: int = 3, overall_ratio_top: int = 10):
    # 0) Prepare data: drop all boolean columns
    dforig = df.copy()
    dfn = drop_boolean_columns(dforig)

    # Keep only numeric columns
    num_cols = [c for c in dfn.columns if pd.api.types.is_numeric_dtype(dfn[c])]
    if not num_cols:
        print("No numeric columns to work with after dropping booleans.")
        return

    # Pick the top-variance numeric columns
    var = dfn[num_cols].var(numeric_only=True).sort_values(ascending=False)
    pick = [c for c in var.index.tolist() if dfn[c].notna().any()][:max_cols]

    print("=== Using numeric columns (top by variance) ===")
    for c in pick:
        print(" -", c)

    # === Class A: TRUE ⇒ col ≥ min(col) ===
    TRUE = TruePred()
    conj_A = []
    eq_preds = []
    ge_preds = []
    print("\n=== Class A: Weakest-true lower-bound conjectures (TRUE ⇒ col ≥ min) ===")
    for col in pick:
        c, eq_mask = weakest_true_lower_and_eq_mask(dfn, col)
        cjt = Conjecture(Ge(to_expr(col), to_expr(c)), TRUE, name=f"{col}_ge_min")
        conj_A.append(cjt)
        print(" -", cjt.pretty(arrow="⇒"))
        # induced atoms
        eqP = atom_eq_col_const(col, c)
        geP = atom_ge_col_const(col, c)
        eq_preds.append(eqP)
        ge_preds.append(geP)

    # equality tightness classes = (col == min)
    print("\n=== Tightness classes induced by minima (H = (col=min)) ===")
    for p in eq_preds:
        supp = int(_mask(dfn, p).sum())
        print(f" - {pretty_pred(p)}   support={supp}")

    # Tightness preds from A (we'll keep equality and strict part but mining will filter by support)
    tight_preds = []
    for cjt in conj_A:
        tight_preds.extend(_tightness_preds(dfn, cjt, atol=1e-9))

    # === Build candidates: equality, ≥min, and cross-column OR/AND
    base_candidates = eq_preds + ge_preds
    candidates = build_candidates(dfn, base_candidates, include_or=True, min_support=4)

    # === Class B1: Mine implications (filters applied)
    mined = mine_implications(
        dfn,
        tight_preds=tight_preds,
        candidate_preds=candidates,
        min_support_H=4,  # avoid singletons
        min_support_C=8
    )

    print("\n=== Class B1: Best single-consequent implications from tightness classes (ranked by IG) ===")
    if not mined:
        print("(none)")
    else:
        for r in mined[:20]:
            arrow = "≡" if r["equiv"] else "⇒"
            print(f"{format_pred(r['H'], unicode_ops=True)} {arrow} "
                  f"{format_pred(r['C'], unicode_ops=True)}   "
                  f"[support(H)={r['support_H']}, support(C)={r['support_C']}, "
                  f"lift={r['lift']:.2f}, IG={r['IG']:.3f}, p(H|C)={r['pH_given_C']:.2f}]")

    # === Class C: Ratio bounds under each equality tightness class
    print("\n=== Class C: Best ratio bounds under learned tightness classes (ranked by IG) ===")
    all_ratio_rows = []
    for H in eq_preds:
        Hm = _mask(dfn, H)
        supH = int(Hm.sum())
        print(f"\n-- Under H={pretty_pred(H)} (support={supH}) --")
        best = ratio_conjectures(dfn, H, pick, top_k_per_H=ratio_top_per_H)
        if not best:
            print("  (no informative ratios)")
            continue
        for C, stats in best:
            all_ratio_rows.append((H, C, stats))
            print(f"  {pretty_pred(H)} ⇒ {pretty_pred(C)}  "
                  f"[support(H)={stats['support_H']}, support(C)={stats['support_C']}, "
                  f"lift={stats['lift']:.2f}, IG={stats['IG']:.3f}]")

    # overall top ratio conjectures
    if all_ratio_rows:
        all_ratio_rows.sort(key=lambda t: (-t[2]["IG"], -t[2]["lift"], -t[2]["support_H"],
                                           pretty_pred(t[0]), pretty_pred(t[1])))
        print(f"\n-- Overall top {overall_ratio_top} ratio conjectures --")
        for (H, C, stats) in all_ratio_rows[:overall_ratio_top]:
            print(f"  {pretty_pred(H)} ⇒ {pretty_pred(C)}  "
                  f"[support(H)={stats['support_H']}, support(C)={stats['support_C']}, "
                  f"lift={stats['lift']:.2f}, IG={stats['IG']:.3f}]")

    # === Promotion: add best consequents (cross-column & high IG) as new hypotheses
    promoted = []
    for r in mined:
        C = r["C"]
        if getattr(C, "_kind", "") == "ratio" or (getattr(C, "_kind", "") in {"ge_const", "eq_const"} and len(getattr(C, "_cols", set())) == 1):
            # prefer cross-column (ratio) or strong eq/thresholds with reasonable support
            if r["IG"] > 0.01 and r["support_C"] >= 12:
                promoted.append(C)
    # dedupe by name
    seen = set(); promoted_unique = []
    for p in promoted:
        k = getattr(p, "name", repr(p))
        if k not in seen:
            seen.add(k)
            promoted_unique.append(p)

    if promoted_unique:
        print("\n=== Promoted hypotheses (new classes to condition on) ===")
        for p in promoted_unique[:10]:
            supp = int(_mask(dfn, p).sum())
            print(f" - {pretty_pred(p)}   [support={supp}]")
    else:
        print("\n=== Promoted hypotheses ===\n(none)")

    # === Re-mine once using promoted classes as antecedents to see “what it would learn next”
    if promoted_unique:
        # Build a small candidate pool again (including original atoms)
        candidates2 = build_candidates(dfn, base_candidates + promoted_unique, include_or=True, min_support=8)
        # Use promoted classes directly as tightness antecedents
        tight2 = promoted_unique
        mined2 = mine_implications(dfn, tight2, candidates2, min_support_H=8, min_support_C=12)
        print("\n=== Re-mined implications from promoted hypotheses (ranked by IG) ===")
        if not mined2:
            print("(none)")
        else:
            for r in mined2[:20]:
                arrow = "≡" if r["equiv"] else "⇒"
                print(f"{format_pred(r['H'], unicode_ops=True)} {arrow} "
                      f"{format_pred(r['C'], unicode_ops=True)}   "
                      f"[support(H)={r['support_H']}, support(C)={r['support_C']}, "
                      f"lift={r['lift']:.2f}, IG={r['IG']:.3f}, p(H|C)={r['pH_given_C']:.2f}]")

if __name__ == "__main__":
    # Tune: raising supports trims noise; increase max_cols to widen search space.
    main(max_cols=10, ratio_top_per_H=3, overall_ratio_top=10)
