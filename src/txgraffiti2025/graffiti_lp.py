# # src/txgraffiti2025/graffiti_lp.py
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Optional, Sequence, Tuple, List, Dict, Union, Iterable
# from itertools import combinations

# import numpy as np
# import pandas as pd

# try:
#     # Fast in-process LP solver
#     from scipy.optimize import linprog as _linprog
#     _HAVE_SCIPY = True
# except Exception:
#     _HAVE_SCIPY = False

# import shutil
# import pulp  # fallback if SciPy unavailable

# from txgraffiti2025.graffiti_relations import GraffitiClassRelations
# from txgraffiti2025.forms.utils import Expr, ColumnTerm, Const
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE
# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.processing.post import morgan_filter

# # ───────────────────────── Config ───────────────────────── #

# @dataclass(frozen=True)
# class LPFitConfig:
#     target: Union[str, Expr]
#     features: Sequence[Union[str, Expr]]
#     direction: str = "both"      # "both" | "lower" | "upper"
#     max_denom: int = 50          # for pretty constants
#     tol: float = 1e-9            # zero-out small coeffs; equality tolerance
#     min_support: int = 3         # rows required after masking
#     touch_atol: float = 0.0      # |lhs-rhs| <= atol + rtol*|rhs|
#     touch_rtol: float = 0.0


# @dataclass(frozen=True)
# class GenerateK2Config:
#     target: Union[str, Expr]
#     hypotheses_limit: Optional[int] = 10
#     min_touch: int = 3
#     max_denom: int = 30
#     tol: float = 1e-9
#     touch_atol: float = 0.0
#     touch_rtol: float = 0.0


# # ───────────────────────── Helpers ───────────────────────── #

# def _to_expr(e: Union[str, Expr], exprs: Dict[str, Expr]) -> Expr:
#     return e if isinstance(e, Expr) else (exprs[e] if e in exprs else ColumnTerm(e))

# def _rconst(x: float, max_denom: int, tol: float) -> Const:
#     if not np.isfinite(x) or abs(x) <= tol:
#         return Const(0.0)
#     from fractions import Fraction
#     fr = Fraction(x).limit_denominator(max_denom)
#     return Const(fr)

# def _affine_expr(a: np.ndarray, feats: Sequence[Expr], b: float, max_denom: int, tol: float) -> Expr:
#     acc: Optional[Expr] = None
#     for coef, f in zip(a, feats):
#         if abs(coef) <= tol:
#             continue
#         term = _rconst(float(coef), max_denom, tol) * f
#         acc = term if acc is None else (acc + term)
#     if abs(b) > tol:
#         acc = (_rconst(float(b), max_denom, tol) if acc is None else (acc + _rconst(float(b), max_denom, tol)))
#     return acc if acc is not None else Const(0.0)

# def _finite_mask(arrs: Sequence[np.ndarray]) -> np.ndarray:
#     m = np.ones_like(arrs[0], dtype=bool)
#     for a in arrs: m &= np.isfinite(a)
#     return m

# def _touch(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> Tuple[int, float, int]:
#     m = mask & np.isfinite(lhs) & np.isfinite(rhs)
#     n = int(m.sum())
#     if n == 0:
#         return 0, 0.0, 0
#     d = np.abs(lhs[m] - rhs[m])
#     t = atol + rtol * np.abs(rhs[m])
#     tc = int((d <= t).sum())
#     return tc, (tc / n), n

# def _solve_sum_slack_lp(X: np.ndarray, y: np.ndarray, *, sense: str) -> Tuple[np.ndarray, float]:
#     """
#     Minimize sum of slacks for y >= a·x + b  (sense='lower') or y <= a·x + b (sense='upper').
#     Variables: a (k), b (1), s (n>=0).  Equality constraints encode slack definition.
#     """
#     n, k = X.shape
#     ones = np.ones((n, 1))
#     I = np.eye(n)

#     if _HAVE_SCIPY:
#         # in-process: SciPy HiGHS
#         c = np.r_[np.zeros(k + 1), np.ones(n)]
#         if sense == "upper":
#             A_eq = np.hstack([X, ones, -I]); b_eq = y
#         elif sense == "lower":
#             A_eq = np.hstack([-X, -ones, -I]); b_eq = -y
#         else:
#             raise ValueError("sense must be 'upper' or 'lower'")
#         bounds = [(None, None)] * (k + 1) + [(0, None)] * n
#         res = _linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
#         if not res.success:
#             raise RuntimeError(f"LP failed: {res.message}")
#         x = res.x
#         return x[:k], float(x[k])

#     # fallback: PuLP external solver (CBC/GLPK)
#     solver = None
#     cbc = shutil.which("cbc")
#     if cbc: solver = pulp.COIN_CMD(path=cbc, msg=False)
#     glp = shutil.which("glpsol")
#     if solver is None and glp: solver = pulp.GLPK_CMD(path=glp, msg=False)
#     if solver is None:
#         raise RuntimeError("No LP solver found (need SciPy or CBC/GLPK).")

#     prob = pulp.LpProblem("sum_slack", pulp.LpMinimize)
#     a = [pulp.LpVariable(f"a_{j}") for j in range(k)]
#     b = pulp.LpVariable("b")
#     s = [pulp.LpVariable(f"s_{i}", lowBound=0) for i in range(n)]
#     prob += pulp.lpSum(s)
#     for i in range(n):
#         lhs = pulp.lpSum(a[j] * float(X[i, j]) for j in range(k)) + b
#         if sense == "upper":
#             prob += lhs - float(y[i]) == s[i]
#         else:
#             prob += float(y[i]) - lhs == s[i]
#     status = prob.solve(solver)
#     if pulp.LpStatus[status] != "Optimal":
#         raise RuntimeError(f"LP not optimal: {pulp.LpStatus[status]}")
#     a_val = np.array([float(v.value()) for v in a], dtype=float)
#     b_val = float(b.value())
#     return a_val, b_val


# # ───────────────────────── Core class ───────────────────────── #

# class GraffitiLP:
#     """
#     Minimal, high-throughput LP fitter for LOWER/UPPER/EQ affine conjectures.
#     Shares masks/exprs with GraffitiClassRelations and works directly on arrays.
#     """

#     def __init__(self, gcr: GraffitiClassRelations) -> None:
#         if not isinstance(gcr, GraffitiClassRelations):
#             raise TypeError("gcr must be a GraffitiClassRelations")
#         self.gcr = gcr
#         self.df: pd.DataFrame = gcr.df
#         self.exprs: Dict[str, Expr] = gcr.get_exprs()
#         self.expr_columns: List[str] = gcr.get_expr_columns()
#         self.base_hypothesis: Predicate = gcr.base_hypothesis
#         self.hypotheses: List[Tuple[str, Predicate]] = gcr.sort_conjunctions_by_generality()
#         self.invariants = list(self.exprs.values())

#     # ---------- public: single fit ----------

#     def fit_affine(self, cfg: LPFitConfig, *, condition: Optional[Predicate] = None)\
#             -> Tuple[List[Conjecture], List[Conjecture], List[Conjecture]]:
#         y_expr = _to_expr(cfg.target, self.exprs)
#         X_exprs = [_to_expr(f, self.exprs) for f in cfg.features]
#         cond = condition or self.base_hypothesis or TRUE

#         # arrays once
#         y_full = y_expr.eval(self.df).to_numpy(dtype=float, copy=False)
#         X_full = [x.eval(self.df).to_numpy(dtype=float, copy=False) for x in X_exprs]

#         # mask
#         if cond is TRUE:
#             mask = np.ones(len(self.df), dtype=bool)
#         else:
#             mask = self.gcr._mask_cached(cond)
#         fin = _finite_mask([y_full, *X_full]) & mask
#         if int(fin.sum()) < cfg.min_support:
#             return [], [], []

#         y = y_full[fin]
#         X = np.column_stack([c[fin] for c in X_full])

#         lowers: List[Conjecture] = []
#         uppers: List[Conjecture] = []
#         equals: List[Conjecture] = []

#         want_lo = cfg.direction in ("both", "lower")
#         want_up = cfg.direction in ("both", "upper")

#         a_lo = b_lo = a_up = b_up = None

#         # lower: y ≥ a·x + b
#         if want_lo:
#             a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
#             rhs_lo = _affine_expr(a_lo, X_exprs, b_lo, cfg.max_denom, cfg.tol)
#             cj = Conjecture(relation=Ge(y_expr, rhs_lo), condition=(None if cond is TRUE else cond))
#             setattr(cj, "coefficient_pairs", [(repr(f), float(c)) for f, c in zip(X_exprs, a_lo)])  # or a_up
#             setattr(cj, "intercept", float(b_lo))  # or b_up
#             # touches on cond
#             rhs_vals = rhs_lo.eval(self.df).to_numpy(dtype=float, copy=False)
#             tc, tr, n = _touch(y_full, rhs_vals, mask, atol=cfg.touch_atol, rtol=cfg.touch_rtol)
#             setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#             lowers.append(cj)

#         # upper: y ≤ a·x + b
#         if want_up:
#             a_up, b_up = _solve_sum_slack_lp(X, y, sense="upper")
#             rhs_up = _affine_expr(a_up, X_exprs, b_up, cfg.max_denom, cfg.tol)
#             cj = Conjecture(relation=Le(y_expr, rhs_up), condition=(None if cond is TRUE else cond))
#             setattr(cj, "coefficient_pairs", [(repr(f), float(c)) for f, c in zip(X_exprs, a_lo)])  # or a_up
#             setattr(cj, "intercept", float(b_lo))  # or b_up
#             rhs_vals = rhs_up.eval(self.df).to_numpy(dtype=float, copy=False)
#             tc, tr, n = _touch(y_full, rhs_vals, mask, atol=cfg.touch_atol, rtol=cfg.touch_rtol)
#             setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#             uppers.append(cj)

#         # equality (degenerate coincident bounds)
#         if want_lo and want_up and (a_lo is not None) and (a_up is not None):
#             if np.allclose(a_lo, a_up, atol=cfg.tol, rtol=0.0) and (abs((b_lo or 0.0) - (b_up or 0.0)) <= cfg.tol):
#                 rhs_eq = _affine_expr(a_lo, X_exprs, float(b_lo or 0.0), cfg.max_denom, cfg.tol)
#                 cj = Conjecture(relation=Eq(y_expr, rhs_eq, tol=cfg.tol), condition=(None if cond is TRUE else cond))
#                 setattr(cj, "coefficient_pairs", [(repr(f), float(c)) for f, c in zip(X_exprs, a_lo)])  # or a_up
#                 setattr(cj, "intercept", float(b_lo))  # or b_up
#                 rhs_vals = rhs_eq.eval(self.df).to_numpy(dtype=float, copy=False)
#                 tc, tr, n = _touch(y_full, rhs_vals, mask, atol=cfg.touch_atol, rtol=cfg.touch_rtol)
#                 setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                 equals.append(cj)

#         # sort by touches
#         lowers.sort(key=lambda c: (getattr(c, "touch_count", 0), getattr(c, "touch_rate", 0.0)), reverse=True)
#         uppers.sort(key=lambda c: (getattr(c, "touch_count", 0), getattr(c, "touch_rate", 0.0)), reverse=True)
#         equals.sort(key=lambda c: (getattr(c, "touch_count", 0), getattr(c, "touch_rate", 0.0)), reverse=True)
#         return lowers, uppers, equals

#     # ---------- public: fast k=2 generator ----------

#     def generate_k2_bounds(self, cfg: GenerateK2Config) -> Dict[str, List[Conjecture]]:
#         # resolve target and candidate features
#         target_expr = _to_expr(cfg.target, self.exprs)
#         target_name = repr(target_expr)
#         feats = [self.exprs[c] for c in self.expr_columns if repr(self.exprs[c]) != target_name]
#         if len(feats) < 2:
#             return {"lowers": [], "uppers": [], "equals": []}

#         # hypotheses to iterate
#         H_iter = list(self.hypotheses)
#         if cfg.hypotheses_limit is not None:
#             H_iter = H_iter[:cfg.hypotheses_limit]

#         df = self.df
#         y_full = target_expr.eval(df).to_numpy(dtype=float, copy=False)

#         # cache feature arrays once
#         arr_cache: Dict[str, np.ndarray] = {repr(f): f.eval(df).to_numpy(dtype=float, copy=False) for f in feats}

#         lowers: List[Conjecture] = []
#         uppers: List[Conjecture] = []
#         equals: List[Conjecture] = []

#         for f1, f2 in combinations(feats, 2):
#             a1 = arr_cache[repr(f1)]; a2 = arr_cache[repr(f2)]
#             finite_all = _finite_mask([y_full, a1, a2])
#             if int(finite_all.sum()) < 3:
#                 continue

#             for H_name, H in H_iter:
#                 mask = (self.gcr._mask_cached(H) if H is not TRUE else np.ones(len(df), dtype=bool)) & finite_all
#                 if int(mask.sum()) < 3:
#                     continue

#                 y = y_full[mask]
#                 X = np.column_stack([a1[mask], a2[mask]])

#                 # lower
#                 a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
#                 rhs_lo_vals = a_lo[0]*a1 + a_lo[1]*a2 + b_lo  # evaluate on full arrays once
#                 tc, tr, n = _touch(y_full, rhs_lo_vals, mask, atol=cfg.touch_atol, rtol=cfg.touch_rtol)
#                 if tc == n and n > 0:
#                     rhs_lo_expr = _affine_expr(a_lo, [f1, f2], b_lo, cfg.max_denom, cfg.tol)
#                     cj = Conjecture(relation=Eq(target_expr, rhs_lo_expr, tol=cfg.tol), condition=H)
#                     setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                     setattr(cj, "coefficient_pairs", [(repr(f1), float(a_lo[0])), (repr(f2), float(a_lo[1]))])
#                     setattr(cj, "intercept", float(b_lo))
#                     equals.append(cj)
#                 elif tc > cfg.min_touch:
#                     rhs_lo_expr = _affine_expr(a_lo, [f1, f2], b_lo, cfg.max_denom, cfg.tol)
#                     cj = Conjecture(relation=Ge(target_expr, rhs_lo_expr), condition=H)
#                     setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                     setattr(cj, "coefficient_pairs", [(repr(f1), float(a_lo[0])), (repr(f2), float(a_lo[1]))])
#                     setattr(cj, "intercept", float(b_lo))
#                     lowers.append(cj)

#                 # upper
#                 a_up, b_up = _solve_sum_slack_lp(X, y, sense="upper")
#                 rhs_up_vals = a_up[0]*a1 + a_up[1]*a2 + b_up
#                 tc, tr, n = _touch(y_full, rhs_up_vals, mask, atol=cfg.touch_atol, rtol=cfg.touch_rtol)
#                 if tc == n and n > 0:
#                     rhs_up_expr = _affine_expr(a_up, [f1, f2], b_up, cfg.max_denom, cfg.tol)
#                     cj = Conjecture(relation=Eq(target_expr, rhs_up_expr, tol=cfg.tol), condition=H)
#                     setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                     setattr(cj, "coefficient_pairs", [(repr(f1), float(a_lo[0])), (repr(f2), float(a_lo[1]))])
#                     setattr(cj, "intercept", float(b_lo))
#                     equals.append(cj)
#                 elif tc > cfg.min_touch:
#                     rhs_up_expr = _affine_expr(a_up, [f1, f2], b_up, cfg.max_denom, cfg.tol)
#                     cj = Conjecture(relation=Le(target_expr, rhs_up_expr), condition=H)
#                     setattr(cj, "coefficient_pairs", [(repr(f1), float(a_lo[0])), (repr(f2), float(a_lo[1]))])
#                     setattr(cj, "intercept", float(b_lo))
#                     setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                     uppers.append(cj)

#         # small stable de-dup by string
#         def _dedup(xs: List[Conjecture]) -> List[Conjecture]:
#             seen, out = set(), []
#             for c in xs:
#                 s = str(c)
#                 if s in seen:
#                     continue
#                 seen.add(s); out.append(c)
#             return out

#         lowers, uppers, equals = morgan_filter(df, _dedup(lowers)).kept, morgan_filter(df, _dedup(uppers)).kept, morgan_filter(df, _dedup(equals)).kept
#         key = lambda c: (getattr(c, "touch_count", 0), getattr(c, "touch_rate", 0.0))
#         lowers.sort(key=key, reverse=True)
#         uppers.sort(key=key, reverse=True)
#         equals.sort(key=key, reverse=True)
#         return {"lowers": lowers, "uppers": uppers, "equals": equals}


#     def generate_k_affine_bounds(
#         self,
#         *,
#         target: Union[str, Expr],
#         k_values: Iterable[int] = (1, 2, 3),
#         hypotheses_limit: Optional[int] = 5,
#         min_touch: int = 3,
#         max_denom: int = 30,
#         tol: float = 1e-9,
#         touch_atol: float = 0.0,
#         touch_rtol: float = 0.0,
#         top_m_by_variance: Optional[int] = 12,
#     ) -> dict[str, list[Conjecture]]:
#         """
#         Fast k-feature affine LOWER/UPPER/EQ search.
#         Fixes:
#         - Use module-level _solve_sum_slack_lp / _affine_expr.
#         - Compare equality touches to valid-row count (n), not H_mask.sum().
#         """
#         # --- resolve target & features ---
#         if isinstance(target, str):
#             y_expr = self.exprs.get(target, ColumnTerm(target))
#             target_name = target
#         else:
#             y_expr = target
#             target_name = repr(target)

#         feats_all: list[Expr] = [e for e in self.invariants if repr(e) != target_name]
#         if not feats_all:
#             return {"lowers": [], "uppers": [], "equals": []}

#         # hypotheses (already sorted by generality from GCR)
#         H_iter = list(self.hypotheses)
#         if hypotheses_limit is not None:
#             H_iter = H_iter[:hypotheses_limit]

#         df = self.df

#         # pre-evaluate arrays once
#         y_full = y_expr.eval(df).to_numpy(dtype=float, copy=False)
#         feat_arrays: Dict[str, np.ndarray] = {repr(fx): fx.eval(df).to_numpy(dtype=float, copy=False)
#                                             for fx in feats_all}

#         lowers: list[Conjecture] = []
#         uppers: list[Conjecture] = []
#         equals: list[Conjecture] = []

#         # touch stats with finite filtering; returns (tc, tr, n)
#         def _touch_stats(lhs_arr: np.ndarray, rhs_arr: np.ndarray, mask: np.ndarray) -> Tuple[int, float, int]:
#             m = mask & np.isfinite(lhs_arr) & np.isfinite(rhs_arr)
#             n = int(m.sum())
#             if n == 0:
#                 return 0, 0.0, 0
#             dif = np.abs(lhs_arr[m] - rhs_arr[m])
#             tol_arr = touch_atol + touch_rtol * np.abs(rhs_arr[m])
#             tc = int((dif <= tol_arr).sum())
#             return tc, (tc / n), n

#         for H_name, H in H_iter:
#             H_mask = self.get_condition_mask(H)

#             # choose features by variance under H (optional)
#             if top_m_by_variance is not None:
#                 scores = []
#                 base = np.isfinite(y_full) & H_mask
#                 if base.sum() < 3:
#                     continue
#                 for fx in feats_all:
#                     a = feat_arrays[repr(fx)]
#                     m = base & np.isfinite(a)
#                     if m.sum() >= 3:
#                         scores.append((float(np.var(a[m])), fx))
#                 scores.sort(key=lambda t: t[0], reverse=True)
#                 feats = [fx for _, fx in scores[:top_m_by_variance]]
#             else:
#                 feats = feats_all

#             if not feats or all(k > len(feats) for k in k_values):
#                 continue

#             # iterate k and combos
#             for k in k_values:
#                 if k < 1 or k > len(feats):
#                     continue
#                 for combo in combinations(feats, k):
#                     # per-combo finite mask
#                     m = H_mask & np.isfinite(y_full)
#                     for fx in combo:
#                         m &= np.isfinite(feat_arrays[repr(fx)])
#                     if int(m.sum()) < 3:
#                         continue

#                     # assemble design
#                     X = np.column_stack([feat_arrays[repr(fx)][m] for fx in combo])
#                     y = y_full[m]
#                     feat_exprs = list(combo)

#                     # LOWER: y >= a·x + b
#                     try:
#                         a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
#                     except Exception:
#                         a_lo = b_lo = None
#                     if a_lo is not None:
#                         rhs_lo_expr = _affine_expr(a_lo, feat_exprs, float(b_lo), max_denom, tol)
#                         rhs_lo_vals = rhs_lo_expr.eval(df).to_numpy(dtype=float, copy=False)
#                         tc, tr, n = _touch_stats(y_full, rhs_lo_vals, H_mask)
#                         if n > 0 and tc == n:
#                             cj = Conjecture(relation=Eq(y_expr, rhs_lo_expr, tol=tol), condition=H)
#                             setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                             equals.append(cj)
#                         elif tc >= min_touch:
#                             cj = Conjecture(relation=Ge(y_expr, rhs_lo_expr), condition=H)
#                             setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                             lowers.append(cj)

#                     # UPPER: y <= a·x + b
#                     try:
#                         a_up, b_up = _solve_sum_slack_lp(X, y, sense="upper")
#                     except Exception:
#                         a_up = b_up = None
#                     if a_up is not None:
#                         rhs_up_expr = _affine_expr(a_up, feat_exprs, float(b_up), max_denom, tol)
#                         rhs_up_vals = rhs_up_expr.eval(df).to_numpy(dtype=float, copy=False)
#                         tc, tr, n = _touch_stats(y_full, rhs_up_vals, H_mask)
#                         if n > 0 and tc == n:
#                             cj = Conjecture(relation=Eq(y_expr, rhs_up_expr, tol=tol), condition=H)
#                             setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                             equals.append(cj)
#                         elif tc >= min_touch:
#                             cj = Conjecture(relation=Le(y_expr, rhs_up_expr), condition=H)
#                             setattr(cj, "touch_count", tc); setattr(cj, "touch_rate", tr); setattr(cj, "support_n", n)
#                             uppers.append(cj)

#         # dedup + sort
#         def _dedup(lst: List[Conjecture]) -> List[Conjecture]:
#             seen, out = set(), []
#             for c in lst:
#                 s = str(c)
#                 if s in seen: continue
#                 seen.add(s); out.append(c)
#             out.sort(key=lambda c: (getattr(c, "touch_count", 0), getattr(c, "support_n", 0)), reverse=True)
#             return out

#         lowers, uppers, equals = morgan_filter(df, _dedup(lowers)).kept, morgan_filter(df, _dedup(uppers)).kept, morgan_filter(df, _dedup(equals)).kept
#         return {
#             "lowers": lowers,
#             "uppers": uppers,
#             "equals": equals,
#         }


#     def get_condition_mask(self, cond: Optional[Predicate]) -> np.ndarray:
#         """
#         Return a boolean mask for `cond` on self.df.
#         - None or TRUE ⇒ all rows
#         - otherwise use GraffitiClassRelations' cached mask
#         """
#         if cond is None or cond is TRUE:
#             return np.ones(len(self.df), dtype=bool)
#         return self.gcr._mask_cached(cond)

#     def _mask_via_gcr(self, pred: Predicate) -> np.ndarray:
#         return self.gcr._mask_cached(pred)


from __future__ import annotations

"""
LP-based affine and k-affine conjecture generation for TxGraffiti 2025.

This module exposes:

  * LPFitConfig / GenerateK2Config – configuration dataclasses,
  * GraffitiLP – main entry point for affine LOWER/UPPER/EQ bounds,
  * helpers to solve "sum-of-slacks" LPs with optional coefficient/intercept bounds,
  * consistent metadata on conjectures (float + Fraction coefficients).

The core pattern is to fit bounds of the form

    y  ≥  a·x + b     (lower bounds)
    y  ≤  a·x + b     (upper bounds)

under a structural hypothesis H, where the variables in a and b can
optionally be constrained to lie in a user-specified box.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, Dict, Union, Iterable
from itertools import combinations
from fractions import Fraction

import numpy as np
import pandas as pd

try:
    # Fast in-process LP solver
    from scipy.optimize import linprog as _linprog
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - SciPy-less environments
    _HAVE_SCIPY = False

import shutil
import pulp  # fallback if SciPy unavailable

from txgraffiti2025.graffiti_relations import GraffitiClassRelations
from txgraffiti2025.forms.utils import Expr, ColumnTerm, Const
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.processing.post import morgan_filter


# ───────────────────────── Config ───────────────────────── #

@dataclass(frozen=True)
class LPFitConfig:
    """
    Configuration for a single affine LP fit.

    Parameters
    ----------
    target :
        Target invariant (column name or Expr).
    features :
        Candidate feature expressions for the RHS.
    direction :
        "both", "lower", or "upper".
    max_denom :
        Maximum denominator used when rationalizing coefficients and intercept.
    tol :
        Numerical tolerance for zeroing out small coefficients and for EQ tests.
    min_support :
        Minimum number of rows in the masked domain required to attempt a fit.
    touch_atol, touch_rtol :
        Tolerances used for counting "touches" (equality events) on the domain.
    coef_bounds :
        Optional (lo, hi) box constraint for each slope coefficient a_j.
        If None, coefficients are unbounded.
    intercept_bounds :
        Optional (lo, hi) box constraint for the intercept b.
        If None, intercept is unbounded.
    """

    target: Union[str, Expr]
    features: Sequence[Union[str, Expr]]
    direction: str = "both"      # "both" | "lower" | "upper"
    max_denom: int = 50          # for pretty constants
    tol: float = 1e-9            # zero-out small coeffs; equality tolerance
    min_support: int = 3         # rows required after masking
    touch_atol: float = 0.0      # |lhs-rhs| <= atol + rtol*|rhs|
    touch_rtol: float = 0.0
    coef_bounds: Optional[Tuple[float, float]] = (-2.5, 2.5)
    intercept_bounds: Optional[Tuple[float, float]] = (-2.75, 2.75)


@dataclass(frozen=True)
class GenerateK2Config:
    """
    Configuration for fast k=2 (two-feature) affine bounds.

    The bounds on coefficients and intercept are optional but, when provided,
    they are passed directly to the underlying LP.
    """

    target: Union[str, Expr]
    hypotheses_limit: Optional[int] = 10
    min_touch: int = 3
    max_denom: int = 30
    tol: float = 1e-9
    touch_atol: float = 0.0
    touch_rtol: float = 0.0
    coef_bounds: Optional[Tuple[float, float]] = None
    intercept_bounds: Optional[Tuple[float, float]] = None


# ───────────────────────── Helpers ───────────────────────── #

def _to_expr(e: Union[str, Expr], exprs: Dict[str, Expr]) -> Expr:
    """
    Resolve a string name or Expr into an Expr using the GraffitiClassRelations
    expression table. If `e` is unknown as a named Expr, treat it as a bare
    ColumnTerm.
    """
    return e if isinstance(e, Expr) else (exprs[e] if e in exprs else ColumnTerm(e))


def _rconst(x: float, max_denom: int, tol: float) -> Const:
    """
    Convert a float into a Const carrying a rational Fraction, with denominator
    bounded by `max_denom`. Values whose magnitude is <= tol are treated as 0.
    """
    if not np.isfinite(x) or abs(x) <= tol:
        return Const(0.0)
    fr = Fraction(x).limit_denominator(max_denom)
    return Const(fr)


def _fractionize(x: float, max_denom: int) -> Fraction:
    """
    Convenience helper: turn a float into a Fraction with controlled denominator.
    Non-finite values are mapped to 0.
    """
    if not np.isfinite(x):
        return Fraction(0)
    return Fraction(x).limit_denominator(max_denom)


def _affine_expr(
    a: np.ndarray,
    feats: Sequence[Expr],
    b: float,
    max_denom: int,
    tol: float,
) -> Expr:
    """
    Build an Expr representing Σ_j a_j · feat_j + b, where each coefficient
    and the intercept are rationalized via _rconst.
    """
    acc: Optional[Expr] = None

    for coef, f in zip(a, feats):
        if abs(coef) <= tol:
            continue
        term = _rconst(float(coef), max_denom, tol) * f  # type: ignore[operator]
        acc = term if acc is None else (acc + term)      # type: ignore[operator]

    if abs(b) > tol:
        const_term = _rconst(float(b), max_denom, tol)
        acc = const_term if acc is None else (acc + const_term)  # type: ignore[operator]

    return acc if acc is not None else Const(0.0)


def _finite_mask(arrs: Sequence[np.ndarray]) -> np.ndarray:
    """
    Return a mask that is True exactly where all arrays in `arrs` are finite.
    """
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _touch(
    lhs: np.ndarray,
    rhs: np.ndarray,
    mask: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> Tuple[int, float, int]:
    """
    Count equality events ("touches") between lhs and rhs inside `mask`.

    Returns
    -------
    touches : int
        Number of rows with |lhs - rhs| <= atol + rtol|rhs|.
    touch_rate : float
        Fraction of valid rows that touch.
    n_valid : int
        Number of valid rows considered.
    """
    m = mask & np.isfinite(lhs) & np.isfinite(rhs)
    n = int(m.sum())
    if n == 0:
        return 0, 0.0, 0
    d = np.abs(lhs[m] - rhs[m])
    t = atol + rtol * np.abs(rhs[m])
    tc = int((d <= t).sum())
    return tc, (tc / n), n


def _solve_sum_slack_lp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sense: str,
    coef_bounds: Optional[Tuple[float, float]] = None,
    intercept_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Solve the "sum of slacks" LP

        minimize   Σ_i s_i
        subject to
            y_i  ≥  a·x_i + b + s_i   (sense='lower')
            y_i  ≤  a·x_i + b + s_i   (sense='upper')
            s_i ≥ 0
            a_j, b within optional box constraints

    where X is n×k, y is length-n, a ∈ R^k, and b ∈ R.

    The constraints are encoded as equalities with signed y/X depending on
    the sense. Bounds on coefficients and intercept are passed through to
    the underlying solver (SciPy HiGHS when available, otherwise PuLP).
    """
    n, k = X.shape
    ones = np.ones((n, 1))
    I = np.eye(n)

    coef_lo, coef_hi = (None, None) if coef_bounds is None else coef_bounds
    b_lo, b_hi = (None, None) if intercept_bounds is None else intercept_bounds

    if _HAVE_SCIPY:
        # Variables: [a(0..k-1), b, s(0..n-1)]
        c = np.r_[np.zeros(k + 1), np.ones(n)]

        if sense == "upper":
            # y_i = a·x_i + b + s_i  ⇒  a·x_i + b - s_i = y_i
            A_eq = np.hstack([X, ones, -I])
            b_eq = y
        elif sense == "lower":
            # y_i = a·x_i + b - s_i  with constraint y_i ≥ a·x_i + b
            # Encode as: -a·x_i - b - s_i = -y_i
            A_eq = np.hstack([-X, -ones, -I])
            b_eq = -y
        else:
            raise ValueError("sense must be 'upper' or 'lower'")

        bounds: List[Tuple[Optional[float], Optional[float]]] = []
        # coefficient bounds
        for _ in range(k):
            bounds.append((coef_lo, coef_hi))
        # intercept bounds
        bounds.append((b_lo, b_hi))
        # slack bounds
        bounds.extend([(0.0, None)] * n)

        res = _linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")
        x = res.x
        return x[:k], float(x[k])

    # Fallback: PuLP external solver (CBC / GLPK)
    solver = None
    cbc = shutil.which("cbc")
    if cbc:
        solver = pulp.COIN_CMD(path=cbc, msg=False)
    glp = shutil.which("glpsol")
    if solver is None and glp:
        solver = pulp.GLPK_CMD(path=glp, msg=False)
    if solver is None:
        raise RuntimeError("No LP solver found (need SciPy or CBC/GLPK).")

    prob = pulp.LpProblem("sum_slack", pulp.LpMinimize)

    # Variables: coefficients a_j, intercept b, slacks s_i
    a_vars = [
        pulp.LpVariable(f"a_{j}", lowBound=coef_lo, upBound=coef_hi)
        for j in range(k)
    ]
    b_var = pulp.LpVariable("b", lowBound=b_lo, upBound=b_hi)
    s_vars = [pulp.LpVariable(f"s_{i}", lowBound=0.0) for i in range(n)]

    # Objective: minimize sum of slacks
    prob += pulp.lpSum(s_vars)

    # Constraints
    for i in range(n):
        lhs = pulp.lpSum(a_vars[j] * float(X[i, j]) for j in range(k)) + b_var
        if sense == "upper":
            # y_i = lhs + s_i  ⇒ lhs - y_i = s_i
            prob += lhs - float(y[i]) == s_vars[i]
        else:
            # y_i = lhs - s_i  with y_i ≥ lhs  ⇒ y_i - lhs = s_i
            prob += float(y[i]) - lhs == s_vars[i]

    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP not optimal: {pulp.LpStatus[status]}")

    a_val = np.array([float(v.value()) for v in a_vars], dtype=float)
    b_val = float(b_var.value())
    return a_val, b_val


# ───────────────────────── Core class ───────────────────────── #

class GraffitiLP:
    """
    Minimal, high-throughput LP fitter for affine LOWER/UPPER/EQ conjectures.

    This class is intentionally light: it shares expressions and hypothesis
    masks with GraffitiClassRelations and only manipulates NumPy arrays
    internally, returning Conjecture objects with rich metadata.
    """

    def __init__(self, gcr: GraffitiClassRelations) -> None:
        if not isinstance(gcr, GraffitiClassRelations):
            raise TypeError("gcr must be a GraffitiClassRelations")
        self.gcr = gcr
        self.df: pd.DataFrame = gcr.df
        self.exprs: Dict[str, Expr] = gcr.get_exprs()
        self.expr_columns: List[str] = gcr.get_expr_columns()
        self.base_hypothesis: Predicate = gcr.base_hypothesis
        self.hypotheses: List[Tuple[str, Predicate]] = gcr.sort_conjunctions_by_generality()
        self.invariants: List[Expr] = list(self.exprs.values())

    # ---------- public: single fit ----------

    def fit_affine(
        self,
        cfg: LPFitConfig,
        *,
        condition: Optional[Predicate] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture], List[Conjecture]]:
        """
        Fit affine LOWER/UPPER/EQ bounds for the given target and features
        under an optional structural condition.
        """
        y_expr = _to_expr(cfg.target, self.exprs)
        X_exprs = [_to_expr(f, self.exprs) for f in cfg.features]

        cond = condition or self.base_hypothesis or TRUE

        # Evaluate arrays once
        y_full = y_expr.eval(self.df).to_numpy(dtype=float, copy=False)
        X_full = [x.eval(self.df).to_numpy(dtype=float, copy=False) for x in X_exprs]

        # Mask and finite filter
        if cond is TRUE:
            mask = np.ones(len(self.df), dtype=bool)
        else:
            mask = self.gcr._mask_cached(cond)

        fin = _finite_mask([y_full, *X_full]) & mask
        if int(fin.sum()) < cfg.min_support:
            return [], [], []

        y = y_full[fin]
        X = np.column_stack([c[fin] for c in X_full])

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []
        equals: List[Conjecture] = []

        want_lo = cfg.direction in ("both", "lower")
        want_up = cfg.direction in ("both", "upper")

        a_lo = b_lo = a_up = b_up = None

        # LOWER: y ≥ a·x + b
        if want_lo:
            a_lo, b_lo = _solve_sum_slack_lp(
                X,
                y,
                sense="lower",
                coef_bounds=cfg.coef_bounds,
                intercept_bounds=cfg.intercept_bounds,
            )
            rhs_lo = _affine_expr(a_lo, X_exprs, float(b_lo), cfg.max_denom, cfg.tol)
            cj = Conjecture(
                relation=Ge(y_expr, rhs_lo),
                condition=(None if cond is TRUE else cond),
            )

            # metadata: floats + Fractions
            coef_pairs = [(repr(f), float(c)) for f, c in zip(X_exprs, a_lo)]
            coef_pairs_frac = [
                (repr(f), _fractionize(float(c), cfg.max_denom))
                for f, c in zip(X_exprs, a_lo)
            ]
            setattr(cj, "coefficient_pairs", coef_pairs)
            setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
            setattr(cj, "intercept", float(b_lo))
            setattr(cj, "intercept_frac", _fractionize(float(b_lo), cfg.max_denom))

            rhs_vals = rhs_lo.eval(self.df).to_numpy(dtype=float, copy=False)
            tc, tr, n = _touch(
                y_full,
                rhs_vals,
                mask,
                atol=cfg.touch_atol,
                rtol=cfg.touch_rtol,
            )
            setattr(cj, "touch_count", tc)
            setattr(cj, "touch_rate", tr)
            setattr(cj, "support_n", n)
            lowers.append(cj)

        # UPPER: y ≤ a·x + b
        if want_up:
            a_up, b_up = _solve_sum_slack_lp(
                X,
                y,
                sense="upper",
                coef_bounds=cfg.coef_bounds,
                intercept_bounds=cfg.intercept_bounds,
            )
            rhs_up = _affine_expr(a_up, X_exprs, float(b_up), cfg.max_denom, cfg.tol)
            cj = Conjecture(
                relation=Le(y_expr, rhs_up),
                condition=(None if cond is TRUE else cond),
            )

            coef_pairs = [(repr(f), float(c)) for f, c in zip(X_exprs, a_up)]
            coef_pairs_frac = [
                (repr(f), _fractionize(float(c), cfg.max_denom))
                for f, c in zip(X_exprs, a_up)
            ]
            setattr(cj, "coefficient_pairs", coef_pairs)
            setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
            setattr(cj, "intercept", float(b_up))
            setattr(cj, "intercept_frac", _fractionize(float(b_up), cfg.max_denom))

            rhs_vals = rhs_up.eval(self.df).to_numpy(dtype=float, copy=False)
            tc, tr, n = _touch(
                y_full,
                rhs_vals,
                mask,
                atol=cfg.touch_atol,
                rtol=cfg.touch_rtol,
            )
            setattr(cj, "touch_count", tc)
            setattr(cj, "touch_rate", tr)
            setattr(cj, "support_n", n)
            uppers.append(cj)

        # Equality: coincident lower and upper bounds
        if (
            want_lo
            and want_up
            and (a_lo is not None)
            and (a_up is not None)
            and (b_lo is not None)
            and (b_up is not None)
        ):
            if np.allclose(a_lo, a_up, atol=cfg.tol, rtol=0.0) and abs(b_lo - b_up) <= cfg.tol:
                rhs_eq = _affine_expr(a_lo, X_exprs, float(b_lo), cfg.max_denom, cfg.tol)
                cj = Conjecture(
                    relation=Eq(y_expr, rhs_eq, tol=cfg.tol),
                    condition=(None if cond is TRUE else cond),
                )

                coef_pairs = [(repr(f), float(c)) for f, c in zip(X_exprs, a_lo)]
                coef_pairs_frac = [
                    (repr(f), _fractionize(float(c), cfg.max_denom))
                    for f, c in zip(X_exprs, a_lo)
                ]
                setattr(cj, "coefficient_pairs", coef_pairs)
                setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                setattr(cj, "intercept", float(b_lo))
                setattr(cj, "intercept_frac", _fractionize(float(b_lo), cfg.max_denom))

                rhs_vals = rhs_eq.eval(self.df).to_numpy(dtype=float, copy=False)
                tc, tr, n = _touch(
                    y_full,
                    rhs_vals,
                    mask,
                    atol=cfg.touch_atol,
                    rtol=cfg.touch_rtol,
                )
                setattr(cj, "touch_count", tc)
                setattr(cj, "touch_rate", tr)
                setattr(cj, "support_n", n)
                equals.append(cj)

        # Sort by touches
        key = lambda c: (getattr(c, "touch_count", 0), getattr(c, "touch_rate", 0.0))
        lowers.sort(key=key, reverse=True)
        uppers.sort(key=key, reverse=True)
        equals.sort(key=key, reverse=True)
        return lowers, uppers, equals

    # ---------- public: fast k=2 generator ----------

    def generate_k2_bounds(self, cfg: GenerateK2Config) -> Dict[str, List[Conjecture]]:
        """
        Special-case fast generator for k=2 bounds over all pairs of features.
        """
        # resolve target and candidate features
        target_expr = _to_expr(cfg.target, self.exprs)
        target_name = repr(target_expr)
        feats: List[Expr] = [
            self.exprs[c] for c in self.expr_columns if repr(self.exprs[c]) != target_name
        ]
        if len(feats) < 2:
            return {"lowers": [], "uppers": [], "equals": []}

        # hypotheses to iterate
        H_iter = list(self.hypotheses)
        if cfg.hypotheses_limit is not None:
            H_iter = H_iter[:cfg.hypotheses_limit]

        df = self.df
        y_full = target_expr.eval(df).to_numpy(dtype=float, copy=False)

        # cache feature arrays once
        arr_cache: Dict[str, np.ndarray] = {
            repr(f): f.eval(df).to_numpy(dtype=float, copy=False) for f in feats
        }

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []
        equals: List[Conjecture] = []

        for f1, f2 in combinations(feats, 2):
            a1 = arr_cache[repr(f1)]
            a2 = arr_cache[repr(f2)]
            finite_all = _finite_mask([y_full, a1, a2])
            if int(finite_all.sum()) < 3:
                continue

            for H_name, H in H_iter:
                mask = (
                    self.gcr._mask_cached(H)
                    if H is not TRUE
                    else np.ones(len(df), dtype=bool)
                ) & finite_all
                if int(mask.sum()) < 3:
                    continue

                y = y_full[mask]
                X = np.column_stack([a1[mask], a2[mask]])

                # LOWER: y ≥ a·x + b
                try:
                    a_lo, b_lo = _solve_sum_slack_lp(
                        X,
                        y,
                        sense="lower",
                        coef_bounds=cfg.coef_bounds,
                        intercept_bounds=cfg.intercept_bounds,
                    )
                except Exception:
                    a_lo = b_lo = None

                if a_lo is not None:
                    rhs_lo_expr = _affine_expr(
                        a_lo, [f1, f2], float(b_lo), cfg.max_denom, cfg.tol
                    )
                    rhs_lo_vals = rhs_lo_expr.eval(df).to_numpy(dtype=float, copy=False)
                    tc, tr, n = _touch(
                        y_full,
                        rhs_lo_vals,
                        mask,
                        atol=cfg.touch_atol,
                        rtol=cfg.touch_rtol,
                    )

                    if n > 0 and tc == n:
                        cj = Conjecture(
                            relation=Eq(target_expr, rhs_lo_expr, tol=cfg.tol),
                            condition=H,
                        )
                        setattr(cj, "touch_count", tc)
                        setattr(cj, "touch_rate", tr)
                        setattr(cj, "support_n", n)
                        coef_pairs = [
                            (repr(f1), float(a_lo[0])),
                            (repr(f2), float(a_lo[1])),
                        ]
                        coef_pairs_frac = [
                            (repr(f1), _fractionize(float(a_lo[0]), cfg.max_denom)),
                            (repr(f2), _fractionize(float(a_lo[1]), cfg.max_denom)),
                        ]
                        setattr(cj, "coefficient_pairs", coef_pairs)
                        setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                        setattr(cj, "intercept", float(b_lo))
                        setattr(cj, "intercept_frac", _fractionize(float(b_lo), cfg.max_denom))
                        equals.append(cj)
                    elif tc > cfg.min_touch:
                        cj = Conjecture(
                            relation=Ge(target_expr, rhs_lo_expr),
                            condition=H,
                        )
                        setattr(cj, "touch_count", tc)
                        setattr(cj, "touch_rate", tr)
                        setattr(cj, "support_n", n)
                        coef_pairs = [
                            (repr(f1), float(a_lo[0])),
                            (repr(f2), float(a_lo[1])),
                        ]
                        coef_pairs_frac = [
                            (repr(f1), _fractionize(float(a_lo[0]), cfg.max_denom)),
                            (repr(f2), _fractionize(float(a_lo[1]), cfg.max_denom)),
                        ]
                        setattr(cj, "coefficient_pairs", coef_pairs)
                        setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                        setattr(cj, "intercept", float(b_lo))
                        setattr(cj, "intercept_frac", _fractionize(float(b_lo), cfg.max_denom))
                        lowers.append(cj)

                # UPPER: y ≤ a·x + b
                try:
                    a_up, b_up = _solve_sum_slack_lp(
                        X,
                        y,
                        sense="upper",
                        coef_bounds=cfg.coef_bounds,
                        intercept_bounds=cfg.intercept_bounds,
                    )
                except Exception:
                    a_up = b_up = None

                if a_up is not None:
                    rhs_up_expr = _affine_expr(
                        a_up, [f1, f2], float(b_up), cfg.max_denom, cfg.tol
                    )
                    rhs_up_vals = rhs_up_expr.eval(df).to_numpy(dtype=float, copy=False)
                    tc, tr, n = _touch(
                        y_full,
                        rhs_up_vals,
                        mask,
                        atol=cfg.touch_atol,
                        rtol=cfg.touch_rtol,
                    )

                    if n > 0 and tc == n:
                        cj = Conjecture(
                            relation=Eq(target_expr, rhs_up_expr, tol=cfg.tol),
                            condition=H,
                        )
                        setattr(cj, "touch_count", tc)
                        setattr(cj, "touch_rate", tr)
                        setattr(cj, "support_n", n)
                        coef_pairs = [
                            (repr(f1), float(a_up[0])),
                            (repr(f2), float(a_up[1])),
                        ]
                        coef_pairs_frac = [
                            (repr(f1), _fractionize(float(a_up[0]), cfg.max_denom)),
                            (repr(f2), _fractionize(float(a_up[1]), cfg.max_denom)),
                        ]
                        setattr(cj, "coefficient_pairs", coef_pairs)
                        setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                        setattr(cj, "intercept", float(b_up))
                        setattr(cj, "intercept_frac", _fractionize(float(b_up), cfg.max_denom))
                        equals.append(cj)
                    elif tc > cfg.min_touch:
                        cj = Conjecture(
                            relation=Le(target_expr, rhs_up_expr),
                            condition=H,
                        )
                        setattr(cj, "touch_count", tc)
                        setattr(cj, "touch_rate", tr)
                        setattr(cj, "support_n", n)
                        coef_pairs = [
                            (repr(f1), float(a_up[0])),
                            (repr(f2), float(a_up[1])),
                        ]
                        coef_pairs_frac = [
                            (repr(f1), _fractionize(float(a_up[0]), cfg.max_denom)),
                            (repr(f2), _fractionize(float(a_up[1]), cfg.max_denom)),
                        ]
                        setattr(cj, "coefficient_pairs", coef_pairs)
                        setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                        setattr(cj, "intercept", float(b_up))
                        setattr(cj, "intercept_frac", _fractionize(float(b_up), cfg.max_denom))
                        uppers.append(cj)

        # Small stable de-dup by string
        def _dedup(xs: List[Conjecture]) -> List[Conjecture]:
            seen, out = set(), []
            for c in xs:
                s = str(c)
                if s in seen:
                    continue
                seen.add(s)
                out.append(c)
            return out

        lowers_d = morgan_filter(self.df, _dedup(lowers)).kept
        uppers_d = morgan_filter(self.df, _dedup(uppers)).kept
        equals_d = morgan_filter(self.df, _dedup(equals)).kept

        key = lambda c: (getattr(c, "touch_count", 0), getattr(c, "touch_rate", 0.0))
        lowers_d.sort(key=key, reverse=True)
        uppers_d.sort(key=key, reverse=True)
        equals_d.sort(key=key, reverse=True)

        return {"lowers": lowers_d, "uppers": uppers_d, "equals": equals_d}

    # ---------- public: general k-affine generator ----------

    def generate_k_affine_bounds(
        self,
        *,
        target: Union[str, Expr],
        k_values: Iterable[int] = (1, 2, 3),
        hypotheses_limit: Optional[int] = 5,
        min_touch: int = 3,
        max_denom: int = 30,
        tol: float = 1e-9,
        touch_atol: float = 0.0,
        touch_rtol: float = 0.0,
        top_m_by_variance: Optional[int] = 12,
        coef_bounds: Optional[Tuple[float, float]] = None,
        intercept_bounds: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, List[Conjecture]]:
        """
        Fast k-feature affine LOWER/UPPER/EQ search.

        The coefficient and intercept bounds are optional and, when provided,
        are passed through to the underlying LP for each combo.
        """
        # Resolve target & features
        if isinstance(target, str):
            y_expr = self.exprs.get(target, ColumnTerm(target))
            target_name = target
        else:
            y_expr = target
            target_name = repr(target)

        feats_all: List[Expr] = [
            e for e in self.invariants if repr(e) != target_name
        ]
        if not feats_all:
            return {"lowers": [], "uppers": [], "equals": []}

        # Hypotheses (already sorted by generality from GCR)
        H_iter = list(self.hypotheses)
        if hypotheses_limit is not None:
            H_iter = H_iter[:hypotheses_limit]

        df = self.df

        # Pre-evaluate arrays once
        y_full = y_expr.eval(df).to_numpy(dtype=float, copy=False)
        feat_arrays: Dict[str, np.ndarray] = {
            repr(fx): fx.eval(df).to_numpy(dtype=float, copy=False)
            for fx in feats_all
        }

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []
        equals: List[Conjecture] = []

        # Local touch-stats helper
        def _touch_stats(
            lhs_arr: np.ndarray,
            rhs_arr: np.ndarray,
            H_mask: np.ndarray,
        ) -> Tuple[int, float, int]:
            m = H_mask & np.isfinite(lhs_arr) & np.isfinite(rhs_arr)
            n = int(m.sum())
            if n == 0:
                return 0, 0.0, 0
            dif = np.abs(lhs_arr[m] - rhs_arr[m])
            tol_arr = touch_atol + touch_rtol * np.abs(rhs_arr[m])
            tc = int((dif <= tol_arr).sum())
            return tc, (tc / n), n

        for H_name, H in H_iter:
            H_mask = self.get_condition_mask(H)

            # Choose features by variance under H (optional)
            if top_m_by_variance is not None:
                scores: List[Tuple[float, Expr]] = []
                base = np.isfinite(y_full) & H_mask
                if base.sum() < 3:
                    continue
                for fx in feats_all:
                    a = feat_arrays[repr(fx)]
                    m = base & np.isfinite(a)
                    if m.sum() >= 3:
                        scores.append((float(np.var(a[m])), fx))
                scores.sort(key=lambda t: t[0], reverse=True)
                feats = [fx for _, fx in scores[:top_m_by_variance]]
            else:
                feats = feats_all

            if not feats or all(k > len(feats) for k in k_values):
                continue

            # Iterate k and combos
            for k in k_values:
                if k < 1 or k > len(feats):
                    continue
                for combo in combinations(feats, k):
                    # Per-combo finite mask
                    m = H_mask & np.isfinite(y_full)
                    for fx in combo:
                        m &= np.isfinite(feat_arrays[repr(fx)])
                    if int(m.sum()) < 3:
                        continue

                    # Assemble design
                    X = np.column_stack([feat_arrays[repr(fx)][m] for fx in combo])
                    y = y_full[m]
                    feat_exprs = list(combo)

                    # LOWER: y ≥ a·x + b
                    try:
                        a_lo, b_lo = _solve_sum_slack_lp(
                            X,
                            y,
                            sense="lower",
                            coef_bounds=coef_bounds,
                            intercept_bounds=intercept_bounds,
                        )
                    except Exception:
                        a_lo = b_lo = None
                    if a_lo is not None:
                        rhs_lo_expr = _affine_expr(
                            a_lo, feat_exprs, float(b_lo), max_denom, tol
                        )
                        rhs_lo_vals = rhs_lo_expr.eval(df).to_numpy(
                            dtype=float, copy=False
                        )
                        tc, tr, n = _touch_stats(y_full, rhs_lo_vals, H_mask)
                        if n > 0 and tc == n:
                            cj = Conjecture(
                                relation=Eq(y_expr, rhs_lo_expr, tol=tol),
                                condition=H,
                            )
                            setattr(cj, "touch_count", tc)
                            setattr(cj, "touch_rate", tr)
                            setattr(cj, "support_n", n)
                            coef_pairs = [
                                (repr(fx), float(c))
                                for fx, c in zip(feat_exprs, a_lo)
                            ]
                            coef_pairs_frac = [
                                (repr(fx), _fractionize(float(c), max_denom))
                                for fx, c in zip(feat_exprs, a_lo)
                            ]
                            setattr(cj, "coefficient_pairs", coef_pairs)
                            setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                            setattr(cj, "intercept", float(b_lo))
                            setattr(cj, "intercept_frac", _fractionize(float(b_lo), max_denom))
                            equals.append(cj)
                        elif tc >= min_touch:
                            cj = Conjecture(
                                relation=Ge(y_expr, rhs_lo_expr),
                                condition=H,
                            )
                            setattr(cj, "touch_count", tc)
                            setattr(cj, "touch_rate", tr)
                            setattr(cj, "support_n", n)
                            coef_pairs = [
                                (repr(fx), float(c))
                                for fx, c in zip(feat_exprs, a_lo)
                            ]
                            coef_pairs_frac = [
                                (repr(fx), _fractionize(float(c), max_denom))
                                for fx, c in zip(feat_exprs, a_lo)
                            ]
                            setattr(cj, "coefficient_pairs", coef_pairs)
                            setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                            setattr(cj, "intercept", float(b_lo))
                            setattr(cj, "intercept_frac", _fractionize(float(b_lo), max_denom))
                            lowers.append(cj)

                    # UPPER: y ≤ a·x + b
                    try:
                        a_up, b_up = _solve_sum_slack_lp(
                            X,
                            y,
                            sense="upper",
                            coef_bounds=coef_bounds,
                            intercept_bounds=intercept_bounds,
                        )
                    except Exception:
                        a_up = b_up = None
                    if a_up is not None:
                        rhs_up_expr = _affine_expr(
                            a_up, feat_exprs, float(b_up), max_denom, tol
                        )
                        rhs_up_vals = rhs_up_expr.eval(df).to_numpy(
                            dtype=float, copy=False
                        )
                        tc, tr, n = _touch_stats(y_full, rhs_up_vals, H_mask)
                        if n > 0 and tc == n:
                            cj = Conjecture(
                                relation=Eq(y_expr, rhs_up_expr, tol=tol),
                                condition=H,
                            )
                            setattr(cj, "touch_count", tc)
                            setattr(cj, "touch_rate", tr)
                            setattr(cj, "support_n", n)
                            coef_pairs = [
                                (repr(fx), float(c))
                                for fx, c in zip(feat_exprs, a_up)
                            ]
                            coef_pairs_frac = [
                                (repr(fx), _fractionize(float(c), max_denom))
                                for fx, c in zip(feat_exprs, a_up)
                            ]
                            setattr(cj, "coefficient_pairs", coef_pairs)
                            setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                            setattr(cj, "intercept", float(b_up))
                            setattr(cj, "intercept_frac", _fractionize(float(b_up), max_denom))
                            equals.append(cj)
                        elif tc >= min_touch:
                            cj = Conjecture(
                                relation=Le(y_expr, rhs_up_expr),
                                condition=H,
                            )
                            setattr(cj, "touch_count", tc)
                            setattr(cj, "touch_rate", tr)
                            setattr(cj, "support_n", n)
                            coef_pairs = [
                                (repr(fx), float(c))
                                for fx, c in zip(feat_exprs, a_up)
                            ]
                            coef_pairs_frac = [
                                (repr(fx), _fractionize(float(c), max_denom))
                                for fx, c in zip(feat_exprs, a_up)
                            ]
                            setattr(cj, "coefficient_pairs", coef_pairs)
                            setattr(cj, "coefficient_pairs_frac", coef_pairs_frac)
                            setattr(cj, "intercept", float(b_up))
                            setattr(cj, "intercept_frac", _fractionize(float(b_up), max_denom))
                            uppers.append(cj)

        # Dedup + sort
        def _dedup(lst: List[Conjecture]) -> List[Conjecture]:
            seen, out = set(), []
            for c in lst:
                s = str(c)
                if s in seen:
                    continue
                seen.add(s)
                out.append(c)
            out.sort(
                key=lambda c: (
                    getattr(c, "touch_count", 0),
                    getattr(c, "support_n", 0),
                ),
                reverse=True,
            )
            return out

        lowers_d = morgan_filter(df, _dedup(lowers)).kept
        uppers_d = morgan_filter(df, _dedup(uppers)).kept
        equals_d = morgan_filter(df, _dedup(equals)).kept

        return {
            "lowers": lowers_d,
            "uppers": uppers_d,
            "equals": equals_d,
        }

    # ---------- condition masks ----------

    def get_condition_mask(self, cond: Optional[Predicate]) -> np.ndarray:
        """
        Return a boolean mask for `cond` on self.df.
        - None or TRUE ⇒ all rows
        - otherwise use GraffitiClassRelations' cached mask.
        """
        if cond is None or cond is TRUE:
            return np.ones(len(self.df), dtype=bool)
        return self.gcr._mask_cached(cond)

    def _mask_via_gcr(self, pred: Predicate) -> np.ndarray:
        """
        Backwards-compatible alias for GraffitiClassRelations mask lookup.
        """
        return self.gcr._mask_cached(pred)
