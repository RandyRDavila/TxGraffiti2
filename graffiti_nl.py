# # src/txgraffiti2025/graffiti_nonlinear.py

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Sequence, Tuple, Union, Optional, Dict, List, Iterable

# import numpy as np
# import pandas as pd
# import pulp
# import shutil
# from itertools import combinations

# from txgraffiti2025.graffiti_relations import GraffitiClassRelations
# from txgraffiti2025.forms.utils import Expr, ColumnTerm, Const
# from txgraffiti2025.forms.predicates import Predicate, Where
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE


# # ─────────────────────────── Touch & array helpers ─────────────────────────── #

# def _expr_array(e: Expr, df: pd.DataFrame) -> np.ndarray:
#     """Evaluate an Expr on df to a float ndarray (no copy where possible)."""
#     return e.eval(df).to_numpy(dtype=float, copy=False)

# def _touch_count_on_mask(lhs: np.ndarray, rhs: np.ndarray, mask: np.ndarray, *, atol: float, rtol: float) -> tuple[int, float, int]:
#     """
#     Count rows on `mask` where |lhs - rhs| <= atol + rtol*|rhs|.
#     Returns (touch_count, touch_rate, n_mask).
#     """
#     if mask is None:
#         m = np.ones_like(lhs, dtype=bool)
#     else:
#         m = mask
#     m = m & np.isfinite(lhs) & np.isfinite(rhs)
#     n = int(m.sum())
#     if n == 0:
#         return 0, 0.0, 0
#     diff = np.abs(lhs[m] - rhs[m])
#     tol = atol + rtol * np.abs(rhs[m])
#     tc = int((diff <= tol).sum())
#     return tc, (tc / n), n

# def _attach_touch_and_sort(conjs: list[Conjecture]) -> list[Conjecture]:
#     """
#     Ensure touch metadata is present; sort by (-touch_count, -touch_rate).
#     """
#     for cj in conjs:
#         if not hasattr(cj, "touch_count"):
#             setattr(cj, "touch_count", 0)
#         if not hasattr(cj, "touch_rate"):
#             setattr(cj, "touch_rate", 0.0)
#         if not hasattr(cj, "support_n"):
#             setattr(cj, "support_n", 0)
#     conjs.sort(key=lambda c: (getattr(c, "touch_count", 0), getattr(c, "touch_rate", 0.0)), reverse=True)
#     return conjs

# def _recompute_touch_for(cj: Conjecture, df: pd.DataFrame, mask: np.ndarray, *, atol: float, rtol: float) -> None:
#     """
#     For compressed/constructed conjectures, recompute touch metadata if possible
#     (only for atomic Ge/Le/Eq with .lhs/.rhs Exprs). Otherwise sets zeros.
#     """
#     rel = cj.relation
#     lhs = getattr(rel, "lhs", None)
#     rhs = getattr(rel, "rhs", None)
#     if lhs is None or rhs is None:
#         setattr(cj, "touch_count", 0)
#         setattr(cj, "touch_rate", 0.0)
#         setattr(cj, "support_n", int(mask.sum()))
#         return
#     lhs_arr = _expr_array(lhs, df)
#     rhs_arr = _expr_array(rhs, df)
#     tc, tr, n = _touch_count_on_mask(lhs_arr, rhs_arr, mask, atol=atol, rtol=rtol)
#     setattr(cj, "touch_count", tc)
#     setattr(cj, "touch_rate", tr)
#     setattr(cj, "support_n", n)

# # Optional: if your class does NOT already have this
# def get_condition_mask(self, cond: Predicate | None) -> np.ndarray:
#     """
#     TRUE/None -> all rows; else delegate to GCR's cached mask.
#     """
#     if cond is None or cond is TRUE:
#         return np.ones(len(self.df), dtype=bool)
#     return self.gcr._mask_cached(cond)

# # ─────────────────────────── Solver discovery ─────────────────────────── #

# def _get_available_solver():
#     """
#     Return a silent CBC/GLPK pulp solver instance if available.
#     """
#     cbc = shutil.which("cbc")
#     if cbc:
#         return pulp.COIN_CMD(path=cbc, msg=False)
#     glpk = shutil.which("glpsol")
#     if glpk:
#         pulp.LpSolverDefault.msg = 0
#         return pulp.GLPK_CMD(path=glpk, msg=False)
#     raise RuntimeError("No LP solver found (install CBC or GLPK)")


# # ───────────────────────── LP: sum-of-slacks fit ───────────────────────── #

# def _solve_sum_slack_lp(X: np.ndarray, y: np.ndarray, *, sense: str) -> Tuple[np.ndarray, float]:
#     """
#     Solve the sum-of-slacks LP for an UPPER or LOWER affine bound.

#     sense: "upper" -> (a·x + b) - y == s, s >= 0  (so y <= a·x + b)
#            "lower" -> y - (a·x + b) == s, s >= 0  (so y >= a·x + b)
#     """
#     n, k = X.shape
#     prob = pulp.LpProblem("sum_slack", pulp.LpMinimize)

#     a = [pulp.LpVariable(f"a_{j}", lowBound=None) for j in range(k)]
#     b = pulp.LpVariable("b", lowBound=None)
#     s = [pulp.LpVariable(f"s_{i}", lowBound=0) for i in range(n)]

#     # minimize total slack
#     prob += pulp.lpSum(s)

#     # constraints
#     for i in range(n):
#         lhs = pulp.lpSum(a[j] * float(X[i, j]) for j in range(k)) + b
#         yi = float(y[i])
#         if sense == "upper":
#             prob += lhs - yi == s[i]
#         elif sense == "lower":
#             prob += yi - lhs == s[i]
#         else:
#             raise ValueError("sense must be 'upper' or 'lower'")

#     solver = _get_available_solver()
#     status = prob.solve(solver)
#     if pulp.LpStatus[status] != "Optimal":
#         raise RuntimeError(f"LP did not solve optimally: {pulp.LpStatus[status]}")

#     def _val(v):
#         vv = v.value()
#         return float(vv) if vv is not None else float("nan")

#     a_sol = np.array([_val(v) for v in a], dtype=float)
#     b_sol = _val(b)
#     return a_sol, b_sol


# # ───────────────────────── Config & utilities ─────────────────────────── #

# @dataclass
# class LPConfig:
#     """
#     Configuration for linear (affine) bound fitting.

#     Parameters
#     ----------
#     features : sequence of column names or Exprs
#         RHS features x_j. Strings will be resolved to ColumnTerm.
#     target : column name or Expr
#         LHS target y.
#     direction : {"both","upper","lower"}, default="both"
#         Which bounds to emit.
#     max_denominator : int, default=50
#         Rationalization denominator cap for a, b when pretty-printing via Const(Fraction).
#     tol : float, default=1e-9
#         Zero-out coefficients with |a_j| <= tol (and b if |b| <= tol).
#     min_support : int, default=3
#         Minimum number of valid rows (after masking & finiteness) required to fit.
#     """
#     features: Sequence[Union[str, Expr]]
#     target: Union[str, Expr]
#     direction: str = "both"
#     max_denominator: int = 50
#     tol: float = 1e-9
#     min_support: int = 3


# def _rational_const(x: float, max_den: int, tol: float) -> Const:
#     """
#     Return a Const that prefers a small-denominator Fraction for display.

#     Always attempts to rationalize any finite x with |x| > tol.
#     """
#     from fractions import Fraction
#     if not np.isfinite(x) or abs(x) <= tol:
#         return Const(0.0)
#     fr = Fraction(x).limit_denominator(max_den)
#     return Const(fr)

# def _to_expr(e: Union[str, Expr], exprs: Dict[str, Expr]) -> Expr:
#     if isinstance(e, Expr):
#         return e
#     if isinstance(e, str):
#         if e not in exprs:
#             # allow on-the-fly wrap for columns not in exprs yet
#             return ColumnTerm(e)
#         return exprs[e]
#     raise TypeError(f"Unsupported feature/target type: {type(e)}")


# def _finite_mask(arrs: Sequence[np.ndarray]) -> np.ndarray:
#     m = np.ones_like(arrs[0], dtype=bool)
#     for a in arrs:
#         m &= np.isfinite(a)
#     return m


# # def _rational_const(x: float, max_den: int, tol: float) -> Const:
# #     """
# #     Return a Const that prefers a small-denominator Fraction if close, else float.
# #     """
# #     if not np.isfinite(x) or abs(x) <= tol:
# #         return Const(0.0)
# #     from fractions import Fraction
# #     fr = Fraction(x).limit_denominator(max_den)
# #     if abs(float(fr) - x) <= 1e-12:
# #         return Const(fr)
# #     return Const(float(x))


# def _affine_expr(a: np.ndarray, feats: Sequence[Expr], b: float, max_den: int, tol: float) -> Expr:
#     """
#     Build sum_j a_j * feats[j] + b as an Expr, dropping ~zero coefficients.
#     """
#     expr: Optional[Expr] = None
#     for coef, fj in zip(a, feats):
#         if abs(coef) <= tol:
#             continue
#         term = _rational_const(coef, max_den, tol) * fj
#         expr = term if expr is None else (expr + term)
#     # append intercept if meaningful
#     if abs(b) > tol:
#         c = _rational_const(b, max_den, tol)
#         expr = c if expr is None else (expr + c)
#     # if everything zeroed, return Const(0)
#     return expr if expr is not None else Const(0.0)


# @dataclass(frozen=True)
# class NonlinearConfig:
#     """
#     Configuration for future nonlinear mining (placeholders for now).

#     Parameters
#     ----------
#     require_finite : bool
#         If True, ignore ±inf/NaN rows before analysis.
#     rationalize_constants : bool
#         Whether to snap constants to small denominators when printing.
#     rationalize_exponents : bool
#         Whether to prefer canonical rationals (e.g., 1/2) for displayed exponents.
#     max_denom : int
#         Max denominator for rationalization via Fraction.limit_denominator.
#     touch_atol : float
#         Absolute tolerance for boundary touch tests.
#     touch_rtol : float
#         Relative tolerance for boundary touch tests.
#     """
#     require_finite: bool = True
#     rationalize_constants: bool = True
#     rationalize_exponents: bool = True
#     max_denom: int = 30
#     touch_atol: float = 0.0
#     touch_rtol: float = 0.0


# class GraffitiNonlinear:
#     """
#     Thin scaffold for (affine now, nonlinear later) conjecture mining that *adopts*
#     the full analysis context already discovered by `GraffitiClassRelations` (GCR).

#     Mirrors:
#       - df
#       - base_hypothesis / base_hypothesis_name
#       - exprs / expr_columns
#       - base_predicates
#       - hypotheses: nonredundant conjunctions sorted by generality
#     """

#     # ───────────────────────────── Constructors ───────────────────────────── #

#     def __init__(
#         self,
#         gcr: GraffitiClassRelations,
#         *,
#         config: Optional[NonlinearConfig] = None,
#     ) -> None:
#         if not isinstance(gcr, GraffitiClassRelations):
#             raise TypeError("gcr must be an instance of GraffitiClassRelations")

#         # Keep a direct handle so we share caches (masks, etc.)
#         self.gcr: GraffitiClassRelations = gcr

#         # Stable mirrors of the GCR context
#         self.df: pd.DataFrame = gcr.df
#         self.base_hypothesis: Predicate = gcr.base_hypothesis
#         self.base_hypothesis_name: str = gcr.base_hypothesis_name
#         self.nonredundant_conjunctions_ = gcr.nonredundant_conjunctions_
#         # Use GCR's sorting by generality as our active hypothesis list
#         self.hypotheses: List[Tuple[str, Predicate]] = gcr.sort_conjunctions_by_generality()

#         # Universes
#         self.exprs: Dict[str, Expr] = gcr.get_exprs()
#         self.expr_columns: List[str] = gcr.get_expr_columns()
#         self.base_predicates: Dict[str, Predicate] = gcr.get_base_predicates()
#         self.invariants = list(self.exprs.values())

#         # Config
#         self.config: NonlinearConfig = config or NonlinearConfig()

#         # Delegate mask computation to GCR (shared cache)
#         self._mask = self._mask_via_gcr

#     @classmethod
#     def from_gcr(
#         cls,
#         gcr: GraffitiClassRelations,
#         *,
#         config: Optional[NonlinearConfig] = None,
#     ) -> "GraffitiNonlinear":
#         return cls(gcr, config=config)

#     @classmethod
#     def from_dataframe(
#         cls,
#         df: pd.DataFrame,
#         *,
#         config: Optional[NonlinearConfig] = None,
#     ) -> "GraffitiNonlinear":
#         gcr = GraffitiClassRelations(df)
#         return cls(gcr, config=config)

#     # ────────────────────────────── Internals ─────────────────────────────── #

#     def _mask_via_gcr(self, pred: Predicate) -> np.ndarray:
#         """Delegate to GraffitiClassRelations' cached mask machinery."""
#         return self.gcr._mask_cached(pred)

#     def condition_or_base(self, cond: Optional[Predicate]) -> Predicate:
#         """Return `cond` if provided, else the base hypothesis (or TRUE)."""
#         return cond or self.base_hypothesis or TRUE

#     def get_condition_mask(self, cond: Optional[Predicate]) -> np.ndarray:
#         """
#         Mask for `cond` or the base hypothesis; `None/TRUE` -> all rows.
#         This avoids passing TRUE (sentinel) into the GCR mask cache.
#         """
#         target = self.condition_or_base(cond)
#         if target is TRUE:  # sentinel: not a Predicate
#             return np.ones(len(self.df), dtype=bool)
#         return self._mask(target)

#     # ───────────────────────── Affine (linear) fits ───────────────────────── #

#     def fit_linear_bounds(
#         self,
#         cfg: LPConfig,
#         *,
#         condition: Optional[Predicate] = None,
#         compress: bool = True,
#         touch_atol: Optional[float] = None,
#         touch_rtol: Optional[float] = None,
#     ) -> Tuple[List[Conjecture], List[Conjecture], List[Conjecture]]:
#         """
#         Fit affine bounds y ≲ a·x + b (upper) and/or y ≳ a·x + b (lower) under a condition.
#         Attaches touch_count/touch_rate/support_n and sorts lowers/uppers by touches (desc).
#         """
#         t_atol = float(touch_atol if touch_atol is not None else getattr(self.config, "touch_atol", 0.0))
#         t_rtol = float(touch_rtol if touch_rtol is not None else getattr(self.config, "touch_rtol", 0.0))

#         y_expr = _to_expr(cfg.target, self.exprs)
#         feat_exprs = [_to_expr(f, self.exprs) for f in cfg.features]

#         cond = self.condition_or_base(condition)
#         mask = self.get_condition_mask(cond)

#         y_arr_full = _expr_array(y_expr, self.df)
#         X_cols_full = [_expr_array(fx, self.df) for fx in feat_exprs]

#         fin = _finite_mask([y_arr_full, *X_cols_full]) & mask
#         if int(fin.sum()) < cfg.min_support:
#             return [], [], []

#         y = y_arr_full[fin]
#         X = np.column_stack([c[fin] for c in X_cols_full])

#         lowers: List[Conjecture] = []
#         uppers: List[Conjecture] = []
#         equals: List[Conjecture] = []

#         want_lower = cfg.direction in ("both", "lower")
#         want_upper = cfg.direction in ("both", "upper")

#         a_lo = b_lo = a_up = b_up = None

#         # LOWER: y ≥ a·x + b
#         if want_lower:
#             a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
#             rhs_lo = _affine_expr(a_lo, feat_exprs, b_lo, cfg.max_denominator, cfg.tol)
#             cj_lo = Conjecture(
#                 relation=Ge(y_expr, rhs_lo),
#                 condition=(None if cond is TRUE else cond),
#                 name=f"{repr(cond)} | lower_affine",
#             )
#             rhs_lo_arr = _expr_array(rhs_lo, self.df)
#             tc, tr, n = _touch_count_on_mask(y_arr_full, rhs_lo_arr, mask, atol=t_atol, rtol=t_rtol)
#             setattr(cj_lo, "touch_count", tc)
#             setattr(cj_lo, "touch_rate", tr)
#             setattr(cj_lo, "support_n", n)
#             lowers.append(cj_lo)

#         # UPPER: y ≤ a·x + b
#         if want_upper:
#             a_up, b_up = _solve_sum_slack_lp(X, y, sense="upper")
#             rhs_up = _affine_expr(a_up, feat_exprs, b_up, cfg.max_denominator, cfg.tol)
#             cj_up = Conjecture(
#                 relation=Le(y_expr, rhs_up),
#                 condition=(None if cond is TRUE else cond),
#                 name=f"{repr(cond)} | upper_affine",
#             )
#             rhs_up_arr = _expr_array(rhs_up, self.df)
#             tc, tr, n = _touch_count_on_mask(y_arr_full, rhs_up_arr, mask, atol=t_atol, rtol=t_rtol)
#             setattr(cj_up, "touch_count", tc)
#             setattr(cj_up, "touch_rate", tr)
#             setattr(cj_up, "support_n", n)
#             uppers.append(cj_up)

#         # Optional equality (degenerate when upper==lower)
#         if want_lower and want_upper and (a_lo is not None) and (a_up is not None):
#             same_a = np.allclose(a_lo, a_up, atol=cfg.tol, rtol=0.0)
#             b_lo_v = (b_lo if b_lo is not None else 0.0)
#             b_up_v = (b_up if b_up is not None else 0.0)
#             same_b = abs(b_lo_v - b_up_v) <= cfg.tol
#             if same_a and same_b:
#                 rhs_eq = _affine_expr(a_lo, feat_exprs, b_lo_v, cfg.max_denominator, cfg.tol)
#                 cj_eq = Conjecture(
#                     relation=Eq(y_expr, rhs_eq, tol=cfg.tol),
#                     condition=(None if cond is TRUE else cond),
#                     name=f"{repr(cond)} | eq_affine",
#                 )
#                 rhs_eq_arr = _expr_array(rhs_eq, self.df)
#                 tc, tr, n = _touch_count_on_mask(y_arr_full, rhs_eq_arr, mask, atol=t_atol, rtol=t_rtol)
#                 setattr(cj_eq, "touch_count", tc)
#                 setattr(cj_eq, "touch_rate", tr)
#                 setattr(cj_eq, "support_n", n)
#                 equals.append(cj_eq)

#         return lowers, uppers, equals

#     # --- small internal helpers ------------------------------------------------

#     def _support_size(self, pred: Optional[Predicate]) -> int:
#         """Number of rows satisfying pred (or base if pred is None)."""
#         m = self.get_condition_mask(pred or self.base_hypothesis)
#         return int(m.sum())

#     def _dedup_by_hash(self, conjs: list[Conjecture]) -> list[Conjecture]:
#         """Stable de-dup by conjecture string form (your Conjecture __str__/pretty is canonical)."""
#         seen = set()
#         out = []
#         for c in conjs:
#             key = str(c)
#             if key in seen:
#                 continue
#             seen.add(key)
#             out.append(c)
#         return out

#     def _sorted_invariants_excluding(self, target_name: str) -> list[Expr]:
#         """All invariants (Expr) except the target (by repr) in a stable order."""
#         return [inv for inv in getattr(self, "invariants", []) or []
#                 if repr(inv) != target_name]

#     # --- public 2-feature generator -------------------------------------------

#     def generate_k2_affine_bounds(
#         self,
#         *,
#         target: Union[str, Expr],
#         hypotheses_limit: Optional[int] = 10,
#         min_touch: int = 3,
#         max_denominator: int = 30,
#         tol: float = 1e-9,
#         touch_atol: float = 0.0,
#         touch_rtol: float = 0.0,
#         apply_morgan_filter: bool = True,
#     ) -> dict[str, list[Conjecture]]:
#         """
#         Enumerate all pairs of invariants (excluding target) and, for each active
#         hypothesis H, fit LOWER and UPPER affine bounds via sum-of-slacks LP:
#             lower: y ≥ a1*x1 + a2*x2 + b
#             upper: y ≤ a1*x1 + a2*x2 + b

#         We compute touch_count on the H-support, promote to equality if all rows
#         under H touch, and sort by touch_count (desc). Results are deduped.

#         Parameters
#         ----------
#         target : str | Expr
#             LHS target invariant.
#         hypotheses_limit : int, optional
#             Only consider the first N hypotheses in `self.hypotheses` (already
#             sorted by generality). None = use all.
#         min_touch : int, default 3
#             Keep a bound if it touches at least this many rows on the H-support.
#         max_denominator : int, default 30
#             Rationalization cap for coefficient pretty-printing.
#         tol : float, default 1e-9
#             Coefficient/const pruning and equality tolerance.
#         touch_atol, touch_rtol : float
#             Absolute/relative tolerances for touch test |lhs-rhs| ≤ atol + rtol*|rhs|.
#         apply_morgan_filter : bool, default True
#             If True and `morgan_filter` is importable, post-prune with it.

#         Returns
#         -------
#         dict
#             {"lowers": [...], "uppers": [...], "equals": [...]}
#         """
#         try:
#             # optional; user already has this in your pipeline
#             from txgraffiti2025.processing.post import morgan_filter
#         except Exception:
#             morgan_filter = None
#             apply_morgan_filter = False

#         # Resolve target & RHS candidate invariants
#         if isinstance(target, str):
#             target_name = target
#             y_expr = ColumnTerm(target)
#         else:
#             y_expr = target
#             target_name = repr(target)

#         feats = self._sorted_invariants_excluding(target_name)
#         if not feats:
#             return {"lowers": [], "uppers": [], "equals": []}

#         # Hypotheses to iterate (name, Predicate)
#         H_iter = list(self.hypotheses)
#         if hypotheses_limit is not None:
#             H_iter = H_iter[:hypotheses_limit]

#         df = self.df
#         y_full = y_expr.eval(df).to_numpy(dtype=float, copy=False)

#         lowers: list[Conjecture] = []
#         uppers: list[Conjecture] = []
#         equals: list[Conjecture] = []

#         # Iterate pairs and hypotheses
#         for x1, x2 in combinations(feats, 2):
#             x1_full = x1.eval(df).to_numpy(dtype=float, copy=False)
#             x2_full = x2.eval(df).to_numpy(dtype=float, copy=False)

#             # Quick global finite mask to avoid recomputing too often
#             finite_all = np.isfinite(y_full) & np.isfinite(x1_full) & np.isfinite(x2_full)
#             if finite_all.sum() < 3:
#                 continue

#             for H_name, H in H_iter:
#                 H_mask = self.get_condition_mask(H)
#                 sup_n = int((finite_all & H_mask).sum())
#                 if sup_n < 3:
#                     continue

#                 y = y_full[finite_all & H_mask]
#                 X = np.column_stack([x1_full[finite_all & H_mask],
#                                      x2_full[finite_all & H_mask]])

#                 # LOWER
#                 a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
#                 rhs_lo = _affine_expr(a_lo, [x1, x2], b_lo, max_denominator, tol)
#                 cj_lo = Conjecture(relation=Ge(y_expr, rhs_lo), condition=H)
#                 # touches on H-support, with tolerances
#                 lhs_arr = y_full[H_mask]
#                 rhs_arr = rhs_lo.eval(df)[H_mask].to_numpy(dtype=float, copy=False)
#                 diff = np.abs(lhs_arr - rhs_arr)
#                 tol_arr = touch_atol + touch_rtol * np.abs(rhs_arr)
#                 touch = int((np.isfinite(lhs_arr) & np.isfinite(rhs_arr) & (diff <= tol_arr)).sum())

#                 if touch == sup_n:
#                     cj_eq = Conjecture(relation=Eq(y_expr, rhs_lo, tol=tol), condition=H)
#                     cj_eq.touch = touch
#                     equals.append(cj_eq)
#                 elif touch > min_touch:
#                     cj_lo.touch = touch
#                     lowers.append(cj_lo)

#                 # UPPER
#                 a_up, b_up = _solve_sum_slack_lp(X, y, sense="upper")
#                 rhs_up = _affine_expr(a_up, [x1, x2], b_up, max_denominator, tol)
#                 cj_up = Conjecture(relation=Le(y_expr, rhs_up), condition=H)

#                 rhs_arr = rhs_up.eval(df)[H_mask].to_numpy(dtype=float, copy=False)
#                 diff = np.abs(lhs_arr - rhs_arr)
#                 tol_arr = touch_atol + touch_rtol * np.abs(rhs_arr)
#                 touch = int((np.isfinite(lhs_arr) & np.isfinite(rhs_arr) & (diff <= tol_arr)).sum())

#                 if touch == sup_n:
#                     cj_eq = Conjecture(relation=Eq(y_expr, rhs_up, tol=tol), condition=H)
#                     cj_eq.touch = touch
#                     equals.append(cj_eq)
#                 elif touch > min_touch:
#                     cj_up.touch = touch
#                     uppers.append(cj_up)

#         # De-dup by string form, then sort by touch desc
#         lowers = self._dedup_by_hash(lowers)
#         uppers = self._dedup_by_hash(uppers)
#         equals = self._dedup_by_hash(equals)

#         lowers.sort(key=lambda c: getattr(c, "touch", 0), reverse=True)
#         uppers.sort(key=lambda c: getattr(c, "touch", 0), reverse=True)
#         equals.sort(key=lambda c: getattr(c, "touch", 0), reverse=True)

#         # Optional Morgan post-filter
#         if apply_morgan_filter and morgan_filter is not None:
#             lowers = list(morgan_filter(self.df, lowers).kept)
#             uppers = list(morgan_filter(self.df, uppers).kept)
#             equals = list(morgan_filter(self.df, equals).kept)

#         return {"lowers": lowers, "uppers": uppers, "equals": equals}


#     def generate_k_affine_bounds(
#         self,
#         *,
#         target: Union[str, Expr],
#         k: Union[int, Sequence[int]] = 1,
#         features: Optional[Sequence[Union[str, Expr]]] = None,
#         hypotheses_limit: Optional[int] = 10,
#         min_touch: int = 3,
#         max_denominator: int = 30,
#         tol: float = 1e-9,
#         touch_atol: float = 0.0,
#         touch_rtol: float = 0.0,
#         apply_morgan_filter: bool = True,
#         return_by_k: bool = False,
#     ) -> Union[
#         Dict[str, List[Conjecture]],
#         Dict[int, Dict[str, List[Conjecture]]]
#     ]:
#         """
#         General k-feature affine bounds (k = 1,2,...). For each hypothesis H and each
#         k-combination of RHS features (excluding `target`), fit:
#             lower: y ≥ a·x + b
#             upper: y ≤ a·x + b
#         via a sum-of-slacks LP. Count touches on H-support; promote to equality when
#         all supported rows touch. Dedup and sort by touch.

#         Parameters
#         ----------
#         target : str | Expr
#             LHS invariant.
#         k : int | sequence of int, default=1
#             k-ary combinations to consider (e.g., 1 or [1,2,3]).
#         features : sequence[str|Expr], optional
#             Restrict RHS feature pool. Defaults to all invariants except target.
#         hypotheses_limit : int, optional
#             Consider only the first N (most general) hypotheses.
#         min_touch : int, default 3
#             Keep bounds with >= min_touch on H-support (non-equality).
#         max_denominator : int, default 30
#             Pretty-print rationalization cap for coefficients.
#         tol : float, default 1e-9
#             Coefficient pruning and equality tolerance.
#         touch_atol, touch_rtol : float
#             Touch test uses |lhs - rhs| ≤ atol + rtol*|rhs|.
#         apply_morgan_filter : bool, default True
#             If available, post-prune with `morgan_filter`.
#         return_by_k : bool, default False
#             If True, returns a dict keyed by k; else returns flattened lists.

#         Returns
#         -------
#         If return_by_k == False:
#             {"lowers": [...], "uppers": [...], "equals": [...]}
#         Else:
#             {k: {"lowers": [...], "uppers": [...], "equals": [...]}, ...}
#         """
#         # Optional Morgan filter
#         try:
#             from txgraffiti2025.processing.post import morgan_filter
#         except Exception:
#             morgan_filter = None
#             apply_morgan_filter = False

#         # Resolve target
#         if isinstance(target, str):
#             target_name = target
#             y_expr = ColumnTerm(target)
#         else:
#             y_expr = target
#             target_name = repr(target)

#         # Build RHS feature universe (Exprs) excluding target
#         if features is None:
#             # Prefer self.invariants if present; fall back to expr_columns
#             cand_exprs: List[Expr] = []
#             if getattr(self, "invariants", None):
#                 cand_exprs = [inv for inv in self.invariants if repr(inv) != target_name]
#             else:
#                 # use ColumnTerm for numeric expr columns
#                 cand_exprs = [self.exprs[c] if c in self.exprs else ColumnTerm(c)
#                               for c in self.expr_columns if c != target_name]
#         else:
#             # user-specified pool
#             cand_exprs = []
#             for f in features:
#                 if isinstance(f, Expr):
#                     if repr(f) != target_name:
#                         cand_exprs.append(f)
#                 elif isinstance(f, str):
#                     if f != target_name:
#                         cand_exprs.append(self.exprs.get(f, ColumnTerm(f)))
#                 else:
#                     raise TypeError(f"Unsupported feature type: {type(f)}")

#         if not cand_exprs:
#             return {"lowers": [], "uppers": [], "equals": []} if not return_by_k else {}

#         # Normalize k(s)
#         if isinstance(k, int):
#             ks = [k]
#         else:
#             ks = sorted(set(int(kk) for kk in k if int(kk) >= 1))

#         # Hypotheses to iterate (name, Predicate)
#         H_iter = list(self.hypotheses)
#         if hypotheses_limit is not None:
#             H_iter = H_iter[:hypotheses_limit]

#         df = self.df
#         y_full = y_expr.eval(df).to_numpy(dtype=float, copy=False)

#         # Pre-evaluate candidate features once
#         # Use repr(expr) as a stable key
#         feat_keys = [repr(e) for e in cand_exprs]
#         feat_by_key: Dict[str, Expr] = {repr(e): e for e in cand_exprs}
#         eval_by_key: Dict[str, np.ndarray] = {
#             key: feat_by_key[key].eval(df).to_numpy(dtype=float, copy=False)
#             for key in feat_keys
#         }

#         def _make_results_dict():
#             return {"lowers": [], "uppers": [], "equals": []}

#         results_by_k: Dict[int, Dict[str, List[Conjecture]]] = {kk: _make_results_dict() for kk in ks}

#         # For each hypothesis and each k-combination
#         for H_name, H in H_iter:
#             H_mask = self.get_condition_mask(H)
#             # Finite y on H-support (we’ll AND with per-combo finiteness later)
#             y_H = y_full[H_mask]

#             for kk in ks:
#                 # Enumerate combos by keys to keep arrays in sync and avoid hashing Exprs
#                 for combo_keys in combinations(feat_keys, kk):
#                     cols = [eval_by_key[k] for k in combo_keys]
#                     # Finite mask across y & all chosen X columns on H-support
#                     finite_all = np.isfinite(y_full)
#                     for c in cols:
#                         finite_all &= np.isfinite(c)
#                     sup_mask = finite_all & H_mask
#                     sup_n = int(sup_mask.sum())
#                     if sup_n < 3:
#                         continue

#                     y = y_full[sup_mask]
#                     X = np.column_stack([c[sup_mask] for c in cols])

#                     # Build Expr combo list (in the same order)
#                     combo_exprs = [feat_by_key[k] for k in combo_keys]

#                     # LOWER: y ≥ a·x + b
#                     a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
#                     rhs_lo = _affine_expr(a_lo, combo_exprs, b_lo, max_denominator, tol)
#                     cj_lo = Conjecture(relation=Ge(y_expr, rhs_lo), condition=H)

#                     # Touches on H-support
#                     lhs_arr = y_full[H_mask]
#                     rhs_arr = rhs_lo.eval(df)[H_mask].to_numpy(dtype=float, copy=False)
#                     diff = np.abs(lhs_arr - rhs_arr)
#                     tol_arr = touch_atol + touch_rtol * np.abs(rhs_arr)
#                     touch = int((np.isfinite(lhs_arr) & np.isfinite(rhs_arr) & (diff <= tol_arr)).sum())

#                     if touch == sup_n:
#                         cj_eq = Conjecture(relation=Eq(y_expr, rhs_lo, tol=tol), condition=H)
#                         cj_eq.touch = touch
#                         results_by_k[kk]["equals"].append(cj_eq)
#                     elif touch > min_touch:
#                         cj_lo.touch = touch
#                         results_by_k[kk]["lowers"].append(cj_lo)

#                     # UPPER: y ≤ a·x + b
#                     a_up, b_up = _solve_sum_slack_lp(X, y, sense="upper")
#                     rhs_up = _affine_expr(a_up, combo_exprs, b_up, max_denominator, tol)
#                     cj_up = Conjecture(relation=Le(y_expr, rhs_up), condition=H)

#                     rhs_arr = rhs_up.eval(df)[H_mask].to_numpy(dtype=float, copy=False)
#                     diff = np.abs(lhs_arr - rhs_arr)
#                     tol_arr = touch_atol + touch_rtol * np.abs(rhs_arr)
#                     touch = int((np.isfinite(lhs_arr) & np.isfinite(rhs_arr) & (diff <= tol_arr)).sum())

#                     if touch == sup_n:
#                         cj_eq = Conjecture(relation=Eq(y_expr, rhs_up, tol=tol), condition=H)
#                         cj_eq.touch = touch
#                         results_by_k[kk]["equals"].append(cj_eq)
#                     elif touch > min_touch:
#                         cj_up.touch = touch
#                         results_by_k[kk]["uppers"].append(cj_up)

#         # Dedup, sort, optional Morgan filter per k
#         for kk in ks:
#             for key in ("lowers", "uppers", "equals"):
#                 results_by_k[kk][key] = self._dedup_by_hash(results_by_k[kk][key])
#                 results_by_k[kk][key].sort(key=lambda c: getattr(c, "touch", 0), reverse=True)
#                 if apply_morgan_filter and morgan_filter is not None:
#                     results_by_k[kk][key] = list(morgan_filter(self.df, results_by_k[kk][key]).kept)

#         if return_by_k:
#             return results_by_k

#         # Flatten across k if user prefers a single bucket
#         flat = _make_results_dict()
#         for kk in ks:
#             flat["lowers"].extend(results_by_k[kk]["lowers"])
#             flat["uppers"].extend(results_by_k[kk]["uppers"])
#             flat["equals"].extend(results_by_k[kk]["equals"])

#         # Final dedup/sort after flatten
#         for key in ("lowers", "uppers", "equals"):
#             flat[key] = self._dedup_by_hash(flat[key])
#             flat[key].sort(key=lambda c: getattr(c, "touch", 0), reverse=True)
#             if apply_morgan_filter and morgan_filter is not None:
#                 flat[key] = list(morgan_filter(self.df, flat[key]).kept)

#         return flat



# src/txgraffiti2025/graffiti_nonlinear.py
# if __name__ == "__main__":
#     # Example usage, guarded for safety.
#     from txgraffiti2025.graffiti_relations import GraffitiClassRelations
#     from txgraffiti2025.graffiti_lp import GraffitiLP

#     # Example: use one of your datasets
#     from txgraffiti.example_data import graph_data as df

#     # Add a new boolean column (as you suggested)
#     df["nontrivial"] = df["connected"]

#     # 1. Create the base class relations (GCR)
#     gcr = GraffitiClassRelations(df)
#     print("=== GraffitiClassRelations summary ===")
#     print("Boolean columns:", gcr.boolean_cols)
#     print("Expr columns:", gcr.expr_cols)
#     print("Base hypothesis:", gcr.base_hypothesis_name)

#     # 2. Create the nonlinear wrapper
#     gn = GraffitiNonlinear(gcr)

#     # 3. Inspect its attributes
#     print("\n=== GraffitiNonlinear summary ===")
#     print(gn)  # triggers __repr__

#     print("\nAttributes:")
#     print("  DataFrame rows:", len(gn.df))
#     print("  Expr columns:", gn.expr_columns[:10])
#     print("  Base predicates:", list(gn.base_predicates.keys()))
#     print("  Base hypothesis:", gn.base_hypothesis_name)

#     # 4. Example: show how masks are delegated
#     print("\nMask for base hypothesis:")
#     base_mask = gn.get_condition_mask(None)
#     print("  Sum =", base_mask.sum(), "of", len(base_mask))

#     # 5. Example: view predicate masks
#     for name, pred in list(gn.hypotheses)[:10]:
#         mask = gn._mask(pred)
#         print(f"  {name:20s} → {mask.sum():4d} true rows")

#     # from txgraffiti2025.processing.post import morgan_filter


#     # TARGET = 'independence_number'
#     # target_expr = ColumnTerm(TARGET)
#     # min_touch = 3
#     # other_invariants = [invariant for invariant in gn.invariants if repr(invariant) != TARGET]

#     # lower_bounds = []
#     # upper_bounds = []
#     # equals = []
#     # for invar1, invar2 in combinations(other_invariants, 2):
#     #     for name, pred in list(gn.hypotheses)[:10]:
#     #         y = df[TARGET].to_numpy()
#     #         X = np.column_stack((invar1.eval(df), invar2.eval(df)))

#     #         a_lo, b_lo = _solve_sum_slack_lp(X, y, sense="lower")
#     #         rhs_lo = _affine_expr(a_lo, [invar1, invar2], b_lo, 30, 0.0000001)
#     #         conjecture = Conjecture(relation=Ge(target_expr, rhs_lo), condition=pred)
#     #         touch_count = conjecture.touch_count(df)
#     #         if touch_count == sum(pred(df)):
#     #             conjecture = Conjecture(relation=Eq(target_expr, rhs_lo), condition=pred)
#     #             conjecture.touch = touch_count
#     #             equals.append(conjecture)
#     #             equals = list(set(equals))
#     #         elif touch_count > min_touch:
#     #             lower_bounds.append(conjecture)
#     #             lower_bounds = list(set(lower_bounds))


#     #         a_hi, b_hi = _solve_sum_slack_lp(X, y, sense="upper")
#     #         rhs_hi = _affine_expr(a_hi, [invar1, invar2], b_hi, 30, 0.0000001)
#     #         conjecture = Conjecture(relation=Le(target_expr, rhs_hi), condition=pred)
#     #         touch_count = conjecture.touch_count(df)
#     #         if touch_count == sum(pred(df)):
#     #             conjecture = Conjecture(relation=Eq(target_expr, rhs_hi), condition=pred)
#     #             conjecture.touch = touch_count
#     #             equals.append(conjecture)
#     #             equals = list(set(equals))
#     #         elif touch_count > min_touch:
#     #             upper_bounds.append(conjecture)
#     #             upper_bounds = list(set(upper_bounds))


#     # equals.sort(reverse=True, key = lambda c : c.touch)
#     # lower_bounds.sort(reverse=True, key = lambda c : c.touch)
#     # upper_bounds.sort(reverse=True, key = lambda c : c.touch)

#     # lower_bounds = morgan_filter(df, lower_bounds)
#     # upper_bounds = morgan_filter(df, upper_bounds)
#     # equals = morgan_filter(df, equals)

#     # for i, conj in enumerate(lower_bounds.kept, 1):
#     #     print(f'Conjecture {i}. {conj.pretty()}. touch count = {conj.touch}')

#     # print()
#     # for i, conj in enumerate(upper_bounds.kept, 1):
#     #     print(f'Conjecture {i}. {conj.pretty()}. touch count = {conj.touch}')

#     # print()
#     # for i, conj in enumerate(equals.kept, 1):
#     #     print(f'Conjecture {i}. {conj.pretty()}. touch count = {conj.touch}')

#     res = gn.generate_k2_affine_bounds(
#             target="independence_number",
#             hypotheses_limit=20,
#             min_touch=3,
#             max_denominator=30,
#             tol=1e-9,
#             touch_atol=0.0,
#             touch_rtol=0.0,
#             apply_morgan_filter=True,
#         )

#     print("\n=== LOWERS (top 15) ===")
#     for i, c in enumerate(res["lowers"][:15], 1):
#         print(f"{i}. {c.pretty()}  | touches={getattr(c,'touch',0)}")

#     print("\n=== UPPERS (top 15) ===")
#     for i, c in enumerate(res["uppers"][:15], 1):
#         print(f"{i}. {c.pretty()}  | touches={getattr(c,'touch',0)}")

#     print("\n=== EQUALS (top 15) ===")
#     for i, c in enumerate(res["equals"][:15], 1):
#         print(f"{i}. {c.pretty()}  | touches={getattr(c,'touch',0)}")

    # k = 1 and 2 together, grouped by k
    # res_by_k = gn.generate_k_affine_bounds(
    #     target="independence_number",
    #     k=[1, 2],
    #     hypotheses_limit=10,
    #     min_touch=3,
    #     max_denominator=30,
    #     touch_atol=0.0,
    #     touch_rtol=0.0,
    #     apply_morgan_filter=True,
    #     return_by_k=False,
    # )

    # # print top 3 per bucket for k=2
    # for i, c in enumerate(res_by_k["lowers"][:15], 1):
    #     print(f"[k=2 LOWER {i}] {c.pretty()} | touches={getattr(c,'touch',0)}")


    # for i, c in enumerate(res_by_k["uppers"][:15], 1):
    #     print(f"[k=2 LOWER {i}] {c.pretty()} | touches={getattr(c,'touch',0)}")


    # for i, c in enumerate(res_by_k["equals"][:15], 1):
    #     print(f"[k=2 LOWER {i}] {c.pretty()} | touches={getattr(c,'touch',0)}")



# if __name__ == "__main__":
#     # Example usage, guarded for safety.
#     from txgraffiti2025.graffiti_relations import GraffitiClassRelations
#     from txgraffiti2025.graffiti_lp import GraffitiLP, GenerateK2Config, LPFitConfig

#     # Example dataset (replace with your own DataFrame)
#     from txgraffiti.example_data import graph_data as df

#     # Example: add a boolean column
#     df["nontrivial"] = df["connected"]

#     # 1. Create the base class relations (GCR)
#     gcr = GraffitiClassRelations(df)
#     print("=== GraffitiClassRelations summary ===")
#     print("Boolean columns:", gcr.boolean_cols)
#     print("Expr columns:", gcr.expr_cols)
#     print("Base hypothesis:", gcr.base_hypothesis_name)

#     gcr.enumerate_conjunctions(max_arity=2)
#     nonred, red, equiv = gcr.find_redundant_conjunctions()

#     print("\nNonredundant:")
#     for n, _ in nonred:
#         print(" ", n)

#     # print("\nRedundant:")
#     # for n, _ in red:
#     #     print(" ", n)

#     # print("\nEquivalent groups:")
#     # for group in equiv:
#     #     names = [n for (n, _) in group]
#     #     print("  {" + ", ".join(names) + "}")

#     # Atomic conjectures
#     atomic = gcr.build_constant_conjectures(tol=0.0, group_per_hypothesis=False)
#     for c in atomic[:5]:
#         print(c.pretty())
#         print()

#     # Grouped conjectures
#     grouped = gcr.build_constant_conjectures(tol=0.0, group_per_hypothesis=True)
#     for c in grouped[:5]:
#         print(c.pretty())
#         print()

#     # Class characterization
#     res = gcr.characterize_constant_classes(tol=0.0, group_per_hypothesis=True, limit=50)
#     gcr.print_class_characterization_summary()

#     # 2. Create the LP fitter
#     lp = GraffitiLP(gcr)

#     # 3. Option A: Single affine fit example
#     cfg = LPFitConfig(
#         target="total_domination_number",
#         features=["domination_number", "order", "minimum_degree"],
#         direction="both",
#         max_denom=20,
#         tol=1e-9,
#         touch_atol=1e-9,
#         touch_rtol=0.0,
#     )
#     lowers, uppers, equals = lp.fit_affine(cfg)
#     print(f"\n=== Single affine fit ===")
#     for c in lowers + uppers + equals:
#         print(c.pretty(show_tol=False), f"| touches={getattr(c, 'touch_count', 0)}")

#     print()
#     res = lp.generate_k_affine_bounds(
#         target="independence_number",
#         k_values=(1,2,3),
#         hypotheses_limit=20,
#         min_touch=3,
#         max_denom=20,
#         touch_atol=1e-9,
#         touch_rtol=0.0,
#         top_m_by_variance=10,

#     )

#     print("\n[LOWERS]", len(res["lowers"]))
#     for c in res["lowers"][:8]:
#         print(" ", c.pretty(show_tol=False),
#             f"| touches={getattr(c, 'touch_count', 0)}",
#             f"| support={getattr(c, 'support_n', 0)}")

#     print("\n[UPPERS]", len(res["uppers"]))
#     for c in res["uppers"][:8]:
#         print(" ", c.pretty(show_tol=False),
#             f"| touches={getattr(c, 'touch_count', 0)}",
#             f"| support={getattr(c, 'support_n', 0)}")

#     print("\n[EQUALS]", len(res["equals"]))
#     for c in res["equals"][:8]:
#         print(" ", c.pretty(show_tol=False),
#             f"| touches={getattr(c, 'touch_count', 0)}",
#             f"| support={getattr(c, 'support_n', 0)}")

#     from txgraffiti2025.graffiti_lp_lift_integer_aware import lift_integer_aware


#     low0, up0, eq0 = res["lowers"], res["uppers"], res["equals"]

#     low1 = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=low0, touch_atol=0.0, touch_rtol=0.0)
#     up1  = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=up0,  touch_atol=0.0, touch_rtol=0.0)
#     # (Eq isn’t integer-lifted here)

#     # print(f"Before vs After (lowers): {len(low0)} → {len(low1)}")
#     # print(f"Before vs After (uppers): {len(up0)} → {len(up1)}")

#     # print()
#     # # show first improved example
#     for old, new in zip(low0, low1):
#         if old.signature() != new.signature():
#             # print("OLD:", old.pretty(), "| touches=", getattr(old, "touch_count", "?"))
#             print("Conjecture:", new.pretty(), "| touches=", getattr(new, "touch_count", "?"))




#     from txgraffiti2025.graffiti_intricate_mixed import GraffitiLPIntricate
#     from txgraffiti.example_data import graph_data as df


#     # Initialize the engine
#     lp = GraffitiLPIntricate(df)

#     print("=== GraffitiLPIntricate summary ===")
#     print("Boolean columns:", lp.bool_columns)
#     print("Numeric columns:", lp.numeric_columns[:8], "...")
#     print("Base hypothesis:", lp.base_hyp)


#     def _fmt(c):
#         try:
#             return c.pretty(show_tol=False)
#         except Exception:
#             return str(c)

#     def show_intricate_sample(res: dict, k: int = 8):
#         print("=== Lower bounds (top few) ===")
#         for c in res["lowers"][:k]:
#             print("•", _fmt(c))

#         print("\n=== Upper bounds (top few) ===")
#         for c in res["uppers"][:k]:
#             print("•", _fmt(c))

#         print("\n=== Equalities (top few) ===")
#         for c in res["equals"][:k]:
#             print("•", _fmt(c))

#     # example:
#     res = lp.run_intricate_mixed_pipeline(
#         target_col="independence_number",
#         weight=0.5,
#         min_touch=3,
#     )
#     show_intricate_sample(res, k=10)


# === TXGRAFFITI SHOWCASE SCRIPT ===
# Clean terminal demo with section separators and spacing.


# ───────────────────────── Conjecture Collector/Finalizer ───────────────────────── #
# ────────────────────────────────────────────────────────────────
# Tee all prints to a UTF-8 report file (nice headers, borders)
# ────────────────────────────────────────────────────────────────
from __future__ import annotations
import sys, os, io, datetime as _dt
from contextlib import contextmanager

class _Tee(io.TextIOBase):
    """Write-through stream that mirrors writes to multiple targets."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)
    def flush(self):
        for st in self._streams:
            st.flush()

def _hr(ch: str = "─", n: int = 80) -> str:
    return ch * n

def _now_stamp() -> str:
    # America/Chicago assumed by your environment; adjust if needed.
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@contextmanager
def tee_report(filepath: str, *, title: str = "TxGraffiti Run Report", include_stderr: bool = True):
    """
    Duplicate all prints to `filepath` (UTF-8). Adds a nice header/footer.
    Usage:
        with tee_report("reports/run.txt", title="My Run"):
            ... your existing code full of print() ...
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    f = open(filepath, "w", encoding="utf-8", newline="\n")
    # header
    header = (
        f"{_hr()}\n"
        f"{title}\n"
        f"{_hr()}\n"
        f"Started: {_now_stamp()}\n"
        f"Working dir: {os.getcwd()}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"{_hr()}\n"
    )
    f.write(header); f.flush()

    old_out = sys.stdout
    old_err = sys.stderr
    try:
        sys.stdout = _Tee(old_out, f)
        if include_stderr:
            sys.stderr = _Tee(old_err, f)
        yield
    finally:
        # footer
        print(_hr())
        print("END OF REPORT")
        print(f"Finished: {_now_stamp()}")
        print(_hr())
        sys.stdout.flush()
        if include_stderr:
            sys.stderr.flush()
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()



from typing import Sequence, Tuple, Union, Optional, Dict, List, Iterable

import numpy as np
import pandas as pd


from txgraffiti2025.graffiti_relations import GraffitiClassRelations
from txgraffiti2025.forms.utils import Expr, ColumnTerm, Const
from txgraffiti2025.forms.predicates import Predicate, Where
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le, Eq, TRUE

def _as_number(v, default=None):
    if v is None:
        return default
    try:
        v = v() if callable(v) else v
        if hasattr(v, "item") and callable(getattr(v, "item")):
            v = v.item()
        return int(v) if isinstance(v, (int, bool)) else float(v)
    except Exception:
        return default

def _mask_for(df: pd.DataFrame, H) -> np.ndarray:
    if H is None or H is TRUE:
        return np.ones(len(df), dtype=bool)
    s = H.mask(df) if hasattr(H, "mask") else H(df)
    return np.asarray(s, dtype=bool)

def _compute_touch_support_batch(df: pd.DataFrame, conjs: Sequence[Conjecture],
                                 rtol: float = 1e-9, atol: float = 1e-9) -> Dict[int, tuple[int,int,float]]:
    """
    Batch compute (touch_count, support_n, touch_rate) for all conjectures.
    Groups by (H, L, R) so we eval each triplet once.
    Returns: {index -> (tc, sup, rate)}
    """
    if not conjs:
        return {}
    # local caches
    def _pred_key(p):
        if p is None or p is TRUE: return "TRUE"
        n = getattr(p, "name", None)
        return f"name:{n}" if n else f"repr:{repr(p)}"
    def _expr_key(e): return repr(e)

    groups: dict[tuple[str, str, str], list[int]] = {}
    for i, c in enumerate(conjs):
        groups.setdefault((_pred_key(c.condition),
                           _expr_key(c.relation.left),
                           _expr_key(c.relation.right)), []).append(i)

    # reusable caches
    mcache = {}
    arrcache = {}
    def _mask(pred):
        k = _pred_key(pred)
        if k in mcache: return mcache[k]
        m = _mask_for(df, pred)
        mcache[k] = m
        return m
    def _arr(expr):
        k = _expr_key(expr)
        a = arrcache.get(k)
        if a is None:
            s = expr.eval(df)
            if hasattr(s, "to_numpy"):
                a = s.to_numpy(dtype=float, copy=False)
            else:
                a = np.asarray(s, dtype=float)
                if a.ndim == 0:
                    a = np.full(len(df), float(a), dtype=float)
            arrcache[k] = a
        return a

    out: Dict[int, tuple[int,int,float]] = {}
    for (Hk, Lk, Rk), idxs in groups.items():
        rep = conjs[idxs[0]]
        Hm = _mask(rep.condition)
        if not np.any(Hm):
            for j in idxs:
                out[j] = (0, 0, 0.0)
            continue
        L = _arr(rep.relation.left)
        R = _arr(rep.relation.right)
        ok = np.isfinite(L) & np.isfinite(R) & Hm
        sup = int(ok.sum())
        if sup == 0:
            tc, rate = 0, 0.0
        else:
            eq = np.isclose(L[ok], R[ok], rtol=rtol, atol=atol)
            tc = int(eq.sum())
            rate = float(tc / sup)
        for j in idxs:
            out[j] = (tc, sup, rate)
    return out

def _dedup_by_string(conjs: Sequence[Conjecture]) -> list[Conjecture]:
    seen, out = set(), []
    for c in conjs:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        out.append(c)
    return out

def _annotate_touch_support(df: pd.DataFrame, conjs: list[Conjecture]) -> None:
    stats = _compute_touch_support_batch(df, conjs)
    for i, c in enumerate(conjs):
        tc, sup, rate = stats.get(i, (0, 0, 0.0))
        setattr(c, "touch_count", int(_as_number(getattr(c, "touch_count", tc), default=tc) or tc))
        setattr(c, "support_n",  int(_as_number(getattr(c, "support_n",  sup), default=sup) or sup))
        setattr(c, "touch_rate", float(_as_number(getattr(c, "touch_rate", rate), default=rate) or rate))

def _sort_by_touch_support(conjs: list[Conjecture]) -> list[Conjecture]:
    def key(c):
        return (int(getattr(c, "touch_count", 0) or 0),
                int(getattr(c, "support_n", 0) or 0))
    return sorted(conjs, key=key, reverse=True)

def _pretty_safe(c) -> str:
    try:
        return c.pretty(show_tol=False)
    except Exception:
        return str(c)

def print_bank(bank: dict[str, list[Conjecture]], k_per_bucket: int = 10, title: str = "FULL CONJECTURE LIST"):
    def section(title: str):
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80 + "\n")
    section(title)
    for name in ("lowers", "uppers", "equals"):
        lst = bank.get(name, [])
        print(f"[{name.upper()}] total={len(lst)}\n")
        for c in lst[:k_per_bucket]:
            print("•", _pretty_safe(c))
            print(f"    touches={getattr(c, 'touch_count', '?')}, support={getattr(c, 'support_n', '?')}")
        print()

def finalize_conjecture_bank(
    df: pd.DataFrame,
    all_lowers: list[Conjecture],
    all_uppers: list[Conjecture],
    all_equals: list[Conjecture],
    *,
    top_k_per_bucket: int = 100,
    apply_morgan: bool = True,
) -> dict[str, list[Conjecture]]:
    """
    1) Dedup per bucket,
    2) annotate touches/support (batch),
    3) sort by (touch_count, support_n),
    4) (optional) Morgan filter per bucket,
    5) take top_k_per_bucket per bucket,
    6) return dict and print summary.
    """
    from txgraffiti2025.processing.post import morgan_filter

    lowers = _dedup_by_string(all_lowers)
    uppers = _dedup_by_string(all_uppers)
    equals = _dedup_by_string(all_equals)

    # annotate
    _annotate_touch_support(df, lowers)
    _annotate_touch_support(df, uppers)
    _annotate_touch_support(df, equals)

    # sort
    lowers = _sort_by_touch_support(lowers)
    uppers = _sort_by_touch_support(uppers)
    equals = _sort_by_touch_support(equals)

    # optional Morgan filter (validity/pruning)
    if apply_morgan:
        lowers = list(morgan_filter(df, lowers).kept)
        uppers = list(morgan_filter(df, uppers).kept)
        equals = list(morgan_filter(df, equals).kept)

        # re-annotate after Morgan (in case of any recomputation)
        _annotate_touch_support(df, lowers)
        _annotate_touch_support(df, uppers)
        _annotate_touch_support(df, equals)

        lowers = _sort_by_touch_support(lowers)
        uppers = _sort_by_touch_support(uppers)
        equals = _sort_by_touch_support(equals)

    # take top-k
    lowers = lowers[:top_k_per_bucket]
    uppers = uppers[:top_k_per_bucket]
    equals = equals[:top_k_per_bucket]

    bank = {"lowers": lowers, "uppers": uppers, "equals": equals}
    return bank



# if __name__ == "__main__":
#     import textwrap
#     from txgraffiti2025.graffiti_relations import GraffitiClassRelations
#     from txgraffiti2025.graffiti_lp import GraffitiLP, LPFitConfig
#     from txgraffiti2025.graffiti_lp_lift_integer_aware import lift_integer_aware
#     from txgraffiti2025.graffiti_intricate_mixed import GraffitiLPIntricate
#     from txgraffiti.example_data import graph_data as df

#     def section(title: str):
#         print("\n" + "-" * 80)
#         print(f"{title.upper()}")
#         print("-" * 80 + "\n")

#     def print_conjs(label: str, conjs, n=6):
#         print(f"[{label.upper()}] total={len(conjs)}\n")
#         for c in conjs[:n]:
#             print("•", c.pretty(show_tol=False))
#             t = getattr(c, "touch_count", "?")
#             s = getattr(c, "support_n", "?")
#             print(f"    touches={t}, support={s}\n")

#     # ────────────────────────────────────────────────────────────────
#     # Dataset setup
#     # ────────────────────────────────────────────────────────────────
#     df["nontrivial"] = df["connected"]

#     # === 1. CLASS RELATIONS ===
#     section("GraffitiClassRelations summary")

#     gcr = GraffitiClassRelations(df)
#     print("Boolean columns:", gcr.boolean_cols)
#     print("Expr columns:", gcr.expr_cols)
#     print("Base hypothesis:", gcr.base_hypothesis_name)

#     gcr.enumerate_conjunctions(max_arity=2)
#     nonred, _, _ = gcr.find_redundant_conjunctions()
#     print("\nNonredundant conjunctions:")
#     for n, _ in nonred[:8]:
#         print(" ", n)
#     print()

#     atomic = gcr.build_constant_conjectures(tol=0.0, group_per_hypothesis=False)
#     print("Example atomic conjectures:\n")
#     for c in atomic[:5]:
#         print(textwrap.indent(c.pretty(), "  "))
#         print()

#     gcr.characterize_constant_classes(tol=0.0, group_per_hypothesis=True, limit=50)
#     gcr.print_class_characterization_summary()

#     # === 2. AFFINE FITS ===
#     section("Affine LP fits")

#     lp = GraffitiLP(gcr)
#     cfg = LPFitConfig(
#         target="total_domination_number",
#         features=["domination_number", "order", "minimum_degree"],
#         direction="both",
#         max_denom=20,
#     )
#     lowers, uppers, equals = lp.fit_affine(cfg)
#     print_conjs("lower", lowers)
#     print_conjs("upper", uppers)
#     print_conjs("equal", equals)

#     # === 3. K-AFFINE BOUNDS ===
#     section("K2 affine bounds generation")

#     res = lp.generate_k_affine_bounds(
#         target="independence_number",
#         k_values=(1, 2, 3),
#         hypotheses_limit=20,
#         min_touch=3,
#         max_denom=20,
#         top_m_by_variance=10,
#     )

#     print_conjs("lower", res["lowers"])
#     print_conjs("upper", res["uppers"])
#     print_conjs("equal", res["equals"])

#     # === 4. INTEGER LIFTING ===
#     section("Integer-aware lifting")

#     low1 = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=res["lowers"])
#     up1 = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=res["uppers"])
#     print(f"Before vs After (lowers): {len(res['lowers'])} → {len(low1)}")
#     print(f"Before vs After (uppers): {len(res['uppers'])} → {len(up1)}\n")

#     for old, new in zip(res["lowers"], low1):
#         if old.signature() != new.signature():
#             print("Refined:", new.pretty(), f"| touches={getattr(new, 'touch_count', '?')}\n")

#     # === 5. INTRICATE MIXED BOUNDS ===
#     section("Intricate mixed inequalities")

#     lp_intr = GraffitiLPIntricate(df)
#     print("Boolean columns:", lp_intr.bool_columns)
#     print("Numeric columns:", lp_intr.numeric_columns[:8], "...")
#     print("Base hypothesis:", lp_intr.base_hyp, "\n")

#     res_intr = lp_intr.run_intricate_mixed_pipeline(
#         target_col="independence_number",
#         weight=0.5,
#         min_touch=3,
#     )

#     print_conjs("lower", res_intr["lowers"])
#     print_conjs("upper", res_intr["uppers"])
#     print_conjs("equal", res_intr["equals"])

#     print("\n" + "-" * 80)
#     print("END OF SHOWCASE")
#     print("-" * 80)
if __name__ == "__main__":
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = f"reports/txgraffiti_run_{stamp}.txt"

    with tee_report(report_path, title="TxGraffiti • Full Program Run Report"):
        # ────────────────────────────────────────────────────────────────
        # your whole block, unchanged:
        # (section(), print_conjs(), all prints, etc. will go to console + file)
        # ────────────────────────────────────────────────────────────────

        import textwrap
        import pandas as pd
        from txgraffiti2025.graffiti_relations import GraffitiClassRelations
        from txgraffiti2025.graffiti_lp import GraffitiLP, LPFitConfig
        from txgraffiti2025.graffiti_lp_lift_integer_aware import lift_integer_aware
        from txgraffiti2025.graffiti_intricate_mixed import GraffitiLPIntricate
        # from txgraffiti.example_data import graph_data as df

        df = pd.read_csv('number_theory_1_to_500.csv')
        # integers_basic.py
        # import math
        # import pandas as pd

        # def _smallest_divisor_ge2(n: int) -> int:
        #     """Return the smallest d≥2 with n % d == 0, or n if none exists (i.e., n is prime).
        #     Uses only arithmetic + mod; checks up to floor(sqrt(n))."""
        #     if n % 2 == 0:
        #         return 2
        #     d = 3
        #     # check odd divisors up to sqrt(n)
        #     while d * d <= n:
        #         if n % d == 0:
        #             return d
        #         d += 2
        #     return n

        # def _tau_sigma(n: int) -> tuple[int, int]:
        #     """Compute tau(n) (#divisors) and sigma(n) (sum of divisors) via mod checks only."""
        #     t = 0
        #     s = 0
        #     r = int(math.isqrt(n))
        #     for d in range(1, r + 1):
        #         if n % d == 0:
        #             q = n // d
        #             if q == d:
        #                 t += 1
        #                 s += d
        #             else:
        #                 t += 2
        #                 s += d + q
        #     return t, s

        # def integers_dataset_basic(N: int = 500) -> pd.DataFrame:
        #     """
        #     Build a dataset over n=1..N using only:
        #     - +, -, *, //, %, integer sqrt
        #     No parity/prime/special predicates—just raw arithmetic signals that let
        #     your miner *discover* classes as equality/ratio structures.
        #     """
        #     rows = []
        #     for n in range(1, N + 1):
        #         one = 1

        #         # Euclidean building blocks: quotients & remainders for small moduli
        #         q2, r2 = n // 2, n % 2
        #         q3, r3 = n // 3, n % 3
        #         q4, r4 = n // 4, n % 4
        #         q5, r5 = n // 5, n % 5
        #         q6, r6 = n // 6, n % 6
        #         q7, r7 = n // 7, n % 7
        #         q8, r8 = n // 8, n % 8
        #         q9, r9 = n // 9, n % 9
        #         q10, r10 = n // 10, n % 10

        #         # divisibility structure from pure mod-arithmetic
        #         sdiv = _smallest_divisor_ge2(n)        # = n if “prime”; else smallest nontrivial factor
        #         ldiv = n // sdiv                        # = 1 if “prime”; else largest proper factor

        #         tau, sigma = _tau_sigma(n)              # counts/sums by divisor testing
        #         proper_sum = sigma - n                  # “aliquot sum” (but purely arithmetic)

        #         # square structure from integer sqrt
        #         r_sq = n - (math.isqrt(n) ** 2)        # = 0 for perfect squares, >0 otherwise

        #         rows.append({
        #             "n": n,
        #             # "integer": one,

        #             # congruence scaffolding (lets miner find even/odd, mod-k classes)
        #             "q2": q2, "r2": r2,
        #             "q3": q3, "r3": r3,
        #             "q4": q4, "r4": r4,
        #             "q5": q5, "r5": r5,
        #             "q6": q6, "r6": r6,
        #             "q7": q7, "r7": r7,
        #             "q8": q8, "r8": r8,
        #             "q9": q9, "r9": r9,
        #             "q10": q10, "r10": r10,

        #             # factor signals (no prime flags; miner can discover from equalities)
        #             "sdiv": sdiv,               # = n on “primes”
        #             "ldiv": ldiv,               # = 1 on “primes”
        #             "tau": tau,                 # = 2 on “primes”
        #             "sigma": sigma,             # = n + 1 on “primes”
        #             "proper_sum": proper_sum,   # = 1 on “primes”

        #             # square signal (no boolean): equality class r_sq = 0
        #             "r_sq": r_sq,

        #             # a few clean ratios (float) to help your ratio miner
        #             "tau_over_n": tau / n,
        #             "sigma_over_n": sigma / n,
        #             "sdiv_over_n": sdiv / n,
        #             "ldiv_over_n": ldiv / n,
        #         })

        #     return pd.DataFrame.from_records(rows)

        # df = integers_dataset_basic()

        # ────────────────────────────────────────────────────────────────
        # Small local printers (compatible with collector at top)
        # ────────────────────────────────────────────────────────────────
        def section(title: str):
            print("\n" + "-" * 80)
            print(f"{title.upper()}")
            print("-" * 80 + "\n")

        def print_conjs(label: str, conjs, n=6):
            print(f"[{label.upper()}] total={len(conjs)}\n")
            for c in conjs[:n]:
                try:
                    s = c.pretty(show_tol=False)
                except Exception:
                    s = str(c)
                print("•", s)
                t = getattr(c, "touch_count", "?")
                s_n = getattr(c, "support_n", "?")
                print(f"    touches={t}, support={s_n}\n")

        # ────────────────────────────────────────────────────────────────
        # 0) ACCUMULATORS for everything we generate along the way
        # ────────────────────────────────────────────────────────────────
        ALL_LOWERS, ALL_UPPERS, ALL_EQUALS = [], [], []

        # ────────────────────────────────────────────────────────────────
        # Dataset setup
        # ────────────────────────────────────────────────────────────────
        # df["nontrivial"] = df["connected"]
        # df.drop(columns=['cograph'], inplace=True)

        from txgraffiti2025.graffiti_relations import GraffitiClassRelations
        from txgraffiti2025.graffiti_qualitative import GraffitiQualitative
        # from txgraffiti.example_data import graph_data as df

        gcr = GraffitiClassRelations(df)
        qual = GraffitiQualitative(gcr)

        TARGET = "n"
        results = qual.generate_qualitative_relations(
            y_targets=[TARGET],      # or None for all numeric targets
            # x_candidates=["order","size","radius"], # or None for all numeric candidates
            method="spearman",
            min_abs_rho=0.4,
            min_n=12,
            top_k_per_hyp=5,
        )

        GraffitiQualitative.print_sample(results, k=12)


        from txgraffiti2025.graffiti_relations import GraffitiClassRelations
        from txgraffiti2025.graffiti_asymptotic import (
            lim_to_infinity, lim_to_zero, lim_to_const, lim_ratio_const
        )
        from fractions import Fraction

        # gcr = GraffitiClassRelations(df)

        # # Example hypothesis: base or a named conjunction
        # H = gcr.base_hypothesis  # or some Predicate from gcr.nonredundant_conjunctions_

        # # 1) lim_{order→∞} independence_number = ∞
        # cj1 = lim_to_infinity(f=TARGET, t="order", condition=H)

        # # 2) lim_{order→∞} independence_number / order = 1/4
        # cj2 = lim_ratio_const(f=TARGET, g="order", t="order", c=Fraction(1,4), condition=H)

        # # 3) lim_{order→∞} algebraic_connectivity = 0
        # cj3 = lim_to_zero(f="algebraic_connectivity", t="order", condition=H)

        # print(cj1.pretty())
        # print(cj2.pretty_latex())


        from txgraffiti2025.graffiti_asymptotic_miner import AsymptoticMiner, AsymptoticSearchConfig

        section("Asymptotic limits (qualitative → definite)")
        miner = AsymptoticMiner(gcr, cfg=AsymptoticSearchConfig(
            min_abs_rho=0.45, tail_quantile=0.75, min_support_n=20
        ))

        asym_conjs = miner.generate_asymptotics_for_target(
            target=TARGET,
            # t_candidates=["order", "size", "diameter"],        # or omit to use all numeric
            # ratio_denominators=["order", "size"],               # optional; omit for all numeric
            hyps=getattr(gcr, "nonredundant_conjunctions_", None),
        )

        print(f"[ASYMPTOTIC] total={len(asym_conjs)}\n")
        for c in asym_conjs[:10]:
            print("•", c.pretty())

        # === 1. CLASS RELATIONS ===
        section("GraffitiClassRelations summary")

        gcr = GraffitiClassRelations(df)
        print("Boolean columns:", gcr.boolean_cols)
        print("Expr columns:", gcr.expr_cols)
        print("Base hypothesis:", gcr.base_hypothesis_name)

        gcr.enumerate_conjunctions(max_arity=2)
        nonred, _, _ = gcr.find_redundant_conjunctions()
        print("\nNonredundant conjunctions:")
        for n, _ in nonred[:8]:
            print(" ", n)
        print()

        atomic = gcr.build_constant_conjectures(tol=0.0, group_per_hypothesis=False)
        print("Example atomic conjectures:\n")
        for c in atomic[:5]:
            print(textwrap.indent(c.pretty(), "  "))
            print()

        gcr.characterize_constant_classes(tol=0.0, group_per_hypothesis=True, limit=50)
        gcr.print_class_characterization_summary()

        # (Optional) If atomic/build_constant_conjectures return Conjecture objects,
        # you can add them to the global lists too:
        # ALL_EQUALS.extend([c for c in atomic if isinstance(c.relation, Eq)])

        # === 2. AFFINE FITS ===
        section("Affine LP fits")


        lp = GraffitiLP(gcr)
        features = [c for c in lp.invariants if c != TARGET]
        cfg = LPFitConfig(
            target=TARGET,
            # features= ['mu', 'liouville', 'omega', 'Omega', 'rad', 'spf', 'lpf', 'v2', 'popcount', 'aliquot_sum', 'abundancy_index'], #["domination_number", "order", "minimum_degree"],
            features = features,
            direction="both",
            max_denom=20,
        )
        lowers, uppers, equals = lp.fit_affine(cfg)
        print_conjs("lower", lowers)
        print_conjs("upper", uppers)
        print_conjs("equal", equals)

        # Collect
        ALL_LOWERS.extend(lowers)
        ALL_UPPERS.extend(uppers)
        ALL_EQUALS.extend(equals)

        # === 3. K-AFFINE BOUNDS ===
        section("K2 affine bounds generation")

        res = lp.generate_k_affine_bounds(
            target=TARGET,
            k_values=(1, 2, 3),
            hypotheses_limit=20,
            min_touch=3,
            max_denom=20,
            top_m_by_variance=10,
        )

        print_conjs("lower", res["lowers"])
        print_conjs("upper", res["uppers"])
        print_conjs("equal", res["equals"])

        # Collect
        ALL_LOWERS.extend(res["lowers"])
        ALL_UPPERS.extend(res["uppers"])
        ALL_EQUALS.extend(res["equals"])

        # === 4. INTEGER LIFTING ===
        section("Integer-aware lifting")

        low1 = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=res["lowers"])
        up1  = lift_integer_aware(df=lp.df, gcr=lp.gcr, conjectures=res["uppers"])
        print(f"Before vs After (lowers): {len(res['lowers'])} → {len(low1)}")
        print(f"Before vs After (uppers): {len(res['uppers'])} → {len(up1)}\n")

        for old, new in zip(res["lowers"], low1):
            if old.signature() != new.signature():
                print("Refined:", new.pretty(), f"| touches={getattr(new, 'touch_count', '?')}\n")

        # Collect the refined ones (you can keep the originals too if you want)
        ALL_LOWERS.extend(low1)
        ALL_UPPERS.extend(up1)

        # === 5. INTRICATE MIXED BOUNDS ===
        section("Intricate mixed inequalities")

        lp_intr = GraffitiLPIntricate(df)
        print("Boolean columns:", lp_intr.bool_columns)
        print("Numeric columns:", lp_intr.numeric_columns[:8], "...")
        print("Base hypothesis:", lp_intr.base_hyp, "\n")

        res_intr = lp_intr.run_intricate_mixed_pipeline(
            target_col=TARGET,
            weight=0.5,
            min_touch=3,
        )

        print_conjs("lower", res_intr["lowers"])
        print_conjs("upper", res_intr["uppers"])
        print_conjs("equal", res_intr["equals"])

        # Collect
        ALL_LOWERS.extend(res_intr["lowers"])
        ALL_UPPERS.extend(res_intr["uppers"])
        ALL_EQUALS.extend(res_intr["equals"])

        # ────────────────────────────────────────────────────────────────
        # REINITIALIZE WITH EQUALITY-CLASS HYPOTHESES (auto-conjunctions by GCR)
        # ────────────────────────────────────────────────────────────────
        section("Equality-class hypotheses bootstrap → rebuild engines")

        import numpy as np

        # --- 1) Mine equality classes on the base hypothesis ---
        summary, _base_conjs = gcr.analyze_ratio_bounds_on_base(
            min_support=0.05,
            positive_denominator=True,
            touch_atol=0.0,
            touch_rtol=0.0,
        )

        def _top_equality_classes(gcr_obj, eq_summary, *, k=12, min_rows=25, side="auto"):
            tmp = eq_summary.copy()
            if tmp.empty:
                return []
            tmp["best_rate"] = np.where(tmp["touch_lower_rate"] >= tmp["touch_upper_rate"],
                                        tmp["touch_lower_rate"], tmp["touch_upper_rate"])
            tmp["best_side"] = np.where(tmp["touch_lower_rate"] >= tmp["touch_upper_rate"], "lower", "upper")
            tmp["score"] = tmp["best_rate"] * np.log1p(tmp["n_rows"])
            tmp = tmp[(tmp["n_rows"] >= min_rows)]
            if tmp.empty:
                return []
            tmp["_pairkey"] = tmp.apply(lambda r: tuple(sorted([r["inv1"], r["inv2"]])), axis=1)
            tmp = tmp.sort_values(["score"], ascending=False).drop_duplicates("_pairkey", keep="first")
            sel = tmp.head(k)

            picks = []
            for _, row in sel.iterrows():
                which = row["best_side"] if side == "auto" else side
                name, pred = gcr_obj.spawn_equality_classes_from_ratio_row(row, which=which, tol=0.0)
                picks.append((name, pred, float(row["best_rate"]), int(row["n_rows"])))
            return picks

        top_eqs = _top_equality_classes(gcr, summary, k=12, min_rows=25, side="auto")

        # --- 2) Build a new dataframe that keeps only *base* booleans + add equality-class booleans ---
        # Determine which booleans are always-true (the base)
        base_true_cols = []
        for c in gcr.boolean_cols:
            m = gcr.predicates[c].mask(df)
            if bool(m.all()):
                base_true_cols.append(c)

        # Keep: base booleans + all numeric/other columns
        keep_cols = base_true_cols + [c for c in df.columns if c not in gcr.boolean_cols]
        df_eq = df[keep_cols].copy()

        # Add equality-class hypothesis columns
        # (GCR will auto-enumerate conjunctions of these with the base)
        # def _safe_colname(name: str) -> str:
        #     # make a readable, collision-safe boolean column name from "A = k·B"
        #     return (name
        #             .replace(" ", "_")
        #             .replace("·", "x")
        #             .replace("=", "_eq_")
        #             .replace("/", "_over_")
        #             .replace("(", "")
        #             .replace(")", "")
        #             )

        print("Selected equality classes:")
        for name, pred, rate, nrows in top_eqs:
            mask = gcr._mask_cached(pred)
            colname = name
            df_eq[colname] = mask
            print(f"  • {name:40s} (tightness≈{rate:.3f}, n={nrows})  → added as boolean '{colname}'")

        # --- 3) Rebuild engines on df_eq (GCR will find new conjunctions automatically) ---
        gcr_eq = GraffitiClassRelations(df_eq)
        lp_eq = GraffitiLP(gcr_eq)
        lp_intr_eq = GraffitiLPIntricate(df_eq)

        features = [c for c in lp_eq.invariants if c != TARGET]
        print("\nBoolean columns in rebuilt frame (includes equality-class hypotheses):")
        print(gcr_eq.boolean_cols)

        # --- 4) Rerun pipelines *unchanged* under the enriched hypothesis space ---
        section("Affine LP fits (equality-class hypotheses)")
        cfg_eq = LPFitConfig(
            target=TARGET,
            # features= ['mu', 'liouville', 'omega', 'Omega', 'rad', 'spf', 'lpf', 'v2', 'popcount', 'aliquot_sum', 'abundancy_index'],#["domination_number", "order", "minimum_degree"],
            # features = [
            #     "n", "one",
            #     "q2", "r2", "q3", "r3", "q4", "r4", "q5", "r5",
            #     "sdiv", "ldiv", "tau", "sigma", "proper_sum", "r_sq",
            #     "tau_over_n", "sigma_over_n", "sdiv_over_n", "ldiv_over_n"
            # ],
            features=features,
            direction="both",
            max_denom=20,
        )
        lowers_eq, uppers_eq, equals_eq = lp_eq.fit_affine(cfg_eq)
        print_conjs("lower", lowers_eq)
        print_conjs("upper", uppers_eq)
        print_conjs("equal", equals_eq)

        ALL_LOWERS.extend(lowers_eq)
        ALL_UPPERS.extend(uppers_eq)
        ALL_EQUALS.extend(equals_eq)

        section("K2 affine bounds (equality-class hypotheses)")
        res_eq = lp_eq.generate_k_affine_bounds(
            target=TARGET,
            k_values=(1, 2, 3),
            hypotheses_limit=20,
            min_touch=3,
            max_denom=20,
            top_m_by_variance=10,
        )
        print_conjs("lower", res_eq["lowers"])
        print_conjs("upper", res_eq["uppers"])
        print_conjs("equal", res_eq["equals"])

        ALL_LOWERS.extend(res_eq["lowers"])
        ALL_UPPERS.extend(res_eq["uppers"])
        ALL_EQUALS.extend(res_eq["equals"])

        section("Intricate mixed inequalities (equality-class hypotheses)")
        res_intr_eq = lp_intr_eq.run_intricate_mixed_pipeline(
            target_col=TARGET,
            weight=0.5,
            min_touch=3,
        )
        print_conjs("lower", res_intr_eq["lowers"])
        print_conjs("upper", res_intr_eq["uppers"])
        print_conjs("equal", res_intr_eq["equals"])

        ALL_LOWERS.extend(res_intr_eq["lowers"])
        ALL_UPPERS.extend(res_intr_eq["uppers"])
        ALL_EQUALS.extend(res_intr_eq["equals"])


        # ────────────────────────────────────────────────────────────────
        # 6) GLOBAL FINALIZATION (dedup → annotate touches/support → rank → Morgan → top-K)
        # Uses the helper utilities you pasted at the top.
        # ────────────────────────────────────────────────────────────────
        FINAL = finalize_conjecture_bank(
            df+df_eq,
            ALL_LOWERS, ALL_UPPERS, ALL_EQUALS,
            top_k_per_bucket=100,  # tweak to your liking
            apply_morgan=True,
        )

        # Nice final summary printout
        print_bank(FINAL, k_per_bucket=12, title="FULL CONJECTURE LIST (DEDUPED • RANKED • TOP-K)")

        print("\n" + "-" * 80)
        print("END OF SHOWCASE")
        print("-" * 80)
        print(f"\n📄 Report saved to: {report_path}")
