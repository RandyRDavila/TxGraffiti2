# from __future__ import annotations
# import numpy as np
# import pandas as pd
# from fractions import Fraction

# from txgraffiti.example_data import graph_data as df

# # Hypotheses
# from txgraffiti2025.processing.pre.hypotheses import (
#     enumerate_boolean_hypotheses,
#     detect_base_hypothesis,
# )
# from txgraffiti2025.processing.pre.simplify_hypotheses import (
#     simplify_and_dedup_hypotheses,
# )

# # Forms
# from txgraffiti2025.forms.utils import to_expr, Expr, Const
# from txgraffiti2025.forms.generic_conjecture import Conjecture
# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.forms.logexp import log_base, sqrt
# from txgraffiti2025.forms.nonlinear import square

# import numpy as np

# from txgraffiti2025.forms.utils import (
#     Expr, Const, BinOp, to_expr, floor, ceil, sqrt
# )
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le


# from txgraffiti2025.forms.utils import to_expr, LinearForm, floor, log, sqrt, ceil
# from txgraffiti2025.forms.generic_conjecture import (
#     Eq, Le, Ge, AllOf, AnyOf, Conjecture, TRUE
# )

# # Generators
# from txgraffiti2025.generators.lp import lp_bounds, LPConfig

# from txgraffiti2025.processing.post import morgan_filter

# # Post: constant finder + generalizers + refinement
# from txgraffiti2025.processing.post.constant_ratios import (
#     extract_ratio_pattern,
#     find_constant_ratios_over_hypotheses,
# )
# from txgraffiti2025.processing.post.generalize_from_constants import (
#     constants_from_ratios_hits,
#     propose_generalizations_from_constants,
# )
# from txgraffiti2025.processing.post.intercept_generalizer import (
#     propose_generalizations_from_intercept,
# )
# from txgraffiti2025.processing.post.refine_numeric import (
#     refine_numeric_bounds, RefinementConfig
# )

# def _touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
#     """Count how many rows achieve equality lhs == rhs (within tolerance) on conj.condition."""
#     try:
#         lhs = conj.relation.left.eval(df)
#         rhs = conj.relation.right.eval(df)
#         mask = conj.condition.mask(df) if conj.condition else np.ones(len(df), bool)
#         touch = np.isclose(lhs[mask], rhs[mask], rtol=1e-8, atol=1e-8)
#         return int(np.sum(touch))
#     except Exception:
#         return 0

# # --- helpers ---------------------------------------------------------------

# from itertools import combinations, product
# from tqdm.auto import tqdm

# def _all_le_on_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> bool:
#     """Check a <= b on rows where mask & both finite."""
#     m = mask & np.isfinite(a) & np.isfinite(b)
#     if not np.any(m):
#         return False
#     return bool(np.all(a[m] <= b[m]))

# def _caro_trivial_masked(W_arr, x_arr, y_arr, z_arr, mask) -> bool:
#     """
#     Trivial if (W<=y & x<=z) OR (W<=z & x<=y) on rows where mask & all four finite.
#     """
#     common = mask & np.isfinite(W_arr) & np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
#     if not np.any(common):
#         return False
#     c1 = _all_le_on_mask(W_arr, y_arr, common) and _all_le_on_mask(x_arr, z_arr, common)
#     if c1:
#         return True
#     c2 = _all_le_on_mask(W_arr, z_arr, common) and _all_le_on_mask(x_arr, y_arr, common)
#     return c2

# # ---------- helpers: floor/ceil variants over Exprs ----------
# def to_frac_const(val: float, max_denom: int = 30):
#     """Convert float to Const(Fraction) with a bounded denominator."""
#     return Const(Fraction(val).limit_denominator(max_denom))

# from fractions import Fraction
# import numpy as np

# # cache container to avoid re-evaluating expressions repeatedly
# class EvalCache:
#     def __init__(self, df_temp, cols):
#         self.df = df_temp
#         self.cols = cols
#         self.arr = {}      # raw column arrays
#         self.sqrt_arr = {} # sqrt(column) arrays
#         self.sq_arr = {}   # column**2 arrays

#     def col(self, name):
#         a = self.arr.get(name)
#         if a is None:
#             s = to_expr(name).eval(self.df)
#             a = s.values.astype(float, copy=False)
#             self.arr[name] = a
#         return a

#     def sqrt_col(self, name):
#         a = self.sqrt_arr.get(name)
#         if a is None:
#             x = self.col(name)
#             # skip invalid (≤0) upstream, so safe:
#             a = np.sqrt(x, dtype=float)
#             self.sqrt_arr[name] = a
#         return a

#     def sq_col(self, name):
#         a = self.sq_arr.get(name)
#         if a is None:
#             x = self.col(name)
#             a = np.square(x, dtype=float)
#             self.sq_arr[name] = a
#         return a


# def _expr_floor(e: Expr) -> Expr:
#     # Wrap with floor() from your utils API
#     try:
#         return floor(e)
#     except Exception:
#         return None

# def _expr_ceil(e: Expr) -> Expr:
#     try:
#         return ceil(e)
#     except Exception:
#         return None

# def _is_add(e: Expr) -> bool:
#     # In your BinOp, the operator is stored as .fn == np.add
#     return isinstance(e, BinOp) and getattr(e, "fn", None) is np.add

# def _split_add(e: Expr):
#     # returns (left, right) if e == left + right else None
#     if _is_add(e):
#         return e.left, e.right
#     return None

# def _all_true(conj: Conjecture, df_frame) -> bool:
#     try:
#         return conj.is_true(df_frame)
#     except Exception:
#         return False

# def _strength_for_true_ge(rhs_expr: Expr, df_frame) -> float:
#     # For target ≥ RHS: larger RHS is stronger
#     try:
#         vals = rhs_expr.eval(df_frame)
#         return float(vals.mean())
#     except Exception:
#         return float("-inf")

# def _strength_for_true_le(rhs_expr: Expr, df_frame) -> float:
#     # For target ≤ RHS: smaller RHS is stronger (use negative mean)
#     try:
#         vals = rhs_expr.eval(df_frame)
#         return -float(vals.mean())
#     except Exception:
#         return float("-inf")

# def _ge_best(target: Expr, base_rhs: Expr, hypothesis, df_frame):
#     """
#     Given target ≥ base_rhs, try floor/ceil (whole expr) and sum-split combos,
#     keep the strongest true one.
#     """
#     candidates = [base_rhs]

#     # whole-expression floor/ceil
#     f_base = _expr_floor(base_rhs)
#     c_base = _expr_ceil(base_rhs)
#     if f_base is not None: candidates.append(f_base)  # weaker but sometimes the only true one
#     if c_base is not None: candidates.append(c_base)  # stronger candidate

#     # sum decomposition
#     split = _split_add(base_rhs)
#     if split is not None:
#         a, b = split
#         fa, fb = _expr_floor(a), _expr_floor(b)
#         ca, cb = _expr_ceil(a), _expr_ceil(b)

#         # ceil(a)+ceil(b)-1  ≤ ceil(a+b)  (good, often tighter lower bound)
#         if ca is not None and cb is not None:
#             try:
#                 candidates.append((ca + cb) - Const(1))
#             except Exception:
#                 pass

#         # floor(a)+floor(b) ≤ floor(a+b) ≤ a+b  (usually weaker; include as fallback)
#         if fa is not None and fb is not None:
#             try:
#                 candidates.append(fa + fb)
#             except Exception:
#                 pass

#     best_rhs, best_score = None, float("-inf")
#     for rhs in candidates:
#         conj = Conjecture(Ge(target, rhs), hypothesis)
#         if _all_true(conj, df_frame):
#             score = _strength_for_true_ge(rhs, df_frame)
#             if score > best_score:
#                 best_rhs, best_score = rhs, score

#     if best_rhs is None:
#         best_rhs = base_rhs
#     return Conjecture(Ge(target, best_rhs), hypothesis)

# def _le_best(target: Expr, base_rhs: Expr, hypothesis, df_frame):
#     """
#     Given target ≤ base_rhs, try floor/ceil (whole expr) and sum-split combos,
#     keep the strongest true one.
#     """
#     candidates = [base_rhs]

#     # whole-expression floor/ceil (floor is tighter for ≤)
#     f_base = _expr_floor(base_rhs)
#     c_base = _expr_ceil(base_rhs)
#     if f_base is not None: candidates.append(f_base)  # stronger for ≤
#     if c_base is not None: candidates.append(c_base)  # looser fallback

#     split = _split_add(base_rhs)
#     if split is not None:
#         a, b = split
#         fa, fb = _expr_floor(a), _expr_floor(b)
#         ca, cb = _expr_ceil(a), _expr_ceil(b)

#         # floor(a)+floor(b) ≤ floor(a+b) ≤ a+b  (tends to be tighter/smaller)
#         if fa is not None and fb is not None:
#             try:
#                 candidates.append(fa + fb)
#             except Exception:
#                 pass

#         # ceil(a)+ceil(b) ≥ ceil(a+b) ≥ a+b (looser; keep as a fallback)
#         if ca is not None and cb is not None:
#             try:
#                 candidates.append(ca + cb)
#             except Exception:
#                 pass

#     best_rhs, best_score = None, float("-inf")
#     for rhs in candidates:
#         conj = Conjecture(Le(target, rhs), hypothesis)
#         if _all_true(conj, df_frame):
#             score = _strength_for_true_le(rhs, df_frame)
#             if score > best_score:
#                 best_rhs, best_score = rhs, score

#     if best_rhs is None:
#         best_rhs = base_rhs
#     return Conjecture(Le(target, best_rhs), hypothesis)

# def _pick_best_ge(t_arr, rhs_variants, strength='mean'):
#     """
#     target >= rhs. rhs_variants: list of (label, rhs_array, make_expr_fn).
#     Returns the chosen (label, make_expr_fn).
#     """
#     best = None
#     best_score = -np.inf
#     for lab, rhs, make_expr in rhs_variants:
#         ok = np.all(t_arr >= rhs)
#         if not ok:
#             continue
#         score = np.mean(rhs) if strength == 'mean' else np.median(rhs)
#         if score > best_score:
#             best = (lab, make_expr)
#             best_score = score
#     return best

# def _pick_best_le(t_arr, rhs_variants, strength='mean'):
#     """
#     target <= rhs. Prefer *smaller* rhs among those true.
#     """
#     best = None
#     best_score = -np.inf
#     for lab, rhs, make_expr in rhs_variants:
#         ok = np.all(t_arr <= rhs)
#         if not ok:
#             continue
#         # smaller rhs is stronger → use negative mean/median
#         score = -np.mean(rhs) if strength == 'mean' else -np.median(rhs)
#         if score > best_score:
#             best = (lab, make_expr)
#             best_score = score
#     return best


# class TxGraffiti:
#     def __init__(self, df : pd.DataFrame):
#         self.df = df
#         self.base = detect_base_hypothesis(df)
#         self.hyps_all = enumerate_boolean_hypotheses(
#             df,
#             treat_binary_ints=True,
#             include_base=True,
#             include_pairs=True,
#             skip_always_false=True,
#         )
#         self._simplify_and_dedup_hypotheses()
#         self._set_columns()

#     def _simplify_and_dedup_hypotheses(self):
#         self.hyps_kept, _ = simplify_and_dedup_hypotheses(
#             self.df,
#             self.hyps_all,
#             min_support=10,
#             treat_binary_ints=True,
#             )

#     def _set_columns(self):
#         bool_cols = []
#         for c in df.columns:
#             s = df[c]
#             if s.dtype == bool:
#                 bool_cols.append(c)
#             elif pd.api.types.is_integer_dtype(s):
#                 vals = pd.unique(s.dropna())
#                 try: ints = set(int(v) for v in vals)
#                 except Exception: continue
#                 if len(ints) <= 2 and ints.issubset({0,1}):
#                     bool_cols.append(c)
#         self.bool_columns = bool_cols
#         self.numeric_columns = [
#             c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
#             and c not in bool_cols
#         ]




# df.drop(columns=['cograph', 'eulerian', 'chordal', 'vertex_cover_number'], inplace=True)

# # TARGET = 'zero_forcing_number'
# ai  = TxGraffiti(df)
# target = to_expr(TARGET)
# upper_conjectures = []
# lower_conjectures = []


# for hypothesis in ai.hyps_kept:
#     for other in ai.numeric_columns:
#         if other != TARGET:
#             df_temp = df[hypothesis(df)]
#             x = to_expr(other)
#             if x.eval(df_temp).min() <= 0:
#                 pass
#             else:
#                 cmin, cmax = (target/x).eval(df_temp).min(), (target/x).eval(df_temp).max()
#                 cmin = Fraction(cmin).limit_denominator(30)
#                 lower_bound = Ge(target, cmin*x)
#                 lower_conj = Conjecture(lower_bound, hypothesis)
#                 lower_conjectures.append(lower_conj)
#                 cmax = Fraction(cmax).limit_denominator(30)
#                 upper_bound = Le(target, cmax*x)
#                 upper_conj = Conjecture(upper_bound, hypothesis)
#                 upper_conjectures.append(upper_conj)


#                 for other2 in ai.numeric_columns:

#                     x2 = to_expr(other2)
#                     if x2.eval(df_temp).min() <= 0:
#                         pass
#                     else:
#                         sqrt_x = sqrt(other2)
#                         sqrt_x_cmin, sqrt_x_cmax = (target/sqrt_x).eval(df_temp).min(), (target/sqrt_x).eval(df_temp).max()
#                         sqrt_x_cmin = Fraction(sqrt_x_cmin).limit_denominator(30)
#                         sqrt_x_cmax = Fraction(sqrt_x_cmax).limit_denominator(30)

#                         lower_bound = Ge(target, (cmin/2)*x + (sqrt_x_cmin/2)*sqrt_x)
#                         lower_conj = Conjecture(lower_bound, hypothesis)

#                         better_lower_bound1 = Ge(target, ceil((cmin/2)*x + (sqrt_x_cmin/2)*sqrt_x))
#                         better_lower_conj1 = Conjecture(better_lower_bound, hypothesis)

#                         better_lower_bound2 = Ge(target, ceil((cmin/2)*x) + ceil((sqrt_x_cmin/2)*sqrt_x)) - 1
#                         better_lower_conj2 = Conjecture(better_lower_bound, hypothesis)

#                         if better_lower_conj1.is_true(df):
#                             lower_conjectures.append(better_lower_conj1)
#                         elif better_lower_conj2.is_true(df):
#                             lower_conjectures.append(better_lower_conj1)
#                         else:
#                             lower_conjectures.append(lower_conj)

#                         upper_bound = Le(target, (cmax/2)*x + (sqrt_x_cmax/2)*sqrt_x)
#                         better_upper_bound = Le(target, floor((cmax/2)*x + (sqrt_x_cmax/2)*sqrt_x))
#                         upper_conj = Conjecture(upper_bound, hypothesis)
#                         better_upper_conj = Conjecture(better_lower_bound, hypothesis)
#                         if better_upper_conj.is_true(df):
#                             upper_conjectures.append(better_upper_conj)
#                         else:
#                             upper_conjectures.append(upper_conj)


#                     sq_x = square(other2)
#                     sq_cmin, sq_cmax = (target/sq_x).eval(df_temp).min(), (target/sq_x).eval(df_temp).max()
#                     sq_cmin = Fraction(sq_cmin).limit_denominator(30)
#                     lower_bound = Ge(target, (cmin/2)*x + (sq_cmin/2)*sq_x)
#                     lower_conj = Conjecture(lower_bound, hypothesis)
#                     lower_conjectures.append(lower_conj)

#                     sq_cmax = Fraction(cmax).limit_denominator(30)
#                     upper_bound = Le(target, (cmax/2)*x + (sq_cmax/2)*sq_x)
#                     upper_conj = Conjecture(upper_bound, hypothesis)
#                     upper_conjectures.append(upper_conj)

# # upper_conjectures.sort(reverse=True, key = lambda c : _touch_count(c, df))
# # upper_conjectures = [c for c in upper_conjectures if _touch_count(c, df) > 2]
# # morgan_filtered_upper = morgan_filter(df, upper_conjectures)

# # from fractions import Fraction
# # import numpy as np
# # from tqdm.auto import tqdm

# # # assumes: to_frac_const, EvalCache, TxGraffiti, to_expr, floor, ceil, sqrt,
# # #          Conjecture, Ge, Le, _pick_best_ge, _pick_best_le are already imported/defined

# TARGET = 'independence_number'
# ai = TxGraffiti(df)
# target = to_expr(TARGET)

# upper_conjectures = []
# lower_conjectures = []

# # progress over hypotheses
# for hypothesis in tqdm(ai.hyps_kept, desc="Hypotheses", unit="hyp"):
#     mask = hypothesis(df)
#     df_temp = df[mask]
#     if df_temp.empty:
#         continue

#     # cache arrays for this hypothesis
#     cache = EvalCache(df_temp, ai.numeric_columns)
#     t_arr = target.eval(df_temp).values.astype(float, copy=False)

#     # progress over primary columns
#     for other in tqdm(ai.numeric_columns, desc="Primary columns", unit="col", leave=False):
#         if other == TARGET:
#             continue

#         x_arr = cache.col(other)
#         if np.min(x_arr) <= 0:
#             continue

#         # ratio constants (floats first)
#         rx = t_arr / x_arr
#         cmin_f = float(np.min(rx))
#         cmax_f = float(np.max(rx))

#         # ---------- ALWAYS EMIT BASIC LINEAR CONJECTURES ----------
#         cmin_frac = Fraction(cmin_f).limit_denominator(30)
#         cmax_frac = Fraction(cmax_f).limit_denominator(30)

#         # y ≥ cmin * x
#         lower_conjectures.append(
#             Conjecture(Ge(target, to_frac_const(cmin_f) * to_expr(other)), hypothesis)
#         )

#         # y ≤ cmax * x
#         base_upper_expr = to_frac_const(cmax_f) * to_expr(other)
#         upper_conjectures.append(
#             Conjecture(Le(target, base_upper_expr), hypothesis)
#         )

#         # y ≤ floor(cmax * x)  (only if it actually remains true)
#         floor_arr = np.floor(cmax_f * x_arr)
#         if np.all(t_arr <= floor_arr) and cmax_frac.denominator > 1:
#             upper_conjectures.append(
#                 Conjecture(Le(target, floor(base_upper_expr)), hypothesis)
#             )

#         # y ≥ ceil(cmin * x)  (only if true)
#         ceil_arr = np.ceil(cmin_f * x_arr)
#         if np.all(t_arr >= ceil_arr) and cmin_frac.denominator > 1:
#             lower_conjectures.append(
#                 Conjecture(Ge(target, ceil(to_frac_const(cmin_f) * to_expr(other))), hypothesis)
#             )

#         # ---------- TWO-FEATURE MIXES: distributed 1/2 ----------
#         # (Try x with sqrt(x2) and x with (x2)^2)
#         for other2 in tqdm(ai.numeric_columns, desc="Secondary columns", unit="col", leave=False):
#             x2_arr = cache.col(other2)
#             if np.min(x2_arr) <= 0 or other2 == TARGET:
#                 continue

#             # ==== sqrt mix ====
#             sqrt_x2_arr = cache.sqrt_col(other2)
#             r_sqrt = t_arr / sqrt_x2_arr
#             s_cmin_f = float(np.min(r_sqrt))
#             s_cmax_f = float(np.max(r_sqrt))

#             # arrays for truth tests (array math can keep the 0.5 outside)
#             mix_lower_arr = 0.5 * (cmin_f * x_arr + s_cmin_f * sqrt_x2_arr)
#             mix_upper_arr = 0.5 * (cmax_f * x_arr + s_cmax_f * sqrt_x2_arr)

#             # --- ≥ variants (choose best true one) ---
#             lower_mix_variants = [
#                 # BASE (distributed 1/2): (1/2*cmin)*x + (1/2*s_cmin)*sqrt(x2)
#                 ("base",
#                  mix_lower_arr,
#                  lambda: (to_frac_const(0.5 * cmin_f) * to_expr(other)
#                           + to_frac_const(0.5 * s_cmin_f) * sqrt(to_expr(other2)))),

#                 # CEIL WHOLE (distributed inside):
#                 ("ceil whole",
#                  np.ceil(mix_lower_arr),
#                  lambda: ceil(
#                      to_frac_const(0.5 * cmin_f) * to_expr(other)
#                      + to_frac_const(0.5 * s_cmin_f) * sqrt(to_expr(other2))
#                  )),

#                 # CEIL-SPLIT-1: ceil(1/2*cmin*x) + ceil(1/2*s_cmin*sqrt(x2)) - 1
#                 ("ceil-split-1",
#                  np.ceil(0.5 * cmin_f * x_arr) + np.ceil(0.5 * s_cmin_f * sqrt_x2_arr) - 1.0,
#                  lambda: (ceil(to_frac_const(0.5 * cmin_f) * to_expr(other))
#                           + ceil(to_frac_const(0.5 * s_cmin_f) * sqrt(to_expr(other2)))
#                           - Const(1))),
#             ]
#             choice = _pick_best_ge(t_arr, lower_mix_variants)
#             if choice is not None:
#                 _, make_expr = choice
#                 lower_conjectures.append(Conjecture(Ge(target, make_expr()), hypothesis))

#             # --- ≤ variants (choose best true one) ---
#             upper_mix_variants = [
#                 # BASE (distributed 1/2)
#                 ("base",
#                  mix_upper_arr,
#                  lambda: (to_frac_const(0.5 * cmax_f) * to_expr(other)
#                           + to_frac_const(0.5 * s_cmax_f) * sqrt(to_expr(other2)))),

#                 # FLOOR WHOLE (distributed inside)
#                 ("floor whole",
#                  np.floor(mix_upper_arr),
#                  lambda: floor(
#                      to_frac_const(0.5 * cmax_f) * to_expr(other)
#                      + to_frac_const(0.5 * s_cmax_f) * sqrt(to_expr(other2))
#                  )),

#                 # FLOOR-SPLIT: floor(1/2*cmax*x) + floor(1/2*s_cmax*sqrt(x2))
#                 ("floor-split",
#                  np.floor(0.5 * cmax_f * x_arr) + np.floor(0.5 * s_cmax_f * sqrt_x2_arr),
#                  lambda: (floor(to_frac_const(0.5 * cmax_f) * to_expr(other))
#                           + floor(to_frac_const(0.5 * s_cmax_f) * sqrt(to_expr(other2))))),
#             ]
#             choice = _pick_best_le(t_arr, upper_mix_variants)
#             if choice is not None:
#                 _, make_expr = choice
#                 upper_conjectures.append(Conjecture(Le(target, make_expr()), hypothesis))

#             # ==== square mix ====
#             sq_x2_arr = cache.sq_col(other2)
#             r_sq = t_arr / sq_x2_arr
#             sq_cmin_f = float(np.min(r_sq))
#             sq_cmax_f = float(np.max(r_sq))

#             mix_lower_sq_arr = 0.5 * (cmin_f * x_arr + sq_cmin_f * sq_x2_arr)
#             mix_upper_sq_arr = 0.5 * (cmax_f * x_arr + sq_cmax_f * sq_x2_arr)

#             # ≥ variants
#             lower_sq_variants = [
#                 ("base",
#                  mix_lower_sq_arr,
#                  lambda: (to_frac_const(0.5 * cmin_f) * to_expr(other)
#                           + to_frac_const(0.5 * sq_cmin_f) * (to_expr(other2) ** to_frac_const(2)))),

#                 ("ceil whole",
#                  np.ceil(mix_lower_sq_arr),
#                  lambda: ceil(
#                      to_frac_const(0.5 * cmin_f) * to_expr(other)
#                      + to_frac_const(0.5 * sq_cmin_f) * (to_expr(other2) ** to_frac_const(2))
#                  )),
#             ]
#             choice = _pick_best_ge(t_arr, lower_sq_variants)
#             if choice is not None:
#                 _, make_expr = choice
#                 lower_conjectures.append(Conjecture(Ge(target, make_expr()), hypothesis))

#             # ≤ variants
#             upper_sq_variants = [
#                 ("base",
#                  mix_upper_sq_arr,
#                  lambda: (to_frac_const(0.5 * cmax_f) * to_expr(other)
#                           + to_frac_const(0.5 * sq_cmax_f) * (to_expr(other2) ** to_frac_const(2)))),

#                 ("floor whole",
#                  np.floor(mix_upper_sq_arr),
#                  lambda: floor(
#                      to_frac_const(0.5 * cmax_f) * to_expr(other)
#                      + to_frac_const(0.5 * sq_cmax_f) * (to_expr(other2) ** to_frac_const(2))
#                  )),
#             ]
#             choice = _pick_best_le(t_arr, upper_sq_variants)
#             if choice is not None:
#                 _, make_expr = choice
#                 upper_conjectures.append(Conjecture(Le(target, make_expr()), hypothesis))

# # rank + filter as before
# upper_conjectures.sort(reverse=True, key=lambda c: _touch_count(c, df))
# upper_conjectures = [c for c in upper_conjectures if _touch_count(c, df) > 2]
# morgan_filtered_upper = morgan_filter(df, upper_conjectures)


# lower_conjectures.sort(reverse=True, key = lambda c : _touch_count(c, df))
# lower_conjectures = [c for c in lower_conjectures if _touch_count(c, df) > 2]
# morgan_filtered_lower = morgan_filter(df, lower_conjectures)


# from itertools import combinations
# import numpy as np
# from tqdm.auto import tqdm


# # ---- main loop -------------------------------------------------------------

# from itertools import combinations
# import numpy as np
# from tqdm.auto import tqdm

# def _all_le_on_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> bool:
#     m = mask & np.isfinite(a) & np.isfinite(b)
#     if not np.any(m):
#         return False
#     return bool(np.all(a[m] <= b[m]))

# def _caro_trivial_masked(W_arr, x_arr, y_arr, z_arr, mask) -> bool:
#     common = mask & np.isfinite(W_arr) & np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
#     if not np.any(common):
#         return False
#     c1 = _all_le_on_mask(W_arr, y_arr, common) and _all_le_on_mask(x_arr, z_arr, common)
#     if c1:
#         return True
#     c2 = _all_le_on_mask(W_arr, z_arr, common) and _all_le_on_mask(x_arr, y_arr, common)
#     return c2

# def _strictly_positive_on_mask(a: np.ndarray, mask: np.ndarray) -> bool:
#     m = mask & np.isfinite(a)
#     if not np.any(m):
#         return False
#     return bool(np.all(a[m] > 0))

# # --- helpers for overlap/cancellation ---------------------------------------

# def _strictly_pos_on(a: np.ndarray, mask: np.ndarray) -> bool:
#     m = mask & np.isfinite(a)
#     return np.any(m) and bool(np.all(a[m] > 0))

# def _strictly_neg_on(a: np.ndarray, mask: np.ndarray) -> bool:
#     m = mask & np.isfinite(a)
#     return np.any(m) and bool(np.all(a[m] < 0))

# def _reduce_by_common_factor(W, x, y, z, arrays, mask):
#     """
#     If exactly one common factor exists and is strictly positive or negative on mask:
#       return (reduced_L_name, reduced_R_name, relation) where relation is 'le' or 'ge'.
#     If two commons (same multiset) -> tautology -> return None
#     If sign mixed/zero -> return None (skip)
#     If no common -> return ('Wx','yz','le') to indicate no reduction (use products).
#     """
#     lhs_set, rhs_set = {W, x}, {y, z}
#     common = lhs_set & rhs_set
#     if len(common) == 2:
#         return None  # tautology, skip
#     if len(common) == 0:
#         return ('Wx', 'yz', 'le')  # no reduction

#     # exactly one common
#     f = next(iter(common))
#     f_arr = arrays[f]
#     if _strictly_pos_on(f_arr, mask):
#         # cancel, keep ≤
#         if f == W:
#             Lother = x
#         elif f == x:
#             Lother = W
#         if f == y:
#             Rother = z
#         elif f == z:
#             Rother = y
#         return (Lother, Rother, 'le')

#     if _strictly_neg_on(f_arr, mask):
#         # cancel, flip to ≥
#         if f == W:
#             Lother = x
#         elif f == x:
#             Lother = W
#         if f == y:
#             Rother = z
#         elif f == z:
#             Rother = y
#         return (Lother, Rother, 'ge')

#     # mixed sign or zeros: skip this pair
#     return None


# product_upper_conjectures = []

# for hypothesis in tqdm(ai.hyps_kept, desc="Hypotheses", unit="hyp"):
#     hyp_mask = hypothesis(df)
#     df_temp = df[hyp_mask]
#     if df_temp.empty:
#         continue

#     cand_cols = list(ai.numeric_columns)  # allow overlaps; filter if you want
#     arrays = {c: to_expr(c).eval(df_temp).values.astype(float, copy=False) for c in cand_cols}
#     finite = {c: np.isfinite(arrays[c]) for c in cand_cols}

#     # LHS pairs
#     for (W, x) in tqdm(list(combinations(cand_cols, 2)), desc="Product LHS pairs", unit="pair", leave=False):
#         W_arr = arrays[W]; x_arr = arrays[x]
#         L_arr = W_arr * x_arr
#         L_fin = finite[W] & finite[x] & np.isfinite(L_arr)
#         if not np.any(L_fin):
#             continue

#         # RHS pairs
#         for (y, z) in tqdm(list(combinations(cand_cols, 2)), desc="Product RHS pairs", unit="pair", leave=False):
#             y_arr = arrays[y]; z_arr = arrays[z]
#             R_arr = y_arr * z_arr
#             R_fin = finite[y] & finite[z] & np.isfinite(R_arr)

#             # rows where BOTH products and all 4 inputs are finite
#             both_fin = L_fin & R_fin
#             if not np.any(both_fin):
#                 continue

#             # Caro non-triviality on the SAME rows
#             if _caro_trivial_masked(W_arr, x_arr, y_arr, z_arr, both_fin):
#                 continue

#             # Final rowwise check (no over-strict min/max prune)
#             if _all_le_on_mask(L_arr, R_arr, both_fin):
#                 lhs = to_expr(W) * to_expr(x)
#                 rhs = to_expr(y) * to_expr(z)
#                 product_upper_conjectures.append(Conjecture(Le(lhs, rhs), hypothesis))

# # (Optionally) merge into your main pool and rank/filter:
# # product_upper_conjectures.extend(product_upper_conjectures)
# product_upper_conjectures.sort(reverse=True, key=lambda c: _touch_count(c, df))
# product_upper_conjectures = [c for c in product_upper_conjectures if _touch_count(c, df) > 2]
# morgan_filtered_product_conjectures= morgan_filter(df, product_upper_conjectures)

# MAX_PRINT = min(100, len(morgan_filtered_product_conjectures.kept))
# print(f"---------- Caro Conjectures ----------")
# for i, c in enumerate(morgan_filtered_product_conjectures.kept[:MAX_PRINT]):
#     if i > 0:
#         print(f"Conjecture Caro.{i}. {c.pretty()} \n")

# print()
# print()
# print()

# print(f"----------- Non-Linear Upper Conjectures ----------")
# for i, c in enumerate(morgan_filtered_upper.kept[:100], 1):
#     print(f"Conjecture NLU.{i}. {c.pretty()} \n")

# print()
# print()
# print()

# print(f"----------- Non-Linear Lower Conjectures ----------")
# for i, c in enumerate(morgan_filtered_lower.kept[:100], 1):
#     print(f"Conjecture NLL.{i}. {c.pretty()} \n")


# # from txgraffiti.example_data import graph_data as df
# # from txgraffiti2025.processing.pre.simplify_hypotheses import (
# #     simplify_predicate_via_df,
# #     simplify_and_dedup_hypotheses,
# # )
# # from txgraffiti2025.processing.pre.hypotheses import enumerate_boolean_hypotheses

# # hyps = enumerate_boolean_hypotheses(df)
# # print(f"Enumerated {len(hyps)} hypotheses")
# # for h in hyps[:5]:
# #     print("-", h)

# # h0 = hyps[0]
# # Hs, eq = simplify_predicate_via_df(h0, df)
# # print("Original:", h0)
# # print("Simplified:", Hs)
# # print("Equivalent?", eq is not None)
# # if eq:
# #     print(eq)

# # kept, eqs = simplify_and_dedup_hypotheses(df, hyps)

# # print(f"\nSimplified to {len(kept)} unique hypotheses")
# # for h in kept:
# #     print("•", h)

# # print(f"\nFound {len(eqs)} equivalence relations:")
# # for e in eqs:
# #     print("  ", e.pretty())

# 1) Parsing with non-linear feature
# smoke_test_constant_ratios.py
# import pandas as pd
# from txgraffiti2025.processing.post.constant_ratios import (
#     find_constant_ratios_over_hypotheses,
#     find_constant_ratios_for_conjecture,
#     extract_ratio_pattern,
# )
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Le
# from txgraffiti2025.forms.predicates import Predicate

# df = pd.DataFrame({
#     "N":  [1,2,3,4,5,6,7,8,9,10],
#     "D":  [2,4,6,8,10,12,14,16,18,20],      # D = 2*N  => N/D = 0.5
#     "Z":  [0,0,0,0,0,1,1,1,1,1],            # boolean-like slice
# })

# H_all = None
# H_Z1  = Predicate.from_column("Z", truthy_only=True)

# hits = find_constant_ratios_over_hypotheses(
#     df,
#     hypotheses=[H_all, H_Z1],
#     numeric_cols=["N","D"],
#     shifts=[-1,0,1],
#     constancy="cv",
#     cv_tol=0.05,
#     min_support=4,
#     max_denominator=20,
#     top_k_per_hypothesis=10,
# )

# assert len(hits) > 0
# for h in hits:
#     assert h.support >= 4
#     assert h.formula_expr is not None
#     assert h.hypothesis in (H_all, H_Z1)

# demo_post_final.py
# from __future__ import annotations

# import numpy as np
# import pandas as pd

# # --- project imports ---
# from txgraffiti2025.forms.utils import to_expr, Const, Expr
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge, TRUE
# from txgraffiti2025.forms.predicates import Predicate
# from txgraffiti2025.forms.pretty import format_expr
# from txgraffiti2025.processing.post.consolidate_intervals import consolidate_interval_chains


# # ------------ helpers for building RHS safely ------------

# def mul(x, k: float) -> Expr:
#     """Return to_expr(x) * Const(k)."""
#     return to_expr(x) * Const(float(k))

# def add(a, b) -> Expr:
#     """Return to_expr(a) + to_expr(b) if needed; accepts Expr or scalar/str."""
#     A = a if isinstance(a, Expr) else (to_expr(a) if isinstance(a, str) else Const(float(a)))
#     if isinstance(b, Expr):
#         B = b
#     elif isinstance(b, str):
#         B = to_expr(b)
#     else:
#         B = Const(float(b))
#     return A + B

# def ratio(a, b) -> Expr:
#     """Return to_expr(a) / to_expr(b) (or Const)."""
#     A = a if isinstance(a, Expr) else (to_expr(a) if isinstance(a, str) else Const(float(a)))
#     B = b if isinstance(b, Expr) else (to_expr(b) if isinstance(b, str) else Const(float(b)))
#     return A / B


# # ------------ pretty-print helpers for this demo ------------

# def fmt(e) -> str:
#     return format_expr(e) if isinstance(e, Expr) else str(e)

# def print_chain(chain):
#     # Header
#     Htxt = repr(chain.hypothesis) if chain.hypothesis is not None else "TRUE"
#     parts = []
#     for L in chain.lowers:
#         parts.append(fmt(L))
#         parts.append(" ≤ ")
#     parts.append(chain.target)
#     for U in chain.uppers:
#         parts.append(" ≤ ")
#         parts.append(fmt(U))
#     print(f"({Htxt}) ⇒ " + "".join(parts))

#     # Optional details
#     if chain.lowers:
#         print("  lowers:")
#         for e in chain.lowers:
#             print("    -", fmt(e))
#     if chain.uppers:
#         print("  uppers:")
#         for e in chain.uppers:
#             print("    -", fmt(e))
#     if chain.pieces:
#         print("  pieces:")
#         for c in chain.pieces:
#             label = c.name or c.relation.__class__.__name__
#             print("    -", label)
#     print()


# # ------------ make a tiny dataset ------------

# def make_df(n=30, seed=0):
#     rng = np.random.default_rng(seed)
#     # numeric columns
#     b = rng.integers(1, 10, size=n)
#     c = rng.integers(2, 20, size=n)
#     # a is roughly between c/2 and 2b+3 so our demo inequalities hold nicely
#     a = np.clip(rng.normal(loc=(c/2 + 2*b)/2, scale=1.5, size=n), 0, None)

#     df = pd.DataFrame({
#         "a": a.astype(float),
#         "b": b.astype(float),
#         "c": c.astype(float),
#         # boolean-ish classes
#         "connected": rng.choice([True, True, True, False], size=n),   # ~75% True
#         "bipartite": rng.choice([True, False], size=n),
#         "planar":    rng.choice([True, False], size=n),
#     })
#     return df

# # ------------ build a few conjectures ------------

# def make_conjectures(df: pd.DataFrame):
#     # Predicates (NA-safe truthy)
#     H_connected = Predicate.from_column("connected", truthy_only=True)
#     H_conn_bip = H_connected & Predicate.from_column("bipartite", truthy_only=True)
#     H_conn_planar = H_connected & Predicate.from_column("planar", truthy_only=True)

#     # a ≤ 2·b + 3  on connected
#     rhs1 = add(mul("b", 2.0), Const(3.0))
#     C1 = Conjecture(Le("a", rhs1), condition=H_connected, name="a ≤ 2b+3")

#     # a ≥ c / 2  on connected ∧ bipartite
#     rhs2 = ratio("c", 2.0)
#     C2 = Conjecture(Ge("a", rhs2), condition=H_conn_bip, name="a ≥ c/2")

#     # Another upper: a ≤ b + 6 on connected ∧ planar
#     rhs3 = add("b", Const(6.0))
#     C3 = Conjecture(Le("a", rhs3), condition=H_conn_planar, name="a ≤ b+6")

#     # A lower bound with target on the right to test normalization:
#     #  (0.4 * c) ≤ a   (i.e., a ≥ 0.4 c) on connected
#     lhs4 = mul("c", 0.4)
#     C4 = Conjecture(Le(lhs4, "a"), condition=H_connected, name="0.4c ≤ a")

#     return [C1, C2, C3, C4]


# # ------------ main demo ------------

# if __name__ == "__main__":
#     df = make_df(n=40, seed=42)
#     print("Dataset head:")
#     print(df.head(), "\n")

#     conjs = make_conjectures(df)
#     print("Conjectures:")
#     for c in conjs:
#         print(" -", c.name, "under", repr(c.condition))
#     print()

#     chains = consolidate_interval_chains(df, conjs, tol=1e-9)

#     print(f"Built {len(chains)} consolidated chains:\n")
#     for ch in chains:
#         print_chain(ch)


# txgraffiti2025/engine/txgraffiti.py
# from __future__ import annotations
# from dataclasses import dataclass, field
# from typing import List, Optional, Sequence, Tuple

# import pandas as pd
# from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

# from txgraffiti2025.processing.pre.hypotheses import (
#     enumerate_boolean_hypotheses,
#     detect_base_hypothesis,
# )
# from txgraffiti2025.processing.pre.simplify_hypotheses import (
#     simplify_and_dedup_hypotheses,
# )
# from txgraffiti2025.forms.predicates import Predicate

# from txgraffiti2025.processing.pre.constants_cache import (
#     precompute_constant_ratios,
#     precompute_constant_ratios_pairs,
#     constants_matching_coeff,
#     ConstantsCache,
#     ConstantRatio as CRItem,   # type for items stored
# )

# from txgraffiti2025.forms.logexp import sqrt     # √(Expr)
# from txgraffiti2025.forms.nonlinear import square  # (Expr)**2

# from fractions import Fraction
# import numpy as np
# import pandas as pd

# from txgraffiti2025.forms.utils import to_expr, Const, floor, ceil
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le

# from txgraffiti2025.processing.pre.constants_cache import (
#     precompute_constant_ratios,
#     precompute_constant_ratios_pairs,
#     constants_matching_coeff,
#     ConstantsCache,
# )

# def _touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
#     """Rows with lhs == rhs (within small tol) under the conjecture's condition."""
#     try:
#         lhs = np.asarray(conj.relation.left.eval(df), dtype=float)
#         rhs = np.asarray(conj.relation.right.eval(df), dtype=float)
#         mask = (conj.condition.mask(df).reindex(df.index, fill_value=False).astype(bool)
#                 if conj.condition else np.ones(len(df), bool))
#         ok = mask & np.isfinite(lhs) & np.isfinite(rhs)
#         return int(np.sum(np.isclose(lhs[ok], rhs[ok], rtol=1e-8, atol=1e-8)))
#     except Exception:
#         return 0

# def to_frac_const(val: float, max_denom: int = 30):
#     return Const(Fraction(val).limit_denominator(max_denom))



# # keep this list in one place so every run uses the same projection
# DEFAULT_GRAPH_COLS: Tuple[str, ...] = (
#     "connected", "cubic", "bipartite", "regular", "tree",
#     "domination_number", "order", "matching_number", "independence_number",
#     "maximum_degree", "minimum_degree",
# )

# from __future__ import annotations
# from dataclasses import dataclass, field
# from typing import List, Optional, Sequence, Tuple

# import numpy as np
# import pandas as pd
# from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
# from fractions import Fraction

# # Predicates / hypotheses
# from txgraffiti2025.processing.pre.hypotheses import (
#     enumerate_boolean_hypotheses,
#     detect_base_hypothesis,
# )
# from txgraffiti2025.processing.pre.simplify_hypotheses import (
#     simplify_and_dedup_hypotheses,
# )

# # Forms
# from txgraffiti2025.forms.utils import to_expr, Expr, Const, floor, ceil
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
# from txgraffiti2025.forms.predicates import Predicate

# # Constants cache
# from txgraffiti2025.processing.pre.constants_cache import (
#     ConstantsCache,
#     precompute_constant_ratios,
#     precompute_constant_ratios_pairs,
#     constants_matching_coeff,
# )

# # If you keep these in a shared module, import here.
# # Otherwise, define them locally where you instantiate the class.
# DEFAULT_GRAPH_COLS = [
#     "connected", "cubic", "bipartite", "regular", "tree",
#     "domination_number", "order", "matching_number", "independence_number",
#     "maximum_degree", "minimum_degree",
# ]


# def _to_frac_const(val: float, max_denom: int = 30) -> Const:
#     """Compact fraction → Const(Fraction), else Const(float)."""
#     try:
#         frac = Fraction(val).limit_denominator(max_denom)
#         # Always store as Const(Fraction) so pretty-printing stays rational
#         return Const(frac)
#     except Exception:
#         return Const(float(val))


# def _touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
#     """How many rows achieve equality lhs == rhs (on conjecture mask)."""
#     try:
#         lhs = conj.relation.left.eval(df)
#         rhs = conj.relation.right.eval(df)
#         mask = conj.condition.mask(df) if conj.condition else np.ones(len(df), bool)
#         touch = np.isclose(lhs[mask], rhs[mask], rtol=1e-8, atol=1e-8)
#         return int(np.sum(touch))
#     except Exception:
#         return 0

# from __future__ import annotations
# from dataclasses import dataclass, field
# from typing import List, Tuple, Optional, Sequence, Dict

# import numpy as np
# import pandas as pd
# from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
# from fractions import Fraction

# # --- forms -------------------------------------------------------------------
# from txgraffiti2025.forms.utils import (
#     Expr, Const, BinOp, to_expr, floor, ceil, sqrt,
# )
# from txgraffiti2025.forms.generic_conjecture import (
#     Conjecture, Ge, Le,
# )
# from txgraffiti2025.forms.predicates import Predicate

# # --- hypotheses + simplifying ------------------------------------------------
# from txgraffiti2025.processing.pre.hypotheses import (
#     enumerate_boolean_hypotheses,
#     detect_base_hypothesis,
# )
# from txgraffiti2025.processing.pre.simplify_hypotheses import (
#     simplify_and_dedup_hypotheses,
# )

# # --- constants cache / constant ratios miner ---------------------------------
# from txgraffiti2025.processing.pre.constants_cache import (
#     ConstantsCache,
#     precompute_constant_ratios,
#     precompute_constant_ratios_pairs,
#     constants_matching_coeff,
# )

# # --- generalizers -------------------------------------------------------------
# from txgraffiti2025.processing.post.generalize_from_constants import (
#     # function name is stable; signature may differ by branch → we wrap it
#     propose_generalizations_from_constants as PGFC,
# )
# from txgraffiti2025.processing.post.reciprocal_generalizer import (
#     propose_generalizations_from_reciprocals,
# )
# from txgraffiti2025.processing.post.intercept_generalizer import (
#     propose_generalizations_from_intercept,
# )

# # --- post: dedup/filter -------------------------------------------------------
# from txgraffiti2025.processing.post import morgan_filter


# # If you have this constant elsewhere, feel free to remove/replace this fallback.
# DEFAULT_GRAPH_COLS = (
#     "connected", "cubic", "bipartite", "regular", "tree",
#     "domination_number", "order", "matching_number",
#     "independence_number", "maximum_degree", "minimum_degree",
# )

# # -----------------------------------------------------------------------------
# # Small helpers retained from your working code
# # -----------------------------------------------------------------------------

# def _touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
#     """How many masked rows achieve equality LHS == RHS (within a tiny tolerance)."""
#     try:
#         lhs = conj.relation.left.eval(df)
#         rhs = conj.relation.right.eval(df)
#         mask = conj.condition.mask(df) if conj.condition else np.ones(len(df), bool)
#         touch = np.isclose(lhs[mask], rhs[mask], rtol=1e-8, atol=1e-8)
#         return int(np.sum(touch))
#     except Exception:
#         return 0

# def _to_frac_const(val: float, *, max_denom: int = 30) -> Const:
#     """
#     Wrap a float as a Const(Fraction) using a bounded denominator
#     so pretty-printing shows nice rational coefficients.
#     """
#     return Const(Fraction(val).limit_denominator(max_denom))


# # -----------------------------------------------------------------------------
# # The class
# # -----------------------------------------------------------------------------

# @dataclass
# class TxGraffiti:
#     df: pd.DataFrame
#     use_cols: Sequence[str] = field(default_factory=lambda: list(DEFAULT_GRAPH_COLS))

#     # discovered context
#     df_proj: pd.DataFrame = field(init=False)
#     base: Predicate = field(init=False)
#     hyps_all: List[Predicate] = field(init=False)
#     hyps_kept: List[Predicate] = field(init=False)
#     equivalences: list = field(init=False)  # ClassEquivalence objects (from simplify)

#     bool_columns: List[str] = field(init=False)
#     numeric_columns: List[str] = field(init=False)
#     constants_cache: Optional[ConstantsCache] = field(default=None, init=False)

#     # stores
#     linear_bounds: dict = field(default_factory=dict, init=False)          # {'upper': [], 'lower': []}
#     nonlinear_bounds: dict = field(default_factory=dict, init=False)       # {'upper': [], 'lower': []}
#     linear_bounds_generalized: List[Conjecture] = field(default_factory=list, init=False)

#     # ──────────────────────────────────────────────────────────────
#     # Lifecycle
#     # ──────────────────────────────────────────────────────────────
#     def __post_init__(self):
#         # 0) project
#         cols = [c for c in self.use_cols if c in self.df.columns]
#         if not cols:
#             raise ValueError("None of the requested columns exist in the DataFrame.")
#         self.df_proj = self.df.loc[:, cols].copy()

#         # 1) hypotheses: base + singles + pairs
#         self.base = detect_base_hypothesis(self.df_proj)
#         self.hyps_all = enumerate_boolean_hypotheses(
#             self.df_proj,
#             treat_binary_ints=True,
#             include_base=True,
#             include_pairs=True,
#             skip_always_false=True,
#         )

#         # 2) simplify & dedup (record equivalences)
#         self.hyps_kept, self.equivalences = simplify_and_dedup_hypotheses(
#             self.df_proj,
#             self.hyps_all,
#             min_support=10,
#             treat_binary_ints=True,
#         )

#         # 3) classify columns
#         self.bool_columns, self.numeric_columns = self._detect_columns(self.df_proj)

#         # 4) init stores
#         self._init_stores()

#     # ──────────────────────────────────────────────────────────────
#     # Column classification
#     # ──────────────────────────────────────────────────────────────
#     @staticmethod
#     def _detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
#         """
#         Boolean-like: bool dtype OR integer {0,1} (NA allowed).
#         Numeric: numeric dtypes excluding boolean-like.
#         """
#         bool_cols: List[str] = []
#         for c in df.columns:
#             s = df[c]
#             if is_bool_dtype(s):
#                 bool_cols.append(c)
#                 continue
#             if is_integer_dtype(s):
#                 vals = pd.unique(s.dropna())
#                 try:
#                     ints = {int(v) for v in vals}
#                 except Exception:
#                     continue
#                 if len(ints) <= 2 and ints.issubset({0, 1}):
#                     bool_cols.append(c)

#         numeric_cols = [
#             c for c in df.columns
#             if is_numeric_dtype(df[c]) and c not in bool_cols
#         ]
#         return bool_cols, numeric_cols

#     # ──────────────────────────────────────────────────────────────
#     # Storage
#     # ──────────────────────────────────────────────────────────────
#     def _init_stores(self):
#         if not self.linear_bounds:
#             self.linear_bounds = {"upper": [], "lower": []}
#         if not self.nonlinear_bounds:
#             self.nonlinear_bounds = {"upper": [], "lower": []}

#     # ──────────────────────────────────────────────────────────────
#     # Linear ratio bounds (pure; optional integral tightening)
#     # ──────────────────────────────────────────────────────────────
#     def make_linear_ratio_bounds(
#         self,
#         target,
#         *,
#         min_support: int = 8,
#         tighten_integral: bool = False,   # default False: keep them purely linear
#         max_denominator: int = 30,
#         verbose: bool = False,
#         equal_atol: float = 1e-12,
#         equal_rtol: float = 1e-12,
#     ) -> None:
#         """
#         For each hypothesis H and each (target, feature) with feature>0 on H:
#           emit target ≥ c_min·feature and target ≤ c_max·feature, where c_min
#           and c_max are computed on the masked slice.

#         - By default, does NOT add floor/ceil tightenings (to keep bounds linear).
#           You can set tighten_integral=True to try ceil/floor variants that remain true.
#         - Skips pairs where target and feature are (essentially) identical on H.
#         - Stores results into self.linear_bounds['upper'|'lower'].
#         """
#         self._init_stores()
#         uppers: List[Conjecture] = []
#         lowers: List[Conjecture] = []

#         num_cols = list(self.numeric_columns)
#         DF = self.df_proj

#         for H in self.hyps_kept:
#             mask = H(DF) if H is not None else np.ones(len(DF), dtype=bool)
#             if not np.any(mask):
#                 continue

#             D = DF.loc[mask]
#             colvals = {c: pd.to_numeric(D[c], errors="coerce").to_numpy(dtype=float) for c in num_cols}

#             for target in num_cols:
#                 t = colvals[target]
#                 t_ok = np.isfinite(t)
#                 if np.count_nonzero(t_ok) < min_support:
#                     continue

#                 for feature in num_cols:
#                     if feature == target:
#                         continue

#                     x = colvals[feature]
#                     both_ok = t_ok & np.isfinite(x)
#                     if np.count_nonzero(both_ok) < min_support:
#                         continue

#                     # Skip essentially identical invariants on the slice
#                     if np.allclose(t[both_ok], x[both_ok], rtol=equal_rtol, atol=equal_atol):
#                         continue

#                     # Require feature > 0 for safe ratio bounds
#                     ok = both_ok & (x > 0)
#                     if np.count_nonzero(ok) < min_support:
#                         continue

#                     r = t[ok] / x[ok]
#                     r = r[np.isfinite(r)]
#                     if r.size == 0:
#                         continue

#                     cmin = float(np.min(r))
#                     cmax = float(np.max(r))

#                     cmin_c = _to_frac_const(cmin, max_denom=max_denominator)
#                     cmax_c = _to_frac_const(cmax, max_denom=max_denominator)

#                     T = to_expr(target)
#                     X = to_expr(feature)

#                     # lower: T ≥ cmin·X
#                     ge_base = Conjecture(Ge(T, cmin_c * X), H)
#                     if ge_base.is_true(DF):
#                         chosen = ge_base
#                         if tighten_integral:
#                             ge_ceil = Conjecture(Ge(T, ceil(cmin_c * X)), H)
#                             if ge_ceil.is_true(DF):
#                                 # choose stronger RHS on the masked slice
#                                 rhs_base = (cmin * x[ok]).mean()
#                                 rhs_ceil = np.ceil(cmin * x[ok]).mean()
#                                 if rhs_ceil >= rhs_base - 1e-12:
#                                     chosen = ge_ceil
#                         lowers.append(chosen)

#                     # upper: T ≤ cmax·X
#                     le_base = Conjecture(Le(T, cmax_c * X), H)
#                     if le_base.is_true(DF):
#                         chosen = le_base
#                         if tighten_integral:
#                             le_floor = Conjecture(Le(T, floor(cmax_c * X)), H)
#                             if le_floor.is_true(DF):
#                                 rhs_base = (cmax * x[ok]).mean()
#                                 rhs_floor = np.floor(cmax * x[ok]).mean()
#                                 if rhs_floor <= rhs_base + 1e-12:
#                                     chosen = le_floor
#                         uppers.append(chosen)

#         # order by touch count, then keep
#         uppers.sort(key=lambda c: _touch_count(c, DF), reverse=True)
#         lowers.sort(key=lambda c: _touch_count(c, DF), reverse=True)
#         self.linear_bounds["upper"].extend(uppers)
#         self.linear_bounds["lower"].extend(lowers)

#         if verbose:
#             print(f"[linear] Built {len(lowers)} lower and {len(uppers)} upper bounds.")

#     # ──────────────────────────────────────────────────────────────
#     # Nonlinear bounds (stored separately)
#     # ──────────────────────────────────────────────────────────────
#     def make_nonlinear_bounds(
#         self,
#         *,
#         min_support: int = 8,
#         max_denominator: int = 30,
#         try_sqrt: bool = True,
#         try_square: bool = True,
#         verbose: bool = False,
#     ) -> None:
#         """
#         Emit 2-feature “½-mix” bounds:
#           T ≥ ½(cmin_x X + cmin_sqrtY sqrt(Y)),  T ≤ ½(cmax_x X + cmax_sqrtY sqrt(Y))
#           T ≥ ½(cmin_x X + cmin_sqY Y^2),        T ≤ ½(cmax_x X + cmax_sqY Y^2)

#         We intentionally keep these *nonlinear* in a separate store so linear
#         generalization can focus only on pure linear bounds.
#         """
#         self._init_stores()
#         DF = self.df_proj
#         num_cols = list(self.numeric_columns)
#         uppers: List[Conjecture] = []
#         lowers: List[Conjecture] = []

#         for H in self.hyps_kept:
#             mask = H(DF) if H is not None else np.ones(len(DF), dtype=bool)
#             if not np.any(mask):
#                 continue
#             D = DF.loc[mask]

#             arr = {c: pd.to_numeric(D[c], errors="coerce").to_numpy(dtype=float) for c in num_cols}

#             for target in num_cols:
#                 t = arr[target]
#                 t_ok = np.isfinite(t)
#                 if np.count_nonzero(t_ok) < min_support:
#                     continue

#                 for Xcol in num_cols:
#                     if Xcol == target:
#                         continue
#                     x = arr[Xcol]
#                     ok_x = t_ok & np.isfinite(x) & (x > 0)
#                     if np.count_nonzero(ok_x) < min_support:
#                         continue

#                     rx = t[ok_x] / x[ok_x]
#                     rx = rx[np.isfinite(rx)]
#                     if rx.size == 0:
#                         continue

#                     cmin_x = float(np.min(rx))
#                     cmax_x = float(np.max(rx))
#                     cmin_x_c = _to_frac_const(0.5 * cmin_x, max_denom=max_denominator)
#                     cmax_x_c = _to_frac_const(0.5 * cmax_x, max_denom=max_denominator)

#                     T = to_expr(target)
#                     X = to_expr(Xcol)

#                     for Ycol in num_cols:
#                         if Ycol == target:
#                             continue
#                         y = arr[Ycol]
#                         ok_y = t_ok & np.isfinite(y) & (y > 0)
#                         if np.count_nonzero(ok_y) < min_support:
#                             continue

#                         # sqrt(Y)
#                         if try_sqrt:
#                             sY = np.sqrt(y, dtype=float)
#                             ok = t_ok & np.isfinite(sY) & np.isfinite(x) & (x > 0) & (y > 0)
#                             if np.count_nonzero(ok) >= min_support:
#                                 r_s = t[ok] / sY[ok]
#                                 r_s = r_s[np.isfinite(r_s)]
#                                 if r_s.size:
#                                     smin = float(np.min(r_s))
#                                     smax = float(np.max(r_s))
#                                     smin_c = _to_frac_const(0.5 * smin, max_denom=max_denominator)
#                                     smax_c = _to_frac_const(0.5 * smax, max_denom=max_denominator)

#                                     base_lower = Conjecture(
#                                         Ge(T, cmin_x_c * X + smin_c * sqrt(to_expr(Ycol))), H
#                                     )
#                                     if base_lower.is_true(DF):
#                                         lowers.append(base_lower)

#                                     base_upper = Conjecture(
#                                         Le(T, cmax_x_c * X + smax_c * sqrt(to_expr(Ycol))), H
#                                     )
#                                     if base_upper.is_true(DF):
#                                         uppers.append(base_upper)

#                         # (Y)^2
#                         if try_square:
#                             Y2 = y * y
#                             ok = t_ok & np.isfinite(Y2) & np.isfinite(x) & (x > 0)
#                             if np.count_nonzero(ok) >= min_support:
#                                 r2 = t[ok] / Y2[ok]
#                                 r2 = r2[np.isfinite(r2)]
#                                 if r2.size:
#                                     qmin = float(np.min(r2))
#                                     qmax = float(np.max(r2))
#                                     qmin_c = _to_frac_const(0.5 * qmin, max_denom=max_denominator)
#                                     qmax_c = _to_frac_const(0.5 * qmax, max_denom=max_denominator)

#                                     base_lower = Conjecture(
#                                         Ge(T, cmin_x_c * X + qmin_c * (to_expr(Ycol) ** Const(2))), H
#                                     )
#                                     if base_lower.is_true(DF):
#                                         lowers.append(base_lower)

#                                     base_upper = Conjecture(
#                                         Le(T, cmax_x_c * X + qmax_c * (to_expr(Ycol) ** Const(2))), H
#                                     )
#                                     if base_upper.is_true(DF):
#                                         uppers.append(base_upper)

#         uppers.sort(key=lambda c: _touch_count(c, DF), reverse=True)
#         lowers.sort(key=lambda c: _touch_count(c, DF), reverse=True)
#         self.nonlinear_bounds["upper"].extend(uppers)
#         self.nonlinear_bounds["lower"].extend(lowers)
#         if verbose:
#             print(f"[nonlinear] Built {len(lowers)} lower and {len(uppers)} upper bounds.")

#     # ──────────────────────────────────────────────────────────────
#     # Robust wrapper for the (branch-varying) PGFC signature
#     # ──────────────────────────────────────────────────────────────
#     def _call_pgfc(self, DF, conj, candidate_hyps, atol, rtol) -> List[Conjecture]:
#         """
#         Call propose_generalizations_from_constants regardless of which signature is installed.
#         Supported variants in the wild include (names may vary):
#         - (df, conj, cache, *, candidate_hypotheses, atol, rtol)
#         - (conj, cache, *, candidate_hypotheses, atol, rtol)
#         - (cache, conj, *, candidate_hypotheses, atol, rtol)
#         - (conj, cache)   # legacy minimal

#         We detect available parameter names and call by **keyword** to avoid mis-ordered
#         positional arguments. The result is normalized to List[Conjecture] (unwrapping
#         .new_conjecture where necessary).
#         """
#         from txgraffiti2025.processing.post.generalize_from_constants import (
#             propose_generalizations_from_constants as PGFC
#         )
#         import inspect

#         # Collect the parameter names the current function actually accepts.
#         sig = inspect.signature(PGFC)
#         names = {p.name for p in sig.parameters.values()}

#         call_kwargs = {}

#         # Map common aliases safely to keywords the function supports
#         df_names = ["df", "DF", "dataframe"]
#         conj_names = ["conj", "conjecture"]
#         cache_names = ["cache", "constants_cache"]

#         # If the function exposes any of these names, provide that argument.
#         if any(n in names for n in df_names):
#             for n in df_names:
#                 if n in names:
#                     call_kwargs[n] = DF
#                     break

#         if any(n in names for n in conj_names):
#             for n in conj_names:
#                 if n in names:
#                     call_kwargs[n] = conj
#                     break

#         if any(n in names for n in cache_names):
#             for n in cache_names:
#                 if n in names:
#                     call_kwargs[n] = self.constants_cache
#                     break

#         # Optional keyword-only knobs (add only if accepted)
#         if "candidate_hypotheses" in names:
#             call_kwargs["candidate_hypotheses"] = list(candidate_hyps)
#         if "atol" in names:
#             call_kwargs["atol"] = float(atol)
#         if "rtol" in names:
#             call_kwargs["rtol"] = float(rtol)

#         # Primary attempt: pure keyword call (prevents positional confusion)
#         try:
#             res = PGFC(**call_kwargs)
#         except TypeError:
#             # Be robust: try minimal, widely-supported permutations.
#             res = None
#             # 1) conj, cache (keywords only)
#             try:
#                 res = PGFC(**{k: v for k, v in call_kwargs.items() if k in {"conj", "conjecture", "cache", "constants_cache"}})
#             except TypeError:
#                 pass
#             # 2) cache, conj (keywords only)
#             if res is None:
#                 try:
#                     alt = {}
#                     if "cache" in names:
#                         alt["cache"] = self.constants_cache
#                     elif "constants_cache" in names:
#                         alt["constants_cache"] = self.constants_cache
#                     if "conj" in names:
#                         alt["conj"] = conj
#                     elif "conjecture" in names:
#                         alt["conjecture"] = conj
#                     res = PGFC(**alt)
#                 except TypeError:
#                     pass
#             # 3) df, conj, cache (keywords only)
#             if res is None:
#                 try:
#                     alt = {}
#                     for n in df_names:
#                         if n in names:
#                             alt[n] = DF
#                             break
#                     for n in conj_names:
#                         if n in names:
#                             alt[n] = conj
#                             break
#                     for n in cache_names:
#                         if n in names:
#                             alt[n] = self.constants_cache
#                             break
#                     res = PGFC(**alt)
#                 except TypeError:
#                     pass
#             # 4) truly-legacy (conj, cache) only
#             if res is None:
#                 # last resort: assume only these two keywords exist
#                 alt = {}
#                 if "conj" in names:
#                     alt["conj"] = conj
#                 elif "conjecture" in names:
#                     alt["conjecture"] = conj
#                 if "cache" in names:
#                     alt["cache"] = self.constants_cache
#                 elif "constants_cache" in names:
#                     alt["constants_cache"] = self.constants_cache
#                 res = PGFC(**alt)

#         # Normalize the result to a list of Conjecture objects
#         if res is None:
#             return []
#         if isinstance(res, Conjecture):
#             return [res]

#         out: List[Conjecture] = []
#         for r in res:
#             out.append(getattr(r, "new_conjecture", r))
#         return out



#     # ──────────────────────────────────────────────────────────────
#     # Generalize the *linear* bounds
#     # ──────────────────────────────────────────────────────────────
#     def generalize_linear_bounds(
#         self,
#         *,
#         use_constants: bool = True,
#         use_reciprocals: bool = True,
#         use_intercepts: bool = True,
#         require_superset: bool = True,
#         atol: float = 1e-9,
#         rtol: float = 1e-9,
#         verbose: bool = False,
#     ) -> None:
#         """
#         Run generalizers on stored *linear* bounds and keep successful results.
#         Constants cache is required for the slope-from-constants path.
#         """
#         DF = self.df_proj
#         if not self.linear_bounds["upper"] and not self.linear_bounds["lower"]:
#             if verbose:
#                 print("No linear bounds to generalize.")
#             self.linear_bounds_generalized = []
#             return

#         cands = self.get_linear_bounds()
#         proposals: List[Conjecture] = []
#         candidate_hyps = list(self.hyps_kept)

#         for conj in cands:
#             # A: slope from constants cache
#             # A: slope from constants cache
#             if use_constants and self.constants_cache is not None:
#                 proposals.extend(self._call_pgfc(DF, conj, candidate_hyps, atol, rtol))



#             # B: slope via reciprocals (1/(B+s))
#             if use_reciprocals:
#                 rs = propose_generalizations_from_reciprocals(
#                     DF, conj, candidate_hypotheses=candidate_hyps
#                 )
#                 proposals.extend(rs)

#             # C: intercept generalizations
#             if use_intercepts:
#                 is_ = propose_generalizations_from_intercept(
#                     DF, conj, self.constants_cache,
#                     candidate_hypotheses=candidate_hyps,
#                     candidate_intercepts=None,
#                     relaxers_Z=None,
#                     require_superset=require_superset,
#                 )
#                 proposals.extend([g.new_conjecture for g in is_])

#         # Deduplicate & prefer more-general via Morgan
#         if proposals:
#             mf = morgan_filter(DF, proposals)
#             self.linear_bounds_generalized = list(mf.kept)
#         else:
#             self.linear_bounds_generalized = []

#         if verbose:
#             print(f"[generalize] kept {len(self.linear_bounds_generalized)} conjectures.")

#     # ──────────────────────────────────────────────────────────────
#     # Summaries
#     # ──────────────────────────────────────────────────────────────
#     def summarize_linear_bounds(self, kind: str = "upper", top: int = 20) -> str:
#         assert kind in ("upper", "lower")
#         pool = self.linear_bounds[kind]
#         if not pool:
#             return f"(no {kind} linear bounds)"
#         DF = self.df_proj
#         lines = [f"{kind.capitalize()} linear bounds (top {top} by touches):"]
#         for c in sorted(pool, key=lambda z: _touch_count(z, DF), reverse=True)[:top]:
#             lines.append(f"  {c.pretty()}   [touches={_touch_count(c, DF)}]")
#         return "\n".join(lines)

#     def summarize_nonlinear_bounds(self, kind: str = "upper", top: int = 20) -> str:
#         assert kind in ("upper", "lower")
#         pool = self.nonlinear_bounds[kind]
#         if not pool:
#             return f"(no {kind} nonlinear bounds)"
#         DF = self.df_proj
#         lines = [f"{kind.capitalize()} nonlinear bounds (top {top} by touches):"]
#         for c in sorted(pool, key=lambda z: _touch_count(z, DF), reverse=True)[:top]:
#             lines.append(f"  {c.pretty()}   [touches={_touch_count(c, DF)}]")
#         return "\n".join(lines)

#     def summarize_generalized_linear(self, top: int = 30) -> str:
#         pool = self.linear_bounds_generalized
#         if not pool:
#             return "(no generalized linear bounds)"
#         DF = self.df_proj
#         lines = [f"Generalized linear bounds (top {top} by touches):"]
#         for c in sorted(pool, key=lambda z: _touch_count(z, DF), reverse=True)[:top]:
#             lines.append(f"  {c.pretty()}   [touches={_touch_count(c, DF)}]")
#         return "\n".join(lines)

#     def get_linear_bounds(self, kind: Optional[str] = None) -> List[Conjecture]:
#         if kind is None:
#             return list(self.linear_bounds["lower"]) + list(self.linear_bounds["upper"])
#         assert kind in ("upper", "lower")
#         return list(self.linear_bounds[kind])

#     # ──────────────────────────────────────────────────────────────
#     # Constants cache (pairs or full hyps), plus helpers
#     # ──────────────────────────────────────────────────────────────
#     def compute_constants_cache(
#         self,
#         *,
#         use_pairs: bool = True,
#         numeric_cols: Optional[List[str]] = None,
#         shifts: Tuple[int, ...] = (-2, -1, 0, 1, 2),
#         atol: float = 1e-9,
#         rtol: float = 1e-9,
#         min_support: int = 8,
#         max_denominator: Optional[int] = 50,
#         skip_identity: bool = True,   # skip (X+a)/(X+a); still allows X/(X+1)
#     ) -> None:
#         DF = self.df_proj
#         if numeric_cols is None:
#             numeric_cols = list(self.numeric_columns)

#         if use_pairs:
#             self.constants_cache = precompute_constant_ratios_pairs(
#                 DF,
#                 base=self.base,
#                 boolean_cols=self.bool_columns,
#                 numeric_cols=numeric_cols,
#                 shifts=shifts,
#                 atol=atol,
#                 rtol=rtol,
#                 min_support=min_support,
#                 max_denominator=max_denominator,
#                 skip_identity=skip_identity,
#             )
#         else:
#             self.constants_cache = precompute_constant_ratios(
#                 DF,
#                 hypotheses=self.hyps_kept,
#                 numeric_cols=numeric_cols,
#                 shifts=shifts,
#                 atol=atol,
#                 rtol=rtol,
#                 min_support=min_support,
#                 max_denominator=max_denominator,
#                 skip_identity=skip_identity,
#             )

#     def constants_summary(self, max_per_hyp: int = 8) -> str:
#         if not self.constants_cache:
#             return "(no constants cache built)"
#         lines: List[str] = []
#         for _, bucket in self.constants_cache.key_to_constants.items():
#             H = bucket.hypothesis
#             lines.append(f"[{repr(H) if H is not None else 'TRUE'}]")
#             if not bucket.constants:
#                 lines.append("  (no constants)")
#                 continue
#             for cr in bucket.constants[:max_per_hyp]:
#                 a = f"+{cr.shift_num}" if cr.shift_num else ""
#                 b = f"+{cr.shift_den}" if cr.shift_den else ""
#                 lines.append(
#                     f"  (N={cr.numerator}{a})/(D={cr.denominator}{b}) "
#                     f"≈ {cr.value_display}  [support={cr.support}]"
#                 )
#         return "\n".join(lines)

#     def match_constants(
#         self,
#         hypothesis,
#         *,
#         coeff: float,
#         atol: float = 1e-9,
#         rtol: float = 1e-9,
#     ):
#         if not self.constants_cache:
#             return []
#         return constants_matching_coeff(
#             self.constants_cache,
#             hypothesis=hypothesis,
#             coeff=coeff,
#             atol=atol,
#             rtol=rtol,
#         )

#     # ──────────────────────────────────────────────────────────────
#     # Quick global summary
#     # ──────────────────────────────────────────────────────────────
#     def summary(self) -> str:
#         lines = []
#         lines.append(f"Projected columns ({len(self.df_proj.columns)}): {list(self.df_proj.columns)}")
#         lines.append(f"Base hypothesis: {repr(self.base)}")
#         lines.append(f"Enumerated hypotheses: {len(self.hyps_all)}")
#         lines.append(f"Simplified hypotheses kept: {len(self.hyps_kept)}")
#         lines.append(f"Equivalences found: {len(self.equivalences)}")
#         lines.append(f"Boolean columns: {self.bool_columns}")
#         lines.append(f"Numeric columns: {self.numeric_columns}")
#         return "\n".join(lines)

# from txgraffiti.example_data import graph_data as df
# # from txgraffiti2025.engine.txgraffiti import TxGraffiti

# # Instantiate (your TxGraffiti.__init__ should already project to the 11 columns)
# tg = TxGraffiti(df)
# print(tg.summary())

# # Build linear bounds first (pure linear; ceil/floor only if true and stronger)
# tg.make_linear_ratio_bounds(tighten_integral=True)
# print(tg.summarize_linear_bounds("upper", top=10))
# print(tg.summarize_linear_bounds("lower", top=10))

# # Nonlinear bounds go to a separate store
# tg.make_nonlinear_bounds()
# print(tg.summarize_nonlinear_bounds("upper", top=10))
# print(tg.summarize_nonlinear_bounds("lower", top=10))

# # Constants cache → then generalize the linear bounds
# tg.compute_constants_cache(use_pairs=True, min_support=2, skip_identity=True)
# # tg.generalize_linear_bounds(verbose=True)
# # print(tg.summarize_generalized_linear(top=20))

# tg.generalize_linear_bounds(verbose=True)
# print(tg.summarize_generalized_linear(top=20))

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
from fractions import Fraction
from itertools import combinations_with_replacement

# --- your modules ---
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)

from txgraffiti2025.forms.utils import to_expr, Expr, Const, floor, ceil, sqrt, safe_sqrt_series
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.processing.post import morgan_filter


from txgraffiti2025.processing.post.reciprocal_generalizer import (
    propose_generalizations_from_reciprocals,
)
from txgraffiti2025.forms.generic_conjecture import Ge, Le
from txgraffiti2025.forms.utils import BinOp, Const as _ConstClass  # for simple structure checks
from typing import Sequence

from txgraffiti2025.forms.generic_conjecture import Ge, Le, TRUE
from txgraffiti2025.forms.utils import Const, sqrt

from dataclasses import dataclass
from typing import Optional, Iterable, List, Tuple
import numpy as np
import pandas as pd

from txgraffiti2025.forms.qualitative import MonotoneRelation, Method as CorrMethod
from txgraffiti2025.forms.predicates import Predicate


from collections import defaultdict
from typing import Iterable, Optional, Tuple, List
import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence


from typing import List, Set
import numpy as np
from txgraffiti2025.forms.predicates import Predicate
try:
    from txgraffiti2025.forms.predicates import AndPred
    _HAS_ANDPRED = True
except Exception:
    _HAS_ANDPRED = False

import numpy as np
import pandas as pd
from typing import Iterable, Sequence, Tuple, List, Optional

from txgraffiti2025.forms.predicates import Predicate, Where, AndPred
from txgraffiti2025.forms.generic_conjecture import Ge, Le, Conjecture
from txgraffiti2025.forms.utils import to_expr, Const

# # --- stable hypothesis key
# def _hyp_key(H) -> str:
#     return H.pretty() if hasattr(H, "pretty") else repr(H)

# import re

# _IFTHEN_RE = re.compile(r"^\[\s*(?P<hyp>.+?)\s*\]\s*::\s*(?P<phi>.+)$")

# import re

# # Matches the derived-predicate naming we used: "[  <hyp>  ] :: <phi>"
# _IFTHEN_RE = re.compile(r"^\[\s*(?P<hyp>.+?)\s*\]\s*::\s*(?P<phi>.+)$")

import re

_IFTHEN_RE = re.compile(r"^\[\s*(?P<hyp>.+?)\s*\]\s*::\s*(?P<phi>.+)$")


import re
from typing import Iterable, Optional, Tuple, List
import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence

# --- existing helpers you already had ---
def _pred_key(p: Predicate) -> str:
    return p.pretty() if hasattr(p, "pretty") else repr(p)

def _mask_bool(df: pd.DataFrame, p: Predicate) -> np.ndarray:
    s = p.mask(df).reindex(df.index, fill_value=False)
    if s.dtype != bool:
        s = s.fillna(False).astype(bool, copy=False)
    return s.to_numpy()

def _support(mask: np.ndarray) -> int:
    return int(mask.sum())

# def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
#     return bool(np.array_equal(a, b))

# --- NEW: robust atomization of predicates (derived + plain) ---

# matches derived: "[  H  ] :: phi"
_IFTHEN_RE = re.compile(r"^\[\s*(?P<hyp>.+?)\s*\]\s*::\s*(?P<phi>.+)$")

def _strip_outer_parens_once(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        return inner if (inner.startswith("(") and inner.endswith(")")) else f"({inner})"
    return s

def _split_top_level_conj(s: str) -> list[str]:
    """Split on top-level '∧' or '\\land' respecting parentheses."""
    s = s.strip()
    parts, buf, depth = [], [], 0
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == '(':
            depth += 1; buf.append(ch); i += 1; continue
        if ch == ')':
            if depth > 0: depth -= 1
            buf.append(ch); i += 1; continue
        if depth == 0:
            if ch == '∧':
                part = ''.join(buf).strip()
                if part: parts.append(part)
                buf = []; i += 1; continue
            if s.startswith(r'\land', i):
                part = ''.join(buf).strip()
                if part: parts.append(part)
                buf = []; i += len(r'\land'); continue
        buf.append(ch); i += 1
    tail = ''.join(buf).strip()
    if tail: parts.append(tail)
    return parts or [s]

def _canon_atom(s: str) -> str:
    """Normalize one atom to '(...)' with collapsed spaces, single outer parens."""
    s = ' '.join(_strip_outer_parens_once(s).split())
    if not (s.startswith('(') and s.endswith(')')):
        s = f'({s})'
    return s

def _atoms_for_pred(p: Predicate) -> set[str]:
    """
    Return a set of canonical atoms for Predicate p.
    - Derived "[ H ] :: phi" -> atoms(H) ∪ {phi}
    - Plain "(A ∧ B)" -> atoms(A) ∪ atoms(B)
    Also honors structured attributes if you later add them:
        p._derived_hypothesis, p._derived_conclusion
    """
    # Prefer structured metadata if present
    hyp = getattr(p, "_derived_hypothesis", None)
    phi_obj = getattr(p, "_derived_conclusion", None)
    if hyp is not None and phi_obj is not None:
        hyp_txt = getattr(hyp, "pretty", lambda: repr(hyp))()
        phi_txt = getattr(phi_obj, "pretty", lambda: repr(phi_obj))()
        hyps = {_canon_atom(x) for x in _split_top_level_conj(hyp_txt)}
        return hyps | {_canon_atom(phi_txt)}

    # Parse name if it matches derived format
    name = getattr(p, "name", None)
    if name:
        m = _IFTHEN_RE.match(name)
        if m:
            hyp_txt = m.group("hyp").strip()
            phi_txt = m.group("phi").strip()
            hyps = {_canon_atom(x) for x in _split_top_level_conj(hyp_txt)}
            return hyps | {_canon_atom(phi_txt)}

    # Fallback: plain predicate string
    s = (p.pretty() if hasattr(p, "pretty") else repr(p)).strip()
    return {_canon_atom(x) for x in _split_top_level_conj(s)}



def _parse_pred_kind(p):
    """
    Return ('derived', hyp_atoms, phi_atom) or ('plain', atoms, None).
    """
    name = getattr(p, "name", None)
    if name:
        m = _IFTHEN_RE.match(name)
        if m:
            hyp = m.group("hyp").strip()
            phi = m.group("phi").strip()
            hyp_parts = _split_top_level_conj(hyp) or [hyp]
            hyp_atoms = [_canon_atom(h) for h in hyp_parts]
            phi_atom  = _canon_atom(phi)
            return ('derived', hyp_atoms, phi_atom)

    # plain predicate path
    s = p.pretty() if hasattr(p, "pretty") else repr(p)
    parts = _split_top_level_conj(s) or [s]
    atoms = [_canon_atom(x) for x in parts]
    return ('plain', atoms, None)

def _rhs_atoms_subset_lhs_atoms(kindA, atomsA, phiA, kindB, atomsB, phiB) -> bool:
    """
    Decide triviality for A ⇒ B by checking if B's atoms are contained in A's atoms.
    Covers:
      - derived→derived with same phi and hyp(B) ⊆ hyp(A)
      - derived→plain where plain(B) is one hyp atom of A
      - plain/derived general case: all B atoms ⊆ (A hyp atoms ∪ {phiA if derived})
    """
    setA = set(atomsA)
    if kindA == 'derived':
        setA = setA | {phiA}

    # B atoms:
    setB = set(atomsB)
    if kindB == 'derived':
        # If phi differs, it's not a trivial "drop-a-conjunct" case
        if phiB != phiA:
            # still, the generic subset test below might catch (rare) formatting-equal atoms
            pass
        else:
            # same phi: trivial if hyp(B) ⊆ hyp(A)
            if set(atomsB).issubset(set(atomsA)):
                return True
        # include phiB for the generic subset test
        setB = setB | {phiB}

    # generic containment test
    return setB.issubset(setA)

def filter_trivial_conj_inclusions(inclusions):
    """
    Remove inclusions A ⇒ B that are trivial because RHS conjuncts are already
    contained in LHS conjuncts (accounting for derived predicate structure).
    """
    out = []
    for inc in inclusions:
        kindA, atomsA, phiA = _parse_pred_kind(inc.A)
        kindB, atomsB, phiB = _parse_pred_kind(inc.B)
        if _rhs_atoms_subset_lhs_atoms(kindA, atomsA, phiA, kindB, atomsB, phiB):
            # drop trivial inclusion
            continue
        out.append(inc)
    return out


# def _strip_outer_parens_once(s: str) -> str:
#     s = s.strip()
#     if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
#         inner = s[1:-1].strip()
#         return inner if (inner.startswith("(") and inner.endswith(")")) else f"({inner})"
#     return s

# def _split_top_level_conj(s: str) -> list[str]:
#     """
#     Split a string on top-level conjunctions (Unicode '∧' or ASCII '\\land'),
#     respecting parentheses nesting. Returns raw parts (not parenthesized).
#     """
#     s = s.strip()
#     parts, buf = [], []
#     depth = 0
#     i = 0
#     while i < len(s):
#         ch = s[i]
#         if ch == '(':
#             depth += 1
#             buf.append(ch)
#             i += 1
#             continue
#         if ch == ')':
#             depth -= 1 if depth > 0 else 0
#             buf.append(ch)
#             i += 1
#             continue

#         # unicode '∧' at top level
#         if depth == 0 and ch == '∧':
#             part = ''.join(buf).strip()
#             if part:
#                 parts.append(part)
#             buf = []
#             i += 1
#             continue

#         # ascii '\land' at top level
#         if depth == 0 and s.startswith(r'\land', i):
#             part = ''.join(buf).strip()
#             if part:
#                 parts.append(part)
#             buf = []
#             i += len(r'\land')
#             continue

#         buf.append(ch)
#         i += 1
#     tail = ''.join(buf).strip()
#     if tail:
#         parts.append(tail)
#     return parts

# def _canon_atom(s: str) -> str:
#     """Normalize one atom to a stable '(...)' form for set comparisons."""
#     s = _strip_outer_parens_once(s)
#     # collapse spaces a bit
#     s = ' '.join(s.split())
#     if not (s.startswith('(') and s.endswith(')')):
#         s = f'({s})'
#     return s


def predicate_to_conjunction(p, *, ascii_ops: bool = False) -> str:
    """
    Render a Predicate as:
      - Derived form "[ H ] :: phi"  →  "((H) ∧ (phi))"
      - Otherwise                     →  "(pretty(p))"
    """
    land = r"\land" if ascii_ops else "∧"

    # Prefer the explicit name; else pretty/repr
    raw = getattr(p, "name", None)
    if raw:
        m = _IFTHEN_RE.match(raw)
        if m:
            hyp = _strip_outer_parens_once(m.group("hyp"))
            phi = m.group("phi").strip()
            # Ensure each side has one clean set of parens
            if not (hyp.startswith("(") and hyp.endswith(")")):
                hyp = f"({hyp})"
            if not (phi.startswith("(") and phi.endswith(")")):
                phi = f"({phi})"
            return f"({hyp} {land} {phi})"

    # Fallback for non-derived predicates
    s = p.pretty() if hasattr(p, "pretty") else repr(p)
    s = _strip_outer_parens_once(s)
    if not (s.startswith("(") and s.endswith(")")):
        s = f"({s})"
    return s


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    # trim one redundant outer layer like "((connected))" -> "(connected)"
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        # only one pass to avoid changing grouping semantics
        return inner if inner.startswith("(") and inner.endswith(")") else f"({inner})"
    return s

import re

# matches our derived predicates: "[ <hyp> ] :: <phi>"
_IFTHEN_RE = re.compile(r"^\[\s*(?P<hyp>.+?)\s*\]\s*::\s*(?P<phi>.+)$")

# def _strip_outer_parens_once(s: str) -> str:
#     s = s.strip()
#     if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
#         inner = s[1:-1].strip()
#         # only one peel for readability; keeps grouping sane
#         return inner if (inner.startswith("(") and inner.endswith(")")) else f"({inner})"
#     return s

# def _canon_atom(s: str) -> str:
#     """Normalize a single conjunct text for set comparisons."""
#     s = _strip_outer_parens_once(s)
#     s = s.strip()
#     # tiny normalization
#     s = s.replace("  ", " ")
#     return s

def _conjunct_atoms_from_pred(p) -> list[str]:

    name = getattr(p, "name", None)

    # Derived predicate: "[ H ] :: phi"
    if name:
        m = _IFTHEN_RE.match(name)
        if m:
            hyp = m.group("hyp").strip()
            phi = m.group("phi").strip()
            hyp_parts = _split_top_level_conj(hyp)
            atoms = [_canon_atom(h) for h in hyp_parts]
            atoms.append(_canon_atom(phi))
            return atoms

    # Non-derived: use pretty()/repr and split on top-level conjunctions
    s = p.pretty() if hasattr(p, "pretty") else repr(p)
    s = s.strip()
    parts = _split_top_level_conj(s) if ('∧' in s or r'\land' in s) else [s]
    return [_canon_atom(x) for x in parts]

def _is_trivial_conjunctive_inclusion(inc) -> bool:
    """
    Inclusion A ⇒ B is trivial if every atom of B appears among atoms of A.
    """
    A_atoms = set(_conjunct_atoms_from_pred(inc.A))
    B_atoms = set(_conjunct_atoms_from_pred(inc.B))
    return B_atoms.issubset(A_atoms)

def _postfilter_conj_trivialities(eqs, incs):
    # Drop inclusions where RHS atoms ⊆ LHS atoms
    incs_filtered = [inc for inc in incs if not _is_trivial_conjunctive_inclusion(inc)]
    return eqs, incs_filtered



def _conjuncts_from_pred(p) -> list[str]:
    """
    Return the list of textual conjuncts that define this predicate:
      - derived "[ H ] :: φ"  -> ["(H)", "(φ)"]
      - plain "(planar)"      -> ["(planar)"]
    """
    name = getattr(p, "name", None)
    if name:
        m = _IFTHEN_RE.match(name)
        if m:
            hyp = m.group("hyp").strip()
            phi = m.group("phi").strip()
            # ensure each is parenthesized once
            if not (hyp.startswith("(") and hyp.endswith(")")):
                hyp = f"({hyp})"
            if not (phi.startswith("(") and phi.endswith(")")):
                phi = f"({phi})"
            return [_canon_atom(hyp), _canon_atom(phi)]

    # fallback for non-derived predicates
    s = p.pretty() if hasattr(p, "pretty") else repr(p)
    if not (s.startswith("(") and s.endswith(")")):
        s = f"({s})"
    return [_canon_atom(s)]

# def _is_trivial_conjunctive_inclusion(inc) -> bool:
#     """
#     True if inclusion A ⇒ B is tautological because all conjuncts of B
#     already appear among the conjuncts of A (i.e., B’s atoms ⊆ A’s atoms).
#     """
#     A_atoms = set(_conjuncts_from_pred(inc.A))
#     B_atoms = set(_conjuncts_from_pred(inc.B))
#     return B_atoms.issubset(A_atoms)


def predicate_to_if_then(p) -> str:
    """
    If this is one of our derived Where predicates named like:
        "[ (hypothesis) ] :: conclusion"
    render it as:
        "If (hypothesis) ⇒ conclusion"
    else fall back to p.pretty()/repr.
    """
    name = getattr(p, "name", None)
    if not name:
        return p.pretty() if hasattr(p, "pretty") else repr(p)

    m = _IFTHEN_RE.match(name)
    if not m:
        return name  # not a derived format, just print the name

    hyp = _strip_outer_parens(m.group("hyp"))
    phi = m.group("phi").strip()
    # ensure hypothesis is wrapped in a single pair of parens
    if not (hyp.startswith("(") and hyp.endswith(")")):
        hyp = f"({hyp})"
    return f"If {hyp} ⇒ {phi}"

def pretty_class_relations_ifthen(title: str, eqs, incs, df, max_items: int = 60):
    print(f"\n=== {title} ===")

    # Equivalences
    print("\n-- Equivalences --")
    for i, e in enumerate(eqs[:max_items], 1):
        left  = predicate_to_if_then(e.A)
        right = predicate_to_if_then(e.B)
        viol  = e.violation_count(df)
        print(f"{i:3d}. {left}  ⇔  {right}    [violations={viol}]")

    # Inclusions
    print("\n-- Inclusions --")
    for i, inc in enumerate(incs[:max_items], 1):
        left   = predicate_to_if_then(inc.A)
        right  = predicate_to_if_then(inc.B)
        suppA  = int(inc.A.mask(df).sum())
        viol   = inc.violation_count(df)
        print(f"{i:3d}. {left}  ⇒  {right}    [support(A)={suppA}, violations={viol}]")

def pretty_class_relations_conj(
    title: str,
    eqs,
    incs,
    df,
    *,
    max_items: int = 60,
    ascii_ops: bool = False,
    show_violations: bool = False,   # you asked to emphasize support(A); violations are usually 0 after filtering
):
    impl = r"\Rightarrow" if ascii_ops else "⇒"
    equiv = r"\Leftrightarrow" if ascii_ops else "⇔"

    print(f"\n=== {title} ===")

    # Equivalences
    print("\n-- Equivalences --")
    for i, e in enumerate(eqs[:max_items], 1):
        left  = predicate_to_conjunction(e.A, ascii_ops=ascii_ops)
        right = predicate_to_conjunction(e.B, ascii_ops=ascii_ops)
        meta  = f"[violations={e.violation_count(df)}]" if show_violations else ""
        sep   = ("  " + meta) if meta else ""
        print(f"{i:3d}. {left} {equiv} {right}{sep}")

    # Inclusions
    print("\n-- Inclusions --")
    for i, inc in enumerate(incs[:max_items], 1):
        left   = predicate_to_conjunction(inc.A, ascii_ops=ascii_ops)
        right  = predicate_to_conjunction(inc.B, ascii_ops=ascii_ops)
        suppA  = int(inc.A.mask(df).sum())
        meta   = f"[support(A)={suppA}]"
        print(f"{i:3d}. {left} {impl} {right}  {meta}")


def _mask_from_pred(df: pd.DataFrame, H: Optional[Predicate]) -> np.ndarray:
    if H is None:
        return np.ones(len(df), dtype=bool)
    m = H.mask(df)
    return np.asarray(m, dtype=bool)

def _as_series(x, index) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.reindex(index)
    return pd.Series(x, index=index)

def _touch_mask(conj: Conjecture, df: pd.DataFrame, rtol=1e-8, atol=1e-8) -> np.ndarray:
    """Rows where lhs == rhs within tolerance AND the base hypothesis holds."""
    lhs = _as_series(conj.relation.left.eval(df), df.index).astype(float)
    rhs = _as_series(conj.relation.right.eval(df), df.index).astype(float)
    Hm  = _mask_from_pred(df, conj.condition)
    eq  = np.isclose(lhs.values, rhs.values, rtol=rtol, atol=atol)
    return Hm & eq

def _strict_mask(conj: Conjecture, df: pd.DataFrame, side: str, rtol=1e-8, atol=1e-8) -> np.ndarray:
    """
    side: 'lt' or 'gt' relative to the conjecture's inequality direction.
      - If conj is 'y ≤ R':
            'lt' means y < R (strictly below)
            'gt' means y > R (violating side)
      - If conj is 'y ≥ R':
            'gt' means y > R (strictly above)
            'lt' means y < R (violating side)
    Returned mask is ANDed with the base hypothesis.
    """
    lhs = _as_series(conj.relation.left.eval(df), df.index).astype(float)
    rhs = _as_series(conj.relation.right.eval(df), df.index).astype(float)
    Hm  = _mask_from_pred(df, conj.condition)

    # Use tolerances to avoid classifying numerical noise as strict
    # Define "strict" as > (rhs + max(atol, rtol*|rhs|)) etc.
    tol = np.maximum(atol, rtol * np.maximum(1.0, np.abs(rhs.values)))
    if isinstance(conj.relation, Le):
        if side == "lt":
            strict = (lhs.values < rhs.values - tol)
        elif side == "gt":
            strict = (lhs.values > rhs.values + tol)
        else:
            raise ValueError("side must be 'lt' or 'gt'")
    elif isinstance(conj.relation, Ge):
        if side == "gt":
            strict = (lhs.values > rhs.values + tol)
        elif side == "lt":
            strict = (lhs.values < rhs.values - tol)
        else:
            raise ValueError("side must be 'lt' or 'gt'")
    else:
        raise ValueError("Conjecture relation must be Le or Ge")

    return Hm & strict

def _format_coeff_expr(expr) -> str:
    """Short, stable textual form for the RHS coefficient*var part (no heavy pretty)."""
    # Fallback to .pretty() which is already designed for human-friendly output.
    return expr.pretty() if hasattr(expr, "pretty") else repr(expr)

def _mk_name(kind: str, conj: Conjecture) -> str:
    """
    kind ∈ {'onbound','strict_lt','strict_gt'}.
    Name includes base hypothesis and the normalized RHS pretty.
    """
    Hname = conj.condition.pretty() if hasattr(conj.condition, "pretty") else repr(conj.condition)
    lhs   = conj.relation.left.pretty()
    rhs   = conj.relation.right.pretty()
    if kind == "onbound":
        tag = "=="
    elif kind == "strict_lt":
        tag = "<"
    elif kind == "strict_gt":
        tag = ">"
    else:
        tag = "?"
    return f"[{Hname}] :: {lhs} {tag} {rhs}"


def _flatten_conjuncts(p: Predicate) -> List[Predicate]:
    """Flatten nested AndPred into a list of primitive conjuncts (best-effort)."""
    if not _HAS_ANDPRED:
        return [p]
    out = []
    stack = [p]
    while stack:
        q = stack.pop()
        if isinstance(q, AndPred):
            # common attributes: .left/.right or .args
            if hasattr(q, "left") and hasattr(q, "right"):
                stack.append(q.left)
                stack.append(q.right)
            elif hasattr(q, "args"):
                stack.extend(list(q.args))
            else:
                out.append(q)  # unknown shape; keep as-is
        else:
            out.append(q)
    return out

# def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
#     return bool(np.array_equal(a, b))

def _subset_mask(a: np.ndarray, b: np.ndarray) -> bool:
    """a ⊆ b  <=> whenever a is True, b is True."""
    return bool(np.all(~a | b))

def _is_conjunction_redundancy(A: Predicate, B: Predicate, mA: np.ndarray, mB: np.ndarray) -> bool:
    """
    True if one side is (other ∧ ... ) but masks are equal, i.e. the extra conjunct(s)
    add no rows. Detect structurally when possible via AndPred flattening.
    """
    if not _same_mask(mA, mB):
        return False

    if not _HAS_ANDPRED:
        # Heuristic fallback: if repr/pretty of one contains ' ∧ ' and the other's string is a substring
        sA = A.pretty() if hasattr(A, "pretty") else repr(A)
        sB = B.pretty() if hasattr(B, "pretty") else repr(B)
        return ("∧" in sA and sB in sA) or ("∧" in sB and sA in sB)

    # Structural: if B is among conjuncts of A, or A is among conjuncts of B
    As = _flatten_conjuncts(A)
    Bs = _flatten_conjuncts(B)
    # compare by mask equality (more robust than object identity)
    def _is_member(target: Predicate, pool: List[Predicate]) -> bool:
        # target is equivalent to ANY in pool?
        # Quick mask-based check using overall df is needed externally; here we just compare .pretty to avoid recursion.
        tname = target.pretty() if hasattr(target, "pretty") else repr(target)
        pool_names = {q.pretty() if hasattr(q, "pretty") else repr(q) for q in pool}
        return tname in pool_names

    return _is_member(B, As) or _is_member(A, Bs)


# def _pred_key(p: Predicate) -> str:
#     # Stable printable id
#     return p.pretty() if hasattr(p, "pretty") else repr(p)

# def _mask_bool(df: pd.DataFrame, p: Predicate) -> np.ndarray:
#     """Aligned bool np.array, NA->False."""
#     s = p.mask(df).reindex(df.index, fill_value=False)
#     if s.dtype != bool:
#         s = s.fillna(False).astype(bool, copy=False)
#     return s.to_numpy()

# def _support(mask: np.ndarray) -> int:
#     return int(mask.sum())

def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))

def _subset(a: np.ndarray, b: np.ndarray) -> bool:
    """
    a ⊆ b  <=> whenever a is True, b is True.
    """
    return bool(np.all(~a | b))


def _is_const_times_var(expr) -> bool:
    """
    Very light structural check: expr == Const * <var-expr> (commutative).
    """
    if not isinstance(expr, BinOp) or getattr(expr, "op", None) not in ("*",):
        return False
    L, R = expr.left, expr.right
    return isinstance(L, _ConstClass) or isinstance(R, _ConstClass)

def _rhs_monomial_var_name(expr) -> str | None:
    """
    If expr is Const * to_expr('col'), return 'col', else None.
    """
    if not _is_const_times_var(expr):
        return None
    L, R = expr.left, expr.right
    if isinstance(L, _ConstClass):
        maybe_var = R
    elif isinstance(R, _ConstClass):
        maybe_var = L
    else:
        return None
    # best-effort: your leaf var Expr normally carries .name or prints to the column
    return getattr(maybe_var, "name", None)


def _all_le_on_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> bool:
    m = mask & np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return False
    return bool(np.all(a[m] <= b[m]))

def _cond_name(H) -> str:
    return H.pretty() if hasattr(H, "pretty") else repr(H)

def _hyp_mask(df: pd.DataFrame, H) -> np.ndarray:
    if H is None:
        return np.ones(len(df), dtype=bool)
    if hasattr(H, "mask"):
        return H.mask(df).astype(bool).to_numpy()
    return np.asarray(H(df), dtype=bool)

def _includes(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    # a is at least as general as b  <=>  whenever b is True, a is True
    return bool(np.all(~mask_b | mask_a))


def _strictly_pos_on(a: np.ndarray, mask: np.ndarray) -> bool:
    m = mask & np.isfinite(a)
    return np.any(m) and bool(np.all(a[m] > 0.0))

def _caro_trivial_masked_targeted(T_arr, x_arr, y_arr, z_arr, mask, *, upper: bool) -> bool:
    """
    For upper:  Tx <= yz is trivial if (T <= y & x <= z) OR (T <= z & x <= y).
    For lower:  Tx >= yz is trivial if (T >= y & x >= z) OR (T >= z & x >= y).
    Only meaningful under positivity (monotone); call this when require_pos=True.
    """
    m = mask & np.isfinite(T_arr) & np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
    if not np.any(m):
        return False
    if upper:
        c1 = _all_le_on_mask(T_arr, y_arr, m) and _all_le_on_mask(x_arr, z_arr, m)
        if c1: return True
        c2 = _all_le_on_mask(T_arr, z_arr, m) and _all_le_on_mask(x_arr, y_arr, m)
        return c2
    else:
        # lower form
        c1 = _all_le_on_mask(y_arr, T_arr, m) and _all_le_on_mask(z_arr, x_arr, m)  # y<=T & z<=x
        if c1: return True
        c2 = _all_le_on_mask(z_arr, T_arr, m) and _all_le_on_mask(y_arr, x_arr, m)  # z<=T & y<=x
        return c2


from txgraffiti2025.forms.utils import to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Eq

def _as_hyp_mask(H, df: pd.DataFrame) -> np.ndarray:
    if hasattr(H, "mask"):
        return np.asarray(H.mask(df), dtype=bool)
    return np.asarray(H(df), dtype=bool)

def _to_rational(x: float, max_denom: int) -> Fraction:
    return Fraction(x).limit_denominator(max_denom)

def _const_ratio_conjecture(H, A: str, a_val: float, B: str, b_val: float, max_denom: int) -> Conjecture | None:
    if not np.isfinite(a_val) or not np.isfinite(b_val) or b_val == 0.0:
        return None
    C = _to_rational(a_val / b_val, max_denom)
    return Conjecture(Eq(to_expr(A), to_expr(B) * Const(C)), H)

def _lex_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)

def _values_on_mask(df: pd.DataFrame, mask: np.ndarray, col: str) -> np.ndarray:
    return df.loc[mask, col].to_numpy(dtype=float, copy=False)

def _finite_nonzero_scalar(x: float) -> bool:
    return np.isfinite(x) and x != 0.0

def _finite_scalar(x: float) -> bool:
    return np.isfinite(x)

def _make_const_bank_for_H(df: pd.DataFrame, mask: np.ndarray, num_cols: list[str], eps: float, min_support: int) -> dict[str, float]:
    bank = {}
    if _support(mask) < min_support:
        return bank
    for c in num_cols:
        vals = _values_on_mask(df, mask, c)
        is_const, val = _constant_value_on_mask(vals, eps)
        if is_const:
            bank[c] = val
    return bank

def _iter_ratio_eqs_from_bank(H, bank: dict[str, float], max_denom: int):
    # unordered pairs (A,B), A != B
    cols = sorted(bank.keys())
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            A, B = cols[i], cols[j]
            a_val, b_val = bank[A], bank[B]
            # prefer "A = C * B" where B ≠ 0
            if _finite_nonzero_scalar(b_val):
                conj = _const_ratio_conjecture(H, A, a_val, B, b_val, max_denom)
                if conj is not None:
                    yield conj
            # optionally also "B = D * A" when A ≠ 0 — unnecessary if we just use lex order;
            # omit to prevent duplication.

def _truthy(c: Conjecture, df: pd.DataFrame) -> bool:
    try:
        return c.is_true(df)
    except Exception:
        return False

def _unique_by_relation_and_condition(conjs: list[Conjecture]) -> list[Conjecture]:
    seen = set()
    out = []
    for c in conjs:
        key = (repr(c.relation), c.condition.pretty() if hasattr(c.condition, "pretty") else str(c.condition))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def _finite_mask(a: np.ndarray) -> np.ndarray:
    return np.isfinite(a)

def _constant_value_on_mask(vals: np.ndarray, eps: float) -> tuple[bool, float]:
    """
    Return (is_constant, value). Constant if max deviation <= eps*(1+|median|).
    """
    f = _finite_mask(vals)
    if not np.any(f):
        return False, np.nan
    v = np.median(vals[f].astype(float))
    dev = np.max(np.abs(vals[f] - v))
    if dev <= eps * (1.0 + abs(v)):
        return True, float(v)
    return False, float(v)

def _mask_includes(a: np.ndarray, b: np.ndarray) -> bool:
    """a ⊆ b for boolean arrays of equal length."""
    if a.shape != b.shape:
        return False
    return bool(np.all(~a | b))  # whenever a is True, b must be True

import re
from typing import Iterable, Optional, Tuple, List, Set
import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence

# ---------- basic helpers you already had ----------
def _pred_key(p: Predicate) -> str:
    return p.pretty() if hasattr(p, "pretty") else repr(p)

def _mask_bool(df: pd.DataFrame, p: Predicate) -> np.ndarray:
    s = p.mask(df).reindex(df.index, fill_value=False)
    if s.dtype != bool:
        s = s.fillna(False).astype(bool, copy=False)
    return s.to_numpy()

def _support(mask: np.ndarray) -> int:
    return int(mask.sum())

def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))

# ---------- NEW: robust atomization ----------
_IFTHEN_RE = re.compile(r"^\[\s*(?P<hyp>.+?)\s*\]\s*::\s*(?P<phi>.+)$")

def _strip_outer_parens_once(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        return inner if (inner.startswith("(") and inner.endswith(")")) else f"({inner})"
    return s

def _split_top_level_conj(s: str) -> list[str]:
    """Split on top-level ∧ or \\land respecting parentheses."""
    s = s.strip()
    parts, buf, depth = [], [], 0
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == '(':
            depth += 1; buf.append(ch); i += 1; continue
        if ch == ')':
            if depth > 0: depth -= 1
            buf.append(ch); i += 1; continue
        if depth == 0:
            if ch == '∧':
                part = ''.join(buf).strip()
                if part: parts.append(part)
                buf = []; i += 1; continue
            if s.startswith(r'\land', i):
                part = ''.join(buf).strip()
                if part: parts.append(part)
                buf = []; i += len(r'\land'); continue
        buf.append(ch); i += 1
    tail = ''.join(buf).strip()
    if tail: parts.append(tail)
    return parts or [s]

def _canon_atom(s: str) -> str:
    """Normalize one atom to '(...)' with collapsed spaces and single outer parens."""
    s = ' '.join(_strip_outer_parens_once(s).split())
    if not (s.startswith('(') and s.endswith(')')):
        s = f'({s})'
    return s

def _atoms_for_pred(p: Predicate) -> Set[str]:
    """
    Return canonical atom set for a predicate p.
    - Derived "[ H ] :: φ" -> atoms(H) ∪ {φ}
    - Plain "(A ∧ B)"     -> atoms(A) ∪ atoms(B)
    If you later attach p._derived_hypothesis / p._derived_conclusion, this
    function will still work (we fall back to name/pretty text).
    """
    # Structured metadata path (if you add it later)
    hyp_obj = getattr(p, "_derived_hypothesis", None)
    phi_obj = getattr(p, "_derived_conclusion", None)
    if hyp_obj is not None and phi_obj is not None:
        hyp_txt = getattr(hyp_obj, "pretty", lambda: repr(hyp_obj))()
        phi_txt = getattr(phi_obj, "pretty", lambda: repr(phi_obj))()
        return {_canon_atom(x) for x in _split_top_level_conj(hyp_txt)} | {_canon_atom(phi_txt)}

    # Parse our derived naming convention
    name = getattr(p, "name", None)
    if name:
        m = _IFTHEN_RE.match(name)
        if m:
            hyp_txt = m.group("hyp").strip()
            phi_txt = m.group("phi").strip()
            return {_canon_atom(x) for x in _split_top_level_conj(hyp_txt)} | {_canon_atom(phi_txt)}

    # Plain predicate string
    s = (p.pretty() if hasattr(p, "pretty") else repr(p)).strip()
    return {_canon_atom(x) for x in _split_top_level_conj(s)}


from txgraffiti2025.forms.utils import BinOp, Const as _ConstClass
import numpy as np

def _split_const_times_expr(expr):
    """
    If expr = Const * something (commutative), return (Const, other_expr) else (None, None).
    Supports both BinOp.op == "*" and BinOp.fn is np.multiply.
    """
    if not isinstance(expr, BinOp):
        return None, None

    # Accept either representation
    op_is_star = getattr(expr, "op", None) == "*"
    fn_is_mul  = getattr(expr, "fn", None) in (np.multiply,)

    if not (op_is_star or fn_is_mul):
        return None, None

    L, R = expr.left, expr.right
    if isinstance(L, _ConstClass):
        return L, R
    if isinstance(R, _ConstClass):
        return R, L
    return None, None


def _safe_inv_scalar(x: float) -> float | None:
    return (1.0 / x) if np.isfinite(x) and x != 0.0 else None

def _hyp_key(H) -> str:
    """Stable string key for a hypothesis object."""
    if hasattr(H, "pretty"):
        try:
            return H.pretty()
        except Exception:
            pass
    return repr(H)


def _candidate_coeffs_from_constants(bank: dict[str, float], cfg: GenerationConfig):
    """
    Yield (expr_symbolic, numeric_value) where expr_symbolic is an Expr
    built from constant columns, and numeric_value is its scalar value on H'.
    """
    cols = list(bank.keys())
    # 1) a/(K+s)
    for K in cols:
        kval = bank[K]
        for a in cfg.numerators:
            for s in cfg.shifts:
                denom = kval + s
                if not _finite_nonzero_scalar(denom):
                    continue
                val = a / denom
                expr = Const(a) / (to_expr(K) + Const(s))
                yield (expr, val)

    # 2) (K1 + s1)/(K2 + s2)   (skip identical exprs)
    for i, K1 in enumerate(cols):
        k1v = bank[K1]
        for j, K2 in enumerate(cols):
            if i == j:
                continue
            k2v = bank[K2]
            for s1 in cfg.shifts:
                for s2 in cfg.shifts:
                    if (K1 == K2) and (s1 == s2):
                        continue  # avoid (K+s)/(K+s)
                    denom = k2v + s2
                    if not _finite_nonzero_scalar(denom):
                        continue
                    val = (k1v + s1) / denom
                    expr = (to_expr(K1) + Const(s1)) / (to_expr(K2) + Const(s2))
                    yield (expr, val)

    # 3) a/sqrt(K)
    for K in cols:
        kval = bank[K]
        if not _finite_scalar(kval) or kval <= 0.0:
            continue
        root = np.sqrt(kval)
        for a in cfg.numerators:
            val = a / root
            expr = Const(a) / sqrt(to_expr(K))
            yield (expr, val)

    # 4) K/(a+s)
    for K in cols:
        kval = bank[K]
        if not _finite_scalar(kval):
            continue
        for a in cfg.numerators:
            for s in cfg.shifts:
                denom = a + s
                if not _finite_nonzero_scalar(denom):
                    continue
                val = kval / denom
                expr = to_expr(K) / Const(denom)  # combine (a+s) into a Const
                yield (expr, val)

def _candidate_coeffs_from_symbolic_cols(df: pd.DataFrame, mask: np.ndarray, cols: list[str], cfg: GenerationConfig):
    """
    Build purely symbolic candidates that *aren't* constant on H'.
    We cannot compute a single numeric value -> we skip numeric closeness
    and only truth-test these later.
    """
    tried = 0
    limit = cfg.symbolic_col_limit
    for K in cols:
        if tried >= limit:
            break
        vals = df.loc[mask, K].to_numpy(dtype=float, copy=False)
        if not np.any(_finite_mask(vals)):
            continue
        # simple guards to avoid absurd expressions
        if np.nanmin(vals) <= 0:  # for sqrt and reciprocal with K+s in denom, we still test truth later
            pass
        # 1) a/(K+s)
        for a in cfg.numerators:
            for s in cfg.shifts:
                expr = Const(a) / (to_expr(K) + Const(s))
                yield (expr, None)
        # 2) a/sqrt(K)
        for a in cfg.numerators:
            expr = Const(a) / sqrt(to_expr(K))
            yield (expr, None)
        # 3) (K)/(a+s)
        for a in cfg.numerators:
            for s in cfg.shifts:
                denom = a + s
                if denom == 0:
                    continue
                expr = to_expr(K) / Const(denom)
                yield (expr, None)
        tried += 1

def _close(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps * (1.0 + max(abs(a), abs(b)))

def _replace_coeff_and_build(base_rel, coeff_expr, var_expr, condition) -> Conjecture | None:
    from txgraffiti2025.forms.generic_conjecture import Ge, Le
    rhs = coeff_expr * var_expr
    if isinstance(base_rel, Ge):
        return Conjecture(Ge(base_rel.left, rhs), condition)
    if isinstance(base_rel, Le):
        return Conjecture(Le(base_rel.left, rhs), condition)
    return None


# ---------------- utilities ----------------

from collections import defaultdict

def consolidate_qualitative(results: list, df: pd.DataFrame) -> list:
    """
    Given a list of QualResult, keep the most general hypothesis per 'conclusion':
      conclusion key := (y, x, method, direction)
    If multiple hypotheses are incomparable by inclusion, keep a minimal antichain,
    preferring larger support and higher |rho|.
    """
    # 1) group by conclusion
    buckets = defaultdict(list)
    for r in results:
        direction = "increasing" if r.rho >= 0 else "decreasing"
        key = (r.y, r.x, r.relation.method, direction)
        buckets[key].append(r)

    consolidated = []

    # 2) within each group, choose most-general hypothesis (by mask inclusion)
    for key, items in buckets.items():
        # precompute masks once
        masks = [ _hyp_mask(df, r.condition) for r in items ]
        supports = [ int(m.sum()) for m in masks ]

        # sort by (support desc, |rho| desc) so larger masks & stronger trends come first
        order = sorted(
            range(len(items)),
            key=lambda i: (supports[i], abs(items[i].rho)),
            reverse=True
        )

        kept_indices = []

        for i in order:
            m_i = masks[i]
            # if any already-kept mask includes this one, skip it (less general)
            dominated = any(_includes(masks[j], m_i) for j in kept_indices)
            if dominated:
                continue
            # remove any previously kept that this mask includes (i is more general)
            kept_indices = [ j for j in kept_indices if not _includes(m_i, masks[j]) ]
            kept_indices.append(i)

        # collect the kept representatives (most-general set)
        consolidated.extend(items[i] for i in kept_indices)

    # (optional) final ranking: by |rho| desc then support desc
    consolidated.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
    return consolidated


def to_frac_const(val: float, max_denom: int = 30) -> Const:
    """Convert float -> Const(Fraction) with bounded denominator."""
    return Const(Fraction(val).limit_denominator(max_denom))


def _touch_count(conj: Conjecture, df: pd.DataFrame) -> int:
    """
    # rows (under conj.condition) where lhs == rhs (numerically close).
    """
    try:
        lhs = conj.relation.left.eval(df)
        rhs = conj.relation.right.eval(df)
        if conj.condition is not None:
            # Predicate-style: condition.mask(df) → boolean Series
            if hasattr(conj.condition, "mask"):
                mask = conj.condition.mask(df).astype(bool).values
            else:
                # Callable-style: condition(df) → boolean Series/array
                mask = np.asarray(conj.condition(df), dtype=bool)
        else:
            mask = np.ones(len(df), dtype=bool)
        touch = np.isclose(lhs[mask], rhs[mask], rtol=1e-8, atol=1e-8)
        return int(touch.sum())
    except Exception:
        return 0

# === add these helpers (top-level, near Phase-1 helpers) =====================

def _pick_best_ge(t_arr: np.ndarray, rhs_variants):
    """
    target >= rhs. rhs_variants: list of (label, rhs_array, make_expr_fn).
    Returns (label, make_expr_fn) for the strongest true candidate:
      - chooses the one with largest mean(rhs_array).
    """
    best = None
    best_score = -np.inf
    for lab, rhs, make_expr in rhs_variants:
        ok = np.all(t_arr >= rhs)
        if not ok:
            continue
        score = float(np.mean(rhs))
        if score > best_score:
            best = (lab, make_expr)
            best_score = score
    return best

def _pick_best_le(t_arr: np.ndarray, rhs_variants):
    """
    target <= rhs. Prefer *smaller* rhs among those true:
      - chooses the one with smallest mean(rhs_array).
    """
    best = None
    best_score = +np.inf
    for lab, rhs, make_expr in rhs_variants:
        ok = np.all(t_arr <= rhs)
        if not ok:
            continue
        score = float(np.mean(rhs))
        if score < best_score:
            best = (lab, make_expr)
            best_score = score
    return best

def _mask_and_support(df: pd.DataFrame, H: Optional[Predicate]) -> Tuple[np.ndarray, int]:
    if H is None or H is TRUE:
        m = np.ones(len(df), dtype=bool)
    else:
        m = np.asarray(H.mask(df), dtype=bool)
    return m, int(m.sum())


# --- structural conjunct extraction & mask-equality guard ---

def _flatten_and_conjuncts(pred: Predicate) -> list[Predicate]:
    """
    Flatten an And-predicate tree into a list of primitive conjunct Predicates.
    If this is a derived Where with _derived_hypothesis attached, flatten that.
    Otherwise return [pred].
    """
    # If we attached _derived_hypothesis when creating Where(..) predicates, use it
    base = getattr(pred, "_derived_hypothesis", None)
    if base is not None:
        pred = base  # inspect the real hypothesis we derived from

    # Try structural flatten using AndPred
    try:
        from txgraffiti2025.forms.predicates import AndPred
        out = []
        stack = [pred]
        while stack:
            q = stack.pop()
            if isinstance(q, AndPred):
                # common shapes: q.left/q.right or q.args
                if hasattr(q, "left") and hasattr(q, "right"):
                    stack.append(q.left); stack.append(q.right)
                elif hasattr(q, "args"):
                    stack.extend(list(q.args))
                else:
                    out.append(q)
            else:
                out.append(q)
        return out
    except Exception:
        # if AndPred not importable or unknown shape
        return [pred]

def _bool_mask(df: pd.DataFrame, p: Predicate) -> np.ndarray:
    """Aligned bool numpy mask for a Predicate (NA→False)."""
    s = p.mask(df).reindex(df.index, fill_value=False)
    if s.dtype != bool:
        s = s.fillna(False).astype(bool, copy=False)
    return s.to_numpy()

def _same_bool_mask(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact equality on booleans."""
    return bool(np.array_equal(a, b))


def predicates_from_conjecture(
    conj: Conjecture,
    *,
    make_eq: bool = True,
    make_strict: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> List[Predicate]:
    """
    From (H ⇒ y ≤ C*x) or (H ⇒ y ≥ C*x), produce Predicates for:
      - on-boundary:   H ∧ (y == C*x)       if make_eq
      - strict side:   H ∧ (y <  C*x) or H ∧ (y > C*x)   if make_strict
        (direction chosen automatically from conj.relation type)

    Returns a list of Where(...) Predicates with readable names.
    """
    preds = []

    if make_eq:
        def fn_eq(df, _conj=conj, _rtol=rtol, _atol=atol):
            return _touch_mask(_conj, df, rtol=_rtol, atol=_atol)
        p_eq = Where(fn=fn_eq, name=_mk_name("onbound", conj))
        p_eq._derived_hypothesis = conj.condition        # <<< attach the base H
        # optional but nice to have:
        # p_eq._derived_conclusion = conj.relation        # the Le/Ge Expr or a pretty string
        preds.append(p_eq)

    if make_strict:
        if isinstance(conj.relation, Le):
            def fn_lt(df, _conj=conj, _rtol=rtol, _atol=atol):
                return _strict_mask(_conj, df, side="lt", rtol=_rtol, atol=_atol)
            p_lt = Where(fn=fn_lt, name=_mk_name("strict_lt", conj))
            p_lt._derived_hypothesis = conj.condition     # <<< attach the base H
            # p_lt._derived_conclusion = conj.relation
            preds.append(p_lt)
        elif isinstance(conj.relation, Ge):
            def fn_gt(df, _conj=conj, _rtol=rtol, _atol=atol):
                return _strict_mask(_conj, df, side="gt", rtol=_rtol, atol=_atol)
            p_gt = Where(fn=fn_gt, name=_mk_name("strict_gt", conj))
            p_gt._derived_hypothesis = conj.condition     # <<< attach the base H
            # p_gt._derived_conclusion = conj.relation
            preds.append(p_gt)

    return preds


class _EvalCache:
    """
    Cache column/transform evaluations on a *filtered* dataframe to avoid
    re-evaluating expressions in tight loops.
    """
    def __init__(self, df_temp: pd.DataFrame):
        self.df = df_temp
        self._col = {}
        self._sqrt = {}
        self._sq = {}

    def col(self, name: str) -> np.ndarray:
        if name not in self._col:
            a = to_expr(name).eval(self.df).values.astype(float, copy=False)
            self._col[name] = a
        return self._col[name]

    def sqrt_col(self, name):
        a = self._sqrt.get(name)
        if a is None:
            x = self.col(name)  # already float numpy array
            a = safe_sqrt_series(x)     # <- no warnings, NaN for invalid rows
            self._sqrt[name] = a
        return a

    def sq_col(self, name: str) -> np.ndarray:
        if name not in self._sq:
            x = self.col(name)
            self._sq[name] = np.square(x, dtype=float)
        return self._sq[name]


@dataclass
class QualResult:
    relation: MonotoneRelation
    condition: Predicate
    rho: float
    n: int
    support: int      # rows where condition is true
    x: str
    y: str

    def pretty(self, unicode_ops: bool = True) -> str:
        arrow = "↑" if (self.rho >= 0) else "↓"
        rho_tag = "ρₛ" if self.relation.method == "spearman" else "ρₚ"
        return (f"{self.y} {arrow} {self.x}  "
                f"({rho_tag}={self.rho:+.3f}, n={self.n}, support={self.support})  "
                f"under {getattr(self.condition,'pretty',lambda:repr(self.condition))()}")


# ---------------- TxGraffiti (Phase-1) ----------------

# @dataclass
# class GenerationConfig:
#     min_touch_keep: int = 3      # require at least this many touches
#     max_denom: int = 30          # fraction denominator cap
#     use_floor_ceil_if_true: bool = True

# extend GenerationConfig (near your Phase-1 class)
from dataclasses import dataclass
from typing import Optional, Iterable, Sequence

@dataclass
class GenerationConfig:
    min_touch_keep: int = 3
    max_denom: int = 30
    use_floor_ceil_if_true: bool = True

    # --- constant bank / reciprocal generalizer ---
    min_support_const: int = 5          # rows needed to declare a constant on H
    eps_const: float = 1e-12            # tolerance to detect "constant" under H
    coeff_eps_match: float = 1e-3       # numeric closeness when matching C to a candidate
    shifts: tuple[int, ...] = (-2, -1, 0, 1, 2)
    numerators: tuple[int, ...] = (1, 2, 3, 4)
    coeff_sources: str = "allow_symbolic_columns"  # or "constants_only"
    symbolic_col_limit: int = 10        # cap how many non-constant columns we try per H'



class TxGraffitiMini:
    """
    Phase-1 minimal class:
      • detects hypotheses and numeric columns
      • generates single-feature ratio bounds y ≥ c_min x and y ≤ c_max x
      • tries floor/ceil variants when they remain true (stronger)
      • ranks by touch count and applies Morgan filter
    """

    def __init__(self, df: pd.DataFrame, *, config: Optional[GenerationConfig] = None):
        self.df = df
        self.config = config or GenerationConfig()
        self.base_hyp = detect_base_hypothesis(df)
        self.hyps_all = enumerate_boolean_hypotheses(
            df,
            treat_binary_ints=True,
            include_base=True,
            include_pairs=True,
            skip_always_false=True,
        )
        self.hyps_kept, _ = simplify_and_dedup_hypotheses(
            df,
            self.hyps_all,
            min_support=10,
            treat_binary_ints=True,
        )
        self.bool_columns, self.numeric_columns = self._split_columns(df)

        # --- new: constant bank + ratio equalities ---
        self.const_bank: dict = {}                   # H -> {col: scalar_value}
        self.constant_ratio_equalities: list[Conjecture] = []
        self._build_constant_banks_and_ratio_equalities()

    @staticmethod
    def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        bool_cols: List[str] = []
        for c in df.columns:
            s = df[c]
            if s.dtype == bool:
                bool_cols.append(c)
            elif pd.api.types.is_integer_dtype(s):
                vals = pd.unique(s.dropna())
                try:
                    ints = set(int(v) for v in vals)
                except Exception:
                    continue
                if len(ints) <= 2 and ints.issubset({0, 1}):
                    bool_cols.append(c)
        num_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and c not in bool_cols]
        return bool_cols, num_cols

    @staticmethod
    def _mask_for(df: pd.DataFrame, hypothesis) -> np.ndarray:
        """
        Support both Predicate-style (has .mask(df)) and callable-style (hyp(df)).
        """
        if hasattr(hypothesis, "mask"):
            return np.asarray(hypothesis.mask(df), dtype=bool)
        return np.asarray(hypothesis(df), dtype=bool)


    # -------- public API --------

    def generate_single_feature_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        candidates: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Produce:
          • Lower bounds:  target ≥ c_min * x
          • Upper bounds:  target ≤ c_max * x
        Optionally add floor/ceil variants if they remain true.
        """
        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        cand_cols = list(candidates or self.numeric_columns)

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask):
                continue
            dfH = self.df.loc[mask]
            cache = _EvalCache(dfH)
            t_arr = target.eval(dfH).values.astype(float, copy=False)

            for xname in cand_cols:
                if xname == target_col:
                    continue
                x_arr = cache.col(xname)
                if np.min(x_arr) <= 0:   # avoid sign/log issues for ratios
                    continue

                rx = t_arr / x_arr
                cmin_f = float(np.min(rx))
                cmax_f = float(np.max(rx))

                cmin = to_frac_const(cmin_f, self.config.max_denom)
                cmax = to_frac_const(cmax_f, self.config.max_denom)

                x_expr = to_expr(xname)

                # y ≥ c_min x
                lb_expr = cmin * x_expr
                lb = Conjecture(Ge(target, lb_expr), H)
                lowers.append(lb)

                # y ≤ c_max x
                ub_expr = cmax * x_expr
                ub = Conjecture(Le(target, ub_expr), H)
                uppers.append(ub)

                if not self.config.use_floor_ceil_if_true:
                    continue

                # Strengthen when true:
                #  - Lower: y ≥ ceil(c_min x)
                ceil_arr = np.ceil(cmin_f * x_arr)
                if np.all(t_arr >= ceil_arr) and cmin.value.denominator > 1:
                    lowers.append(Conjecture(Ge(target, ceil(lb_expr)), H))

                #  - Upper: y ≤ floor(c_max x)
                floor_arr = np.floor(cmax_f * x_arr)
                if np.all(t_arr <= floor_arr) and cmax.value.denominator > 1:
                    uppers.append(Conjecture(Le(target, floor(ub_expr)), H))

        return lowers, uppers

    def _build_constant_banks_and_ratio_equalities(self):
        eps = self.config.eps_const
        min_sup = self.config.min_support_const
        max_denom = self.config.max_denom

        ratio_eqs: list[Conjecture] = []
        for H in self.hyps_kept:
            mask = _as_hyp_mask(H, self.df)
            bank = _make_const_bank_for_H(self.df, mask, self.numeric_columns, eps, min_sup)

            # store keyed by string (not by the unhashable object)
            self.const_bank[_hyp_key(H)] = bank

            # Emit ratio equalities right away
            for conj in _iter_ratio_eqs_from_bank(H, bank, max_denom):
                if _truthy(conj, self.df):
                    ratio_eqs.append(conj)

        self.constant_ratio_equalities = _unique_by_relation_and_condition(ratio_eqs)


    def generate_mixed_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        primary: Optional[Iterable[str]] = None,
        secondary: Optional[Iterable[str]] = None,
        weight: float = 0.5,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Phase-2: Build 2-feature mixes for bounds on target:
          • LOWER:  target ≥ w*(c_min*x) + w*(s_cmin*sqrt(y))   and   w*(c_min*x) + w*(q_cmin*y^2)
          • UPPER:  target ≤ w*(c_max*x) + w*(s_cmax*sqrt(y))   and   w*(c_max*x) + w*(q_cmax*y^2)
        We generate base + whole ceil/floor + split ceil/floor variants,
        pick the strongest one that stays true, and add it.

        Notes:
          - c_min/max from target / x
          - s_cmin/max from target / sqrt(y)
          - q_cmin/max from target / (y^2)
          - Requires x>0, y>0 on the hypothesis mask (to avoid weirdness).
        """
        assert 0 < weight <= 1.0, "weight must be in (0, 1]"

        target = to_expr(target_col)
        hyps_iter = (hyps or self.hyps_kept)
        prim_cols = list(primary or self.numeric_columns)
        sec_cols  = list(secondary or self.numeric_columns)

        lowers: List[Conjecture] = []
        uppers: List[Conjecture] = []

        w = float(weight)
        w_const = to_frac_const(weight, self.config.max_denom)

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask):
                continue
            dfH = self.df.loc[mask]
            cache = _EvalCache(dfH)
            t_arr = target.eval(dfH).values.astype(float, copy=False)

            for xname in prim_cols:
                if xname == target_col:
                    continue
                x_arr = cache.col(xname)
                if np.min(x_arr) <= 0:
                    continue

                # ratio constants wrt x
                rx = t_arr / x_arr
                cmin_f = float(np.min(rx))
                cmax_f = float(np.max(rx))

                # iterate over secondaries
                for yname in sec_cols:
                    if yname == target_col:
                        continue

                    y_arr = cache.col(yname)
                    if np.min(y_arr) <= 0:
                        continue

                    # ----- sqrt mix -------------------------------------------------
                    sqrt_y_arr = cache.sqrt_col(yname)
                    r_sqrt = t_arr / sqrt_y_arr
                    s_cmin_f = float(np.min(r_sqrt))
                    s_cmax_f = float(np.max(r_sqrt))

                    # arrays (keep array-side pure floats)
                    mix_lower_arr = w * (cmin_f * x_arr + s_cmin_f * sqrt_y_arr)
                    mix_upper_arr = w * (cmax_f * x_arr + s_cmax_f * sqrt_y_arr)

                    # candidates: LOWER (>=)
                    lower_mix_variants = [
                        ("base",
                         mix_lower_arr,
                         lambda xname=xname, yname=yname, cmin=cmin_f, smin=s_cmin_f: (
                             w_const * to_frac_const(cmin) * to_expr(xname)
                           + w_const * to_frac_const(smin) * sqrt(to_expr(yname))
                         )),
                        ("ceil whole",
                         np.ceil(mix_lower_arr),
                         lambda xname=xname, yname=yname, cmin=cmin_f, smin=s_cmin_f: ceil(
                             w_const * to_frac_const(cmin) * to_expr(xname)
                           + w_const * to_frac_const(smin) * sqrt(to_expr(yname))
                         )),
                        ("ceil-split-1",
                         np.ceil(w * cmin_f * x_arr) + np.ceil(w * s_cmin_f * sqrt_y_arr) - 1.0,
                         lambda xname=xname, yname=yname, cmin=cmin_f, smin=s_cmin_f: (
                             ceil(w_const * to_frac_const(cmin) * to_expr(xname))
                           + ceil(w_const * to_frac_const(smin) * sqrt(to_expr(yname)))
                           - Const(1)
                         )),
                    ]
                    choice = _pick_best_ge(t_arr, lower_mix_variants)
                    if choice is not None:
                        _, make_expr = choice
                        lowers.append(Conjecture(Ge(target, make_expr()), H))

                    # candidates: UPPER (<=)
                    upper_mix_variants = [
                        ("base",
                         mix_upper_arr,
                         lambda xname=xname, yname=yname, cmax=cmax_f, smax=s_cmax_f: (
                             w_const * to_frac_const(cmax) * to_expr(xname)
                           + w_const * to_frac_const(smax) * sqrt(to_expr(yname))
                         )),
                        ("floor whole",
                         np.floor(mix_upper_arr),
                         lambda xname=xname, yname=yname, cmax=cmax_f, smax=s_cmax_f: floor(
                             w_const * to_frac_const(cmax) * to_expr(xname)
                           + w_const * to_frac_const(smax) * sqrt(to_expr(yname))
                         )),
                        ("floor-split",
                         np.floor(w * cmax_f * x_arr) + np.floor(w * s_cmax_f * sqrt_y_arr),
                         lambda xname=xname, yname=yname, cmax=cmax_f, smax=s_cmax_f: (
                             floor(w_const * to_frac_const(cmax) * to_expr(xname))
                           + floor(w_const * to_frac_const(smax) * sqrt(to_expr(yname)))
                         )),
                    ]
                    choice = _pick_best_le(t_arr, upper_mix_variants)
                    if choice is not None:
                        _, make_expr = choice
                        uppers.append(Conjecture(Le(target, make_expr()), H))

                    # ----- square mix ------------------------------------------------
                    y_sq_arr = cache.sq_col(yname)
                    r_sq = t_arr / y_sq_arr
                    q_cmin_f = float(np.min(r_sq))
                    q_cmax_f = float(np.max(r_sq))

                    mix_lower_sq_arr = w * (cmin_f * x_arr + q_cmin_f * y_sq_arr)
                    mix_upper_sq_arr = w * (cmax_f * x_arr + q_cmax_f * y_sq_arr)

                    # ≥ variants (base + whole ceil)
                    lower_sq_variants = [
                        ("base",
                         mix_lower_sq_arr,
                         lambda xname=xname, yname=yname, cmin=cmin_f, qmin=q_cmin_f: (
                             w_const * to_frac_const(cmin) * to_expr(xname)
                           + w_const * to_frac_const(qmin) * (to_expr(yname) ** to_frac_const(2))
                         )),
                        ("ceil whole",
                         np.ceil(mix_lower_sq_arr),
                         lambda xname=xname, yname=yname, cmin=cmin_f, qmin=q_cmin_f: ceil(
                             w_const * to_frac_const(cmin) * to_expr(xname)
                           + w_const * to_frac_const(qmin) * (to_expr(yname) ** to_frac_const(2))
                         )),
                    ]
                    choice = _pick_best_ge(t_arr, lower_sq_variants)
                    if choice is not None:
                        _, make_expr = choice
                        lowers.append(Conjecture(Ge(target, make_expr()), H))

                    # ≤ variants (base + whole floor)
                    upper_sq_variants = [
                        ("base",
                         mix_upper_sq_arr,
                         lambda xname=xname, yname=yname, cmax=cmax_f, qmax=q_cmax_f: (
                             w_const * to_frac_const(cmax) * to_expr(xname)
                           + w_const * to_frac_const(qmax) * (to_expr(yname) ** to_frac_const(2))
                         )),
                        ("floor whole",
                         np.floor(mix_upper_sq_arr),
                         lambda xname=xname, yname=yname, cmax=cmax_f, qmax=q_cmax_f: floor(
                             w_const * to_frac_const(cmax) * to_expr(xname)
                           + w_const * to_frac_const(qmax) * (to_expr(yname) ** to_frac_const(2))
                         )),
                    ]
                    choice = _pick_best_le(t_arr, upper_sq_variants)
                    if choice is not None:
                        _, make_expr = choice
                        uppers.append(Conjecture(Le(target, make_expr()), H))

        return lowers, uppers

    def generate_targeted_product_bounds(
        self,
        target_col: str,
        *,
        hyps=None,
        x_candidates=None,
        yz_candidates=None,
        require_pos: bool = True,
        enable_cancellation: bool = True,
        allow_x_equal_yz: bool = True,
    ):
        """
        Phase-3: targeted product bounds.
        Emit only:
            (H) ⇒ T*x ≤ y*z    and    (H) ⇒ T*x ≥ y*z
        with optional triviality skipping and safe cancellation when x == y or x == z.

        Parameters
        ----------
        target_col : str
            The fixed target invariant T.
        hyps : iterable, optional
            Hypotheses to use (defaults to self.hyps_kept).
        x_candidates : iterable[str], optional
            Columns allowed as x (defaults to numeric columns except target).
        yz_candidates : iterable[str], optional
            Columns to form pairs (y,z) (defaults to all numeric columns).
        require_pos : bool
            If True, truth tests and triviality checks only use rows where T,x,y,z > 0
            and finite; this makes monotone reasoning valid and avoids sign flips.
        enable_cancellation : bool
            If True and (x==y or x==z) is strictly positive on the valid mask,
            emit the reduced inequality (T ≤ other or T ≥ other) instead of the product.
            T is never canceled.
        allow_x_equal_yz : bool
            If False, we skip pairs where x == y or x == z entirely (stronger pruning).
        """
        target_expr = to_expr(target_col)
        hyps_iter = hyps or self.hyps_kept

        # default candidate sets
        x_cands = list(x_candidates or [c for c in self.numeric_columns if c != target_col])
        yz_cands = list(yz_candidates or list(self.numeric_columns))

        uppers = []  # (H) ⇒ T*x ≤ y*z
        lowers = []  # (H) ⇒ T*x ≥ y*z

        for H in hyps_iter:
            mask = self._mask_for(self.df, H)
            if not np.any(mask):
                continue
            dfH = self.df.loc[mask]
            cache = _EvalCache(dfH)

            T_arr = target_expr.eval(dfH).values.astype(float, copy=False)
            T_pos_guard = (T_arr > 0.0) if require_pos else np.ones_like(T_arr, dtype=bool)

            # pre-evaluate all candidate columns once
            arrays = {c: cache.col(c) for c in set(x_cands) | set(yz_cands)}
            finite = {c: np.isfinite(arrays[c]) for c in arrays}

            for x in x_cands:
                x_arr = arrays[x]
                x_ok = finite[x] & (x_arr > 0.0 if require_pos else np.isfinite(x_arr))
                if not np.any(x_ok):
                    continue

                # pairs (y,z) with replacement (y==z allowed)
                for (y, z) in combinations_with_replacement(yz_cands, 2):
                    if not allow_x_equal_yz and (x == y or x == z):
                        continue

                    y_arr = arrays[y]; z_arr = arrays[z]
                    base_valid = finite[y] & finite[z] & finite[x] & np.isfinite(T_arr)
                    if require_pos:
                        base_valid &= (T_pos_guard & (y_arr > 0.0) & (z_arr > 0.0) & (x_arr > 0.0))

                    if not np.any(base_valid):
                        continue

                    # Caro-triviality (only meaningful with positivity)
                    if require_pos:
                        if _caro_trivial_masked_targeted(T_arr, x_arr, y_arr, z_arr, base_valid, upper=True):
                            pass  # skip the upper trivial candidate check later
                        # we won't "continue" yet because lower form triviality differs.
                        # We'll check each orientation separately below.

                    # ----- optional cancellation (stronger form) -----
                    # If x == y (resp. x == z) and that factor is strictly positive on valid rows,
                    # we can reduce T*x ≤ y*z to T ≤ z (resp. T ≤ y). Similarly for ≥.
                    canceled_upper_appended = False
                    canceled_lower_appended = False

                    if enable_cancellation:
                        # x == y ?
                        if x == y and _strictly_pos_on(x_arr, base_valid):
                            # upper: T*x ≤ x*z  -> T ≤ z
                            if _all_le_on_mask(T_arr, z_arr, base_valid):
                                lowers_or_uppers = Conjecture(Le(target_expr, to_expr(z)), H)
                                uppers.append(lowers_or_uppers)
                                canceled_upper_appended = True
                            # lower: T*x ≥ x*z  -> T ≥ z
                            if _all_le_on_mask(z_arr, T_arr, base_valid):
                                lowers.append(Conjecture(Ge(target_expr, to_expr(z)), H))
                                canceled_lower_appended = True

                        # x == z ?
                        if x == z and _strictly_pos_on(x_arr, base_valid):
                            # upper: T*x ≤ y*x  -> T ≤ y
                            if _all_le_on_mask(T_arr, y_arr, base_valid):
                                uppers.append(Conjecture(Le(target_expr, to_expr(y)), H))
                                canceled_upper_appended = True
                            # lower: T*x ≥ y*x  -> T ≥ y
                            if _all_le_on_mask(y_arr, T_arr, base_valid):
                                lowers.append(Conjecture(Ge(target_expr, to_expr(y)), H))
                                canceled_lower_appended = True

                        # If cancellation yielded a valid conjecture, we prefer it over the product form.
                        # (Still allow the non-canceled product too, if you want both—here we prefer the reduced one.)

                    # ----- non-canceled product forms -----
                    L = T_arr * x_arr
                    R = y_arr * z_arr

                    # upper: T*x ≤ y*z
                    if not canceled_upper_appended:
                        # if require_pos: skip trivial uppers
                        if not (require_pos and _caro_trivial_masked_targeted(T_arr, x_arr, y_arr, z_arr, base_valid, upper=True)):
                            if _all_le_on_mask(L, R, base_valid):
                                lhs = target_expr * to_expr(x)
                                rhs = to_expr(y) * to_expr(z)
                                uppers.append(Conjecture(Le(lhs, rhs), H))

                    # lower: T*x ≥ y*z
                    if not canceled_lower_appended:
                        if not (require_pos and _caro_trivial_masked_targeted(T_arr, x_arr, y_arr, z_arr, base_valid, upper=False)):
                            if _all_le_on_mask(R, L, base_valid):
                                lhs = target_expr * to_expr(x)
                                rhs = to_expr(y) * to_expr(z)
                                lowers.append(Conjecture(Ge(lhs, rhs), H))

        return lowers, uppers

    def generate_qualitative_relations(
        self,
        *,
        y_targets: Optional[Iterable[str]] = None,
        x_candidates: Optional[Iterable[str]] = None,
        hyps: Optional[Iterable[Predicate]] = None,
        method: CorrMethod = "spearman",
        min_abs_rho: float = 0.35,
        min_n: int = 12,
        drop_constant: bool = True,
        top_k_per_hyp: Optional[int] = None,
    ) -> List[QualResult]:
        """
        Mine qualitative/monotone tendencies y vs x under each hypothesis.

        - Picks direction by sign(rho) automatically.
        - Accepts iff |rho| >= min_abs_rho and n >= min_n on that hypothesis' mask.
        - Optionally keeps only top_k_per_hyp by |rho| for readability.

        Returns a list of QualResult, sorted by |rho| desc then support desc.
        """
        y_cols = list(y_targets or self.numeric_columns)
        x_cols = list(x_candidates or self.numeric_columns)
        hyps_iter = list(hyps or self.hyps_kept)

        results: List[QualResult] = []

        for H in hyps_iter:
            mask, support = _mask_and_support(self.df, H)
            if support < min_n:
                continue
            dfH = self.df.loc[mask]

            # pre-coerce numerics (fast and uniform)
            num_df = dfH.apply(pd.to_numeric, errors="coerce")

            hyp_results: List[QualResult] = []
            for y in y_cols:
                ys = num_df.get(y)
                if ys is None:
                    continue
                if drop_constant and ys.nunique(dropna=True) <= 1:
                    continue

                for x in x_cols:
                    if x == y:
                        continue
                    xs = num_df.get(x)
                    if xs is None:
                        continue
                    if drop_constant and xs.nunique(dropna=True) <= 1:
                        continue

                    # drop NaNs pairwise
                    valid = xs.notna() & ys.notna()
                    n = int(valid.sum())
                    if n < min_n:
                        continue

                    # compute rho (re-using MonotoneRelation._corr logic via the class)
                    mr = MonotoneRelation(x=x, y=y, direction="increasing",
                                        method=method, min_abs_rho=min_abs_rho, min_n=min_n)
                    rho = mr._corr(xs[valid].to_numpy(), ys[valid].to_numpy())
                    if not np.isfinite(rho):
                        continue

                    if abs(rho) < float(min_abs_rho):
                        continue

                    # set direction by sign
                    mr.direction = "increasing" if rho >= 0 else "decreasing"

                    hyp_results.append(QualResult(
                        relation=mr,
                        condition=H,
                        rho=float(rho),
                        n=n,
                        support=support,
                        x=x,
                        y=y
                    ))

            # optional per-hypothesis truncation
            if top_k_per_hyp is not None and top_k_per_hyp > 0:
                hyp_results.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
                hyp_results = hyp_results[:top_k_per_hyp]

            results.extend(hyp_results)

        # global ranking
        results.sort(key=lambda r: (abs(r.rho), r.support), reverse=True)
        return results

    @staticmethod
    def pretty_qualitative_block(title: str, items: List[QualResult], max_items: int = 40):
        print(f"\n=== {title} ===")
        for i, r in enumerate(items[:max_items], 1):
            Hs = getattr(r.condition, "pretty", lambda: repr(r.condition))()
            arrow = "↑" if r.rho >= 0 else "↓"
            tag = "ρₛ" if r.relation.method == "spearman" else "ρₚ"
            print(f"{i:3d}. [{Hs}]  {r.y} {arrow} {r.x}   ({tag}={r.rho:+.3f}, n={r.n}, support={r.support})")

    @staticmethod
    def pretty_qualitative_block_descriptive(title: str, items: list, max_items: int = 40):
        print(f"\n=== {title} ===")
        for i, r in enumerate(items[:max_items], 1):
            Hs = getattr(r.condition, "pretty", lambda: repr(r.condition))()
            rho_tag = "ρₛ" if r.relation.method == "spearman" else "ρₚ"
            dir_text = (
                "tends to increase (monotone increasing trend)"
                if r.rho >= 0
                else "tends to decrease (monotone decreasing trend)"
            )

            print(f"\n{i:3d}. Hypothesis: {Hs}")
            print(f"     Response: {r.y}")
            print(f"     Predictor: {r.x}")
            print(f"     Observation: As {r.x} increases, {r.y} {dir_text}.")
            print(f"     Correlation: {rho_tag} = {r.rho:+.3f} (n={r.n}, support={r.support})")

    def _postfilter_conj_trivialities(self, eqs, incs):
        # Drop inclusions that are trivial by conjunct containment
        incs_filtered = [inc for inc in incs if not _is_trivial_conjunctive_inclusion(inc)]
        # (Optionally) drop equivalences that are just A ≡ (A ∧ Q) — you may already filter these.
        return eqs, incs_filtered

    def _transitive_reduce_inclusions(
        self,
        inclusions: List[ClassInclusion],
        mask_cache: dict[str, np.ndarray],
    ) -> List[ClassInclusion]:
        """
        Remove inclusions implied by others: if A⊆B and B⊆C exist, drop A⊆C.
        Uses mask inclusion; fine for tens/hundreds of classes.
        """
        # Map predicate key to index in a compact list
        nodes = sorted({ _pred_key(inc.A) for inc in inclusions } | { _pred_key(inc.B) for inc in inclusions })
        idx = {k:i for i,k in enumerate(nodes)}
        mks = {k: mask_cache[k] for k in nodes}

        # Build adjacency (A -> B)
        adj = [[False]*len(nodes) for _ in nodes]
        edge_obj = {}  # (i,j) -> ClassInclusion
        for inc in inclusions:
            i = idx[_pred_key(inc.A)]
            j = idx[_pred_key(inc.B)]
            adj[i][j] = True
            edge_obj[(i,j)] = inc

        # For each edge i->k, check if there is a j so that i->j and j->k; if yes, i->k is redundant
        keep = set(edge_obj.keys())
        for i in range(len(nodes)):
            for k in range(len(nodes)):
                if not adj[i][k]:
                    continue
                # Is there a middle node j with i->j and j->k?
                redundant = any(adj[i][j] and adj[j][k] for j in range(len(nodes)) if j != i and j != k)
                if redundant:
                    keep.discard((i,k))

        # Return the kept edges
        out = []
        seen = set()
        for (i,j) in keep:
            inc = edge_obj[(i,j)]
            sig = inc.signature()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(inc)
        return out

    def discover_class_relations(
        self,
        *,
        predicates: Optional[Iterable[Predicate]] = None,
        include_bool_columns: bool = False,
        min_support_A: int = 3,
        skip_trivial_equiv: bool = True,
        # NEW:
        disallow_shared_atoms: bool = True,
        ambient_atoms: Optional[Set[str]] = None,
    ) -> Tuple[List[ClassEquivalence], List[ClassInclusion]]:
        """
        R₁ discovery with structural & mask guards:

        - Equivalences A ≡ B are kept only when masks equal AND neither side's
        atom set is a proper subset of the other's ((A∧Q) ≡ A is skipped).
        - Inclusions A ⊆ B are created only if:
            • no violations (mask test),
            • NOT (atoms(B) ⊆ atoms(A))  [blocks subset/tautologies],
            • if disallow_shared_atoms: (atoms(A) ∩ atoms(B)) \ ambient_atoms == ∅,
            • and (NEW) B is NOT already one of A’s conjuncts (mask-equality test).
        """
        # 1) candidates
        cand: List[Predicate] = list(predicates or self.hyps_kept)
        if include_bool_columns:
            from txgraffiti2025.forms.predicates import Predicate as _P
            for c in getattr(self, "bool_columns", []):
                try:
                    cand.append(_P.from_column(c))
                except Exception:
                    pass

        # 2) caches
        mask_cache: dict[str, np.ndarray] = {}
        obj_cache:  dict[str, Predicate]  = {}
        atoms_cache: dict[str, Set[str]]  = {}

        for p in cand:
            k = _pred_key(p)
            if k in mask_cache:
                continue
            try:
                m = _mask_bool(self.df, p)
            except Exception:
                continue
            mask_cache[k]  = m
            obj_cache[k]   = p
            atoms_cache[k] = _atoms_for_pred(p)  # you already added this helper

        keys = list(mask_cache.keys())
        n = len(self.df)
        all_true  = np.ones(n, dtype=bool)
        all_false = np.zeros(n, dtype=bool)

        # Ambient exemptions for the shared-atoms rule
        ambient_atoms = set() if ambient_atoms is None else { _canon_atom(a) for a in ambient_atoms }

        equivalences: List[ClassEquivalence] = []
        inclusions:   List[ClassInclusion]   = []

        # 3a) Equivalences (unordered pairs)
        for i, ki in enumerate(keys):
            mi = mask_cache[ki]; Ai = obj_cache[ki]; atoms_i = atoms_cache[ki]
            for kj in keys[i+1:]:
                mj = mask_cache[kj]; Aj = obj_cache[kj]; atoms_j = atoms_cache[kj]
                if not _same_mask(mi, mj):
                    continue
                if skip_trivial_equiv and (np.array_equal(mi, all_true) or np.array_equal(mi, all_false)):
                    continue
                # Skip (A∧Q) ≡ A kind of trivialities (one atom set strictly contains the other)
                if (atoms_i.issubset(atoms_j) and atoms_i != atoms_j) or (atoms_j.issubset(atoms_i) and atoms_i != atoms_j):
                    continue
                equivalences.append(ClassEquivalence(Ai, Aj))

        # 3b) Inclusions (directed)
        for ki in keys:
            Ai = obj_cache[ki]; mi = mask_cache[ki]; atoms_i = atoms_cache[ki]
            suppA = _support(mi)
            if suppA < int(min_support_A) or np.array_equal(mi, all_false):
                continue

            # Precompute masks for *conjuncts* of Ai once per Ai
            lhs_conj = _flatten_and_conjuncts(Ai)  # you added this helper
            lhs_conj_masks = [ _bool_mask(self.df, c) for c in lhs_conj ]

            for kj in keys:
                if kj == ki:
                    continue
                Aj = obj_cache[kj]; mj = mask_cache[kj]; atoms_j = atoms_cache[kj]

                # Not informative
                if np.array_equal(mj, all_true):
                    continue

                # Mask inclusion test: A ⊆ B iff no violations (A & ~B) anywhere
                if np.any(mi & ~mj):
                    continue

                # Structural guard 1: block subset/tautology
                if atoms_j.issubset(atoms_i):
                    continue

                # Structural guard 2: disallow shared atoms (except ambient)
                if disallow_shared_atoms:
                    inter = (atoms_i & atoms_j) - ambient_atoms
                    if inter:
                        continue

                # NEW Mask guard: skip if RHS is already one of LHS conjuncts (by mask equality)
                if any(_same_bool_mask(cm, mj) for cm in lhs_conj_masks):
                    continue

                inclusions.append(ClassInclusion(Ai, Aj))

        # 4) de-dup
        def _uniq_by_sig(items):
            seen = set(); out = []
            for x in items:
                s = x.signature()
                if s in seen: continue
                seen.add(s); out.append(x)
            return out

        equivalences = _uniq_by_sig(equivalences)
        inclusions   = _uniq_by_sig(inclusions)

        # 5) transitive reduction (optional, mask-based)
        inclusions = self._transitive_reduce_inclusions(inclusions, mask_cache)

        # 6) sort
        def _equiv_support(eq: ClassEquivalence) -> int:
            return _support(mask_cache[_pred_key(eq.A)])
        equivalences.sort(key=lambda e: (_equiv_support(e), e.pretty()), reverse=True)
        inclusions.sort(key=lambda inc: (_support(mask_cache[_pred_key(inc.A)]), inc.pretty()), reverse=True)

        return equivalences, inclusions

    def _postfilter_class_relations(self, eqs, incs, mask_cache):
        # 2.1 Suppress trivial equivalences of the form (A ∧ Q) ≡ A
        filtered_eqs = []
        for e in eqs:
            kA = e.A.pretty() if hasattr(e.A, "pretty") else repr(e.A)
            kB = e.B.pretty() if hasattr(e.B, "pretty") else repr(e.B)
            mA = mask_cache[kA]; mB = mask_cache[kB]
            if _is_conjunction_redundancy(e.A, e.B, mA, mB):
                continue
            filtered_eqs.append(e)

        # 2.2 Build a quick set of equivalent pairs (unordered) to suppress mirrored inclusions
        eq_key = set()
        for e in filtered_eqs:
            a = e.A.pretty() if hasattr(e.A, "pretty") else repr(e.A)
            b = e.B.pretty() if hasattr(e.B, "pretty") else repr(e.B)
            eq_key.add(tuple(sorted([a, b])))

        # 2.3 Suppress inclusions that are part of an equivalence, and trivial conjunction inclusions
        filtered_incs = []
        for inc in incs:
            a = inc.A.pretty() if hasattr(inc.A, "pretty") else repr(inc.A)
            b = inc.B.pretty() if hasattr(inc.B, "pretty") else repr(inc.B)
            # Drop A⊆B if A≡B exists
            if tuple(sorted([a, b])) in eq_key:
                continue
            mA = mask_cache[a]; mB = mask_cache[b]
            # Drop inclusions like A ⊆ (A ∧ Q) or (A ∧ Q) ⊆ A (they are trivial strengthen/weaken)
            if _is_conjunction_redundancy(inc.A, inc.B, mA, mB):
                continue
            filtered_incs.append(inc)

        return filtered_eqs, filtered_incs


    @staticmethod
    def pretty_class_relations(title: str, eqs: List[ClassEquivalence], incs: List[ClassInclusion], df: pd.DataFrame, max_items: int = 60):
        print(f"\n=== {title} ===")
        print("\n-- Equivalences (A ≡ B) --")
        for i, e in enumerate(eqs[:max_items], 1):
            print(f"{i:3d}. {e.pretty()}  [violations={e.violation_count(df)}]")

        print("\n-- Inclusions (A ⊆ B) --")
        for i, inc in enumerate(incs[:max_items], 1):
            suppA = int(inc.A.mask(df).sum())
            viol  = inc.violation_count(df)
            print(f"{i:3d}. {inc.pretty()}  [support(A)={suppA}, violations={viol}]")


    def run_targeted_product_pipeline(
        self,
        target_col: str,
        *,
        require_pos: bool = True,
        enable_cancellation: bool = True,
        allow_x_equal_yz: bool = True,
        x_candidates=None,
        yz_candidates=None,
    ):
        """
        Convenience wrapper: generate targeted product bounds, then rank/filter each side.
        """
        lowers, uppers = self.generate_targeted_product_bounds(
            target_col,
            require_pos=require_pos,
            enable_cancellation=enable_cancellation,
            allow_x_equal_yz=allow_x_equal_yz,
            x_candidates=x_candidates,
            yz_candidates=yz_candidates,
        )
        lowers_final = self.rank_and_filter(lowers)
        uppers_final = self.rank_and_filter(uppers)
        return lowers_final, uppers_final

    def add_derived_predicates_from_top_conjectures(
        self,
        conjs: Sequence[Conjecture],
        *,
        top_quantile: float = 0.10,   # top 10% by touch count
        min_support: int = 10,        # keep derived predicates only if they have at least this many rows
        make_eq: bool = True,
        make_strict: bool = True,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        dedupe_by_mask: bool = True,
    ) -> List[Predicate]:
        """
        - Rank input conjectures by *touch count* on their own hypothesis.
        - Take top `top_quantile`.
        - For each selected conjecture, produce boundary and strict predicates.
        - Keep only those with support >= `min_support`.
        - Optionally de-duplicate predicates by mask equality.
        - Extend `self.hyps_kept` with these new predicates.
        - Return the list of newly added predicates.
        """
        if not conjs:
            return []

        # compute touch counts
        def _touch_count_local(c: Conjecture) -> int:
            try:
                return int(_touch_mask(c, self.df, rtol=rtol, atol=atol).sum())
            except Exception:
                return 0

        scored = [(c, _touch_count_local(c)) for c in conjs]
        scored.sort(key=lambda t: t[1], reverse=True)

        # take top quantile (at least 1 if non-empty)
        k = max(1, int(np.ceil(len(scored) * float(top_quantile))))
        top = [c for (c, _) in scored[:k]]

        # build predicates
        derived: List[Predicate] = []
        for c in top:
            for p in predicates_from_conjecture(c, make_eq=make_eq, make_strict=make_strict, rtol=rtol, atol=atol):
                # filter by support
                m = _mask_from_pred(self.df, p)
                if int(m.sum()) >= int(min_support):
                    derived.append(p)

        # optional de-duplication by mask equality (predicates can collide)
        if dedupe_by_mask and derived:
            # compute masks
            masks = [ _mask_from_pred(self.df, p) for p in derived ]
            keep_idx = []
            for i, mi in enumerate(masks):
                duplicate = False
                for j in keep_idx:
                    if np.array_equal(mi, masks[j]):
                        duplicate = True
                        break
                if not duplicate:
                    keep_idx.append(i)
            derived = [derived[i] for i in keep_idx]

        # extend hypothesis set
        self.hyps_kept.extend(derived)

        return derived



    def run_mixed_pipeline(
        self,
        target_col: str,
        *,
        weight: float = 0.5,
        primary: Optional[Iterable[str]] = None,
        secondary: Optional[Iterable[str]] = None,
    ):
        """
        Convenience: generate mixes and return ranked/filtered lists.
        """
        lows, ups = self.generate_mixed_bounds(
            target_col,
            weight=weight,
            primary=primary,
            secondary=secondary,
        )
        lows_final = self.rank_and_filter(lows)
        ups_final  = self.rank_and_filter(ups)
        return lows_final, ups_final

    def rank_and_filter(
        self,
        conjs: List[Conjecture],
        *,
        min_touch: Optional[int] = None
    ) -> List[Conjecture]:
        """
        Sort by touch count (desc), drop below threshold, apply Morgan filter.
        """
        m = min_touch if min_touch is not None else self.config.min_touch_keep
        conjs_sorted = sorted(conjs, key=lambda c: _touch_count(c, self.df), reverse=True)
        conjs_kept = [c for c in conjs_sorted if _touch_count(c, self.df) >= m]
        mf = morgan_filter(self.df, conjs_kept)
        return list(mf.kept)

    # convenience: run Phase-1 end-to-end
    def run_single_feature_pipeline(self, target_col: str):
        lows, ups = self.generate_single_feature_bounds(target_col)
        lows_final = self.rank_and_filter(lows)
        ups_final = self.rank_and_filter(ups)
        return lows_final, ups_final

    # pretty printer
    @staticmethod
    def pretty_block(title: str, conjs: List[Conjecture], max_items: int = 100) -> None:
        print(f"\n=== {title} ===")
        for i, c in enumerate(conjs[:max_items], start=1):
            print(f"{i:3d}. {c.pretty(arrow='⇒')}")

    def generalize_from_reciprocal_patterns(
        self,
        base_conjectures: Sequence[Conjecture],
        *,
        coeff_sources: Optional[str] = None,   # "constants_only" or "allow_symbolic_columns"
    ) -> list[Conjecture]:
        """
        Try to replace a numeric coefficient C in monomial bounds (T ≥ C·X or T ≤ C·X)
        with a symbolic expression built from:
        (A) columns that are constant on a broader hypothesis H', and/or
        (B) (optional) symbolic columns on H' (not constant),
        accepting only if the new conjecture is TRUE on H'.

        IMPORTANT: If the proposed hypothesis H' is NOT strictly broader than the base
        hypothesis H (mask inclusion + strictly more rows), we KEEP the base hypothesis H.

        Returns a de-duplicated list of accepted proposals.
        """
        cfg = self.config
        mode = coeff_sources or cfg.coeff_sources

        proposals: list[Conjecture] = []

        for base in base_conjectures:
            rel = base.relation
            C_const, var_expr = _split_const_times_expr(rel.right)
            if C_const is None:
                # not a numeric-constant times variable; skip
                continue

            C_float = float(C_const.value)
            base_mask = _as_hyp_mask(base.condition, self.df)
            base_sup = int(base_mask.sum())

            # iterate broader hypotheses H' (sorted by largest support first)
            for Hp in self._broader_hypotheses(base.condition):
                p_mask = _as_hyp_mask(Hp, self.df)

                # Skip if Hp has too little support to be meaningful
                if int(p_mask.sum()) < cfg.min_support_const:
                    continue

                # candidate set (A): from constants on H'
                bank = self.const_bank.get(_hyp_key(Hp), {})
                found_any_for_this_Hp = False

                for coeff_expr, coeff_val in _candidate_coeffs_from_constants(bank, cfg):
                    # numeric closeness gate to original C
                    if not _close(coeff_val, C_float, cfg.coeff_eps_match):
                        continue

                    # Build proposal with Hp as the target condition
                    prop = _replace_coeff_and_build(rel, coeff_expr, var_expr, Hp)
                    if prop is None:
                        continue

                    # Accept only if TRUE on Hp
                    if not _truthy(prop, self.df):
                        continue

                    # --- keep base hypothesis unless Hp is STRICTLY broader ---
                    if _mask_includes(base_mask, p_mask) and (p_mask.sum() > base_sup):
                        # proper generalization
                        pass  # keep Hp
                        if not getattr(prop, "name", None):
                            prop.name = f"recip-generalized:{getattr(Hp,'pretty',lambda:repr(Hp))()}"
                    else:
                        # not strictly broader -> keep base H
                        prop.condition = base.condition
                        if not getattr(prop, "name", None):
                            prop.name = "recip-same"

                    proposals.append(prop)
                    found_any_for_this_Hp = True

                if found_any_for_this_Hp:
                    # We prefer the broadest Hp where we found valid constant-driven matches.
                    # Move to next base conjecture (or keep scanning other Hp if you want multiple).
                    continue

                # candidate set (B): from symbolic (non-constant) columns on H'
                if mode == "allow_symbolic_columns":
                    sym_hits_for_Hp = []
                    for coeff_expr, _ in _candidate_coeffs_from_symbolic_cols(self.df, p_mask, self.numeric_columns, cfg):
                        prop = _replace_coeff_and_build(rel, coeff_expr, var_expr, Hp)
                        if prop is None:
                            continue
                        if not _truthy(prop, self.df):
                            continue

                        # --- keep base hypothesis unless Hp is STRICTLY broader ---
                        if _mask_includes(base_mask, p_mask) and (p_mask.sum() > base_sup):
                            if not getattr(prop, "name", None):
                                prop.name = f"recip-generalized:{getattr(Hp,'pretty',lambda:repr(Hp))()}"
                        else:
                            prop.condition = base.condition
                            if not getattr(prop, "name", None):
                                prop.name = "recip-same"

                        sym_hits_for_Hp.append(prop)

                    if sym_hits_for_Hp:
                        proposals.extend(sym_hits_for_Hp)
                        # prefer the broadest Hp where we found valid symbolic matches
                        continue

        # de-dup final set
        return _unique_by_relation_and_condition(proposals)


    def run_reciprocal_generalizer(
        self,
        base_conjectures: Sequence[Conjecture],
        *,
        coeff_sources: Optional[str] = None,
    ):
        props = self.generalize_from_reciprocal_patterns(
            base_conjectures,
            coeff_sources=coeff_sources,
        )
        kept = self.rank_and_filter(props)
        return kept

    def _broader_hypotheses(self, base_H) -> list:
        """Return all candidate H' with mask(base_H) ⊆ mask(H'), ordered by decreasing support."""
        base_mask = _as_hyp_mask(base_H, self.df)
        cands = []
        # allow all known hypotheses; include TRUE (if not already present)
        hyps = list(self.hyps_kept)
        if TRUE not in hyps:
            hyps.append(TRUE)
        for Hp in hyps:
            pmask = _as_hyp_mask(Hp, self.df)
            if _mask_includes(base_mask, pmask) and _support(pmask) >= self.config.min_support_const:
                cands.append((Hp, _support(pmask)))
        # prefer broadest first
        return [h for h, _ in sorted(cands, key=lambda t: -t[1])]


def _pretty_safe(expr):
    s = expr.pretty()
    # If pattern like " / " followed by " · ", wrap left side in parentheses
    if " / " in s and "·" in s:
        parts = s.split("·")
        lhs = parts[0].strip()
        rhs = "·".join(parts[1:]).strip()
        return f"({lhs}) · {rhs}"
    return s

def _cond_str(H) -> str:
    # Robust name for any Predicate (Where, AndPred, TRUE, etc.)
    if hasattr(H, "pretty"):
        try:
            return H.pretty()
        except Exception:
            pass
    if hasattr(H, "name") and H.name:
        return str(H.name)
    return repr(H)

def _support_mask(H, df):
    if hasattr(H, "mask"):
        return H.mask(df).astype(bool).to_numpy()
    return np.asarray(H(df), dtype=bool)

def pretty_block_with_hyp(title: str, conjs: list, df: pd.DataFrame, max_items: int = 40):
    print(f"\n=== {title} ===")
    for i, c in enumerate(conjs[:max_items], 1):
        H = getattr(c, "condition", None)
        cond = _cond_str(H) if H is not None else "TRUE"
        mask = _support_mask(H, df) if H is not None else np.ones(len(df), dtype=bool)
        sup = int(mask.sum())
        pct = 100.0 * sup / len(df) if len(df) else 0.0

        lhs = _pretty_safe(c.relation.left)
        rhs = _pretty_safe(c.relation.right)
        rel_symbol = "≤" if c.relation.__class__.__name__ == "Le" else "≥" if c.relation.__class__.__name__ == "Ge" else c.relation.__class__.__name__

        touch = None
        try:
            # If you want, show “touches” (tight rows) on this hypothesis
            lhs_vals = c.relation.left.eval(df)[mask].to_numpy()
            rhs_vals = c.relation.right.eval(df)[mask].to_numpy()
            touch = int(np.isclose(lhs_vals, rhs_vals, rtol=1e-8, atol=1e-8).sum())
        except Exception:
            pass

        print(f"\n{i:3d}. Hypothesis [{sup}/{len(df)} = {pct:4.1f}%]: {cond}")
        if getattr(c, "name", None):
            print(f"     Name      : {c.name}")
        print(f"     Relation  : {lhs} {rel_symbol} {rhs}")
        if touch is not None:
            print(f"     Tight rows: {touch}")



from txgraffiti.example_data import graph_data as df

# (Optionally) drop columns as you were doing:
df = df.drop(columns=['cograph', 'eulerian', 'chordal', 'vertex_cover_number'])


ai = TxGraffitiMini(df)

# Phase-1 (single feature) if you want:
lower1, upper1 = ai.run_single_feature_pipeline("independence_number")

from fractions import Fraction
# Phase-2 (two-feature mixes):
# lower2, upper2 = ai.run_mixed_pipeline("independence_number", weight=Fraction(0.5))

# TxGraffitiMini.pretty_block("Mixed Lower Conjectures (x with sqrt(y), y^2)", lower2[:40])
# TxGraffitiMini.pretty_block("Mixed Upper Conjectures (x with sqrt(y), y^2)", upper2[:40])

low_prod, up_prod = ai.run_targeted_product_pipeline(
    "independence_number",
    require_pos=True,          # monotone comparisons + triviality are valid
    enable_cancellation=True,  # prefer reduced forms T ≤ y or T ≥ y when x==y (or z)
    allow_x_equal_yz=True,     # allow overlaps; triviality/cancellation will prune
)

TxGraffitiMini.pretty_block("Targeted Product LOWER (T*x ≥ y*z)", low_prod[:60])
TxGraffitiMini.pretty_block("Targeted Product UPPER (T*x ≤ y*z)", up_prod[:60])


lower_lin, upper_lin = ai.generate_single_feature_bounds("independence_number")

TxGraffitiMini.pretty_block("Lower Conjectures", lower_lin[:40])

# # b) Run reciprocal generalizer (discovery-first; no hard-coded columns)
recip_from_lowers = ai.run_reciprocal_generalizer(lower_lin[:20])   # uses cfg.coeff_sources
recip_from_uppers = ai.run_reciprocal_generalizer(upper_lin[:20])



# # c) Constant ratio equalities are available right away
const_ratio_eqs = ai.constant_ratio_equalities

# # d) Inspect
TxGraffitiMini.pretty_block("Constant Ratio Equalities", const_ratio_eqs[:40])
pretty_block_with_hyp("Reciprocal Generalizations (from uppers)", recip_from_uppers, df, max_items=60)

pretty_block_with_hyp("Reciprocal Generalizations (from lowers)", recip_from_lowers, df, max_items=60)


# ai = TxGraffitiMini(df)

# find monotone relations for a few targets vs all numeric predictors
quals_raw = ai.generate_qualitative_relations(
    y_targets=["independence_number", "domination_number"],
    method="spearman",
    min_abs_rho=0.35,
    min_n=12,
    top_k_per_hyp=10,
)

# TxGraffitiMini.pretty_qualitative_block_descriptive("Monotone tendencies", quals, max_items=30)

quals = consolidate_qualitative(quals_raw, df)
TxGraffitiMini.pretty_qualitative_block_descriptive("Monotone tendencies", quals, max_items=30)


# eqs, incs = ai.discover_class_relations(
#     predicates=None,              # defaults to self.hyps_kept
#     include_bool_columns=True,    # optional: add boolean columns as classes
#     min_support_A=5,
#     skip_trivial_equiv=True
# )

# TxGraffitiMini.pretty_class_relations("Class relations (R₁)", eqs, incs, df)

lower_lin, upper_lin = ai.generate_single_feature_bounds('independence_number')


new_preds_upper = ai.add_derived_predicates_from_top_conjectures(
    upper_lin,
    top_quantile=0.10,     # top 10% by touch
    min_support=10,
    make_eq=True,
    make_strict=True,      # makes y < Cx (for ≤) or y > Cx (for ≥)
)

new_preds_lower = ai.add_derived_predicates_from_top_conjectures(
    lower_lin,
    top_quantile=0.10,
    min_support=10,
    make_eq=True,
    make_strict=True,
)


eqs, incs = ai.discover_class_relations(
    predicates=new_preds_lower,              # defaults to self.hyps_kept
    include_bool_columns=True,    # optional: add boolean columns as classes
    min_support_A=5,
    skip_trivial_equiv=True
)

pretty_class_relations_conj("Class relations (R₁)", eqs, incs, df, ascii_ops=False, show_violations=False)

