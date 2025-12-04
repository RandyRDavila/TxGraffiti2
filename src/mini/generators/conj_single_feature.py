# from __future__ import annotations
# from typing import Iterable, Tuple, List
# import numpy as np
# import pandas as pd
# from fractions import Fraction

# from mini.core.expr import to_expr, Const, floor, ceil
# from mini.core.generic_conjecture import Conjecture, Ge, Le
# from mini.core.predicates import Predicate

# from mini.workbench.config import GenerationConfig
# from mini.workbench.caches import _EvalCache

# """
# Single-feature bound generation.

# This workbench component enumerates simple inequalities of the form

#     (H) ⇒  target ≥ c_min · x      and      (H) ⇒  target ≤ c_max · x,

# optionally adding ``ceil``/``floor`` variants when enabled in
# :class:`GenerationConfig`.
# """

# def to_frac_const(val: float, max_denom: int = 30) -> Const:
#     """
#     Convert a floating value into a small-denominator `Const`.
#     """
#     return Const(Fraction(val).limit_denominator(max_denom))

# def generate_single_feature_bounds(
#     df: pd.DataFrame,
#     target_col: str,
#     *,
#     hyps: Iterable[Predicate],
#     numeric_columns: Iterable[str],
#     config: GenerationConfig,
# ) -> Tuple[List[Conjecture], List[Conjecture]]:
#     """
#     Generate simple single-feature conjectures ``target ≥ c·x`` and
#     ``target ≤ c·x`` for each hypothesis and numeric feature.
#     """
#     target = to_expr(target_col)
#     lowers: List[Conjecture] = []
#     uppers: List[Conjecture] = []

#     for H in hyps:
#         mask = H.mask(df).reindex(df.index, fill_value=False).astype(bool, copy=False).to_numpy()
#         if not np.any(mask):
#             continue

#         dfH = df.loc[mask]
#         cache = _EvalCache(dfH)
#         t_arr = target.eval(dfH).values.astype(float, copy=False)

#         for xname in numeric_columns:
#             if xname == target_col:
#                 continue
#             x_arr = cache.col(xname)
#             if np.min(x_arr) <= 0:
#                 # skip nonpositive domains; ratio t/x ill-defined for our usage
#                 continue

#             rx = t_arr / x_arr
#             cmin_f = float(np.min(rx))
#             cmax_f = float(np.max(rx))

#             cmin = to_frac_const(cmin_f, config.max_denom)
#             cmax = to_frac_const(cmax_f, config.max_denom)
#             x_expr = to_expr(xname)

#             # Base bounds
#             lowers.append(Conjecture(Ge(target, cmin * x_expr), H))
#             uppers.append(Conjecture(Le(target, cmax * x_expr), H))

#             # Optional ceil/floor variants when the discrete statement holds for all rows
#             if not config.use_floor_ceil_if_true:
#                 continue

#             if np.all(t_arr >= np.ceil(cmin_f * x_arr)) and getattr(cmin.value, "denominator", 1) > 1:
#                 lowers.append(Conjecture(Ge(target, ceil(cmin * x_expr)), H))
#             if np.all(t_arr <= np.floor(cmax_f * x_arr)) and getattr(cmax.value, "denominator", 1) > 1:
#                 uppers.append(Conjecture(Le(target, floor(cmax * x_expr)), H))

#     return lowers, uppers
