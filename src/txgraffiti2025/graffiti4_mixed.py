# # # src/txgraffiti2025/graffiti4_mixed.py

# # from __future__ import annotations

# # from fractions import Fraction
# # from typing import Dict, Iterable, List, Sequence, TYPE_CHECKING, Callable

# # import numpy as np
# # import pandas as pd

# # from txgraffiti2025.forms.utils import (
# #     Expr,
# #     to_expr,
# #     sqrt,
# #     floor,
# #     ceil,
# #     Const,
# # )
# # from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le

# # if TYPE_CHECKING:
# #     # Only used for static typing; avoids circular import at runtime.
# #     from txgraffiti2025.graffiti4 import HypothesisInfo


# # # ───────────────────────── helpers ───────────────────────── #


# # def _to_const_fraction(x: float, max_denom: int) -> Const:
# #     """Return a Const representing a bounded-denominator rational close to x."""
# #     return Const(Fraction(x).limit_denominator(max_denom))


# # def _pick_best_ge(
# #     t_arr: np.ndarray,
# #     rhs_variants: Sequence[tuple[str, np.ndarray, Callable]],
# # ) -> Callable | None:
# #     """
# #     For lower bounds t >= rhs, pick the variant whose rhs is:
# #       - valid on all finite rows (t >= rhs), and
# #       - has the *largest mean rhs* (least conservative but still valid).
# #     Returns the Expr-constructor (callable) or None.
# #     """
# #     best = None
# #     best_score = -np.inf

# #     for _tag, rhs, make_expr in rhs_variants:
# #         ok = np.isfinite(t_arr) & np.isfinite(rhs)
# #         if not np.any(ok):
# #             continue
# #         if np.all(t_arr[ok] >= rhs[ok]):
# #             score = float(np.mean(rhs[ok]))
# #             if score > best_score:
# #                 best = make_expr
# #                 best_score = score
# #     return best


# # def _pick_best_le(
# #     t_arr: np.ndarray,
# #     rhs_variants: Sequence[tuple[str, np.ndarray, callable]],
# # ) -> callable | None:
# #     """
# #     For upper bounds t <= rhs, pick the variant whose rhs is:
# #       - valid on all finite rows (t <= rhs), and
# #       - has the *smallest mean rhs* (tightest upper bound).
# #     Returns the Expr-constructor (callable) or None.
# #     """
# #     best = None
# #     best_score = np.inf

# #     for _tag, rhs, make_expr in rhs_variants:
# #         ok = np.isfinite(t_arr) & np.isfinite(rhs)
# #         if not np.any(ok):
# #             continue
# #         if np.all(t_arr[ok] <= rhs[ok]):
# #             score = float(np.mean(rhs[ok]))
# #             if score < best_score:
# #                 best = make_expr
# #                 best_score = score
# #     return best


# # # ───────────────────────── main runner ───────────────────────── #


# # def mixed_runner(
# #     *,
# #     target_col: str,
# #     target_expr: Expr,
# #     primaries: Dict[str, Expr],
# #     secondaries: Dict[str, Expr],
# #     hypotheses: Sequence["HypothesisInfo"],
# #     df: pd.DataFrame,
# #     weight: float = 0.5,
# #     min_support: int = 8,
# #     max_denom: int = 20,
# #     exclude_nonpositive_x: bool = True,
# #     exclude_nonpositive_y: bool = True,
# #     max_coef_abs: float = 4.0,
# # ) -> List[Conjecture]:
# #     """
# #     Intricate mixed ratio-style bounds:

# #         t ≈ w * (c_x * x + c_y * sqrt(y))      and
# #         t ≈ w * (c_x * x + c_y * y^2)

# #     where the coefficients c_x, c_y come from min/max ratios over each
# #     hypothesis, and w ∈ (0,1] is a mixing weight.

# #     For each hypothesis H, primary x, secondary y, we construct candidate
# #     lower and upper bounds of the form

# #         H ⇒ t ≥ RHS(x, y)
# #         H ⇒ t ≤ RHS(x, y),

# #     where RHS uses combinations of x, sqrt(y), y^2 plus ceil/floor variants.
# #     Among these, we keep only the variants that are valid on H and “best”
# #     according to the pickers above.

# #     Parameters
# #     ----------
# #     target_col : str
# #         Column name of the dependent variable t.
# #     target_expr : Expr
# #         Expr for t (usually to_expr(target_col)).
# #     primaries : dict[str, Expr]
# #         Candidate x-invariants.
# #     secondaries : dict[str, Expr]
# #         Candidate y-invariants.
# #     hypotheses : sequence[HypothesisInfo]
# #         Hypotheses H, each with .mask (np.ndarray[bool]) and .pred (Predicate).
# #     df : DataFrame
# #         Numeric invariant table.
# #     weight : float in (0,1]
# #         Mixing weight w. Smaller w shrinks the combination.
# #     min_support : int
# #         Minimum number of valid rows under a hypothesis before considering a pair (H, x, y).
# #     max_denom : int
# #         Max denominator for rationalizing coefficients.
# #     exclude_nonpositive_x, exclude_nonpositive_y : bool
# #         If True, ignore x or y that are nonpositive on H (to keep ratios sensible).
# #     max_coef_abs : float
# #         Discard mixed bounds whose |coefficients| exceed this value (avoids insane fractions).

# #     Returns
# #     -------
# #     list[Conjecture]
# #         Combined list of lower- and upper-bound conjectures.
# #     """
# #     assert 0.0 < weight <= 1.0, "weight must be in (0,1]"

# #     conjs: List[Conjecture] = []
# #     t_all = df[target_col].to_numpy(dtype=float)
# #     w = float(weight)
# #     w_const = _to_const_fraction(weight, max_denom)

# #     for hyp in hypotheses:
# #         mask = np.asarray(hyp.mask, dtype=bool)
# #         if not mask.any():
# #             continue

# #         t_arr = t_all[mask]

# #         for xname, x_expr in primaries.items():
# #             if xname == target_col:
# #                 continue

# #             # Evaluate x
# #             try:
# #                 x_all = x_expr.eval(df).to_numpy(dtype=float)
# #             except Exception:
# #                 continue

# #             x_arr = x_all[mask]
# #             if x_arr.size == 0:
# #                 continue

# #             if exclude_nonpositive_x and (np.nanmin(x_arr) <= 0.0):
# #                 continue

# #             # Ratios r_x = t/x
# #             rx = t_arr / x_arr
# #             f_rx = np.isfinite(rx)
# #             if not np.any(f_rx):
# #                 continue

# #             cmin_f = float(np.min(rx[f_rx]))
# #             cmax_f = float(np.max(rx[f_rx]))

# #             for yname, y_expr in secondaries.items():
# #                 if yname == target_col:
# #                     continue

# #                 # Evaluate y
# #                 try:
# #                     y_all = y_expr.eval(df).to_numpy(dtype=float)
# #                 except Exception:
# #                     continue

# #                 y_arr = y_all[mask]
# #                 if y_arr.size == 0:
# #                     continue

# #                 if exclude_nonpositive_y and (np.nanmin(y_arr) <= 0.0):
# #                     continue

# #                 # ── sqrt mix: sqrt(y) component ──────────────────────
# #                 sqrt_y_arr = np.sqrt(y_arr, dtype=float)
# #                 r_sqrt = t_arr / sqrt_y_arr
# #                 f_sq = np.isfinite(r_sqrt)

# #                 lows: List[Conjecture] = []
# #                 ups: List[Conjecture] = []

# #                 if np.any(f_sq):
# #                     s_cmin_f = float(np.min(r_sqrt[f_sq]))
# #                     s_cmax_f = float(np.max(r_sqrt[f_sq]))

# #                     # Drop crazy coefficients early
# #                     if (
# #                         abs(cmin_f) > max_coef_abs
# #                         or abs(cmax_f) > max_coef_abs
# #                         or abs(s_cmin_f) > max_coef_abs
# #                         or abs(s_cmax_f) > max_coef_abs
# #                     ):
# #                         pass  # skip this combination
# #                     else:
# #                         base_lo = w * (cmin_f * x_arr + s_cmin_f * sqrt_y_arr)
# #                         base_up = w * (cmax_f * x_arr + s_cmax_f * sqrt_y_arr)

# #                         ceil_whole = np.ceil(base_lo)
# #                         floor_whole = np.floor(base_up)

# #                         ceil_split = (
# #                             np.ceil(w * cmin_f * x_arr)
# #                             + np.ceil(w * s_cmin_f * sqrt_y_arr)
# #                             - 1.0
# #                         )
# #                         floor_split = (
# #                             np.floor(w * cmax_f * x_arr)
# #                             + np.floor(w * s_cmax_f * sqrt_y_arr)
# #                         )

# #                         # Symbolic mirrors
# #                         def _lo_base():
# #                             kx = w_const * _to_const_fraction(cmin_f, max_denom)
# #                             ky = w_const * _to_const_fraction(s_cmin_f, max_denom)
# #                             return kx * x_expr + ky * sqrt(y_expr)

# #                         def _lo_ceil_whole():
# #                             return ceil(_lo_base())

# #                         def _lo_ceil_split():
# #                             kx = w_const * _to_const_fraction(cmin_f, max_denom)
# #                             ky = w_const * _to_const_fraction(s_cmin_f, max_denom)
# #                             return ceil(kx * x_expr) + ceil(ky * sqrt(y_expr)) - Const(1)

# #                         def _up_base():
# #                             kx = w_const * _to_const_fraction(cmax_f, max_denom)
# #                             ky = w_const * _to_const_fraction(s_cmax_f, max_denom)
# #                             return kx * x_expr + ky * sqrt(y_expr)

# #                         def _up_floor_whole():
# #                             return floor(_up_base())

# #                         def _up_floor_split():
# #                             kx = w_const * _to_const_fraction(cmax_f, max_denom)
# #                             ky = w_const * _to_const_fraction(s_cmax_f, max_denom)
# #                             return floor(kx * x_expr) + floor(ky * sqrt(y_expr))

# #                         # Lower bound choice (t >= RHS)
# #                         lo_choice = _pick_best_ge(
# #                             t_arr,
# #                             [
# #                                 ("base", base_lo, _lo_base),
# #                                 ("ceil_whole", ceil_whole, _lo_ceil_whole),
# #                                 ("ceil_split", ceil_split, _lo_ceil_split),
# #                             ],
# #                         )
# #                         if lo_choice is not None:
# #                             lows.append(
# #                                 Conjecture(
# #                                     relation=Ge(target_expr, lo_choice()),
# #                                     condition=hyp.pred,
# #                                     name=(
# #                                         f"[mixed-sqrt-lower] {target_col} "
# #                                         f"vs {xname}, sqrt({yname}) under {hyp.name}"
# #                                     ),
# #                                 )
# #                             )

# #                         # Upper bound choice (t <= RHS)
# #                         up_choice = _pick_best_le(
# #                             t_arr,
# #                             [
# #                                 ("base", base_up, _up_base),
# #                                 ("floor_whole", floor_whole, _up_floor_whole),
# #                                 ("floor_split", floor_split, _up_floor_split),
# #                             ],
# #                         )
# #                         if up_choice is not None:
# #                             ups.append(
# #                                 Conjecture(
# #                                     relation=Le(target_expr, up_choice()),
# #                                     condition=hyp.pred,
# #                                     name=(
# #                                         f"[mixed-sqrt-upper] {target_col} "
# #                                         f"vs {xname}, sqrt({yname}) under {hyp.name}"
# #                                     ),
# #                                 )
# #                             )

# #                 # ── square mix: y^2 component ────────────────────────
# #                 y_sq_arr = np.square(y_arr, dtype=float)
# #                 r_sq = t_arr / y_sq_arr
# #                 f_rsq = np.isfinite(r_sq)

# #                 if np.any(f_rsq):
# #                     q_cmin_f = float(np.min(r_sq[f_rsq]))
# #                     q_cmax_f = float(np.max(r_sq[f_rsq]))

# #                     if (
# #                         abs(cmin_f) > max_coef_abs
# #                         or abs(cmax_f) > max_coef_abs
# #                         or abs(q_cmin_f) > max_coef_abs
# #                         or abs(q_cmax_f) > max_coef_abs
# #                     ):
# #                         pass
# #                     else:
# #                         base_lo_sq = w * (cmin_f * x_arr + q_cmin_f * y_sq_arr)
# #                         base_up_sq = w * (cmax_f * x_arr + q_cmax_f * y_sq_arr)

# #                         ceil_whole_sq = np.ceil(base_lo_sq)
# #                         floor_whole_sq = np.floor(base_up_sq)

# #                         def _lo_sq_base():
# #                             kx = w_const * _to_const_fraction(cmin_f, max_denom)
# #                             ky = w_const * _to_const_fraction(q_cmin_f, max_denom)
# #                             return kx * x_expr + ky * (y_expr ** Const(Fraction(2, 1)))

# #                         def _lo_sq_ceil_whole():
# #                             return ceil(_lo_sq_base())

# #                         def _up_sq_base():
# #                             kx = w_const * _to_const_fraction(cmax_f, max_denom)
# #                             ky = w_const * _to_const_fraction(q_cmax_f, max_denom)
# #                             return kx * x_expr + ky * (y_expr ** Const(Fraction(2, 1)))

# #                         def _up_sq_floor_whole():
# #                             return floor(_up_sq_base())

# #                         lo_sq_choice = _pick_best_ge(
# #                             t_arr,
# #                             [
# #                                 ("base", base_lo_sq, _lo_sq_base),
# #                                 ("ceil_whole", ceil_whole_sq, _lo_sq_ceil_whole),
# #                             ],
# #                         )
# #                         if lo_sq_choice is not None:
# #                             lows.append(
# #                                 Conjecture(
# #                                     relation=Ge(target_expr, lo_sq_choice()),
# #                                     condition=hyp.pred,
# #                                     name=(
# #                                         f"[mixed-square-lower] {target_col} "
# #                                         f"vs {xname}, {yname}^2 under {hyp.name}"
# #                                     ),
# #                                 )
# #                             )

# #                         up_sq_choice = _pick_best_le(
# #                             t_arr,
# #                             [
# #                                 ("base", base_up_sq, _up_sq_base),
# #                                 ("floor_whole", floor_whole_sq, _up_sq_floor_whole),
# #                             ],
# #                         )
# #                         if up_sq_choice is not None:
# #                             ups.append(
# #                                 Conjecture(
# #                                     relation=Le(target_expr, up_sq_choice()),
# #                                     condition=hyp.pred,
# #                                     name=(
# #                                         f"[mixed-square-upper] {target_col} "
# #                                         f"vs {xname}, {yname}^2 under {hyp.name}"
# #                                     ),
# #                                 )
# #                             )

# #                 # add for this (H, x, y)
# #                 conjs.extend(lows)
# #                 conjs.extend(ups)

# #     return conjs


# # src/txgraffiti2025/graffiti4_mixed.py

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import (
    Expr,
    to_expr,
    sqrt,
    floor,
    ceil,
    Const,
)
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le

if TYPE_CHECKING:
    # Only for static typing; avoids circular import at runtime.
    from txgraffiti2025.graffiti4 import HypothesisInfo


# ───────────────────────── helpers ───────────────────────── #


def _to_const_fraction(x: float, max_denom: int) -> Const:
    """Return a Const representing a bounded-denominator rational close to x."""
    return Const(Fraction(x).limit_denominator(max_denom))


def _pick_best_ge(
    t_arr: np.ndarray,
    rhs_variants: Sequence[tuple[str, np.ndarray, Callable]],
) -> Callable | None:
    """
    For lower bounds t >= rhs, pick the variant whose rhs is:
      - valid on all finite rows (t >= rhs), and
      - has the *largest mean rhs* (least conservative but still valid).
    Returns the Expr-constructor (callable) or None.
    """
    best = None
    best_score = -np.inf

    for _tag, rhs, make_expr in rhs_variants:
        ok = np.isfinite(t_arr) & np.isfinite(rhs)
        if not np.any(ok):
            continue
        if np.all(t_arr[ok] >= rhs[ok]):
            score = float(np.mean(rhs[ok]))
            if score > best_score:
                best = make_expr
                best_score = score
    return best


def _pick_best_le(
    t_arr: np.ndarray,
    rhs_variants: Sequence[tuple[str, np.ndarray, Callable]],
) -> Callable | None:
    """
    For upper bounds t <= rhs, pick the variant whose rhs is:
      - valid on all finite rows (t <= rhs), and
      - has the *smallest mean rhs* (tightest upper bound).
    Returns the Expr-constructor (callable) or None.
    """
    best = None
    best_score = np.inf

    for _tag, rhs, make_expr in rhs_variants:
        ok = np.isfinite(t_arr) & np.isfinite(rhs)
        if not np.any(ok):
            continue
        if np.all(t_arr[ok] <= rhs[ok]):
            score = float(np.mean(rhs[ok]))
            if score < best_score:
                best = make_expr
                best_score = score
    return best


def _safe_sqrt_array(y_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 'safe' sqrt array and its validity mask.

    Parameters
    ----------
    y_arr : np.ndarray
        Raw y values on a hypothesis mask.

    Returns
    -------
    sqrt_y : np.ndarray
        Same shape as y_arr; sqrt applied only where y is finite and >= 0,
        elsewhere 0.
    valid : np.ndarray[bool]
        Mask where sqrt_y is genuinely valid (finite and > 0), suitable
        for forming ratios.
    """
    sqrt_y = np.zeros_like(y_arr, dtype=float)
    domain = np.isfinite(y_arr) & (y_arr >= 0.0)
    sqrt_y[domain] = np.sqrt(y_arr[domain], dtype=float)

    valid = domain & np.isfinite(sqrt_y) & (sqrt_y != 0.0)
    return sqrt_y, valid


# ───────────────────────── main runner ───────────────────────── #


def mixed_runner(
    *,
    target_col: str,
    target_expr: Expr,
    primaries: Dict[str, Expr],
    secondaries: Dict[str, Expr],
    hypotheses: Sequence["HypothesisInfo"],
    df: pd.DataFrame,
    weight: float = 0.5,
    min_support: int = 8,
    max_denom: int = 20,
    exclude_nonpositive_x: bool = True,
    exclude_nonpositive_y: bool = True,
    max_coef_abs: float = 4.0,
) -> List[Conjecture]:
    """
    Intricate mixed ratio-style bounds:

        t ≈ w * (c_x * x + c_y * sqrt(y))      and
        t ≈ w * (c_x * x + c_y * y^2)

    where the coefficients c_x, c_y come from min/max ratios over each
    hypothesis, and w ∈ (0,1] is a mixing weight.

    For each hypothesis H, primary x, secondary y, we construct candidate
    lower and upper bounds of the form

        H ⇒ t ≥ RHS(x, y)
        H ⇒ t ≤ RHS(x, y),

    where RHS uses combinations of x, sqrt(y), y^2 plus ceil/floor variants.
    Among these, we keep only the variants that are valid on H and “best”
    according to the pickers above.

    This version:
      - uses a safe sqrt helper to avoid warnings when y ≤ 0 or NaN,
      - enforces `min_support` for all ratio estimates,
      - clips large coefficients via `max_coef_abs`.

    Parameters
    ----------
    target_col : str
        Column name of the dependent variable t.
    target_expr : Expr
        Expr for t (usually to_expr(target_col)).
    primaries : dict[str, Expr]
        Candidate x-invariants.
    secondaries : dict[str, Expr]
        Candidate y-invariants.
    hypotheses : sequence[HypothesisInfo]
        Hypotheses H, each with .mask (np.ndarray[bool]) and .pred (Predicate).
    df : DataFrame
        Numeric invariant table.
    weight : float in (0,1]
        Mixing weight w. Smaller w shrinks the combination.
    min_support : int
        Minimum number of valid rows under a hypothesis before considering a
        ratio-based coefficient.
    max_denom : int
        Max denominator for rationalizing coefficients.
    exclude_nonpositive_x, exclude_nonpositive_y : bool
        If True, ignore x or y that are nonpositive on H (to keep ratios sensible).
    max_coef_abs : float
        Discard mixed bounds whose |coefficients| exceed this value (avoid insane fractions).

    Returns
    -------
    list[Conjecture]
        Combined list of lower- and upper-bound conjectures.
    """
    assert 0.0 < weight <= 1.0, "weight must be in (0,1]"

    conjs: List[Conjecture] = []
    t_all = df[target_col].to_numpy(dtype=float)
    w = float(weight)
    w_const = _to_const_fraction(weight, max_denom)

    for hyp in hypotheses:
        mask = np.asarray(hyp.mask, dtype=bool)
        if not mask.any():
            continue

        t_arr = t_all[mask]

        for xname, x_expr in primaries.items():
            if xname == target_col:
                continue

            # Evaluate x
            try:
                x_all = x_expr.eval(df).to_numpy(dtype=float)
            except Exception:
                continue

            x_arr = x_all[mask]
            if x_arr.size == 0:
                continue

            if exclude_nonpositive_x and (np.nanmin(x_arr) <= 0.0):
                continue

            # Ratios r_x = t/x
            rx = t_arr / x_arr
            f_rx = np.isfinite(rx)
            if f_rx.sum() < min_support:
                continue

            cmin_f = float(np.min(rx[f_rx]))
            cmax_f = float(np.max(rx[f_rx]))

            for yname, y_expr in secondaries.items():
                if yname == target_col:
                    continue

                # Evaluate y
                try:
                    y_all = y_expr.eval(df).to_numpy(dtype=float)
                except Exception:
                    continue

                y_arr = y_all[mask]
                if y_arr.size == 0:
                    continue

                if exclude_nonpositive_y and (np.nanmin(y_arr) <= 0.0):
                    continue

                lows: List[Conjecture] = []
                ups: List[Conjecture] = []

                # ── sqrt mix: sqrt(y) component (safe) ──────────────────────
                sqrt_y_arr, sqrt_valid = _safe_sqrt_array(y_arr)
                if sqrt_valid.any():
                    r_sqrt = np.full_like(t_arr, np.nan, dtype=float)
                    r_sqrt[sqrt_valid] = t_arr[sqrt_valid] / sqrt_y_arr[sqrt_valid]
                    f_sq = np.isfinite(r_sqrt)

                    if f_sq.sum() >= min_support:
                        s_cmin_f = float(np.min(r_sqrt[f_sq]))
                        s_cmax_f = float(np.max(r_sqrt[f_sq]))

                        # Drop crazy coefficients early
                        if (
                            abs(cmin_f) <= max_coef_abs
                            and abs(cmax_f) <= max_coef_abs
                            and abs(s_cmin_f) <= max_coef_abs
                            and abs(s_cmax_f) <= max_coef_abs
                        ):
                            base_lo = w * (cmin_f * x_arr + s_cmin_f * sqrt_y_arr)
                            base_up = w * (cmax_f * x_arr + s_cmax_f * sqrt_y_arr)

                            ceil_whole = np.ceil(base_lo)
                            floor_whole = np.floor(base_up)

                            ceil_split = (
                                np.ceil(w * cmin_f * x_arr)
                                + np.ceil(w * s_cmin_f * sqrt_y_arr)
                                - 1.0
                            )
                            floor_split = (
                                np.floor(w * cmax_f * x_arr)
                                + np.floor(w * s_cmax_f * sqrt_y_arr)
                            )

                            # Symbolic mirrors
                            def _lo_base():
                                kx = w_const * _to_const_fraction(cmin_f, max_denom)
                                ky = w_const * _to_const_fraction(
                                    s_cmin_f, max_denom
                                )
                                return kx * x_expr + ky * sqrt(y_expr)

                            def _lo_ceil_whole():
                                return ceil(_lo_base())

                            def _lo_ceil_split():
                                kx = w_const * _to_const_fraction(cmin_f, max_denom)
                                ky = w_const * _to_const_fraction(
                                    s_cmin_f, max_denom
                                )
                                return (
                                    ceil(kx * x_expr)
                                    + ceil(ky * sqrt(y_expr))
                                    - Const(1)
                                )

                            def _up_base():
                                kx = w_const * _to_const_fraction(cmax_f, max_denom)
                                ky = w_const * _to_const_fraction(
                                    s_cmax_f, max_denom
                                )
                                return kx * x_expr + ky * sqrt(y_expr)

                            def _up_floor_whole():
                                return floor(_up_base())

                            def _up_floor_split():
                                kx = w_const * _to_const_fraction(cmax_f, max_denom)
                                ky = w_const * _to_const_fraction(
                                    s_cmax_f, max_denom
                                )
                                return (
                                    floor(kx * x_expr)
                                    + floor(ky * sqrt(y_expr))
                                )

                            # Lower bound choice (t >= RHS)
                            lo_choice = _pick_best_ge(
                                t_arr,
                                [
                                    ("base", base_lo, _lo_base),
                                    ("ceil_whole", ceil_whole, _lo_ceil_whole),
                                    ("ceil_split", ceil_split, _lo_ceil_split),
                                ],
                            )
                            if lo_choice is not None:
                                lows.append(
                                    Conjecture(
                                        relation=Ge(target_expr, lo_choice()),
                                        condition=hyp.pred,
                                        name=(
                                            f"[mixed-sqrt-lower] {target_col} "
                                            f"vs {xname}, sqrt({yname}) under {hyp.name}"
                                        ),
                                    )
                                )

                            # Upper bound choice (t <= RHS)
                            up_choice = _pick_best_le(
                                t_arr,
                                [
                                    ("base", base_up, _up_base),
                                    ("floor_whole", floor_whole, _up_floor_whole),
                                    ("floor_split", floor_split, _up_floor_split),
                                ],
                            )
                            if up_choice is not None:
                                ups.append(
                                    Conjecture(
                                        relation=Le(target_expr, up_choice()),
                                        condition=hyp.pred,
                                        name=(
                                            f"[mixed-sqrt-upper] {target_col} "
                                            f"vs {xname}, sqrt({yname}) under {hyp.name}"
                                        ),
                                    )
                                )

                # ── square mix: y^2 component ────────────────────────
                y_sq_arr = np.square(y_arr, dtype=float)
                r_sq = t_arr / y_sq_arr
                f_rsq = np.isfinite(r_sq)

                if f_rsq.sum() >= min_support:
                    q_cmin_f = float(np.min(r_sq[f_rsq]))
                    q_cmax_f = float(np.max(r_sq[f_rsq]))

                    if (
                        abs(cmin_f) <= max_coef_abs
                        and abs(cmax_f) <= max_coef_abs
                        and abs(q_cmin_f) <= max_coef_abs
                        and abs(q_cmax_f) <= max_coef_abs
                    ):
                        base_lo_sq = w * (cmin_f * x_arr + q_cmin_f * y_sq_arr)
                        base_up_sq = w * (cmax_f * x_arr + q_cmax_f * y_sq_arr)

                        ceil_whole_sq = np.ceil(base_lo_sq)
                        floor_whole_sq = np.floor(base_up_sq)

                        def _lo_sq_base():
                            kx = w_const * _to_const_fraction(cmin_f, max_denom)
                            ky = w_const * _to_const_fraction(q_cmin_f, max_denom)
                            return kx * x_expr + ky * (y_expr ** Const(Fraction(2, 1)))

                        def _lo_sq_ceil_whole():
                            return ceil(_lo_sq_base())

                        def _up_sq_base():
                            kx = w_const * _to_const_fraction(cmax_f, max_denom)
                            ky = w_const * _to_const_fraction(q_cmax_f, max_denom)
                            return kx * x_expr + ky * (y_expr ** Const(Fraction(2, 1)))

                        def _up_sq_floor_whole():
                            return floor(_up_sq_base())

                        lo_sq_choice = _pick_best_ge(
                            t_arr,
                            [
                                ("base", base_lo_sq, _lo_sq_base),
                                ("ceil_whole", ceil_whole_sq, _lo_sq_ceil_whole),
                            ],
                        )
                        if lo_sq_choice is not None:
                            lows.append(
                                Conjecture(
                                    relation=Ge(target_expr, lo_sq_choice()),
                                    condition=hyp.pred,
                                    name=(
                                        f"[mixed-square-lower] {target_col} "
                                        f"vs {xname}, {yname}^2 under {hyp.name}"
                                    ),
                                )
                            )

                        up_sq_choice = _pick_best_le(
                            t_arr,
                            [
                                ("base", base_up_sq, _up_sq_base),
                                ("floor_whole", floor_whole_sq, _up_sq_floor_whole),
                            ],
                        )
                        if up_sq_choice is not None:
                            ups.append(
                                Conjecture(
                                    relation=Le(target_expr, up_sq_choice()),
                                    condition=hyp.pred,
                                    name=(
                                        f"[mixed-square-upper] {target_col} "
                                        f"vs {xname}, {yname}^2 under {hyp.name}"
                                    ),
                                )
                            )

                # add for this (H, x, y)
                conjs.extend(lows)
                conjs.extend(ups)

    return conjs


# src/txgraffiti2025/graffiti4_mixed.py

# from __future__ import annotations

# from fractions import Fraction
# from typing import Dict, List, Sequence, TYPE_CHECKING, Callable, Optional, Tuple

# import numpy as np
# import pandas as pd

# from txgraffiti2025.forms.utils import Expr, sqrt, Const
# from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le

# if TYPE_CHECKING:
#     # Only used for typing; avoids circular import at runtime.
#     from txgraffiti2025.graffiti4_types import HypothesisInfo

# # Try to get an LP solver
# try:
#     from scipy.optimize import linprog
# except Exception:  # pragma: no cover
#     linprog = None


# # ───────────────────────── small helpers ───────────────────────── #


# def _to_const_fraction(x: float, max_denom: int) -> Const:
#     """Return a Const representing a bounded-denominator rational close to x."""
#     return Const(Fraction(x).limit_denominator(max_denom))


# def _build_expr_from_coeffs(
#     beta: np.ndarray,
#     c0: float,
#     feature_exprs: Sequence[Expr],
#     *,
#     max_denom: int,
#     coef_tol: float = 1e-8,
# ) -> Expr:
#     """
#     Turn numeric coefficients + intercept into an Expr, dropping tiny coefficients
#     and rationalizing the rest.
#     """
#     assert len(beta) == len(feature_exprs)

#     terms: List[Expr] = []

#     # linear terms a_i * feature_i
#     for coeff, e in zip(beta, feature_exprs):
#         if abs(coeff) < coef_tol:
#             continue
#         c_const = _to_const_fraction(float(coeff), max_denom)
#         terms.append(c_const * e)

#     # intercept term
#     if abs(c0) >= coef_tol:
#         terms.append(_to_const_fraction(float(c0), max_denom))

#     if not terms:
#         # fall back to a zero constant; caller can decide to ignore
#         return Const(0)

#     expr = terms[0]
#     for t in terms[1:]:
#         expr = expr + t
#     return expr


# def _prepare_valid_rows(
#     t_arr: np.ndarray,
#     x_arr: np.ndarray,
#     y_arr: np.ndarray,
#     *,
#     require_nonneg_y: bool,
#     min_support: int,
# ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
#     """
#     Restrict to rows with finite t, x, y (and y >= 0 if require_nonneg_y)
#     and enforce a minimum support.
#     """
#     mask = np.isfinite(t_arr) & np.isfinite(x_arr) & np.isfinite(y_arr)
#     if require_nonneg_y:
#         mask &= (y_arr >= 0.0)

#     if mask.sum() < min_support:
#         return None

#     return t_arr[mask], x_arr[mask], y_arr[mask]


# # ───────────────────────── LP solvers ───────────────────────── #


# def _solve_lp_upper(
#     t_arr: np.ndarray,
#     F: np.ndarray,
#     *,
#     coef_bound: float,
# ) -> Optional[Tuple[np.ndarray, float]]:
#     """
#     Solve for an upper bound t <= F beta + c0 via LP:

#         minimize   sum_i (F_i · beta + c0)
#         subject to F_i · beta + c0 >= t_i   for all i
#                   |beta_j| <= coef_bound
#                   |c0|     <= coef_bound
#     """
#     if linprog is None:
#         return None

#     n, k = F.shape
#     if n == 0:
#         return None

#     # Variables: v = [beta_1, ..., beta_k, c0]
#     # Objective: minimize sum_i (F_i · beta + c0)
#     #           = sum_j beta_j * sum_i F_ij  +  c0 * n
#     c_vec = np.zeros(k + 1, dtype=float)
#     c_vec[:k] = F.sum(axis=0)
#     c_vec[-1] = float(n)

#     # Constraints: F_i · beta + c0 >= t_i  =>  -F_i · beta - c0 <= -t_i
#     A_ub = np.empty((n, k + 1), dtype=float)
#     A_ub[:, :k] = -F
#     A_ub[:, -1] = -1.0
#     b_ub = -t_arr.astype(float)

#     bounds = [(-coef_bound, coef_bound)] * (k + 1)

#     res = linprog(
#         c=c_vec,
#         A_ub=A_ub,
#         b_ub=b_ub,
#         bounds=bounds,
#         method="highs",
#     )

#     if not res.success:
#         return None

#     v = res.x
#     beta = v[:k]
#     c0 = v[-1]

#     # Numerical safety check
#     rhs = F @ beta + c0
#     if not np.all(t_arr <= rhs + 1e-7):
#         return None

#     return beta, c0


# def _solve_lp_lower(
#     t_arr: np.ndarray,
#     F: np.ndarray,
#     *,
#     coef_bound: float,
# ) -> Optional[Tuple[np.ndarray, float]]:
#     """
#     Solve for a lower bound t >= F beta + c0 via LP:

#         maximize   sum_i (F_i · beta + c0)
#         subject to F_i · beta + c0 <= t_i   for all i
#                   |beta_j| <= coef_bound
#                   |c0|     <= coef_bound

#     Implemented as:

#         minimize  -sum_i (F_i · beta + c0)
#         subject to F_i · beta + c0 <= t_i.
#     """
#     if linprog is None:
#         return None

#     n, k = F.shape
#     if n == 0:
#         return None

#     # Variables: v = [beta_1, ..., beta_k, c0]
#     # Objective: minimize -sum_i (F_i · beta + c0)
#     #           = sum_j beta_j * (-sum_i F_ij)  +  c0 * (-n)
#     c_vec = np.zeros(k + 1, dtype=float)
#     c_vec[:k] = -F.sum(axis=0)
#     c_vec[-1] = -float(n)

#     # Constraints: F_i · beta + c0 <= t_i
#     A_ub = np.empty((n, k + 1), dtype=float)
#     A_ub[:, :k] = F
#     A_ub[:, -1] = 1.0
#     b_ub = t_arr.astype(float)

#     bounds = [(-coef_bound, coef_bound)] * (k + 1)

#     res = linprog(
#         c=c_vec,
#         A_ub=A_ub,
#         b_ub=b_ub,
#         bounds=bounds,
#         method="highs",
#     )

#     if not res.success:
#         return None

#     v = res.x
#     beta = v[:k]
#     c0 = v[-1]

#     # Numerical safety check
#     rhs = F @ beta + c0
#     if not np.all(t_arr >= rhs - 1e-7):
#         return None

#     return beta, c0


# # ───────────────────────── main runner ───────────────────────── #


# def mixed_runner(
#     *,
#     target_col: str,
#     target_expr: Expr,
#     primaries: Dict[str, Expr],
#     secondaries: Dict[str, Expr],
#     hypotheses: Sequence["HypothesisInfo"],
#     df: pd.DataFrame,
#     weight: float = 0.5,          # currently unused; kept for API compatibility
#     min_support: int = 8,
#     max_denom: int = 20,
#     exclude_nonpositive_x: bool = True,
#     exclude_nonpositive_y: bool = True,
#     max_coef_abs: float = 4.0,
# ) -> List[Conjecture]:
#     """
#     LP-based mixed bounds of the form

#         H ⇒ t ≥ a x + b sqrt(y) + c
#         H ⇒ t ≤ a x + b sqrt(y) + c

#         H ⇒ t ≥ a x + b y^2 + c
#         H ⇒ t ≤ a x + b y^2 + c,

#     where (a, b, c) are found via small linear programs under each hypothesis H.

#     For each hypothesis H and each pair (x, y) from primaries × secondaries:

#       * Collect rows in H where all quantities are finite (and y >= 0 for sqrt).
#       * Solve an LP to find the tightest valid upper bound (minimizing the RHS).
#       * Solve an LP to find the tightest valid lower bound (maximizing the RHS).
#       * Coefficients are capped in magnitude by `max_coef_abs` and rationalized
#         with denominator at most `max_denom`.

#     Parameters
#     ----------
#     target_col : str
#         Column name of the dependent variable t.
#     target_expr : Expr
#         Expression corresponding to t (usually to_expr(target_col)).
#     primaries : dict[str, Expr]
#         Candidate x-invariants.
#     secondaries : dict[str, Expr]
#         Candidate y-invariants.
#     hypotheses : sequence[HypothesisInfo]
#         Hypotheses H, each with .mask (np.ndarray[bool]), .pred (Predicate), .name.
#     df : DataFrame
#         Numeric invariant table.
#     weight : float
#         Currently unused in LP version; kept for backwards compatibility.
#     min_support : int
#         Minimum number of valid rows under a hypothesis.
#     max_denom : int
#         Maximum denominator for rational approximations of coefficients.
#     exclude_nonpositive_x, exclude_nonpositive_y : bool
#         If True, skip variables that are nonpositive on H (for semantics / sqrt).
#     max_coef_abs : float
#         Coefficient bound |a|, |b|, |c| <= max_coef_abs.

#     Returns
#     -------
#     list[Conjecture]
#         List of mixed lower and upper conjectures.
#     """
#     conjs: List[Conjecture] = []

#     t_all = df[target_col].to_numpy(dtype=float)

#     if linprog is None:
#         # Graceful fallback: no LP solver available.
#         # In that case, return an empty list; caller still runs.
#         return conjs

#     for hyp in hypotheses:
#         mask = np.asarray(hyp.mask, dtype=bool)
#         if not mask.any():
#             continue

#         t_arr_full = t_all[mask]

#         for xname, x_expr in primaries.items():
#             if xname == target_col:
#                 continue

#             # Evaluate x on entire df, then restrict to H.
#             try:
#                 x_all = x_expr.eval(df).to_numpy(dtype=float)
#             except Exception:
#                 continue
#             x_arr_full = x_all[mask]

#             if x_arr_full.size == 0:
#                 continue

#             if exclude_nonpositive_x and (np.nanmin(x_arr_full) <= 0.0):
#                 continue

#             for yname, y_expr in secondaries.items():
#                 if yname == target_col:
#                     continue

#                 # Evaluate y on entire df, then restrict to H.
#                 try:
#                     y_all = y_expr.eval(df).to_numpy(dtype=float)
#                 except Exception:
#                     continue
#                 y_arr_full = y_all[mask]

#                 if y_arr_full.size == 0:
#                     continue

#                 if exclude_nonpositive_y and (np.nanmin(y_arr_full) <= 0.0):
#                     continue

#                 # ── sqrt mix: features [x, sqrt(y)] ──────────────────
#                 prepared = _prepare_valid_rows(
#                     t_arr_full,
#                     x_arr_full,
#                     y_arr_full,
#                     require_nonneg_y=True,
#                     min_support=min_support,
#                 )
#                 if prepared is not None:
#                     t_arr, x_arr, y_arr = prepared
#                     sqrt_y = np.sqrt(y_arr, dtype=float)

#                     F_sqrt = np.column_stack([x_arr, sqrt_y])

#                     # lower bound: t >= a x + b sqrt(y) + c
#                     res_lo = _solve_lp_lower(
#                         t_arr,
#                         F_sqrt,
#                         coef_bound=max_coef_abs,
#                     )
#                     if res_lo is not None:
#                         beta_lo, c0_lo = res_lo
#                         rhs_expr_lo = _build_expr_from_coeffs(
#                             beta_lo,
#                             c0_lo,
#                             feature_exprs=[
#                                 x_expr,
#                                 sqrt(y_expr),
#                             ],
#                             max_denom=max_denom,
#                         )
#                         conjs.append(
#                             Conjecture(
#                                 relation=Ge(target_expr, rhs_expr_lo),
#                                 condition=hyp.pred,
#                                 name=(
#                                     f"[mixed-sqrt-lower] {target_col} "
#                                     f"vs {xname}, sqrt({yname}) under {hyp.name}"
#                                 ),
#                             )
#                         )

#                     # upper bound: t <= a x + b sqrt(y) + c
#                     res_up = _solve_lp_upper(
#                         t_arr,
#                         F_sqrt,
#                         coef_bound=max_coef_abs,
#                     )
#                     if res_up is not None:
#                         beta_up, c0_up = res_up
#                         rhs_expr_up = _build_expr_from_coeffs(
#                             beta_up,
#                             c0_up,
#                             feature_exprs=[
#                                 x_expr,
#                                 sqrt(y_expr),
#                             ],
#                             max_denom=max_denom,
#                         )
#                         conjs.append(
#                             Conjecture(
#                                 relation=Le(target_expr, rhs_expr_up),
#                                 condition=hyp.pred,
#                                 name=(
#                                     f"[mixed-sqrt-upper] {target_col} "
#                                     f"vs {xname}, sqrt({yname}) under {hyp.name}"
#                                 ),
#                             )
#                         )

#                 # ── square mix: features [x, y^2] ────────────────────
#                 prepared_sq = _prepare_valid_rows(
#                     t_arr_full,
#                     x_arr_full,
#                     y_arr_full,
#                     require_nonneg_y=False,
#                     min_support=min_support,
#                 )
#                 if prepared_sq is not None:
#                     t_arr, x_arr, y_arr = prepared_sq
#                     y_sq = np.square(y_arr, dtype=float)

#                     F_sq = np.column_stack([x_arr, y_sq])

#                     # lower bound: t >= a x + b y^2 + c
#                     res_lo_sq = _solve_lp_lower(
#                         t_arr,
#                         F_sq,
#                         coef_bound=max_coef_abs,
#                     )
#                     if res_lo_sq is not None:
#                         beta_lo_sq, c0_lo_sq = res_lo_sq
#                         rhs_expr_lo_sq = _build_expr_from_coeffs(
#                             beta_lo_sq,
#                             c0_lo_sq,
#                             feature_exprs=[
#                                 x_expr,
#                                 (y_expr ** Const(Fraction(2, 1))),
#                             ],
#                             max_denom=max_denom,
#                         )
#                         conjs.append(
#                             Conjecture(
#                                 relation=Ge(target_expr, rhs_expr_lo_sq),
#                                 condition=hyp.pred,
#                                 name=(
#                                     f"[mixed-square-lower] {target_col} "
#                                     f"vs {xname}, {yname}^2 under {hyp.name}"
#                                 ),
#                             )
#                         )

#                     # upper bound: t <= a x + b y^2 + c
#                     res_up_sq = _solve_lp_upper(
#                         t_arr,
#                         F_sq,
#                         coef_bound=max_coef_abs,
#                     )
#                     if res_up_sq is not None:
#                         beta_up_sq, c0_up_sq = res_up_sq
#                         rhs_expr_up_sq = _build_expr_from_coeffs(
#                             beta_up_sq,
#                             c0_up_sq,
#                             feature_exprs=[
#                                 x_expr,
#                                 (y_expr ** Const(Fraction(2, 1))),
#                             ],
#                             max_denom=max_denom,
#                         )
#                         conjs.append(
#                             Conjecture(
#                                 relation=Le(target_expr, rhs_expr_up_sq),
#                                 condition=hyp.pred,
#                                 name=(
#                                     f"[mixed-square-upper] {target_col} "
#                                     f"vs {xname}, {yname}^2 under {hyp.name}"
#                                 ),
#                             )
#                         )

#     return conjs
