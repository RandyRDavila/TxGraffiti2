# src/txgraffiti/graffiti3/runners/mixed.py

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from txgraffiti.graffiti3.exprs import (
    Expr,
    sqrt,
    floor,
    ceil,
    Const,
)
from txgraffiti.graffiti3.relations import Conjecture, Ge, Le
from txgraffiti.graffiti3.runners.utils import (
    _to_const_fraction,
    _pick_best_ge,
    _pick_best_le,
    _safe_sqrt_array,
)

if TYPE_CHECKING:
    from txgraffiti.graffiti3.types import HypothesisInfo


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
    Mixed ratio-style bounds:

        t ≈ w * (c_x * x + c_y * sqrt(y))
        t ≈ w * (c_x * x + c_y * y^2)

    where the coefficients c_x, c_y come from min/max ratios over each
    hypothesis, and w ∈ (0,1] is a mixing weight.

    IMPORTANT: symbolic coefficients are now constructed as single Consts like
        (1/2)·x, (1/4)·√y
    instead of nested products ((1/2)·1)·x, ((1/2)·(1/2))·√y.
    """
    assert 0.0 < weight <= 1.0, "weight must be in (0,1]"

    def _scaled_const(w: float, c: float) -> Const:
        """Return Const for w*c with bounded denominator."""
        return _to_const_fraction(w * c, max_denom)

    # NEW: helper to build linear combos while dropping zero-coefficient terms
    def _lincomb_expr(
        w: float,
        terms: List[tuple[float, Expr]],
    ) -> Expr:
        """
        Given [(coef, expr), ...], return sum (scaled_const(w, coef) * expr),
        skipping any term whose *rationalized* coefficient is exactly zero.

        This avoids outputs like 0 · (n)² when _to_const_fraction rounds a
        small float to the zero fraction.
        """
        acc: Expr | None = None
        for coef, term_expr in terms:
            const = _scaled_const(w, coef)  # Const(Fraction(...))

            # IMPORTANT: drop if the rational coefficient really is 0.
            # (Adjust attribute name if Const stores the Fraction differently.)
            if const.value == 0:  # type: ignore[attr-defined]
                continue

            piece = const * term_expr
            acc = piece if acc is None else acc + piece

        if acc is None:
            # Degenerate case: all terms rounded to zero
            return Const(Fraction(0, 1))
        return acc


    conjs: List[Conjecture] = []
    t_all = df[target_col].to_numpy(dtype=float)
    w = float(weight)

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

                # ── sqrt mix: sqrt(y) component ──────────────────────
                sqrt_y_arr, sqrt_valid = _safe_sqrt_array(y_arr)
                if sqrt_valid.any():
                    r_sqrt = np.full_like(t_arr, np.nan, dtype=float)
                    r_sqrt[sqrt_valid] = (
                        t_arr[sqrt_valid] / sqrt_y_arr[sqrt_valid]
                    )
                    f_sq = np.isfinite(r_sqrt)

                    if f_sq.sum() >= min_support:
                        s_cmin_f = float(np.min(r_sqrt[f_sq]))
                        s_cmax_f = float(np.max(r_sqrt[f_sq]))

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

                            # Symbolic mirrors, now with combined constants
                            # and dropping zero-coefficient terms.

                            def _lo_base():
                                return _lincomb_expr(
                                    w,
                                    [
                                        (cmin_f, x_expr),
                                        (s_cmin_f, sqrt(y_expr)),
                                    ],
                                )

                            def _lo_ceil_whole():
                                return ceil(_lo_base())

                            def _lo_ceil_split():
                                parts: List[Expr] = []

                                # old: if abs(cmin_f) >= 1e-12:
                                cx = _scaled_const(w, cmin_f)
                                if cx.value != 0:  # type: ignore[attr-defined]
                                    parts.append(ceil(cx * x_expr))

                                cy = _scaled_const(w, s_cmin_f)
                                if cy.value != 0:  # type: ignore[attr-defined]
                                    parts.append(ceil(cy * sqrt(y_expr)))

                                if not parts:
                                    # both coefficients zero ⇒ numeric is -1
                                    return Const(Fraction(-1, 1))

                                acc_expr = parts[0]
                                for p in parts[1:]:
                                    acc_expr = acc_expr + p
                                return acc_expr - Const(Fraction(1, 1))

                            def _up_base():
                                return _lincomb_expr(
                                    w,
                                    [
                                        (cmax_f, x_expr),
                                        (s_cmax_f, sqrt(y_expr)),
                                    ],
                                )

                            def _up_floor_whole():
                                return floor(_up_base())

                            def _up_floor_split():
                                parts: List[Expr] = []

                                cx = _scaled_const(w, cmax_f)
                                if cx.value != 0:  # type: ignore[attr-defined]
                                    parts.append(floor(cx * x_expr))

                                cy = _scaled_const(w, s_cmax_f)
                                if cy.value != 0:  # type: ignore[attr-defined]
                                    parts.append(floor(cy * sqrt(y_expr)))

                                if not parts:
                                    # both coefficients zero ⇒ numeric is 0
                                    return Const(Fraction(0, 1))

                                acc_expr = parts[0]
                                for p in parts[1:]:
                                    acc_expr = acc_expr + p
                                return acc_expr

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
                            return _lincomb_expr(
                                w,
                                [
                                    (cmin_f, x_expr),
                                    (
                                        q_cmin_f,
                                        y_expr ** Const(Fraction(2, 1)),
                                    ),
                                ],
                            )

                        def _lo_sq_ceil_whole():
                            return ceil(_lo_sq_base())

                        def _up_sq_base():
                            return _lincomb_expr(
                                w,
                                [
                                    (cmax_f, x_expr),
                                    (
                                        q_cmax_f,
                                        y_expr ** Const(Fraction(2, 1)),
                                    ),
                                ],
                            )

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

                conjs.extend(lows)
                conjs.extend(ups)

    return conjs
