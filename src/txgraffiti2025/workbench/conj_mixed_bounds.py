# src/txgraffiti2025/workbench/conj_mixed_bounds.py

from __future__ import annotations
from typing import Iterable, Tuple, List, Sequence
import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import to_expr, Const, floor, ceil, sqrt
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate
from .config import GenerationConfig
from .caches import _EvalCache
from .conj_single_feature import to_frac_const

"""
Two-feature (mixed) bound generation: linear + sqrt and linear + square.

This module proposes inequalities of the form

    (H) ⇒  target ≥ w·( c_min·x + s_min·sqrt(y) )
    (H) ⇒  target ≤ w·( c_max·x + s_max·sqrt(y) )

and similarly for a **square** mix,

    (H) ⇒  target ≥ w·( c_min·x + q_min·y^2 )
    (H) ⇒  target ≤ w·( c_max·x + q_max·y^2 ),

where constants are chosen from extrema of per-row ratios over the subset
defined by hypothesis predicate H. Optionally, “whole-expression” ceil/floor
variants are considered when they hold across all rows.

Algorithm (per H, per (x, y))
-----------------------------
1. Require strictly positive columns for denominators (x > 0, y > 0).
2. Compute ratios:
   - linear part:   r_x     = target / x
   - sqrt part:     r_sqrt  = target / sqrt(y)
   - square part:   r_sq    = target / (y^2)
   Take min/max of each to obtain c_min/c_max, s_min/s_max, q_min/q_max.
3. Build candidate RHS arrays for lower/upper bounds and select the **tightest
   valid** option using mean(RHS) as a tie-breaker:
   - For ≥ (lower): choose the **largest** RHS that never exceeds target.
   - For ≤ (upper): choose the **smallest** RHS that never falls below target.
4. Emit symbolic Conjectures using :func:`to_expr`, :func:`sqrt`, and exponentiation.

Examples
--------
>>> import pandas as pd, numpy as np
>>> from txgraffiti2025.workbench.config import GenerationConfig
>>> from txgraffiti2025.forms.predicates import Predicate  # doctest: +SKIP
>>> from txgraffiti2025.workbench.conj_mixed_bounds import generate_mixed_bounds
>>> df = pd.DataFrame({"t": [4., 9., 16.], "x": [1., 2., 4.], "y": [1., 4., 9.]})
>>> H = Predicate(lambda df: pd.Series(True, index=df.index))  # doctest: +SKIP
>>> lowers, uppers = generate_mixed_bounds(
...     df, target_col="t", hyps=[H],
...     primary=["x"], secondary=["y"], config=GenerationConfig(), weight=0.5
... )  # doctest: +SKIP
>>> len(lowers) > 0 and len(uppers) > 0  # doctest: +SKIP
True
"""


def _pick_best_ge(
    t_arr: np.ndarray,
    rhs_variants: Sequence[tuple[str, np.ndarray, "callable"]],
):
    """
    Select the tightest valid RHS for a ≥ inequality.

    Among provided candidates ``rhs_variants = [(label, rhs_array, make_expr), ...]``,
    choose the one that satisfies ``t_arr >= rhs_array`` for all rows and has the
    **largest mean(rhs_array)** (to avoid trivial weak bounds). Returns a tuple
    ``(label, make_expr)`` or ``None`` if none qualify.

    Parameters
    ----------
    t_arr : numpy.ndarray
        Target values on the masked subset.
    rhs_variants : sequence of (str, numpy.ndarray, callable)
        Each variant comprises a label, a candidate RHS array, and a thunk
        ``make_expr() -> Expr`` that builds the symbolic RHS if selected.

    Returns
    -------
    tuple[str, callable] or None
        The winning label and expression factory, or ``None`` if no candidate
        satisfies the inequality across all rows.

    Notes
    -----
    - Uses ``np.all(t_arr >= rhs)`` with no additional finiteness filtering;
      upstream construction should ensure finiteness/alignment.
    """
    best = None
    best_score = -np.inf
    for lab, rhs, make_expr in rhs_variants:
        if np.all(t_arr >= rhs):
            score = float(np.mean(rhs))
            if score > best_score:
                best = (lab, make_expr)
                best_score = score
    return best


def _pick_best_le(
    t_arr: np.ndarray,
    rhs_variants: Sequence[tuple[str, np.ndarray, "callable"]],
):
    """
    Select the tightest valid RHS for a ≤ inequality.

    Among candidates ``rhs_variants = [(label, rhs_array, make_expr), ...]``,
    pick the one that satisfies ``t_arr <= rhs_array`` for all rows and has the
    **smallest mean(rhs_array)** (tightest upper bound). Returns a tuple
    ``(label, make_expr)`` or ``None`` if none qualify.

    Parameters
    ----------
    t_arr : numpy.ndarray
        Target values on the masked subset.
    rhs_variants : sequence of (str, numpy.ndarray, callable)
        See :func:`_pick_best_ge`.

    Returns
    -------
    tuple[str, callable] or None
        Winning label and expression factory, or ``None``.
    """
    best = None
    best_score = +np.inf
    for lab, rhs, make_expr in rhs_variants:
        if np.all(t_arr <= rhs):
            score = float(np.mean(rhs))
            if score < best_score:
                best = (lab, make_expr)
                best_score = score
    return best


def generate_mixed_bounds(
    df: pd.DataFrame,
    target_col: str,
    *,
    hyps: Iterable[Predicate],
    primary: Iterable[str],
    secondary: Iterable[str],
    config: GenerationConfig,
    weight: float = 0.5,
) -> Tuple[List[Conjecture], List[Conjecture]]:
    """
    Generate 2-feature mixed bounds (linear + sqrt, linear + square).

    For each hypothesis ``H``, each primary feature ``x`` and secondary
    feature ``y`` (distinct from the target and strictly positive on the
    masked subset), compute min/max ratio coefficients and form weighted
    mixtures with ``weight`` (typically in ``(0, 1]``). Emit both base
    and whole-expression ceil/floor variants (see Notes).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of numeric invariants/features.
    target_col : str
        Target column name.
    hyps : Iterable[Predicate]
        Hypothesis predicates.
    primary : Iterable[str]
        Candidate RHS “primary” features (the linear component).
    secondary : Iterable[str]
        Candidate RHS “secondary” features (used in sqrt / square components).
    config : GenerationConfig
        Controls rational approximation (``max_denom``) and (optionally) whether
        ceil/floor should be used for whole-expression variants.
    weight : float, default=0.5
        Mixture weight ``w``, used as a scalar multiplier for RHS mixes.
        Must satisfy ``0 < weight ≤ 1``.

    Returns
    -------
    lowers, uppers : tuple[list[Conjecture], list[Conjecture]]
        Generated lower- and upper-bound conjectures.

    Notes
    -----
    - Positivity checks: rows with nonpositive ``x`` or ``y`` are skipped to
      avoid divisions by zero/negatives in ratio definitions.
    - Coefficients are rationalized via :func:`to_frac_const`. The call sites
      use either the given ``config.max_denom`` or the default (30); see
      discussion in the review if you want full consistency.
    - Whole-expression variants use ``np.ceil``/``np.floor`` on the **numeric**
      RHS arrays and then map to symbolic ``ceil(...)`` / ``floor(...)`` if valid.
      (Currently these are always considered; see review for gating by
      ``config.use_floor_ceil_if_true`` to match single-feature behavior.)

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.workbench.config import GenerationConfig
    >>> from txgraffiti2025.forms.predicates import Predicate  # doctest: +SKIP
    >>> df = pd.DataFrame({"t": [10., 12., 14.], "x": [2., 3., 4.], "y": [1., 4., 9.]})
    >>> H = Predicate(lambda df: pd.Series(True, index=df.index))  # doctest: +SKIP
    >>> lowers, uppers = generate_mixed_bounds(
    ...     df, "t", hyps=[H], primary=["x"], secondary=["y"],
    ...     config=GenerationConfig(max_denom=16), weight=0.7
    ... )  # doctest: +SKIP
    >>> len(lowers) >= 2 and len(uppers) >= 2  # doctest: +SKIP
    True
    """
    assert 0 < weight <= 1.0
    target = to_expr(target_col)
    lowers: List[Conjecture] = []
    uppers: List[Conjecture] = []
    w = float(weight)
    w_const = to_frac_const(weight, config.max_denom)

    for H in hyps:
        mask = H.mask(df).astype(bool).to_numpy()
        if not np.any(mask):
            continue
        dfH = df.loc[mask]
        cache = _EvalCache(dfH)
        t_arr = target.eval(dfH).values.astype(float, copy=False)

        for xname in primary:
            if xname == target_col:
                continue
            x_arr = cache.col(xname)
            if np.min(x_arr) <= 0:
                continue

            rx = t_arr / x_arr
            cmin_f = float(np.min(rx))
            cmax_f = float(np.max(rx))

            for yname in secondary:
                if yname == target_col:
                    continue
                y_arr = cache.col(yname)
                if np.min(y_arr) <= 0:
                    continue

                # sqrt mix
                sqrt_y = cache.sqrt_col(yname)
                r_sqrt = t_arr / sqrt_y
                s_cmin_f = float(np.min(r_sqrt))
                s_cmax_f = float(np.max(r_sqrt))
                lower_arr = w * (cmin_f * x_arr + s_cmin_f * sqrt_y)
                upper_arr = w * (cmax_f * x_arr + s_cmax_f * sqrt_y)

                lower_variants = [
                    ("base", lower_arr, lambda: w_const * to_frac_const(cmin_f) * to_expr(xname) +
                                               w_const * to_frac_const(s_cmin_f) * sqrt(to_expr(yname))),
                    ("ceil whole", np.ceil(lower_arr), lambda: ceil(
                        w_const * to_frac_const(cmin_f) * to_expr(xname) +
                        w_const * to_frac_const(s_cmin_f) * sqrt(to_expr(yname))
                    )),
                ]
                choice = _pick_best_ge(t_arr, lower_variants)
                if choice:
                    lowers.append(Conjecture(Ge(target, choice[1]()), H))

                upper_variants = [
                    ("base", upper_arr, lambda: w_const * to_frac_const(cmax_f) * to_expr(xname) +
                                               w_const * to_frac_const(s_cmax_f) * sqrt(to_expr(yname))),
                    ("floor whole", np.floor(upper_arr), lambda: floor(
                        w_const * to_frac_const(cmax_f) * to_expr(xname) +
                        w_const * to_frac_const(s_cmax_f) * sqrt(to_expr(yname))
                    )),
                ]
                choice = _pick_best_le(t_arr, upper_variants)
                if choice:
                    uppers.append(Conjecture(Le(target, choice[1]()), H))

                # square mix
                y_sq = cache.sq_col(yname)
                r_sq = t_arr / y_sq
                q_cmin_f = float(np.min(r_sq))
                q_cmax_f = float(np.max(r_sq))
                lower_sq_arr = w * (cmin_f * x_arr + q_cmin_f * y_sq)
                upper_sq_arr = w * (cmax_f * x_arr + q_cmax_f * y_sq)

                l2 = [("base", lower_sq_arr, lambda: w_const * to_frac_const(cmin_f) * to_expr(xname) +
                                                  w_const * to_frac_const(q_cmin_f) * (to_expr(yname) ** to_frac_const(2)))]
                u2 = [("base", upper_sq_arr, lambda: w_const * to_frac_const(cmax_f) * to_expr(xname) +
                                                  w_const * to_frac_const(q_cmax_f) * (to_expr(yname) ** to_frac_const(2)))]
                choice = _pick_best_ge(t_arr, l2)
                if choice:
                    lowers.append(Conjecture(Ge(target, choice[1]()), H))
                choice = _pick_best_le(t_arr, u2)
                if choice:
                    uppers.append(Conjecture(Le(target, choice[1]()), H))

    return lowers, uppers
