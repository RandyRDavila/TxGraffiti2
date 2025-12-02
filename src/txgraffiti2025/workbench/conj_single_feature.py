# src/txgraffiti2025/workbench/conj_single_feature.py

from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd
from fractions import Fraction

from txgraffiti2025.forms.utils import to_expr, Const, floor, ceil
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate

from .config import GenerationConfig
from .caches import _EvalCache

"""
Single-feature bound generation.

This workbench component enumerates simple inequalities of the form

    (H) ⇒  target ≥ c_min · x      and      (H) ⇒  target ≤ c_max · x,

optionally adding ``ceil``/``floor`` variants when enabled in
:class:`GenerationConfig`.

Algorithm sketch
----------------
1.  For each hypothesis predicate ``H``:
    - Build its mask on the dataset.
    - Skip empty masks.
2.  For each numeric feature ``x`` distinct from ``target``:
    - Skip if any nonpositive entries (ratios would be ill-defined).
    - Compute elementwise ratios ``t_arr / x_arr`` under ``H``.
    - Extract extrema and approximate them by small rational constants
      (see :func:`to_frac_const`).
    - Emit the two conjectures ``target ≥ c_min x`` and
      ``target ≤ c_max x``.
    - Optionally add ceil/floor-wrapped forms if they hold on *all*
      masked rows.

Examples
--------
>>> import pandas as pd, numpy as np
>>> from txgraffiti2025.forms.predicates import Predicate  # doctest: +SKIP
>>> from txgraffiti2025.workbench.config import GenerationConfig
>>> from txgraffiti2025.workbench.conj_single_feature import generate_single_feature_bounds
>>> # Dummy predicate accepting all rows:
>>> H = Predicate(lambda df: pd.Series(True, index=df.index))  # doctest: +SKIP
>>> df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
>>> cfg = GenerationConfig()
>>> lowers, uppers = generate_single_feature_bounds(df, "a", hyps=[H], numeric_columns=["b"], config=cfg)  # doctest: +SKIP
>>> len(lowers), len(uppers)  # doctest: +SKIP
(1, 1)
"""


def to_frac_const(val: float, max_denom: int = 30) -> Const:
    """
    Convert a floating value into a small-denominator :class:`Const`.

    Uses :class:`fractions.Fraction.limit_denominator` to approximate
    ``val`` by a rational constant with denominator ≤ ``max_denom``.

    Parameters
    ----------
    val : float
        Numeric value to approximate.
    max_denom : int, default=30
        Maximum denominator permitted for the rational approximation.

    Returns
    -------
    Const
        Constant wrapper encapsulating the rational approximation
        (see :class:`txgraffiti2025.forms.utils.Const`).

    Examples
    --------
    >>> from txgraffiti2025.workbench.conj_single_feature import to_frac_const
    >>> c = to_frac_const(0.33333, max_denom=5)
    >>> str(c.value)
    '1/3'
    """
    return Const(Fraction(val).limit_denominator(max_denom))


def generate_single_feature_bounds(
    df: pd.DataFrame,
    target_col: str,
    *,
    hyps: Iterable[Predicate],
    numeric_columns: Iterable[str],
    config: GenerationConfig,
) -> Tuple[List[Conjecture], List[Conjecture]]:
    """
    Generate simple single-feature conjectures ``target ≥ c·x`` and
    ``target ≤ c·x`` for each hypothesis and numeric feature.

    For each predicate ``H`` and each numeric feature ``x`` distinct
    from ``target_col``, this routine computes
    elementwise ratios ``target/x`` over rows where ``H`` holds. The
    minimum and maximum ratios give the best scalar bounds achievable
    by monotonic linear inequalities under that hypothesis.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the numeric invariants/features.
    target_col : str
        Column name of the target invariant being bounded.
    hyps : Iterable[Predicate]
        Hypothesis predicates defining boolean subsets of rows.
    numeric_columns : Iterable[str]
        Names of candidate numeric columns used as right-hand-side
        variables.
    config : GenerationConfig
        Configuration object controlling tolerances, denominators,
        and optional floor/ceil behavior.

    Returns
    -------
    lowers, uppers : tuple[list[Conjecture], list[Conjecture]]
        Two lists of :class:`Conjecture` objects corresponding to
        lower and upper bounds respectively. Each conjecture stores
        its symbolic inequality and the predicate ``H`` it was derived
        from.

    Notes
    -----
    - Rows with nonpositive ``x`` are skipped (ratio undefined).
    - Each bound uses the smallest-denominator rational constant
      approximating the numeric extrema within ``config.max_denom``.
    - When ``config.use_floor_ceil_if_true`` is ``True``, additional
      variants are emitted:
        * lower bounds with ``ceil`` if all data satisfy the discrete
          version ``target ≥ ceil(c_min * x)``,
        * upper bounds with ``floor`` if all data satisfy
          ``target ≤ floor(c_max * x)``.
    - This function is deterministic given the same DataFrame and
      configuration.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from txgraffiti2025.workbench.config import GenerationConfig
    >>> from txgraffiti2025.workbench.conj_single_feature import generate_single_feature_bounds
    >>> from txgraffiti2025.forms.predicates import Predicate  # doctest: +SKIP
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    >>> H = Predicate(lambda df: pd.Series(True, index=df.index))  # doctest: +SKIP
    >>> cfg = GenerationConfig(max_denom=10)
    >>> lowers, uppers = generate_single_feature_bounds(df, "y", hyps=[H], numeric_columns=["x"], config=cfg)  # doctest: +SKIP
    >>> len(lowers), len(uppers)  # doctest: +SKIP
    (1, 1)
    >>> print(lowers[0])  # doctest: +SKIP
    ((H)) ⇒ y ≥ 2·x
    """
    target = to_expr(target_col)
    lowers: List[Conjecture] = []
    uppers: List[Conjecture] = []

    for H in hyps:
        mask = H.mask(df).astype(bool).to_numpy()
        if not np.any(mask):
            continue
        dfH = df.loc[mask]
        cache = _EvalCache(dfH)
        t_arr = target.eval(dfH).values.astype(float, copy=False)

        for xname in numeric_columns:
            if xname == target_col:
                continue
            x_arr = cache.col(xname)
            if np.min(x_arr) <= 0:
                continue

            rx = t_arr / x_arr
            cmin_f = float(np.min(rx))
            cmax_f = float(np.max(rx))
            cmin = to_frac_const(cmin_f, config.max_denom)
            cmax = to_frac_const(cmax_f, config.max_denom)
            x_expr = to_expr(xname)

            lowers.append(Conjecture(Ge(target, cmin * x_expr), H))
            uppers.append(Conjecture(Le(target, cmax * x_expr), H))

            if not config.use_floor_ceil_if_true:
                continue
            if np.all(t_arr >= np.ceil(cmin_f * x_arr)) and cmin.value.denominator > 1:
                lowers.append(Conjecture(Ge(target, ceil(cmin * x_expr)), H))
            if np.all(t_arr <= np.floor(cmax_f * x_arr)) and cmax.value.denominator > 1:
                uppers.append(Conjecture(Le(target, floor(cmax * x_expr)), H))

    return lowers, uppers
