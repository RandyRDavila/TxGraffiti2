# src/txgraffiti2025/workbench/conj_targeted_products.py

from __future__ import annotations
from itertools import combinations_with_replacement
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import to_expr
from txgraffiti2025.forms.generic_conjecture import Conjecture, Ge, Le
from txgraffiti2025.forms.predicates import Predicate
from .arrays import all_le_on_mask, strictly_pos_on

"""
Targeted product bounds: (H) ⇒ T·x ≤ y·z and (H) ⇒ T·x ≥ y·z.

This module proposes multiplicative inequalities that “target” a specific
column ``T = target_col`` via product comparisons involving a chosen
factor ``x`` and two candidate factors ``y, z``. It supports:

- **Cancellation** when x matches y or z and x is strictly positive under H,
  reducing T·x ≤ y·z to T ≤ z (or T ≤ y), and analogously for ≥.
- **Triviality** filtering for the product inequalities under positivity,
  avoiding statements that follow immediately from pairwise comparisons.

High-level idea
---------------
For each hypothesis H, for each ``x`` in ``x_candidates``, and for each
pair ``(y, z)`` (with replacement) from ``yz_candidates``, we attempt:

- Upper:  (H) ⇒ T·x ≤ y·z
- Lower:  (H) ⇒ T·x ≥ y·z

We first emit cancellation-based bounds when possible, then emit product
bounds only if they are nontrivial and hold on all finite, valid rows.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.predicates import Predicate  # doctest: +SKIP
>>> from txgraffiti2025.workbench.conj_targeted_products import generate_targeted_product_bounds
>>> df = pd.DataFrame({"T":[2.,3.,4.], "x":[1.,1.,1.], "y":[2.,2.,2.], "z":[2.,2.,2.]})
>>> H = Predicate(lambda df: pd.Series(True, index=df.index))  # doctest: +SKIP
>>> lowers, uppers = generate_targeted_product_bounds(
...     df, target_col="T", hyps=[H],
...     x_candidates=["x"], yz_candidates=["y","z"]
... )  # doctest: +SKIP
>>> len(lowers) >= 1 and len(uppers) >= 1  # doctest: +SKIP
True
"""


def _caro_trivial_masked_targeted(
    T_arr: np.ndarray,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    z_arr: np.ndarray,
    mask: np.ndarray,
    *,
    upper: bool,
) -> bool:
    """
    Detect triviality of the targeted product under a finite mask.

    Under the assumption of **positivity** (the caller enforces it),
    the statements
        - Upper:  T·x ≤ y·z
        - Lower:  T·x ≥ y·z
    are trivially true if certain pairwise inequalities already hold.
    This helper checks those pairwise implications on the subset
    ``mask & finite`` and returns ``True`` iff the product inequality
    is **trivial**.

    Parameters
    ----------
    T_arr, x_arr, y_arr, z_arr : numpy.ndarray
        Arrays of equal length for target and factors.
    mask : numpy.ndarray
        Boolean array selecting rows to consider.
    upper : bool
        If ``True``, test triviality for the upper product bound
        ``T·x ≤ y·z``; otherwise for the lower bound ``T·x ≥ y·z``.

    Returns
    -------
    bool
        ``True`` if the product inequality is trivial on the masked,
        finite subset; ``False`` otherwise.

    Notes
    -----
    - This function **does not** enforce positivity itself. The caller
      should ensure positivity prior to calling (see usage in
      :func:`generate_targeted_product_bounds`).
    - Finiteness is enforced internally: rows with non-finite values
      in any of the four arrays are excluded from consideration.

    Examples
    --------
    >>> import numpy as np
    >>> T = np.array([2., 3.])
    >>> x = np.array([1., 1.])
    >>> y = np.array([2., 3.])
    >>> z = np.array([2., 3.])
    >>> m = np.array([True, True])
    >>> _caro_trivial_masked_targeted(T, x, y, z, m, upper=True)
    True
    """
    m = mask & np.isfinite(T_arr) & np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)
    if not np.any(m):
        return False
    if upper:
        c1 = all_le_on_mask(T_arr, y_arr, m) and all_le_on_mask(x_arr, z_arr, m)
        if c1:
            return True
        c2 = all_le_on_mask(T_arr, z_arr, m) and all_le_on_mask(x_arr, y_arr, m)
        return c2
    else:
        c1 = all_le_on_mask(y_arr, T_arr, m) and all_le_on_mask(z_arr, x_arr, m)
        if c1:
            return True
        c2 = all_le_on_mask(z_arr, T_arr, m) and all_le_on_mask(y_arr, x_arr, m)
        return c2


def generate_targeted_product_bounds(
    df: pd.DataFrame,
    target_col: str,
    *,
    hyps: Iterable[Predicate],
    x_candidates: Iterable[str],
    yz_candidates: Iterable[str],
    require_pos: bool = True,
    enable_cancellation: bool = True,
    allow_x_equal_yz: bool = True,
) -> Tuple[List[Conjecture], List[Conjecture]]:
    """
    Emit product-style inequalities with triviality checks and optional cancellation.

    For each hypothesis ``H``, each ``x`` in ``x_candidates``, and each pair
    ``(y, z)`` drawn with replacement from ``yz_candidates``, this function:

    1. Builds per-column arrays on the masked subset.
    2. Optionally emits **cancellation** bounds when ``x == y`` or ``x == z`` and
       ``x`` is strictly positive on the valid rows (e.g., from ``T·x ≤ y·z`` to
       ``T ≤ z``).
    3. Emits **non-canceled** product bounds only if:
       - the rows used are finite (and positive when ``require_pos=True``), and
       - the product inequality holds on **all** selected rows, and
       - the inequality is **nontrivial** under positivity (checked by
         :func:`_caro_trivial_masked_targeted`).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing ``target_col`` and all candidates.
    target_col : str
        Name of the target column ``T``.
    hyps : Iterable[Predicate]
        Hypothesis predicates evaluated against ``df``.
    x_candidates : Iterable[str]
        Candidate columns for the single factor ``x``.
    yz_candidates : Iterable[str]
        Candidate columns from which pairs ``(y, z)`` are drawn (with replacement).
    require_pos : bool, default=True
        If ``True``, only consider rows where ``T, x, y, z > 0``. This enables
        cancellation logic and the triviality implications used here.
    enable_cancellation : bool, default=True
        If ``True``, attempt to simplify product bounds into unary bounds on ``T``
        when ``x`` cancels with ``y`` or ``z`` and positivity holds.
    allow_x_equal_yz : bool, default=True
        If ``False``, skip pairs where ``x`` equals ``y`` or ``z`` (disabling
        possible cancellations).

    Returns
    -------
    lowers, uppers : tuple[list[Conjecture], list[Conjecture]]
        Lower and upper conjectures respectively.  Lower bounds correspond to
        ``Ge(...)`` relations and upper bounds to ``Le(...)`` relations.  Note
        these may include statements on products (``T·x`` vs ``y·z``) if no
        cancellation applied.

    Notes
    -----
    - All comparisons are performed on **finite** rows only (NaN/±inf excluded).
      The functions from :mod:`txgraffiti2025.workbench.arrays` also require at
      least one valid row to evaluate to ``True``.
    - Product inequalities can overflow intermediate products; this is handled
      safely because :func:`all_le_on_mask` re-checks finiteness for both sides.
    - If you prefer to normalize every product inequality back to a unary bound
      on ``T``, set ``require_pos=True`` and keep ``enable_cancellation=True``;
      otherwise, you may get conjectures comparing products symbolically.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Predicate  # doctest: +SKIP
    >>> df = pd.DataFrame({
    ...     "T": [2., 3., 4.], "x": [1., 1., 1.], "y": [2., 3., 4.], "z": [2., 3., 4.]
    ... })
    >>> H = Predicate(lambda df: pd.Series(True, index=df.index))  # doctest: +SKIP
    >>> lowers, uppers = generate_targeted_product_bounds(
    ...     df, target_col="T", hyps=[H],
    ...     x_candidates=["x"], yz_candidates=["y", "z"],
    ...     require_pos=True, enable_cancellation=True
    ... )  # doctest: +SKIP
    >>> len(lowers) >= 1 and len(uppers) >= 1  # doctest: +SKIP
    True
    """
    target = to_expr(target_col)
    uppers: List[Conjecture] = []
    lowers: List[Conjecture] = []

    for H in hyps:
        Hmask = H.mask(df).astype(bool).to_numpy()
        if not np.any(Hmask):
            continue
        dfH = df.loc[Hmask]

        arrays = {
            c: to_expr(c).eval(dfH).values.astype(float, copy=False)
            for c in set(x_candidates) | set(yz_candidates) | {target_col}
        }
        T_arr = arrays[target_col]

        for x in x_candidates:
            if require_pos and np.min(arrays[x]) <= 0:
                continue
            for (y, z) in combinations_with_replacement(yz_candidates, 2):
                if not allow_x_equal_yz and (x == y or x == z):
                    continue

                x_arr, y_arr, z_arr = arrays[x], arrays[y], arrays[z]

                base_valid = (
                    np.isfinite(T_arr) & np.isfinite(x_arr) &
                    np.isfinite(y_arr) & np.isfinite(z_arr)
                )
                if require_pos:
                    base_valid &= (T_arr > 0.0) & (x_arr > 0.0) & (y_arr > 0.0) & (z_arr > 0.0)
                if not np.any(base_valid):
                    continue

                # cancellation: x == y or x == z and x strictly positive on valid rows
                canceled_upper = canceled_lower = False
                if enable_cancellation:
                    if x == y and strictly_pos_on(x_arr, base_valid):
                        if all_le_on_mask(T_arr, z_arr, base_valid):
                            uppers.append(Conjecture(Le(target, to_expr(z)), H))
                            canceled_upper = True
                        if all_le_on_mask(z_arr, T_arr, base_valid):
                            lowers.append(Conjecture(Ge(target, to_expr(z)), H))
                            canceled_lower = True

                    if x == z and strictly_pos_on(x_arr, base_valid):
                        if all_le_on_mask(T_arr, y_arr, base_valid):
                            uppers.append(Conjecture(Le(target, to_expr(y)), H))
                            canceled_upper = True
                        if all_le_on_mask(y_arr, T_arr, base_valid):
                            lowers.append(Conjecture(Ge(target, to_expr(y)), H))
                            canceled_lower = True

                # non-canceled product inequalities
                if not canceled_upper:
                    if not (require_pos and _caro_trivial_masked_targeted(T_arr, x_arr, y_arr, z_arr, base_valid, upper=True)):
                        if all_le_on_mask(T_arr * x_arr, y_arr * z_arr, base_valid):
                            uppers.append(Conjecture(Le(target * to_expr(x), to_expr(y) * to_expr(z)), H))

                if not canceled_lower:
                    if not (require_pos and _caro_trivial_masked_targeted(T_arr, x_arr, y_arr, z_arr, base_valid, upper=False)):
                        if all_le_on_mask(y_arr * z_arr, T_arr * x_arr, base_valid):
                            lowers.append(Conjecture(Ge(target * to_expr(x), to_expr(y) * to_expr(z)), H))

    return lowers, uppers
