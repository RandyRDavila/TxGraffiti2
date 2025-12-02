# src/txgraffiti2025/workbench/arrays.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from txgraffiti2025.forms.predicates import Predicate

"""
Array and mask helpers used across the workbench.

This module provides small, NumPy-centric utilities for:
- converting a :class:`Predicate` into a boolean mask aligned to a DataFrame,
- converting inputs to aligned :class:`pandas.Series`,
- boolean-mask logic (support, inclusion, equality),
- numeric checks under a mask (finite-guarded comparisons, positivity),
- finite-value detection.

Design goals
------------
- All masks returned/consumed are boolean NumPy arrays of shape ``(len(df),)``.
- Functions are NaN/inf-stable where relevant (see examples).
- Behavior on empty masks is explicit (see Notes of relevant functions).

Examples
--------
>>> import numpy as np, pandas as pd
>>> from txgraffiti2025.workbench.arrays import (
...     mask_from_pred, as_series, support, same_mask, includes,
...     all_le_on_mask, strictly_pos_on, finite_mask
... )
>>> df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "flag": [True, False, True]})

# mask_from_pred with H=None returns all True:
>>> mask = mask_from_pred(df, None)
>>> mask.dtype, mask.shape, mask.sum()
(dtype('bool'), (3,), 3)

# as_series aligns input to an index:
>>> as_series([10, 20, 30], df.index).tolist()
[10, 20, 30]

# support and same_mask:
>>> support(np.array([True, False, True]))
2
>>> same_mask(np.array([True, False]), np.array([True, False]))
True

# includes: a includes b  <=>  b => a
>>> A = np.array([True, True, False, True])
>>> B = np.array([True, False, False, True])
>>> includes(A, B)
True
>>> includes(B, A)
False

# all_le_on_mask: finite-guarded <= check
>>> a = np.array([1.0, 2.0, np.inf, 4.0])
>>> b = np.array([1.0, 2.5, 100.0, np.nan])
>>> m = np.array([True, True, True, True])
>>> all_le_on_mask(a, b, m)  # ignores non-finite pairs; requires at least one valid comparison
True

# strictly_pos_on: all positive on finite, masked entries
>>> strictly_pos_on(np.array([1.0, 2.0, np.nan]), np.array([True, True, True]))
True
>>> strictly_pos_on(np.array([1.0, 0.0, 3.0]), np.array([True, True, True]))
False

# finite_mask:
>>> finite_mask(np.array([1.0, np.inf, np.nan, -5]))
array([ True, False, False,  True])
"""


def mask_from_pred(df: pd.DataFrame, H: Optional[Predicate]) -> np.ndarray:
    """
    Build a boolean mask from a Predicate aligned to ``df.index``.

    If ``H`` is ``None``, returns an all-``True`` mask of length ``len(df)``.
    Otherwise, calls ``H.mask(df)`` (expected to be a boolean ``Series``)
    and reindexes it to ``df.index`` with ``fill_value=False``, ensuring a
    boolean NumPy array output.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame against which the predicate is evaluated.
    H : Predicate or None
        A boolean predicate with a ``mask(df) -> Series[bool]`` method, or ``None``
        to indicate no filtering.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(len(df),)`` aligned to ``df.index``.

    Notes
    -----
    - If ``H.mask(df)`` has missing indices, they are treated as ``False``.
    - Non-boolean dtypes from ``H.mask(df)`` are coerced to boolean with
      ``fillna(False).astype(bool)``.
    - If ``len(df) == 0``, the result is an empty boolean array.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2, 3]})
    >>> mask_from_pred(df, None).sum()  # no predicate -> keep all
    3
    """
    if H is None:
        return np.ones(len(df), dtype=bool)
    s = H.mask(df).reindex(df.index, fill_value=False)
    if s.dtype != bool:
        s = s.fillna(False).astype(bool, copy=False)
    return s.to_numpy()


def as_series(x, index) -> pd.Series:
    """
    Convert ``x`` to a :class:`pandas.Series` aligned to ``index``.

    If ``x`` is already a Series, it is reindexed to ``index``.
    Otherwise, builds a new Series with ``index``. Scalars will be
    broadcast by pandas; sequences must match the index length.

    Parameters
    ----------
    x : Any
        A scalar, sequence, or Series-like.
    index : pandas.Index
        Target index for alignment.

    Returns
    -------
    pandas.Series
        Series aligned to ``index``.

    Raises
    ------
    ValueError
        If ``x`` is a sequence whose length does not match ``len(index)``.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.Index(["u", "v", "w"])
    >>> as_series([10, 20, 30], idx).tolist()
    [10, 20, 30]
    >>> as_series(7, idx).tolist()  # scalar broadcast
    [7, 7, 7]
    """
    if isinstance(x, pd.Series):
        return x.reindex(index)
    return pd.Series(x, index=index)


def support(mask: np.ndarray) -> int:
    """
    Count ``True`` values in a boolean mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean array.

    Returns
    -------
    int
        Number of ``True`` entries.

    Examples
    --------
    >>> import numpy as np
    >>> support(np.array([True, False, True]))
    2
    """
    return int(mask.sum())


def same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Test whether two boolean masks are exactly equal.

    Parameters
    ----------
    a, b : numpy.ndarray
        Boolean arrays of the same shape.

    Returns
    -------
    bool
        ``True`` iff arrays are equal elementwise.

    Notes
    -----
    This uses :func:`numpy.array_equal`.

    Examples
    --------
    >>> import numpy as np
    >>> same_mask(np.array([True, False]), np.array([True, False]))
    True
    >>> same_mask(np.array([True, False]), np.array([False, True]))
    False
    """
    return bool(np.array_equal(a, b))


def includes(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check set-inclusion of boolean masks: ``a`` includes ``b`` iff (``b`` => ``a``).

    This is equivalent to verifying that wherever ``b`` is ``True``,
    ``a`` is also ``True``: i.e., ``all(~b | a)``.

    Parameters
    ----------
    a, b : numpy.ndarray
        Boolean arrays of the same shape.

    Returns
    -------
    bool
        ``True`` iff for all i, ``b[i]`` implies ``a[i]``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([True, True, False])
    >>> b = np.array([True, False, False])
    >>> includes(a, b)  # every True in b is True in a
    True
    >>> includes(b, a)  # not vice versa
    False
    """
    # a includes b  <=> whenever b is True, a is True
    return bool(np.all(~b | a))


def all_le_on_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> bool:
    """
    Test whether ``a <= b`` holds for at least one **finite** pair under ``mask``,
    and for **all** finite pairs selected by ``mask``.

    This function first refines the mask to finite entries via
    ``np.isfinite(a) & np.isfinite(b)``. If the refined mask is empty,
    it returns ``False`` (no meaningful comparison). Otherwise, it
    checks ``a[m] <= b[m]`` for all valid entries.

    Parameters
    ----------
    a, b : numpy.ndarray
        Numeric arrays of the same shape.
    mask : numpy.ndarray
        Boolean mask selecting the rows to check.

    Returns
    -------
    bool
        ``True`` iff there exists at least one finite pair under ``mask`` and
        all finite pairs satisfy ``a <= b``; otherwise ``False``.

    Notes
    -----
    - Non-finite pairs (NaN/±inf) are ignored; at least one finite pair must exist.
    - Shape mismatches raise from NumPy operations.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1.0, 2.0, np.inf])
    >>> b = np.array([1.0, 2.1, 10.0])
    >>> m = np.array([True, True, True])
    >>> all_le_on_mask(a, b, m)
    True
    >>> all_le_on_mask(np.array([np.nan]), np.array([0.0]), np.array([True]))
    False
    """
    m = mask & np.isfinite(a) & np.isfinite(b)
    return bool(np.any(m) and np.all(a[m] <= b[m]))


def strictly_pos_on(a: np.ndarray, mask: np.ndarray) -> bool:
    """
    Check that all **finite** entries of ``a`` selected by ``mask`` are strictly positive.

    Non-finite values are ignored. If no finite values remain under the mask,
    returns ``False``.

    Parameters
    ----------
    a : numpy.ndarray
        Numeric array.
    mask : numpy.ndarray
        Boolean mask.

    Returns
    -------
    bool
        ``True`` iff there exists at least one finite value under ``mask`` and
        all such values are ``> 0``; otherwise ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> strictly_pos_on(np.array([1.0, 2.0]), np.array([True, True]))
    True
    >>> strictly_pos_on(np.array([1.0, 0.0]), np.array([True, True]))
    False
    >>> strictly_pos_on(np.array([np.nan]), np.array([True]))
    False
    """
    m = mask & np.isfinite(a)
    return bool(np.any(m) and np.all(a[m] > 0.0))


def finite_mask(a: np.ndarray) -> np.ndarray:
    """
    Elementwise finiteness check.

    Parameters
    ----------
    a : numpy.ndarray
        Numeric array.

    Returns
    -------
    numpy.ndarray
        Boolean array where entries are ``True`` iff the corresponding
        element of ``a`` is finite (not NaN and not ±inf).

    Examples
    --------
    >>> import numpy as np
    >>> finite_mask(np.array([1.0, np.nan, np.inf, -3.0]))
    array([ True, False, False,  True])
    """
    return np.isfinite(a)
