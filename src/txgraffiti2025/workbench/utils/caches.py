# caches.py

from __future__ import annotations
import numpy as np
import pandas as pd
from txgraffiti2025.forms.utils import to_expr, safe_sqrt_series

"""
Lightweight evaluators with memoization for column/transform lookups.

This module provides a small, in-memory cache used by the workbench when it needs
to repeatedly evaluate columns or simple transforms (square roots, squares) on a
**filtered/temporary** DataFrame. The goal is to avoid recomputing the same
expressions during candidate generation, ranking, or checking phases.

Design
------
- The cache is **per-DataFrame-slice**. If the underlying DataFrame changes
  (e.g., rows filtered differently, column values mutated), create a new cache.
- Evaluations are memoized by the *string key* (column name / expression name).
- All results are returned as NumPy arrays of dtype ``float`` for downstream
  numeric operations and NaN/inf-stability.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from txgraffiti2025.workbench.caches import _EvalCache
>>> df = pd.DataFrame({"x": [1.0, 4.0, 9.0], "y": [2.0, np.nan, 8.0]})
>>> ec = _EvalCache(df)

# Raw column as float array:
>>> ec.col("x")
array([1., 4., 9.])

# Square root via the sqrt cache (delegates to safe_sqrt_series):
>>> ec.sqrt_col("x")
array([1., 2., 3.])

# Square (vectorized):
>>> ec.sq_col("y")
array([ 4., nan, 64.])

# Repeated calls are served from memoized results:
>>> ec.col("x") is ec.col("x")
True
"""


class _EvalCache:
    """
    Cache column/transform evaluations on a fixed DataFrame to avoid recomputation.

    The cache holds:
    - raw column/evaluable expression results (``_col``),
    - square roots of columns (``_sqrt``),
    - squares of columns (``_sq``).

    Parameters
    ----------
    df_temp : pandas.DataFrame
        The DataFrame (often an already-filtered view/slice) on which evaluations
        will be performed. Treat it as **immutable** for the lifetime of this cache.

    Notes
    -----
    - Keys are strings (column names or expression names accepted by ``to_expr``).
    - Results are stored as NumPy arrays with ``dtype=float``.
    - This class is not thread-safe.
    - If you mutate ``df_temp`` after constructing the cache, cached values may
      become stale. Prefer creating a fresh instance of :class:`_EvalCache` instead.

    See Also
    --------
    txgraffiti2025.forms.utils.to_expr
        Converts a column/expression string into an evaluable expression object with
        ``.eval(df) -> pandas.Series``.
    txgraffiti2025.forms.utils.safe_sqrt_series
        Applies a NaN/inf-safe square root transform to an array/Series.

    Examples
    --------
    See module examples above.
    """

    def __init__(self, df_temp: pd.DataFrame):
        self.df = df_temp
        self._col: dict[str, np.ndarray] = {}
        self._sqrt: dict[str, np.ndarray] = {}
        self._sq: dict[str, np.ndarray] = {}

    def col(self, name: str) -> np.ndarray:
        """
        Evaluate a column or expression on ``self.df`` and memoize the float array.

        Delegates to :func:`to_expr` and then calls ``.eval(self.df)``. The result
        is coerced to a NumPy array with ``dtype=float`` (using a view when possible).

        Parameters
        ----------
        name : str
            Column name or an expression string accepted by :func:`to_expr`.

        Returns
        -------
        numpy.ndarray
            A 1D float NumPy array aligned with ``self.df.index``.

        Raises
        ------
        KeyError
            If the column/expression references missing columns.
        TypeError
            If ``to_expr(name).eval(self.df)`` does not return a 1D Series.
        ValueError
            If the evaluated length does not match ``len(self.df)``.

        Notes
        -----
        - If the evaluated series has an object dtype, a float coercion will be
          attempted; failures will raise.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2]})
        >>> _EvalCache(df).col("a")
        array([1., 2.])
        """
        if name not in self._col:
            s = to_expr(name).eval(self.df)
            if not isinstance(s, pd.Series):
                raise TypeError("to_expr(name).eval(df) must return a pandas Series.")
            if len(s) != len(self.df):
                raise ValueError("Evaluated result length does not match DataFrame length.")
            self._col[name] = s.values.astype(float, copy=False)
        return self._col[name]

    def sqrt_col(self, name: str) -> np.ndarray:
        """
        Evaluate and memoize the **square root** of a column/expression.

        Uses :meth:`col` for the base array, then applies
        :func:`safe_sqrt_series` to obtain a NaN/inf-safe square root.

        Parameters
        ----------
        name : str
            Column name or expression string.

        Returns
        -------
        numpy.ndarray
            A 1D float NumPy array containing the elementwise square roots.

        Notes
        -----
        Behavior on negative inputs depends on :func:`safe_sqrt_series`.
        Typically, it returns ``NaN`` for negatives and preserves shape.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> df = pd.DataFrame({"x": [0.0, 4.0, 9.0]})
        >>> ec = _EvalCache(df)
        >>> ec.sqrt_col("x")
        array([0., 2., 3.])
        """
        if name not in self._sqrt:
            self._sqrt[name] = safe_sqrt_series(self.col(name))
        return self._sqrt[name]

    def sq_col(self, name: str) -> np.ndarray:
        """
        Evaluate and memoize the **square** of a column/expression.

        Uses :meth:`col` for the base array and returns ``np.square(base, dtype=float)``.

        Parameters
        ----------
        name : str
            Column name or expression string.

        Returns
        -------
        numpy.ndarray
            A 1D float NumPy array containing the elementwise squares.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> df = pd.DataFrame({"y": [2.0, np.nan, -3.0]})
        >>> ec = _EvalCache(df)
        >>> ec.sq_col("y")
        array([ 4., nan,  9.])
        """
        if name not in self._sq:
            self._sq[name] = np.square(self.col(name), dtype=float)
        return self._sq[name]
