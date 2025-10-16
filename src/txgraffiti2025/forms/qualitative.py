# src/txgraffiti2025/forms/qualitative.py

"""
Qualitative / monotone relations (R6) over a dataset.

Checks whether a response column `y` tends to increase or decrease with a
predictor column `x`, optionally restricted to a class mask `C`. Two
correlation methods are supported:

- ``"spearman"``  (rank correlation; robust to monotone nonlinear trends)
- ``"pearson"``   (linear correlation)

NaNs and constant sequences are handled gracefully; undefined correlations are
treated as 0.0. You can also require a minimum absolute correlation magnitude
via ``min_abs_rho``.

Examples
--------
Basic usage:

>>> import pandas as pd
>>> from txgraffiti2025.forms.qualitative import MonotoneRelation
>>> df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 5, 8]})
>>> rel = MonotoneRelation("x", "y", direction="increasing", method="spearman", min_abs_rho=0.5)
>>> rel.evaluate_global(df)["ok"]
True

Restrict to a class mask (e.g., only rows where `flag` is True):

>>> from txgraffiti2025.forms.predicates import Predicate
>>> df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [8, 6, 4, 2], "flag": [True, False, True, True]})
>>> mask = Predicate.from_column("flag").mask(df)
>>> MonotoneRelation("x", "y", direction="decreasing").evaluate_global(df, mask=mask)["ok"]
True
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any
import numpy as np
import pandas as pd

Direction = Literal["increasing", "decreasing"]
Method = Literal["spearman", "pearson"]


@dataclass
class MonotoneRelation:
    """
    Test whether ``y`` is (weakly) monotone in ``x`` over a DataFrame.

    Parameters
    ----------
    x : str
        Name of the predictor column.
    y : str
        Name of the response column.
    direction : {"increasing", "decreasing"}, default "increasing"
        Expected monotone tendency.
    method : {"spearman", "pearson"}, default "spearman"
        Correlation to use: rank-based (Spearman) or linear (Pearson).
    min_abs_rho : float, default 0.0
        Minimum absolute correlation magnitude required to declare success.
    name : str, default "Monotone"
        Display name.

    Attributes
    ----------
    x : str
    y : str
    direction : {"increasing", "decreasing"}
    method : {"spearman", "pearson"}
    min_abs_rho : float
    name : str

    Notes
    -----
    - Constant or NaN-only inputs yield a correlation of ``0.0`` by design.
    - Spearman uses average ties via pandas ranking to avoid spurious ±1.
    - This implements an **R6 qualitative form** (trend-style statement),
      complementing the algebraic R2–R5 modules.

    Examples
    --------
    Increasing trend (Spearman):

    >>> import pandas as pd
    >>> from txgraffiti2025.forms.qualitative import MonotoneRelation
    >>> df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 1.5, 3]})
    >>> MonotoneRelation("x", "y", "increasing", "spearman", 0.7).evaluate_global(df)["ok"]
    True

    Non-monotone case:

    >>> df2 = pd.DataFrame({"x":[0,1,2,3], "y":[0,2,1,3]})
    >>> MonotoneRelation("x","y","increasing","spearman",0.8).evaluate_global(df2)["ok"]
    False
    """
    x: str
    y: str
    direction: Direction = "increasing"   # or "decreasing"
    method: Method = "spearman"           # 'spearman' | 'pearson'
    min_abs_rho: float = 0.0              # require at least this magnitude
    name: str = "Monotone"

    def _corr(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """
        Compute correlation per selected method, returning a finite float.

        Parameters
        ----------
        xs, ys : np.ndarray
            Numeric vectors (aligned, NaNs removed upstream).

        Returns
        -------
        float
            Correlation coefficient. Returns ``0.0`` when undefined.

        Notes
        -----
        - Spearman is computed by ranking with average ties, then Pearson on ranks.
        - For either method, if a side is constant or the result is non-finite,
          the value is coerced to ``0.0``.
        """
        if self.method == "spearman":
            # Tied ranks via pandas (average method) to avoid spurious ±1 on constants
            rx = pd.Series(xs).rank(method="average").to_numpy()
            ry = pd.Series(ys).rank(method="average").to_numpy()
            # If either side is constant after ranking, correlation undefined -> 0.0
            if np.std(rx) == 0.0 or np.std(ry) == 0.0:
                return 0.0
            rho = float(np.corrcoef(rx, ry)[0, 1])
        else:  # pearson
            # If either side is constant, Pearson correlation undefined -> 0.0
            if np.std(xs) == 0.0 or np.std(ys) == 0.0:
                return 0.0
            rho = float(np.corrcoef(xs, ys)[0, 1])

        if not np.isfinite(rho):
            rho = 0.0
        return rho

    def evaluate_global(
        self,
        df: pd.DataFrame,
        mask: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the monotonic tendency of ``y`` vs ``x`` on the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing columns `x` and `y`. Non-numeric entries are
            coerced via ``pd.to_numeric(..., errors="coerce")``.
        mask : pd.Series of bool, optional
            Optional class restriction aligned to ``df.index``. If provided,
            rows with ``mask == True`` are used; others are ignored.

        Returns
        -------
        dict
            Dictionary with keys:
                - ``ok`` : bool
                - ``rho`` : float (correlation used)
                - ``direction`` : {"increasing", "decreasing"}
                - ``method`` : {"spearman", "pearson"}
                - ``n`` : int (number of valid rows after filtering & NaN drop)
                - ``x`` : str
                - ``y`` : str

        Notes
        -----
        Success criteria:
        - Directional check: sign(rho) ≥ 0 for ``"increasing"``, ≤ 0 for ``"decreasing"``.
        - Magnitude check: ``abs(rho) >= min_abs_rho``.

        Examples
        --------
        With a mask:

        >>> import pandas as pd
        >>> from txgraffiti2025.forms.predicates import Predicate
        >>> from txgraffiti2025.forms.qualitative import MonotoneRelation
        >>> df = pd.DataFrame({
        ...     "x": [1, 2, 3, 4, 5],
        ...     "y": [5, 4, 3, 2, 1],
        ...     "use": [True, True, False, True, True],
        ... })
        >>> m = Predicate.from_column("use").mask(df)
        >>> MonotoneRelation("x", "y", "decreasing", "spearman", 0.8).evaluate_global(df, mask=m)["ok"]
        True
        """
        if mask is not None:
            df = df.loc[mask.fillna(False)]

        # Pull vectors and drop rows with NaN in either column
        xs = pd.to_numeric(df[self.x], errors="coerce")
        ys = pd.to_numeric(df[self.y], errors="coerce")
        valid = xs.notna() & ys.notna()
        xs = xs[valid].to_numpy()
        ys = ys[valid].to_numpy()
        n = xs.size

        # Not enough data to judge monotonic trend
        if n < 2:
            return {
                "ok": False,
                "rho": 0.0,
                "direction": self.direction,
                "method": self.method,
                "n": int(n),
                "x": self.x,
                "y": self.y,
            }

        # Short-circuit: if either series is constant, correlation is undefined -> set to 0.0
        if pd.Series(xs).nunique(dropna=True) <= 1 or pd.Series(ys).nunique(dropna=True) <= 1:
            rho = 0.0
        else:
            rho = self._corr(xs, ys)

        # Directional check (+ for increasing, − for decreasing) plus magnitude threshold
        dir_ok = (rho >= 0.0) if self.direction == "increasing" else (rho <= 0.0)
        mag_ok = abs(rho) >= float(self.min_abs_rho)
        ok = bool(dir_ok and mag_ok)

        return {
            "ok": ok,
            "rho": float(rho),
            "direction": self.direction,
            "method": self.method,
            "n": int(n),
            "x": self.x,
            "y": self.y,
        }

    def __repr__(self) -> str:
        return (f"Monotone({self.x}→{self.y}, dir={self.direction}, "
                f"method={self.method}, min|rho|={self.min_abs_rho})")

