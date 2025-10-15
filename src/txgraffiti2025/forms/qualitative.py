"""
Qualitative / monotone relations (R6) over the dataset.

Checks whether y tends to increase/decrease with x on the provided DataFrame,
optionally restricted to rows satisfying a predicate C.

Design choices:
- Supports 'spearman' (rank correlation) or 'pearson' (linear correlation).
- Handles NaNs and constant sequences gracefully.
- Optional minimum correlation magnitude (min_abs_rho) before declaring 'ok'.
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
    x: str
    y: str
    direction: Direction = "increasing"   # or "decreasing"
    method: Method = "spearman"           # 'spearman' | 'pearson'
    min_abs_rho: float = 0.0              # require at least this magnitude
    name: str = "Monotone"

    def _corr(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """Compute correlation per method, returning a finite float (NaN/constant -> 0.0)."""
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
        mask: Optional[pd.Series] = None,  # optional class restriction C
    ) -> Dict[str, Any]:
        """
        Evaluate monotonic tendency globally.

        Args:
            df: DataFrame containing columns x and y.
            mask: optional boolean Series (aligned to df.index). If provided,
                  restricts the evaluation to rows where mask is True.

        Returns:
            dict with keys:
                ok: bool  (does the observed correlation match the direction and threshold?)
                rho: float (correlation value used)
                direction: "increasing" | "decreasing"
                method: "spearman" | "pearson"
                n: int (number of valid rows after filtering)
                x: str, y: str (column names)
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
