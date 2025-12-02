# src/txgraffiti2025/forms/qualitative.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Union

import numpy as np
import pandas as pd

from .predicates import Predicate
# NEW: import the Relation base so we can adapt to DSL
from .generic_conjecture import Relation

__all__ = ["MonotoneRelation", "R6MonotoneRelation"]

Direction = Literal["increasing", "decreasing"]
Method = Literal["spearman", "pearson"]


def _coerce_mask(mask: Optional[Union[pd.Series, Predicate]], df: pd.DataFrame) -> Optional[pd.Series]:
    """Accept a Predicate or boolean Series; return aligned boolean Series or None."""
    if mask is None:
        return None
    if isinstance(mask, Predicate):
        m = mask.mask(df)
    else:
        m = pd.Series(mask, index=df.index)
    return m.reindex(df.index, fill_value=False).astype(bool, copy=False)


@dataclass
class MonotoneRelation:
    """
    Test whether `y` is (weakly) monotone in `x` over a DataFrame.

    Parameters
    ----------
    x : str
        Name of the predictor column.
    y : str
        Name of the response column.
    direction : {"increasing", "decreasing"}, default "increasing"
        Expected monotone tendency (sign of correlation).
    method : {"spearman", "pearson"}, default "spearman"
        Correlation to use: rank-based (Spearman) or linear (Pearson).
    min_abs_rho : float, default 0.0
        Minimum absolute correlation magnitude required to declare success.
    min_n : int, default 2
        Minimum number of valid (x,y) pairs required to attempt correlation.
    name : str, default "Monotone"
        Display name.
    """
    x: str
    y: str
    direction: Direction = "increasing"
    method: Method = "spearman"
    min_abs_rho: float = 0.0
    min_n: int = 2
    name: str = "Monotone"

    # -----------------------------
    # Pretty / identity
    # -----------------------------
    def pretty(self, *, unicode_ops: bool = True, show_threshold: bool = True) -> str:
        up = "↑" if unicode_ops else "up"
        down = "↓" if unicode_ops else "down"
        arrow = up if self.direction == "increasing" else down

        if unicode_ops:
            rho_tag = "ρₛ" if self.method == "spearman" else "ρₚ"
            ge = "≥"
            abs_open, abs_close = "|", "|"
        else:
            rho_tag = "rho_s" if self.method == "spearman" else "rho_p"
            ge = ">="
            abs_open, abs_close = "|", "|"

        tail = f" ({rho_tag}"
        if show_threshold and float(self.min_abs_rho) > 0.0:
            tail += f", {abs_open}ρ{abs_close} {ge} {self.min_abs_rho:g}"
        tail += ")"
        return f"{self.y} {arrow} {self.x}{tail}"

    def signature(self) -> str:
        return self.pretty(unicode_ops=True, show_threshold=True)

    def __repr__(self) -> str:
        return (f"Monotone({self.x}→{self.y}, dir={self.direction}, "
                f"method={self.method}, min|rho|={self.min_abs_rho}, min_n={self.min_n})")

    # -----------------------------
    # Core correlation
    # -----------------------------
    def _corr(self, xs: np.ndarray, ys: np.ndarray) -> float:
        if self.method == "spearman":
            rx = pd.Series(xs).rank(method="average").to_numpy()
            ry = pd.Series(ys).rank(method="average").to_numpy()
            if np.std(rx) == 0.0 or np.std(ry) == 0.0:
                return 0.0
            rho = float(np.corrcoef(rx, ry)[0, 1])
        else:
            if np.std(xs) == 0.0 or np.std(ys) == 0.0:
                return 0.0
            rho = float(np.corrcoef(xs, ys)[0, 1])
        return rho if np.isfinite(rho) else 0.0

    # -----------------------------
    # Public evaluation API
    # -----------------------------
    def evaluate_global(
        self,
        df: pd.DataFrame,
        mask: Optional[Union[pd.Series, Predicate]] = None,
    ) -> Dict[str, Any]:
        m = _coerce_mask(mask, df)
        if m is not None:
            df = df.loc[m]

        xs = pd.to_numeric(df[self.x], errors="coerce")
        ys = pd.to_numeric(df[self.y], errors="coerce")
        valid = xs.notna() & ys.notna()
        xs = xs[valid].to_numpy()
        ys = ys[valid].to_numpy()
        n = xs.size

        if n < int(self.min_n):
            return {
                "ok": False, "rho": 0.0, "direction": self.direction, "method": self.method,
                "n": int(n), "x": self.x, "y": self.y,
            }

        if pd.Series(xs).nunique(dropna=True) <= 1 or pd.Series(ys).nunique(dropna=True) <= 1:
            rho = 0.0
        else:
            rho = self._corr(xs, ys)

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


# ─────────────────────────────────────────────────────────────────────
# NEW: Adapter so R6 qualitative behaves like a Relation in the DSL
# ─────────────────────────────────────────────────────────────────────

@dataclass
class R6MonotoneRelation(Relation):
    """
    Row-wise Relation adapter for qualitative monotonicity (R6).

    It evaluates the dataset-level monotone test and broadcasts the result:
      - evaluate(df): boolean Series
          True on rows inside `condition` if the monotone test passes;
          vacuously True outside `condition`.
      - slack(df): float Series
          On rows in `condition`:  signed_rho - min_abs_rho
              where signed_rho = rho  (increasing) or -rho (decreasing).
          Elsewhere: 0.0.

    Note: This makes R6 usable in Conjecture(...) and with GraffitiBase utilities.
    """
    mono: MonotoneRelation
    # Optional: keep a display name consistent with other Relations
    name: str = "R6Monotone"

    def _result(self, df: pd.DataFrame, condition: Optional[Predicate] = None) -> Dict[str, Any]:
        return self.mono.evaluate_global(df, mask=condition)

    def evaluate(self, df: pd.DataFrame, condition: Optional[Predicate] = None) -> pd.Series:
        # If a Conjecture wraps us, it supplies the condition; for standalone use, pass None.
        res = self._result(df, condition)
        if condition is None:
            applicable = pd.Series(True, index=df.index, dtype=bool)
        else:
            applicable = condition.mask(df).reindex(df.index, fill_value=False).astype(bool)
        ok = bool(res["ok"])
        out = pd.Series(True, index=df.index, dtype=bool)
        out.loc[applicable] = ok
        return out

    def slack(self, df: pd.DataFrame, condition: Optional[Predicate] = None) -> pd.Series:
        res = self._result(df, condition)
        rho = float(res["rho"])
        signed = rho if self.mono.direction == "increasing" else -rho
        margin = signed - float(self.mono.min_abs_rho)

        if condition is None:
            applicable = pd.Series(True, index=df.index, dtype=bool)
        else:
            applicable = condition.mask(df).reindex(df.index, fill_value=False).astype(bool)

        s = pd.Series(0.0, index=df.index, dtype=float)
        s.loc[applicable] = margin
        return s

    # Pretty mirrors the dataset-level description
    def pretty(self, *, unicode_ops: bool = True, show_tol: bool = False) -> str:
        return self.mono.pretty(unicode_ops=unicode_ops, show_threshold=True)

    def __repr__(self) -> str:
        return f"{self.pretty(unicode_ops=True)}"
