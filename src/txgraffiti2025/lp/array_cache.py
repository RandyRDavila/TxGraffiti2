from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import Expr


class ArrayCache:
    """
    Simple per-DataFrame cache for Expr.eval(...) → np.ndarray lookups.
    Uses repr(Expr) as the cache key to align with your existing usage.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self._cache: Dict[str, np.ndarray] = {}

    def arr(self, e: Expr) -> np.ndarray:
        key = repr(e)
        v = self._cache.get(key)
        if v is None:
            v = e.eval(self.df).to_numpy(dtype=float, copy=False)
            self._cache[key] = v
        return v

    def clear(self) -> None:
        self._cache.clear()
