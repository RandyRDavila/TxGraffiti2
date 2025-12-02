# src/txgraffiti2025/graffiti/core.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..forms.utils import Expr, ColumnTerm, to_expr
from ..forms.predicates import Predicate, Where
from ..forms.generic_conjecture import TRUE

__all__ = ["DataModel", "MaskCache"]

# ── tiny local helper (NA→False, aligned) ─────────────────────────────────────
def _as_bool_series_local(arr, index: pd.Index) -> pd.Series:
    if isinstance(arr, pd.Series):
        return arr.reindex(index).fillna(False).astype(bool, copy=False)
    if isinstance(arr, (bool, np.bool_)):
        return pd.Series(bool(arr), index=index, dtype=bool)
    a = np.asarray(arr)
    if a.ndim != 1 or len(a) != len(index):
        raise ValueError("Mask must be 1D and match DataFrame length.")
    return pd.Series(a.astype(bool, copy=False), index=index, dtype=bool)

def _is_boolean_like(s: pd.Series) -> bool:
    if is_bool_dtype(s) or str(s.dtype).lower().startswith("boolean"):
        return True
    if is_numeric_dtype(s):
        vals = s.dropna()
        if len(vals) == 0:
            return False
        if len(vals) > 128:  # small, stable sample to keep O(1)
            vals = vals.sample(128, random_state=0)
        return bool(vals.isin((0, 1, 0.0, 1.0)).all())
    return False

# ── minimal, mutable base for conjecturing ────────────────────────────────────
@dataclass(slots=True)
class DataModel:
    """
    Minimal, mutable base:
      - invariants: List[Expr]
      - booleans  : List[Predicate]

    Names:
      - invariants are looked up by repr(expr) and, for ColumnTerm, by raw column name
      - booleans are looked up by their .name (bare column or custom)
    """
    df: pd.DataFrame
    invariants: List[Expr] = field(default_factory=list)
    booleans:   List[Predicate] = field(default_factory=list)

    # name → index caches
    _inv_idx: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _bool_idx: Dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError("DataModel requires a pandas DataFrame.")
        self.rescan()

    # ── core mutations ───────────────────────────────────────────────────────
    def set_df(self, df: pd.DataFrame, *, rescan: bool = True) -> None:
        self.df = df
        if rescan:
            self.rescan()

    def rescan(self) -> None:
        invs: List[Expr] = []
        bools: List[Predicate] = []
        for c in self.df.columns:
            s = self.df[c]
            if _is_boolean_like(s):
                bools.append(Where(lambda d, col=c: d[col], name=c))
            elif is_numeric_dtype(s):
                invs.append(ColumnTerm(c))
        self.invariants = invs
        self.booleans   = bools
        self._reindex()

    def _reindex(self) -> None:
        inv_idx: Dict[str, int] = {}
        for i, e in enumerate(self.invariants):
            inv_idx[repr(e)] = i
            if isinstance(e, ColumnTerm):
                inv_idx[e.col] = i
        self._inv_idx = inv_idx
        self._bool_idx = {p.name: i for i, p in enumerate(self.booleans)}

    # ── minimal add/drop ─────────────────────────────────────────────────────
    def add_invariant(self, x: Union[str, Expr]) -> str:
        if isinstance(x, str):
            if x not in self.df.columns:
                raise KeyError(f"Column {x!r} not in df.")
            e = ColumnTerm(x)
        else:
            e = to_expr(x)
        key = repr(e)
        if key in self._inv_idx:
            return key
        self.invariants.append(e)
        self._reindex()
        return key

    def add_boolean(self, x: Union[str, Predicate]) -> str:
        if isinstance(x, str):
            if x not in self.df.columns:
                raise KeyError(f"Column {x!r} not in df.")
            p = Where(lambda d, c=x: d[c], name=x)
        else:
            p = x
            if not getattr(p, "name", ""):
                p.name = f"pred_{len(self.booleans)}"
        if p.name in self._bool_idx:
            return p.name
        self.booleans.append(p)
        self._reindex()
        return p.name

    def drop_symbol(self, name: str) -> None:
        if name in self._bool_idx:
            del self.booleans[self._bool_idx[name]]
            self._reindex()
            return
        if name in self._inv_idx:
            del self.invariants[self._inv_idx[name]]
            self._reindex()
            return
        raise KeyError(f"No symbol named {name!r}")

    # ── lookups & names ──────────────────────────────────────────────────────
    def to_expr(self, x) -> Expr:
        return self.invariant(x) if isinstance(x, str) and x in self._inv_idx else to_expr(x)

    @property
    def invariant_names(self) -> List[str]:
        return [repr(e) for e in self.invariants]

    @property
    def boolean_names(self) -> List[str]:
        return [p.name for p in self.booleans]

    def invariant(self, name: str) -> Expr:
        try:
            return self.invariants[self._inv_idx[name]]
        except KeyError as e:
            raise KeyError(f"Unknown invariant: {name!r}") from e

    def boolean(self, name: str) -> Predicate:
        try:
            return self.booleans[self._bool_idx[name]]
        except KeyError as e:
            raise KeyError(f"Unknown boolean: {name!r}") from e

    # ── tiny conveniences used downstream ────────────────────────────────────
    @property
    def base_true(self) -> Predicate:
        return TRUE

    def auto_base_from_always_true(self) -> Predicate:
        cols: List[str] = []
        for name in self.boolean_names:
            m = self.boolean(name).mask(self.df)
            if bool(_as_bool_series_local(m, self.df.index).all()):
                cols.append(name)
        if not cols:
            return TRUE
        pred = Where(lambda d, c=cols[0]: d[c], name=cols[0])
        for c in cols[1:]:
            pred = pred & Where(lambda d, c=c: d[c], name=c)
        pred.name = " ∧ ".join(cols)
        return pred

    def support(self, pred: Predicate) -> int:
        return int(_as_bool_series_local(pred.mask(self.df), self.df.index).sum())

# ── tiny, index-aware mask cache ─────────────────────────────────────────────
@dataclass(slots=True)
class MaskCache:
    model: DataModel
    _cache: Dict[int, pd.Series] = field(default_factory=dict)
    _index_id: int = field(init=False)

    def __post_init__(self):
        self._index_id = id(self.model.df.index)

    def _ensure_fresh_index(self):
        if id(self.model.df.index) != self._index_id:
            self._cache.clear()
            self._index_id = id(self.model.df.index)

    def _key(self, pred: Predicate) -> int:
        try:
            return hash(pred)
        except Exception:
            return id(pred)

    def mask(self, pred: Predicate) -> pd.Series:
        self._ensure_fresh_index()
        k = self._key(pred)
        if k in self._cache:
            return self._cache[k]
        m = _as_bool_series_local(pred.mask(self.model.df), self.model.df.index)
        self._cache[k] = m
        return m

    def clear(self) -> None:
        self._cache.clear()
