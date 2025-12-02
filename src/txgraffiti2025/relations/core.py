# src/txgraffiti2025/relations/core.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_categorical_dtype

from ..forms.utils import Expr, ColumnTerm
from ..forms.predicates import Predicate, Where
from ..forms.generic_conjecture import TRUE

__all__ = [
    "DataModel",
    "MaskCache",
]

# ──────────────────────────────────────────────────────────────────────────────
# Local helpers
# ──────────────────────────────────────────────────────────────────────────────

def _as_bool_series_local(arr, index: pd.Index) -> pd.Series:
    """
    Normalize any array-like/Series/bool to a boolean Series aligned to `index`.
    NA -> False. Mirrors predicates._as_bool_series semantics.
    """
    if isinstance(arr, pd.Series):
        s = arr.reindex(index)
        # Nullable BooleanDtype or bool -> fast path
        if is_bool_dtype(s) or str(s.dtype).lower().startswith("boolean"):
            return s.fillna(False).astype(bool, copy=False)
        return s.fillna(False).astype(bool, copy=False)

    if isinstance(arr, (bool, np.bool_)):
        return pd.Series(bool(arr), index=index, dtype=bool)

    a = np.asarray(arr)
    if a.ndim != 1 or len(a) != len(index):
        raise ValueError("Predicate mask must be 1D and match DataFrame length.")
    if a.dtype != bool:
        a = a.astype(bool, copy=False)
    # Construct with index; no NA possible in pure bool, but keep fillna for parity
    return pd.Series(a, index=index, dtype=bool).fillna(False)


def _is_boolean_like(s: pd.Series) -> bool:
    """
    Return True if `s` behaves like a boolean indicator column.

    Accepts:
    • bool dtype or pandas nullable 'boolean' dtype
    • integer columns with only {0,1} (ignoring NA)
    • float columns with only {0.0,1.0} (ignoring NA)
    • categorical columns whose categories are subset of {0,1,True,False}
    """
    try:
        if is_bool_dtype(s) or str(s.dtype).lower().startswith("boolean"):
            return True

        if is_categorical_dtype(s.dtype):
            cats = pd.Series(s.cat.categories, dtype="object")
            return bool(cats.isin([0, 1, True, False]).all())

        vals = s.dropna()
        if vals.empty:
            return False

        if pd.api.types.is_integer_dtype(vals):
            return bool(vals.isin([0, 1]).all())

        if pd.api.types.is_float_dtype(vals):
            return bool(vals.isin([0.0, 1.0]).all())

        return False
    except Exception:
        return False


def _is_numeric_like(s: pd.Series) -> bool:
    """
    Return True if `s` can reasonably participate in numeric Expr arithmetic.
    Strategy: allow numeric dtypes or strings that coerce to numeric with finite support.
    """
    if pd.api.types.is_numeric_dtype(s):
        return True
    if s.dtype == "object":
        # Try a small sample to avoid O(n) coercion on huge columns
        sample = s.dropna().astype(str).head(64)
        if sample.empty:
            return False
        try:
            coerced = pd.to_numeric(sample, errors="coerce")
            return bool(coerced.notna().mean() > 0.75)  # mostly numeric strings
        except Exception:
            return False
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Core data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DataModel:
    """
    Immutable view of a dataset used by the relations pipeline.

    Responsibilities
    ----------------
    • Owns the source DataFrame `df`.
    • Partitions columns into boolean-like, numeric-like, and "other"/text.
    • Wraps numeric-like columns as Expr (ColumnTerm).
    • Wraps boolean-like columns as Predicate (truthy-only).
    • Provides a simple mined-artifact registry (append-only lists).
    • Exposes helpers for supports and an auto base predicate.

    Registry schema (append dicts with the following keys)
    ------------------------------------------------------
    registry["absdiff"]    : {inv1, inv2, expr_name, hypothesis, support, stats?...}
    registry["minmax"]     : {inv1, inv2, key_min, key_max, hypothesis, support, stats?...}
    registry["constants"]  : {inv, value, hypothesis, support, tol}
    registry["pairs"]      : {inv1, inv2, hypothesis, support, rate_eq, tol}
    registry["conjectures"]: {kind, payload, hypothesis, support, meta?...}
    """

    df: pd.DataFrame
    boolean_cols: List[str]
    numeric_cols: List[str]
    text_cols: List[str]
    exprs: Dict[str, Expr]
    preds: Dict[str, Predicate]
    invariants: List[Expr]
    booleans: List[Predicate]
    registry: Dict[str, List[Dict[str, Any]]]

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataModel requires a pandas DataFrame.")
        object.__setattr__(self, "df", df)

        bool_cols: List[str] = []
        num_cols: List[str] = []
        txt_cols: List[str] = []

        # Preserve original order
        for c in df.columns:
            s = df[c]
            if _is_boolean_like(s):
                bool_cols.append(c)
            elif _is_numeric_like(s):
                num_cols.append(c)
            else:
                txt_cols.append(c)

        # Wraps: numeric-like -> Expr, boolean-like -> Predicate
        exprs: Dict[str, Expr] = {c: ColumnTerm(c) for c in num_cols}
        invariants: List[Expr] = [ColumnTerm(c) for c in num_cols]

        preds: Dict[str, Predicate] = {c: Predicate.from_column(c, truthy_only=True) for c in bool_cols}
        booleans: List[Predicate] = [Predicate.from_column(c, truthy_only=True) for c in bool_cols]

        object.__setattr__(self, "boolean_cols", bool_cols)
        object.__setattr__(self, "numeric_cols", num_cols)
        object.__setattr__(self, "text_cols", txt_cols)
        object.__setattr__(self, "exprs", exprs)
        object.__setattr__(self, "preds", preds)

        object.__setattr__(self, "invariants", invariants)
        object.__setattr__(self, "booleans", booleans)

        # Append-only mined artifacts registry
        object.__setattr__(self, "registry", {
            "absdiff": [],
            "minmax": [],
            "constants": [],
            "pairs": [],
            "conjectures": [],
            "classes": [],
        })

    def record_class(self, **payload: Any) -> None:
        """
        Append a hypothesis-class record. Recommended keys:
          name, arity, support, extras, base_parts, mask_sig (bytes), pred
        """
        self.registry["classes"].append(dict(payload))

    # Convenience accessors
    def expr(self, name: str) -> Expr:
        try:
            return self.exprs[name]
        except KeyError as e:
            raise KeyError(f"Unknown numeric-like column for Expr: {name!r}") from e

    def pred(self, name: str) -> Predicate:
        try:
            return self.preds[name]
        except KeyError as e:
            raise KeyError(f"Unknown boolean-like column for Predicate: {name!r}") from e

    @property
    def base_true(self) -> Predicate:
        """The universal TRUE predicate."""
        return TRUE

    def auto_base_from_always_true(self) -> Predicate:
        """
        Build the conjunction of all boolean-like columns that are always True
        (treat NA as False). If none, return TRUE.
        """
        always_true_cols: List[str] = []
        for c in self.boolean_cols:
            s = self.df[c]
            # Use the same semantics as predicates.Where + truthy_only=True
            m = Predicate.from_column(c, truthy_only=True).mask(self.df)
            if bool(m.fillna(False).all()):
                always_true_cols.append(c)

        if not always_true_cols:
            return TRUE

        preds = [Where(lambda d, col=c: d[col], name=f"{c}") for c in always_true_cols]
        p = preds[0]
        for q in preds[1:]:
            p = (p & q)
        # Friendly compound name
        p.name = " ∧ ".join(always_true_cols)
        return p

    def iter_numeric(self) -> Iterable[Tuple[str, Expr]]:
        yield from self.exprs.items()

    def iter_boolean(self) -> Iterable[Tuple[str, Predicate]]:
        yield from self.preds.items()

    # Registry helpers (append-only; safe on frozen dataclass via internal mutability)
    def record_absdiff(self, **payload: Any) -> None:
        self.registry["absdiff"].append(dict(payload))

    def record_minmax(self, **payload: Any) -> None:
        self.registry["minmax"].append(dict(payload))

    def record_constant(self, **payload: Any) -> None:
        self.registry["constants"].append(dict(payload))

    def record_pair_equality(self, **payload: Any) -> None:
        self.registry["pairs"].append(dict(payload))

    def record_conjecture(self, **payload: Any) -> None:
        self.registry["conjectures"].append(dict(payload))

    def clear_registry(self) -> None:
        for k in self.registry.keys():
            self.registry[k].clear()

    def registry_frame(self, kind: str) -> pd.DataFrame:
        rows = self.registry.get(kind, [])
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def summary(self) -> Dict[str, Any]:
        """Small dict summary useful in logs/demos."""
        return {
            "rows": int(len(self.df)),
            "numeric_cols": list(self.numeric_cols),
            "boolean_cols": list(self.boolean_cols),
            "text_cols": list(self.text_cols),
            "registry_counts": {k: len(v) for k, v in self.registry.items()},
        }

    def support(self, predicate: Predicate) -> int:
        """Return number of rows where `predicate` holds."""
        return int(_as_bool_series_local(predicate.mask(self.df), self.df.index).sum())


@dataclass(slots=True)
class MaskCache:
    """
    Small per-run cache for predicate masks aligned to `model.df.index`.

    Notes
    -----
    • Keys use Predicate.__hash__ (structural via cache_key()).
    • If the DataFrame's index object changes (not just values), the cache is cleared.
      This cheaply protects against common alignment errors when callers rebuild df.
    """
    model: DataModel
    _cache: Dict[int, pd.Series] = field(default_factory=dict)
    _index_id: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "_index_id", id(self.model.df.index))

    def _ensure_fresh_index(self):
        idx_id = id(self.model.df.index)
        if idx_id != self._index_id:
            self._cache.clear()
            object.__setattr__(self, "_index_id", idx_id)

    def _key(self, pred: Predicate) -> int:
        try:
            return hash(pred)
        except Exception:
            return id(pred)

    def mask(self, pred: Predicate) -> pd.Series:
        self._ensure_fresh_index()
        kid = self._key(pred)
        m = self._cache.get(kid)
        if m is not None:
            return m
        raw = pred.mask(self.model.df)   # may be Series/ndarray/list/bool
        m = _as_bool_series_local(raw, self.model.df.index)
        self._cache[kid] = m
        return m

    def clear(self) -> None:
        self._cache.clear()
