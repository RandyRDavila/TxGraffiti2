# src/txgraffiti2025/graffiti_base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import reduce
import operator

import numpy as np
import pandas as pd

# --- New DSL imports only ---
from .graffiti_utils import Expr, ColumnTerm
from .graffiti_predicates import Predicate, Where
from .graffiti_generic_conjecture import (
    TRUE, Conjecture, AllOf, BoolFormula, coerce_formula,
)

__all__ = ["GraffitiBase"]


# Optional utility: a predicate backed by a fixed mask (rarely needed)
@dataclass(frozen=True)
class _MaskPredicate(Predicate):
    _m: pd.Series
    def mask(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._m, index=df.index).fillna(False).astype(bool, copy=False)
    def pretty(self) -> str:
        return "mask(...)"

class GraffitiBase:
    """
    Core container for dataset state and shared DSL utilities (new DSL only).

    • Partitions columns into boolean-like and numeric.
    • Wraps numeric columns as `Expr` and boolean-like as `Predicate`.
    • Detects the base hypothesis (conjunction of all universally true boolean columns).
    • Caches predicate/formula masks keyed by the current df identity.

    Notes
    -----
    - Boolean-like columns are: pandas bool/nullable-boolean OR numeric {0,1}.
    - If you mutate the SAME DataFrame in place, call `clear_cache()`; otherwise
      pass a new DataFrame to `refresh(df)` to auto-reset caches.
    """

    # ───────── lifecycle ───────── #

    def __init__(self, df: pd.DataFrame):
        self.refresh(df)

    def refresh(self, df: Optional[pd.DataFrame] = None) -> None:
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")
            self.df = df
        elif not hasattr(self, "df"):
            raise ValueError("refresh(None) is only valid after initialization.")

        # Reset caches bound to this df identity
        self._mask_cache_version: int = id(self.df)
        self._mask_cache: Dict[int, np.ndarray] = {}
        # Memo of formula→Predicate (Where) so mask_of_formula hits cache keys stably
        self._formula_pred_cache: Dict[str, Predicate] = {}

        # Partitions
        self.boolean_cols: List[str] = [
            c for c in self.df.columns if self._is_boolean_like(self.df[c])
        ]
        self.expr_cols: List[str] = [c for c in self.df.columns if c not in self.boolean_cols]

        # Wrappers (new DSL)
        self.exprs: Dict[str, Expr] = {c: ColumnTerm(c) for c in self.expr_cols}
        self.predicates: Dict[str, Predicate] = {
            c: Predicate.from_column(c, truthy_only=True) for c in self.boolean_cols
        }

        # Base hypothesis = ∧ of universally-true boolean columns, else TRUE
        ats = self._always_true_boolean_cols()
        if ats:
            ats.sort()
            preds = [self.predicates[c] for c in ats]
            if len(preds) == 1:
                self.base_hypothesis: Predicate = preds[0]
            else:
                self.base_hypothesis = reduce(operator.and_, preds)
            self.base_hypothesis_name: str = " ∧ ".join(ats)
        else:
            self.base_hypothesis = TRUE
            self.base_hypothesis_name = "TRUE"

        # Non-base predicates
        self.base_predicates: Dict[str, Predicate] = {
            k: v for k, v in self.predicates.items() if k not in ats
        }

        # Hooks for higher layers (optional bookkeeping)
        self.synthetic_expr_names_: set[str] = set()
        self.abs_exprs: List[Tuple[str, Expr]] = []

    # ───────── lightweight API ───────── #

    def get_expr_columns(self) -> List[str]:
        return list(self.expr_cols)

    def get_boolean_columns(self) -> List[str]:
        return list(self.boolean_cols)

    def get_base_predicates(self) -> Dict[str, Predicate]:
        return dict(self.base_predicates)

    def expr(self, name: str) -> Expr:
        return self.exprs[name]

    # By design, .pred() exposes only *non-base* preds (as in your original).
    def pred(self, name: str) -> Predicate:
        return self.base_predicates[name]

    # Convenience that accesses ANY predicate (base or not)
    def pred_any(self, name: str) -> Predicate:
        return self.predicates[name]

    def mask(self, predicate: Predicate) -> np.ndarray:
        return self._mask_cached(predicate)

    def clear_cache(self) -> None:
        self._mask_cache_clear()
        self._formula_pred_cache.clear()

    # ───────── utilities ───────── #

    @staticmethod
    def _is_boolean_like(s: pd.Series) -> bool:
        # pandas bool or nullable BooleanDtype
        if pd.api.types.is_bool_dtype(s) or str(s.dtype).lower().startswith("boolean"):
            return True
        # numeric 0/1 style
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            nonna = s.dropna()
            if nonna.empty:
                return False
            vals = pd.unique(nonna.astype(float))
            return bool(len(vals) <= 2 and set(vals).issubset({0.0, 1.0}))
        return False

    def _always_true_boolean_cols(self) -> List[str]:
        out: List[str] = []
        for c, pred in self.predicates.items():
            if self._mask_cached(pred).all():
                out.append(c)
        return out

    def _mask_cache_clear(self) -> None:
        self._mask_cache = {}

    def _mask_cached(self, pred: Predicate) -> np.ndarray:
        # If df identity changed, clear the cache
        if self._mask_cache_version != id(self.df):
            self._mask_cache_version = id(self.df)
            self._mask_cache_clear()
            self._formula_pred_cache.clear()

        key = id(pred)
        m = self._mask_cache.get(key)
        if m is None:
            m = pred.mask(self.df).to_numpy(dtype=np.bool_, copy=False)
            self._mask_cache[key] = m
        return m

    # ───────── formulas & masking (new DSL only) ───────── #

    def _formula_signature(self, f: Union[BoolFormula, Predicate]) -> str:
        """Stable string signature used to memoize formula→Predicate."""
        bf: BoolFormula = coerce_formula(f)
        pretty_fn = getattr(bf, "pretty", None)
        return (pretty_fn() if callable(pretty_fn) else None) or repr(bf)

    def formula_to_predicate(self, f: Union[BoolFormula, Predicate]) -> Predicate:
        """Wrap a BoolFormula/Predicate as a memoized `Predicate` (via Where) for caching."""
        sig = self._formula_signature(f)
        cached = self._formula_pred_cache.get(sig)
        if cached is not None:
            return cached

        bf: BoolFormula = coerce_formula(f)

        def _eval_mask(df: pd.DataFrame) -> pd.Series:
            m = bf.evaluate(df)
            if isinstance(m, pd.Series):
                s = m.reindex(df.index)
            else:
                s = pd.Series(m, index=df.index)
            return s.fillna(False).astype(bool, copy=False)

        pred = Where(_eval_mask, name=sig)
        self._formula_pred_cache[sig] = pred
        return pred

    def mask_of_formula(self, f: Union[BoolFormula, Predicate]) -> np.ndarray:
        """Evaluate a BoolFormula/Predicate to a boolean mask using the cache."""
        return self._mask_cached(self.formula_to_predicate(f))

    # If older *call sites* use this name, keep the alias (still new DSL underneath).
    def mask_of_relation(self, f: Union[BoolFormula, Predicate]) -> np.ndarray:
        return self.mask_of_formula(f)

    # ───────── conjecture utilities ───────── #

    def compress_conjectures(self, conjs: List[Conjecture]) -> List[Conjecture]:
        """
        Merge conjectures by identical conditions and deduplicate relations/formulas
        by their printed signature.
        """
        if not conjs:
            return []

        groups: Dict[str, List[BoolFormula]] = {}
        cond_objs: Dict[str, Optional[BoolFormula]] = {}

        for cj in conjs:
            cond = cj.condition
            key = repr(cond) if cond is not None else "TRUE"
            groups.setdefault(key, []).append(cj.relation)
            cond_objs.setdefault(key, cond)

        out: List[Conjecture] = []
        for key, rels in groups.items():
            cond = cond_objs[key]
            unique: List[BoolFormula] = []
            seen = set()
            for r in rels:
                sig = (getattr(r, "pretty", None) and r.pretty()) or repr(r)
                if sig not in seen:
                    seen.add(sig)
                    unique.append(r)
            if len(unique) == 1:
                out.append(Conjecture(relation=unique[0], condition=cond,
                                      name=f"{key} | {unique[0].__class__.__name__}"))
            else:
                out.append(Conjecture(relation=AllOf(unique), condition=cond, name=f"Const[{key}]"))
        return out

    # ───────── summary ───────── #

    def _describe_df_column_types(self) -> pd.DataFrame:
        info = []
        for col in self.df.columns:
            s = self.df[col]
            info.append(dict(
                column=col,
                dtype=str(s.dtype),
                nonnull=int(s.notna().sum()),
                na=int(s.isna().sum()),
                distinct=int(s.nunique(dropna=True)),
                boolean_like=self._is_boolean_like(s),
            ))
        return pd.DataFrame(info)

    def summary(self, verbose: bool = True) -> str:
        lines: list[str] = []
        lines.append("──────────────────────────────────────────────")
        lines.append("GraffitiBase • Initialization Summary")
        lines.append("──────────────────────────────────────────────")
        lines.append(f"DataFrame shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        lines.append("")
        lines.append(f"Boolean-like columns ({len(self.boolean_cols)}):")
        lines.append("  " + (", ".join(self.boolean_cols) if self.boolean_cols else "(none)"))
        lines.append(f"Numeric columns ({len(self.expr_cols)}):")
        lines.append("  " + (", ".join(self.expr_cols) if self.expr_cols else "(none)"))
        lines.append("")
        lines.append("Base hypothesis:")
        lines.append(f"  Name: {self.base_hypothesis_name}")
        lines.append(f"  Type: {type(self.base_hypothesis).__name__}")
        lines.append("")
        lines.append(f"Predicates built: {len(self.predicates)} total")
        lines.append(f"  Non-base predicates: {len(self.base_predicates)}")
        lines.append(f"Exprs built: {len(self.exprs)}")
        lines.append("")
        lines.append("Mask cache:")
        lines.append(f"  Cached entries: {len(self._mask_cache)}")
        lines.append(f"  Cache bound to df id: {self._mask_cache_version}")
        lines.append("")
        if self.synthetic_expr_names_:
            lines.append(f"Synthesized expressions ({len(self.synthetic_expr_names_)}):")
            lines.append("  " + ", ".join(sorted(self.synthetic_expr_names_)))
        else:
            lines.append("Synthesized expressions: (none)")
        lines.append("──────────────────────────────────────────────")
        s = "\n".join(lines)
        if verbose:
            print(s)
        return s
