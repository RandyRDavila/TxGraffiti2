from __future__ import annotations
from typing import List, Sequence, Optional
import pandas as pd

from .engine import WorkbenchEngine
from .config import GenerationConfig
from .conj_single_feature import generate_single_feature_bounds
from .conj_mixed_bounds import generate_mixed_bounds
from .conj_targeted_products import generate_targeted_product_bounds
from .ranking import rank_and_filter
from .class_relations import discover_class_relations

class TxGraffitiMini:
    def _bool_column_predicates(self):
        from txgraffiti2025.forms.predicates import Where
        preds = []
        for col in self.df.columns:
            ser = self.df[col]
            try:
                vals = set(ser.dropna().unique().tolist())
            except Exception:
                continue
            is_boolish = (ser.dtype == bool) or vals.issubset({0,1,True,False})
            if is_boolish:
                preds.append(Where(fn=lambda df, c=col: (df[c] == 1) | (df[c] == True), name=f"({col})"))
        return preds

    """Thin, convenient façade around WorkbenchEngine + free functions."""
    def __init__(self, df: pd.DataFrame, config: Optional[GenerationConfig] = None):
        self.df = df
        self.config = config or GenerationConfig()
        try:
            self.engine = WorkbenchEngine(df, self.config)
        except TypeError:
            # Fallback for engines expecting only df
            self.engine = WorkbenchEngine(df)

    # ---- Phase 1: single-feature bounds
    def generate_single_feature_bounds(self, target: str):
        numeric_columns = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
        hyps = getattr(self.engine, "hyps_kept", [])
        return generate_single_feature_bounds(
            self.df, target, hyps=hyps, numeric_columns=numeric_columns, config=self.config
        )

    # ---- Phase 2: mixed (two-feature) bounds
    def run_mixed_pipeline(self, target: str, primary: Optional[Sequence[str]] = None,
                           secondary: Optional[Sequence[str]] = None):
        if primary is None:
            primary = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c]) and c != target]
        if secondary is None:
            secondary = primary
        hyps = getattr(self.engine, "hyps_kept", [])
        return generate_mixed_bounds(
            self.df, target, hyps=hyps, primary=list(primary), secondary=list(secondary), config=self.config
        )

    # ---- Targeted product bounds
    def run_targeted_product_pipeline(self, target: str, **kwargs):
        x_cands = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c]) and c != target]
        yz_cands = x_cands
        hyps = getattr(self.engine, "hyps_kept", [])
        return generate_targeted_product_bounds(
            self.df, target, hyps=hyps, x_candidates=x_cands, yz_candidates=yz_cands, **kwargs
        )

    # ---- Ranking helper (touch-based)
    def rank_and_filter(self, conjs, min_touch: int = None):
        if min_touch is None:
            min_touch = self.config.min_touch_keep
        return rank_and_filter(conjs, self.df, min_touch=min_touch)

    # ---- Derive predicates from conjectures (delegate to engine implementation if present)
    def add_derived_predicates_from_top_conjectures(self, conjs, **kwargs):
        if hasattr(self.engine, "add_derived_predicates_from_top_conjectures"):
            return self.engine.add_derived_predicates_from_top_conjectures(conjs, **kwargs)
        # Fallback: return empty if engine doesn’t provide it yet
        return []

    # ---- Discover class relations
    def discover_class_relations(self, predicates=None, include_bool_columns=False, **kwargs):
        if predicates is None:
            predicates = getattr(self.engine, "hyps_kept", [])
        return discover_class_relations(self.df, predicates=predicates, **kwargs)

    # ---- Pretty helpers (minimal, safe)
    @staticmethod
    def pretty_block(title: str, conjs: Sequence, max_items: int = 10):
        print(f"\n=== {title} ===")
        for i, c in enumerate(conjs[:max_items], 1):
            try:
                print(f"{i:3d}. {c.pretty()}")
            except Exception:
                print(f"{i:3d}. {c}")
