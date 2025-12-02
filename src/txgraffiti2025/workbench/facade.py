from __future__ import annotations

import numpy as np
import pandas as pd
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple

from .engine import WorkbenchEngine
from .config import GenerationConfig
from .conj_single_feature import generate_single_feature_bounds as _gen_single
from .conj_mixed_bounds import generate_mixed_bounds as _gen_mixed
from .conj_targeted_products import generate_targeted_product_bounds as _gen_products
from .ranking import rank_and_filter, touch_count
from .textparse import predicate_to_conjunction, predicate_to_if_then  # optional—handy printers

# Light imports for derived-predicate helpers
from txgraffiti2025.forms.predicates import Where, Predicate
from txgraffiti2025.forms.generic_conjecture import Conjecture, Le, Ge

def _mask_from_pred(df: pd.DataFrame, H) -> np.ndarray:
    if H is None:
        return np.ones(len(df), dtype=bool)
    if hasattr(H, "mask"):
        s = H.mask(df)
        return np.asarray(s, dtype=bool)
    return np.asarray(H(df), dtype=bool)

def _touch_mask(conj: Conjecture, df: pd.DataFrame, rtol=1e-8, atol=1e-8) -> np.ndarray:
    lhs = np.asarray(conj.relation.left.eval(df), dtype=float)
    rhs = np.asarray(conj.relation.right.eval(df), dtype=float)
    Hm  = _mask_from_pred(df, conj.condition)
    eq  = np.isclose(lhs, rhs, rtol=rtol, atol=atol)
    return Hm & eq

def _strict_mask(conj: Conjecture, df: pd.DataFrame, side: str, rtol=1e-8, atol=1e-8) -> np.ndarray:
    lhs = np.asarray(conj.relation.left.eval(df), dtype=float)
    rhs = np.asarray(conj.relation.right.eval(df), dtype=float)
    Hm  = _mask_from_pred(df, conj.condition)
    tol = np.maximum(atol, rtol * np.maximum(1.0, np.abs(rhs)))
    if isinstance(conj.relation, Le):
        strict = (lhs < rhs - tol) if side == "lt" else (lhs > rhs + tol)
    elif isinstance(conj.relation, Ge):
        strict = (lhs > rhs + tol) if side == "gt" else (lhs < rhs - tol)
    else:
        raise ValueError("Conjecture relation must be Le or Ge")
    return Hm & strict

def _mk_name(kind: str, conj: Conjecture) -> str:
    H = conj.condition
    Hname = H.pretty() if hasattr(H, "pretty") else (H.name if hasattr(H, "name") else repr(H))
    lhs   = conj.relation.left.pretty()
    rhs   = conj.relation.right.pretty()
    sym   = "==" if kind == "on" else ("<" if kind == "lt" else ">")
    return f"[{Hname}] :: {lhs} {sym} {rhs}"

class TxGraffitiMini:
    """
    Facade mirroring your original class API, implemented on top of the
    workbench functions we built. This lets you keep your old calling style.
    """

    def __init__(self, df: pd.DataFrame, *, config: Optional[GenerationConfig] = None):
        self.df = df
        self.config = config or GenerationConfig()
        self.engine = WorkbenchEngine(df, config=self.config)
        # Make these accessible like the original
        self.hyps_kept = self.engine.hyps_kept
        self.numeric_columns = self.engine.numeric_columns

    # ---------- Phase 1 ----------
    def generate_single_feature_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        numeric_columns: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        hyps = list(hyps or self.hyps_kept)
        cols = [c for c in (numeric_columns or self.numeric_columns) if c != target_col]
        return _gen_single(self.df, target_col, hyps=hyps, numeric_columns=cols, config=self.config)

    def run_single_feature_pipeline(self, target_col: str) -> Tuple[List[Conjecture], List[Conjecture]]:
        lows, ups = self.generate_single_feature_bounds(target_col)
        lows_final = rank_and_filter(lows, self.df, min_touch=self.config.min_touch_keep)
        ups_final  = rank_and_filter(ups,  self.df, min_touch=self.config.min_touch_keep)
        return lows_final, ups_final

    # ---------- Phase 2 ----------
    def generate_mixed_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        primary: Optional[Iterable[str]] = None,
        secondary: Optional[Iterable[str]] = None,
        weight: Fraction | float = 0.5,
    ):
        hyps = list(hyps or self.hyps_kept)
        prim = [c for c in (primary   or self.numeric_columns) if c != target_col]
        sec  = [c for c in (secondary or self.numeric_columns) if c != target_col]
        return _gen_mixed(self.df, target_col, hyps=hyps, primary=prim, secondary=sec, weight=weight, config=self.config)

    def run_mixed_pipeline(
        self, target_col: str, *, weight: Fraction | float = 0.5,
        primary: Optional[Iterable[str]] = None, secondary: Optional[Iterable[str]] = None
    ):
        lows, ups = self.generate_mixed_bounds(target_col, weight=weight, primary=primary, secondary=secondary)
        return (
            rank_and_filter(lows, self.df, min_touch=self.config.min_touch_keep),
            rank_and_filter(ups,  self.df, min_touch=self.config.min_touch_keep),
        )

    # ---------- Phase 3 ----------
    def generate_targeted_product_bounds(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable] = None,
        x_candidates: Optional[Iterable[str]] = None,
        yz_candidates: Optional[Iterable[str]] = None,
        require_pos: bool = True,
        enable_cancellation: bool = True,
        allow_x_equal_yz: bool = True,
    ):
        hyps = list(hyps or self.hyps_kept)
        xs   = [c for c in (x_candidates  or self.numeric_columns) if c != target_col]
        yz   = [c for c in (yz_candidates or self.numeric_columns)]
        return _gen_products(
            self.df,
            target_col,
            hyps=hyps,
            x_candidates=xs,
            yz_candidates=yz,
            require_pos=require_pos,
            enable_cancellation=enable_cancellation,
            allow_x_equal_yz=allow_x_equal_yz,
        )

    def run_targeted_product_pipeline(
        self,
        target_col: str,
        *,
        require_pos: bool = True,
        enable_cancellation: bool = True,
        allow_x_equal_yz: bool = True,
        x_candidates: Optional[Iterable[str]] = None,
        yz_candidates: Optional[Iterable[str]] = None,
    ):
        lows, ups = self.generate_targeted_product_bounds(
            target_col,
            hyps=None,
            x_candidates=x_candidates,
            yz_candidates=yz_candidates,
            require_pos=require_pos,
            enable_cancellation=enable_cancellation,
            allow_x_equal_yz=allow_x_equal_yz,
        )
        return (
            rank_and_filter(lows, self.df, min_touch=self.config.min_touch_keep),
            rank_and_filter(ups,  self.df, min_touch=self.config.min_touch_keep),
        )

    # ---------- Derived predicates from conjectures ----------
    def add_derived_predicates_from_top_conjectures(
        self,
        conjs: Sequence[Conjecture],
        *,
        top_quantile: float = 0.10,
        min_support: int = 10,
        make_eq: bool = True,
        make_strict: bool = True,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        dedupe_by_mask: bool = True,
    ) -> List[Predicate]:
        if not conjs:
            return []

        # score by touches
        scored = [(c, touch_count(c, self.df)) for c in conjs]
        scored.sort(key=lambda t: t[1], reverse=True)
        k = max(1, int(np.ceil(len(scored) * float(top_quantile))))
        top = [c for (c, _) in scored[:k]]

        preds: List[Predicate] = []
        for c in top:
            if make_eq:
                p_eq = Where(fn=lambda df, _c=c: _touch_mask(_c, df, rtol=rtol, atol=atol), name=_mk_name("on", c))
                preds.append(p_eq)
            if make_strict:
                if isinstance(c.relation, Le):
                    p_lt = Where(fn=lambda df, _c=c: _strict_mask(_c, df, side="lt", rtol=rtol, atol=atol), name=_mk_name("lt", c))
                    preds.append(p_lt)
                elif isinstance(c.relation, Ge):
                    p_gt = Where(fn=lambda df, _c=c: _strict_mask(_c, df, side="gt", rtol=rtol, atol=atol), name=_mk_name("gt", c))
                    preds.append(p_gt)

        # filter by support
        kept = []
        for p in preds:
            m = _mask_from_pred(self.df, p)
            if int(m.sum()) >= int(min_support):
                kept.append(p)

        # dedupe by mask
        if dedupe_by_mask and kept:
            masks = [ _mask_from_pred(self.df, p) for p in kept ]
            uniq_idx = []
            for i, mi in enumerate(masks):
                if not any(np.array_equal(mi, masks[j]) for j in uniq_idx):
                    uniq_idx.append(i)
            kept = [kept[i] for i in uniq_idx]

        # extend hypotheses with new derived predicates
        self.hyps_kept.extend(kept)
        return kept

    # ---------- Pretty printing (light wrappers) ----------
    @staticmethod
    def pretty_block(title: str, conjs: Sequence[Conjecture], max_items: int = 40):
        print(f"\n=== {title} ===")
        for i, c in enumerate(conjs[:max_items], 1):
            try:
                print(f"{i:3d}. {c.pretty(arrow='⇒')}")
            except Exception:
                print(f"{i:3d}. {repr(c)}")
