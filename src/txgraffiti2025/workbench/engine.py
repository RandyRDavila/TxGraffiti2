# src/txgraffiti2025/workbench/engine.py

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.generic_conjecture import Conjecture
from txgraffiti2025.processing.pre.hypotheses import (
    enumerate_boolean_hypotheses,
    detect_base_hypothesis,
)
from txgraffiti2025.processing.pre.simplify_hypotheses import (
    simplify_and_dedup_hypotheses,
)

from .config import GenerationConfig
from .ranking import rank_and_filter
from .conj_single_feature import generate_single_feature_bounds
from .conj_mixed_bounds import generate_mixed_bounds
from .conj_targeted_products import generate_targeted_product_bounds
from .generalize_reciprocal import generalize_from_reciprocal_patterns

# Optional: LP two-feature generator (present if you added the file)
try:
    from .conj_lp_two_feature import generate_lp_two_feature_bounds  # type: ignore
except Exception:
    generate_lp_two_feature_bounds = None  # type: ignore[assignment]

# --- provide a tolerant TRUE predicate if not available from forms.predicates ---
try:
    from txgraffiti2025.forms.predicates import TRUE  # type: ignore
except Exception:  # fallback for projects without a TRUE constant
    from txgraffiti2025.forms.predicates import Where as _WhereTRUE
    TRUE = _WhereTRUE(fn=lambda df: np.ones(len(df), dtype=bool), name="TRUE")


class WorkbenchEngine:
    """
    Orchestrates hypothesis discovery and generation passes.

    This class discovers boolean hypotheses, splits columns into boolean/numeric,
    builds constant-banks, and exposes uniform pipelines for generating and
    ranking conjectures. All pipelines accept optional overrides for hypotheses
    and feature pools so you can reuse the engine in second-pass runs or for
    controlled experiments.
    """

    def __init__(self, df: pd.DataFrame, *, config: Optional[GenerationConfig] = None):
        self.df = df
        self.config = config or GenerationConfig()

        # hypotheses discovery (base + boolean classes + simplified)
        self.base_hyp = detect_base_hypothesis(df)
        self.hyps_all = enumerate_boolean_hypotheses(
            df,
            treat_binary_ints=True,
            include_base=True,
            include_pairs=True,
            skip_always_false=True,
        )
        self.hyps_kept, _ = simplify_and_dedup_hypotheses(
            df, self.hyps_all, min_support=10, treat_binary_ints=True
        )

        # columns
        self.bool_columns, self.numeric_columns = self._split_columns(df)

        # constant-banks per hypothesis
        self.const_bank = {}  # Hkey -> {col: scalar}
        self._build_constant_banks()

    # ------------- utilities -------------

    @staticmethod
    def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        bool_cols: List[str] = []
        for c in df.columns:
            s = df[c]
            if s.dtype == bool:
                bool_cols.append(c)
            elif pd.api.types.is_integer_dtype(s):
                vals = pd.unique(s.dropna())
                try:
                    ints = set(int(v) for v in vals)
                except Exception:
                    continue
                if len(ints) <= 2 and ints.issubset({0, 1}):
                    bool_cols.append(c)
        num_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in bool_cols
        ]
        return bool_cols, num_cols

    def _build_constant_banks(self) -> None:
        from fractions import Fraction
        import numpy as np
        from .arrays import support as mask_support

        def _hyp_key(H) -> str:
            if hasattr(H, "pretty"):
                try:
                    return H.pretty()
                except Exception:
                    pass
            return repr(H)

        def _finite_mask(a): return np.isfinite(a)

        def _constant_value_on_mask(vals, eps: float):
            f = _finite_mask(vals)
            if not any(f):
                return False, float("nan")
            v = float(np.median(vals[f].astype(float)))
            dev = float(np.max(np.abs(vals[f] - v)))
            return (dev <= eps * (1.0 + abs(v))), v

        def _values_on_mask(df, mask, col):
            return df.loc[mask, col].to_numpy(dtype=float, copy=False)

        eps = self.config.eps_const
        min_sup = self.config.min_support_const
        for H in self.hyps_kept:
            mask = H.mask(self.df).astype(bool).to_numpy()
            if mask_support(mask) < min_sup:
                self.const_bank[_hyp_key(H)] = {}
                continue
            bank = {}
            for c in self.numeric_columns:
                vals = _values_on_mask(self.df, mask, c)
                is_const, val = _constant_value_on_mask(vals, eps)
                if is_const:
                    bank[c] = val
            self.const_bank[_hyp_key(H)] = bank

    # ------------- public pipelines -------------

    def run_single_feature_pipeline(
        self,
        target_col: str,
        *,
        hyps: Optional[Iterable[Predicate]] = None,
        numeric_columns: Optional[Iterable[str]] = None,
        rank: bool = True,
        min_touch: Optional[int] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Generate `target ≥ c_min·x` and `target ≤ c_max·x` (and ceil/floor variants).

        Parameters
        ----------
        target_col : str
            Target column.
        hyps : iterable of Predicate, optional
            Hypotheses to use; defaults to engine-discovered hypotheses.
        numeric_columns : iterable of str, optional
            Numeric feature pool; defaults to engine-detected numeric columns.
        rank : bool, default True
            If True, return rank_and_filter(...) results; else raw conjectures.
        min_touch : int or None
            Minimum touch passed to rank_and_filter. Defaults to config.min_touch_keep.
        """
        Hs = list(hyps or self.hyps_kept)
        nums = list(numeric_columns or self.numeric_columns)
        lows, ups = generate_single_feature_bounds(
            self.df,
            target_col,
            hyps=Hs,
            numeric_columns=nums,
            config=self.config,
        )
        if not rank:
            return lows, ups
        mt = self.config.min_touch_keep if min_touch is None else int(min_touch)
        return (
            rank_and_filter(self.df, lows, min_touch=mt),
            rank_and_filter(self.df, ups,  min_touch=mt),
        )

    def run_mixed_pipeline(
        self,
        target_col: str,
        *,
        primary: Optional[Iterable[str]] = None,
        secondary: Optional[Iterable[str]] = None,
        hyps: Optional[Iterable[Predicate]] = None,
        weight: float = 0.5,
        rank: bool = True,
        min_touch: Optional[int] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Generate two-feature mixed bounds with sqrt/square variants (and ceil/floor).
        """
        prim = list(primary or self.numeric_columns)
        sec  = list(secondary or self.numeric_columns)
        Hs   = list(hyps or self.hyps_kept)
        lows, ups = generate_mixed_bounds(
            self.df, target_col,
            hyps=Hs, primary=prim, secondary=sec,
            config=self.config, weight=weight
        )
        if not rank:
            return lows, ups
        mt = self.config.min_touch_keep if min_touch is None else int(min_touch)
        return (
            rank_and_filter(self.df, lows, min_touch=mt),
            rank_and_filter(self.df, ups,  min_touch=mt),
        )

    def run_targeted_product_pipeline(
        self,
        target_col: str,
        *,
        x_candidates: Optional[Iterable[str]] = None,
        yz_candidates: Optional[Iterable[str]] = None,
        hyps: Optional[Iterable[Predicate]] = None,
        require_pos: bool = True,
        enable_cancellation: bool = True,
        allow_x_equal_yz: bool = True,
        rank: bool = True,
        min_touch: Optional[int] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        Generate (H) ⇒ T·x ≤ y·z and ≥ y·z with triviality checks and optional cancellation.
        """
        Hs = list(hyps or self.hyps_kept)
        xs = list(x_candidates or [c for c in self.numeric_columns if c != target_col])
        yz = list(yz_candidates or list(self.numeric_columns))
        lows, ups = generate_targeted_product_bounds(
            self.df, target_col,
            hyps=Hs, x_candidates=xs, yz_candidates=yz,
            require_pos=require_pos,
            enable_cancellation=enable_cancellation,
            allow_x_equal_yz=allow_x_equal_yz,
        )
        if not rank:
            return lows, ups
        mt = self.config.min_touch_keep if min_touch is None else int(min_touch)
        return (
            rank_and_filter(self.df, lows, min_touch=mt),
            rank_and_filter(self.df, ups,  min_touch=mt),
        )

    def run_reciprocal_generalizer(
        self,
        base_conjectures: Sequence[Conjecture],
        *,
        rank: bool = True,
        min_touch: Optional[int] = None,
    ) -> List[Conjecture]:
        """
        Generalize from reciprocal patterns using constant-banks and broader hypotheses.
        """
        def broader(H0):
            base_mask = H0.mask(self.df).astype(bool).to_numpy()
            pairs = []
            hyps = self.hyps_kept if TRUE in self.hyps_kept else (self.hyps_kept + [TRUE])
            for Hp in hyps:
                pmask = Hp.mask(self.df).astype(bool).to_numpy()
                if ((~base_mask | pmask).all()) and (pmask.sum() >= self.config.min_support_const):
                    pairs.append((Hp, int(pmask.sum())))
            pairs.sort(key=lambda t: -t[1])
            return [h for h,_ in pairs]

        props = generalize_from_reciprocal_patterns(
            self.df,
            base_conjectures,
            const_banks=self.const_bank,
            hyps_sorted_broadest_first=broader,
            config=self.config,
        )
        if not rank:
            return props
        mt = self.config.min_touch_keep if min_touch is None else int(min_touch)
        return rank_and_filter(self.df, props, min_touch=mt)

    def run_lp_two_feature(
        self,
        target_col: str,
        *,
        features: Optional[Iterable[str]] = None,
        hyps: Optional[Iterable[Predicate]] = None,
        secondary_transforms: Iterable[str] = ("identity",),
        allow_same_feature_when_transformed: bool = True,
        require_finite: bool = True,
        require_min_rows: int = 3,
        nonneg_coeffs: bool = False,
        rationalize: bool = True,
        log_base: Optional[float] = None,
        log_epsilon: float = 0.0,
        rank: bool = True,
        min_touch: Optional[int] = None,
    ) -> Tuple[List[Conjecture], List[Conjecture]]:
        """
        LP-based two-feature linear bounds with optional transforms and ceil/floor.

        Requires `workbench.conj_lp_two_feature.generate_lp_two_feature_bounds`.
        """
        if generate_lp_two_feature_bounds is None:
            raise RuntimeError("LP generator not available (conj_lp_two_feature.py not imported).")
        feats = list(features or self.numeric_columns)
        Hs    = list(hyps or self.hyps_kept)
        lows, ups = generate_lp_two_feature_bounds(
            self.df,
            target_col=target_col,
            hyps=Hs,
            features=feats,
            config=self.config,
            require_finite=require_finite,
            require_min_rows=require_min_rows,
            nonneg_coeffs=nonneg_coeffs,
            rationalize=rationalize,
            secondary_transforms=secondary_transforms,
            allow_same_feature_when_transformed=allow_same_feature_when_transformed,
            log_base=log_base,
            log_epsilon=log_epsilon,
        )
        if not rank:
            return lows, ups
        mt = self.config.min_touch_keep if min_touch is None else int(min_touch)
        return (
            rank_and_filter(self.df, lows, min_touch=mt),
            rank_and_filter(self.df, ups,  min_touch=mt),
        )
