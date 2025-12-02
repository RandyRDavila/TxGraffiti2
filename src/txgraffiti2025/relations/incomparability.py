# src/txgraffiti2025/relations/incomparability.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd

from .core import DataModel, MaskCache
from ..forms.predicates import Predicate
from ..forms.utils import abs_, min_, max_

__all__ = ["IncomparabilityAnalyzer"]


@dataclass(slots=True)
class IncomparabilityAnalyzer:
    """
    Analyze unordered numeric-invariant pairs (x, y) for two-sided
    incomparability on a domain, and (optionally) register useful
    derived expressions in the model:

      • abs(x - y)      for meaningfully incomparable pairs with a real gap
      • min(x, y), max(x, y)
         – either for “often unequal” pairs, or
         – for “meaningfully incomparable” pairs (balanced two-sided)

    All registration writes both to `model.exprs[...]` and to
    `model.registry[...]` with compact, queryable rows.
    """

    model: DataModel
    cache: MaskCache

    # quick-inspection handles
    abs_exprs_top: Optional[List[Tuple[str, Any]]] = field(default=None)
    minmax_exprs_top: Optional[List[Tuple[str, str]]] = field(default=None)

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────
    def _domain_mask(
        self,
        condition: Optional[Predicate],
        base_fallback: Optional[Predicate]
    ) -> np.ndarray:
        """
        Resolve the domain mask: condition if given,
        else base_fallback if given, else all True.
        """
        if condition is not None:
            return self.cache.mask(condition).to_numpy(dtype=np.bool_, copy=False)
        if base_fallback is not None:
            return self.cache.mask(base_fallback).to_numpy(dtype=np.bool_, copy=False)
        return np.ones(len(self.model.df), dtype=bool)

    def _series_arrays(
        self,
        require_finite: bool
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Pre-evaluate numeric columns and return arrays + finite masks.
        """
        num_cols = list(self.model.numeric_cols)
        eval_cache: Dict[str, pd.Series] = {
            c: self.model.exprs[c].eval(self.model.df) for c in num_cols
        }

        arrays: Dict[str, np.ndarray] = {}
        finite_masks: Dict[str, np.ndarray] = {}
        for c in num_cols:
            s = eval_cache[c]
            if require_finite:
                s = s.replace([np.inf, -np.inf], np.nan)
            a = s.to_numpy(dtype=float, copy=False)
            arrays[c] = a
            finite_masks[c] = np.isfinite(a) if require_finite else np.ones_like(a, dtype=bool)
        return arrays, finite_masks

    @staticmethod
    def _uniq_key(base: str, taken: Dict[str, Any]) -> str:
        if base not in taken:
            return base
        i = 2
        key = f"{base}__{i}"
        while key in taken:
            i += 1
            key = f"{base}__{i}"
        return key

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics only
    # ──────────────────────────────────────────────────────────────────────
    def analyze(
        self,
        condition: Optional[Predicate] = None,
        *,
        base_fallback: Optional[Predicate] = None,
        require_finite: bool = True,
        include_all_pairs: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with pairwise directionality counts/rates on a domain.

        Columns:
          inv1, inv2, n_rows, n_lt, n_gt, n_eq,
          rate_lt, rate_gt, rate_eq,
          incomparable (two-sided), balance (=min(rate_lt, rate_gt)), support
        """
        domain = self._domain_mask(condition, base_fallback)
        domain_rows = int(domain.sum())
        if domain_rows == 0:
            return pd.DataFrame(columns=[
                "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
                "rate_lt","rate_gt","rate_eq","incomparable","balance","support"
            ])

        arrays, finite_masks = self._series_arrays(require_finite)
        num_cols = list(self.model.numeric_cols)
        rows: List[Dict[str, Any]] = []

        for i in range(len(num_cols)):
            x = num_cols[i]; ax = arrays[x]; fx = finite_masks[x]
            for j in range(i + 1, len(num_cols)):
                y = num_cols[j]; ay = arrays[y]; fy = finite_masks[y]

                m = domain & fx & fy
                n = int(m.sum())
                if n == 0:
                    continue

                axm = ax[m]; aym = ay[m]
                lt = axm < aym
                gt = axm > aym
                eq = ~(lt | gt)

                n_lt = int(lt.sum())
                n_gt = int(gt.sum())
                n_eq = int(eq.sum())

                incomparable = (n_lt > 0) and (n_gt > 0)
                if include_all_pairs or incomparable:
                    rate_lt = n_lt / n
                    rate_gt = n_gt / n
                    rate_eq = n_eq / n
                    rows.append({
                        "inv1": x, "inv2": y,
                        "n_rows": n,
                        "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                        "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                        "incomparable": bool(incomparable),
                        "balance": min(rate_lt, rate_gt),
                        "support": n / domain_rows,
                    })

        out = pd.DataFrame(rows, columns=[
            "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
            "rate_lt","rate_gt","rate_eq","incomparable","balance","support"
        ])
        if out.empty:
            return out

        return out.sort_values(
            ["incomparable","balance","support","inv1","inv2"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # Register abs(x - y) for “meaningfully incomparable” pairs
    # ──────────────────────────────────────────────────────────────────────
    def register_absdiff_exprs_for_meaningful_pairs(
        self,
        condition: Optional[Predicate] = None,
        *,
        base_fallback: Optional[Predicate] = None,
        require_finite: bool = True,
        # incomparability thresholds
        min_support: float = 0.10,
        min_side_rate: float = 0.10,
        min_side_count: int = 5,
        # gap-size thresholds
        min_median_gap: float = 0.5,
        min_mean_gap: float = 0.5,
        # naming
        key_prefix: str = "abs_",
        key_suffix: str = "",
        # registry behavior
        overwrite_existing: bool = False,
        top_n_store: int = 20,
        # optional label for registry
        hypothesis_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Register abs(x - y) into model.exprs for pairs that are:
          – meaningfully incomparable (two-sided with coverage), and
          – exhibit a real gap (median or mean >= threshold)

        Writes rows into model.registry["absdiff"] with:
          {inv1, inv2, expr_name, hypothesis, support, median_gap, mean_gap}
        """
        domain = self._domain_mask(condition, base_fallback)
        domain_rows = int(domain.sum())
        if domain_rows == 0:
            self.abs_exprs_top = None
            return pd.DataFrame(columns=[
                "inv1","inv2","n_rows",
                "n_lt","n_gt","n_eq",
                "rate_lt","rate_gt","rate_eq","support",
                "median_gap","mean_gap","selected","expr_name"
            ])

        arrays, finite_masks = self._series_arrays(require_finite)
        num_cols = list(self.model.numeric_cols)

        rows: List[Dict[str, Any]] = []
        registered: List[Tuple[str, Any]] = []

        for i in range(len(num_cols)):
            x = num_cols[i]; ax = arrays[x]; fx = finite_masks[x]
            for j in range(i + 1, len(num_cols)):
                y = num_cols[j]; ay = arrays[y]; fy = finite_masks[y]

                m = domain & fx & fy
                n = int(m.sum())
                if n == 0:
                    continue

                axm = ax[m]; aym = ay[m]
                lt = axm < aym
                gt = axm > aym
                eq = ~(lt | gt)

                n_lt = int(lt.sum())
                n_gt = int(gt.sum())
                n_eq = int(eq.sum())

                rate_lt = n_lt / n
                rate_gt = n_gt / n
                rate_eq = n_eq / n
                support = n / domain_rows

                both_sides = (n_lt > 0) and (n_gt > 0)
                side_ok = (min(rate_lt, rate_gt) >= float(min_side_rate)) and (min(n_lt, n_gt) >= int(min_side_count))
                support_ok = (support >= float(min_support))

                gaps = np.abs(axm - aym)
                median_gap = float(np.nanmedian(gaps)) if n > 0 else 0.0
                mean_gap   = float(np.nanmean(gaps))   if n > 0 else 0.0
                gap_ok = (median_gap >= float(min_median_gap)) or (mean_gap >= float(min_mean_gap))

                selected = bool(both_sides and side_ok and support_ok and gap_ok)

                expr_name = None
                if selected:
                    expr = abs_(self.model.exprs[x] - self.model.exprs[y])
                    base = f"{key_prefix}{x}_minus_{y}{key_suffix}"
                    key = base if (overwrite_existing or base not in self.model.exprs) \
                          else self._uniq_key(base, self.model.exprs)
                    self.model.exprs[key] = expr
                    expr_name = key
                    registered.append((key, expr))

                    # registry row
                    self.model.record_absdiff(
                        inv1=x, inv2=y,
                        expr_name=key,
                        hypothesis=hypothesis_label,
                        support=float(support),
                        median_gap=median_gap,
                        mean_gap=mean_gap,
                    )

                rows.append({
                    "inv1": x, "inv2": y,
                    "n_rows": n,
                    "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                    "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                    "support": support,
                    "median_gap": median_gap, "mean_gap": mean_gap,
                    "selected": selected, "expr_name": expr_name,
                })

        out = pd.DataFrame(rows, columns=[
            "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
            "rate_lt","rate_gt","rate_eq","support",
            "median_gap","mean_gap","selected","expr_name"
        ])

        if out.empty:
            self.abs_exprs_top = None
            return out

        out_sorted = out.sort_values("mean_gap", ascending=False, kind="mergesort").reset_index(drop=True)
        self.abs_exprs_top = registered[: min(top_n_store, len(registered))] if top_n_store > 0 and registered else None
        return out_sorted

    # ──────────────────────────────────────────────────────────────────────
    # Register min(x,y), max(x,y) for pairs that are often unequal
    # ──────────────────────────────────────────────────────────────────────
    def register_minmax_for_often_unequal_pairs(
        self,
        condition: Optional[Predicate] = None,
        *,
        base_fallback: Optional[Predicate] = None,
        require_finite: bool = True,
        min_support: float = 0.10,
        min_neq_rate: float = 0.50,
        min_neq_count: int = 8,
        must_be_incomparable: bool = True,
        key_prefix_min: str = "min_",
        key_prefix_max: str = "max_",
        key_suffix: str = "",
        overwrite_existing: bool = False,
        top_n_store: int = 20,
        hypothesis_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Register min/max for pairs that are sufficiently often unequal
        (and optionally must be two-sided incomparable).
        Writes rows into model.registry["minmax"].
        """
        domain = self._domain_mask(condition, base_fallback)
        domain_rows = int(domain.sum())
        if domain_rows == 0:
            self.minmax_exprs_top = None
            return pd.DataFrame(columns=[
                "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
                "rate_lt","rate_gt","rate_eq","support",
                "often_unequal","selected",
                "expr_min_name","expr_max_name"
            ])

        arrays, finite_masks = self._series_arrays(require_finite)
        num_cols = list(self.model.numeric_cols)

        rows: List[Dict[str, Any]] = []
        registered_pairs: List[Tuple[str, str]] = []

        for i in range(len(num_cols)):
            x = num_cols[i]; ax = arrays[x]; fx = finite_masks[x]
            for j in range(i + 1, len(num_cols)):
                y = num_cols[j]; ay = arrays[y]; fy = finite_masks[y]

                m = domain & fx & fy
                n = int(m.sum())
                if n == 0:
                    continue

                axm = ax[m]; aym = ay[m]
                lt = axm < aym
                gt = axm > aym
                eq = ~(lt | gt)

                n_lt = int(lt.sum())
                n_gt = int(gt.sum())
                n_eq = int(eq.sum())

                rate_lt = n_lt / n
                rate_gt = n_gt / n
                rate_eq = n_eq / n
                support = n / domain_rows

                incomparable = (n_lt > 0) and (n_gt > 0)
                if must_be_incomparable and not incomparable:
                    selected = False
                    often_unequal = False
                else:
                    neq = n - n_eq
                    often_unequal = (n > 0) and ((neq / n) >= float(min_neq_rate)) and (neq >= int(min_neq_count))
                    selected = bool((support >= float(min_support)) and often_unequal)

                expr_min_name = None
                expr_max_name = None
                if selected:
                    e_min = min_(self.model.exprs[x], self.model.exprs[y])
                    e_max = max_(self.model.exprs[x], self.model.exprs[y])

                    base_min = f"{key_prefix_min}{x}_{y}{key_suffix}"
                    base_max = f"{key_prefix_max}{x}_{y}{key_suffix}"
                    kmin = base_min if (overwrite_existing or base_min not in self.model.exprs) \
                           else self._uniq_key(base_min, self.model.exprs)
                    kmax = base_max if (overwrite_existing or base_max not in self.model.exprs) \
                           else self._uniq_key(base_max, self.model.exprs)

                    self.model.exprs[kmin] = e_min
                    self.model.exprs[kmax] = e_max
                    expr_min_name, expr_max_name = kmin, kmax
                    registered_pairs.append((kmin, kmax))

                    # registry row
                    self.model.record_minmax(
                        inv1=x, inv2=y,
                        key_min=kmin, key_max=kmax,
                        hypothesis=hypothesis_label,
                        support=float(support),
                        often_unequal=bool(often_unequal),
                    )

                rows.append({
                    "inv1": x, "inv2": y,
                    "n_rows": n,
                    "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                    "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                    "support": support,
                    "often_unequal": bool((n - n_eq) / n >= float(min_neq_rate)) if n > 0 else False,
                    "selected": selected,
                    "expr_min_name": expr_min_name,
                    "expr_max_name": expr_max_name,
                })

        out = pd.DataFrame(rows, columns=[
            "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
            "rate_lt","rate_gt","rate_eq","support",
            "often_unequal","selected","expr_min_name","expr_max_name"
        ])

        self.minmax_exprs_top = registered_pairs[: min(top_n_store, len(registered_pairs))] if top_n_store > 0 and registered_pairs else None

        return out.sort_values(
            ["selected","support","inv1","inv2"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # Meaningfully incomparable → register min/max (balanced two-sided)
    # ──────────────────────────────────────────────────────────────────────
    def register_minmax_exprs_for_meaningful_pairs(
        self,
        condition: Optional[Predicate] = None,
        *,
        base_fallback: Optional[Predicate] = None,
        require_finite: bool = True,
        # incomparability & coverage
        min_support: float = 0.10,
        min_side_rate: float = 0.10,
        min_side_count: int = 5,
        max_eq_rate: float = 0.70,  # skip if equality dominates
        # key style
        style: str = "slug",        # "slug" -> min_a_b / max_a_b ; "pretty" -> min(a, b)/max(a, b)
        overwrite_existing: bool = False,
        hypothesis_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        For each (x,y) that is meaningfully incomparable on the domain,
        register min(x,y) and max(x,y).

        Writes rows into model.registry["minmax"].
        """
        domain = self._domain_mask(condition, base_fallback)
        domain_rows = int(domain.sum())
        if domain_rows == 0:
            return pd.DataFrame(columns=[
                "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
                "rate_lt","rate_gt","rate_eq","support",
                "selected","key_min","key_max"
            ])

        arrays, finite_masks = self._series_arrays(require_finite)
        num_cols = list(self.model.numeric_cols)

        def _keypair(x: str, y: str) -> Tuple[str, str]:
            a, b = sorted([x, y])
            if style == "pretty":
                return f"min({a}, {b})", f"max({a}, {b})"
            return f"min_{a}_{b}", f"max_{a}_{b}"

        rows: List[Dict[str, Any]] = []
        for i in range(len(num_cols)):
            x = num_cols[i]; ax = arrays[x]; fx = finite_masks[x]
            for j in range(i + 1, len(num_cols)):
                y = num_cols[j]; ay = arrays[y]; fy = finite_masks[y]

                m = domain & fx & fy
                n = int(m.sum())
                if n == 0:
                    continue

                axm = ax[m]; aym = ay[m]
                lt = axm < aym
                gt = axm > aym
                eq = ~(lt | gt)

                n_lt = int(lt.sum())
                n_gt = int(gt.sum())
                n_eq = int(eq.sum())

                rate_lt = n_lt / n
                rate_gt = n_gt / n
                rate_eq = n_eq / n
                support = n / domain_rows

                both_sides = (n_lt > 0) and (n_gt > 0)
                side_ok = (min(rate_lt, rate_gt) >= float(min_side_rate)) and (min(n_lt, n_gt) >= int(min_side_count))
                support_ok = (support >= float(min_support))
                not_mostly_equal = (rate_eq <= float(max_eq_rate))

                selected = bool(both_sides and side_ok and support_ok and not_mostly_equal)

                key_min = None
                key_max = None
                if selected:
                    e_min = min_(self.model.exprs[x], self.model.exprs[y])
                    e_max = max_(self.model.exprs[x], self.model.exprs[y])

                    kmin, kmax = _keypair(x, y)
                    if not overwrite_existing:
                        kmin = kmin if kmin not in self.model.exprs else self._uniq_key(kmin, self.model.exprs)
                        kmax = kmax if kmax not in self.model.exprs else self._uniq_key(kmax, self.model.exprs)

                    self.model.exprs[kmin] = e_min
                    self.model.exprs[kmax] = e_max
                    key_min, key_max = kmin, kmax

                    self.model.record_minmax(
                        inv1=x, inv2=y,
                        key_min=kmin, key_max=kmax,
                        hypothesis=hypothesis_label,
                        support=float(support),
                        often_unequal=None,  # not applicable in this pathway
                    )

                rows.append({
                    "inv1": x, "inv2": y,
                    "n_rows": n,
                    "n_lt": n_lt, "n_gt": n_gt, "n_eq": n_eq,
                    "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                    "support": support,
                    "selected": selected,
                    "key_min": key_min, "key_max": key_max,
                })

        out = pd.DataFrame(rows, columns=[
            "inv1","inv2","n_rows","n_lt","n_gt","n_eq",
            "rate_lt","rate_gt","rate_eq","support",
            "selected","key_min","key_max",
        ])
        return out.sort_values(
            ["selected","support","inv1","inv2"],
            ascending=[False, False, True, True]
        ).reset_index(drop=True)
