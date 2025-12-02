from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from .core import DataModel, MaskCache
from ..forms.predicates import Predicate, LT, GT, LE, GE, EQ
from ..forms.generic_conjecture import TRUE


__all__ = ["OrderHypothesisMiner"]


def _series_sig_bool(s: pd.Series) -> bytes:
    """Stable byte signature for boolean-like Series (NA -> False)."""
    a = np.asarray(s.fillna(False).astype(bool).to_numpy(copy=False), dtype=np.uint8)
    return a.tobytes()


@dataclass(slots=True)
class OrderHypothesisMiner:
    """
    Mine *informative* boolean hypotheses from numeric pairs (x, y):

      • If both sides occur (incomparability) *and* one side dominates
        (e.g., rate(x<y) ≥ min_side_rate), register (x < y) or (y < x).
      • If an ordering is nearly universal (e.g., x ≤ y ~ 100%),
        skip tautology; optionally register equality core (x == y)
        when its support is nontrivial.
    """
    model: DataModel
    cache: MaskCache

    # --------------- internal helpers ----------------
    def _domain_mask(self, condition: Optional[Predicate], use_base_if_none: bool) -> np.ndarray:
        if condition is not None:
            return np.asarray(self.cache.mask(condition), dtype=bool)
        base = getattr(self.model, "base_pred", TRUE)
        if use_base_if_none and (base is not TRUE):
            return np.asarray(self.cache.mask(base), dtype=bool)
        return np.ones(len(self.model.df), dtype=bool)

    def _already_have_mask(self, m: pd.Series) -> bool:
        """Check if a mask is identical to any existing predicate mask on the full df."""
        # Note: compare vs current catalog of preds to avoid duplicates.
        sig = _series_sig_bool(m)
        df = self.model.df
        for name, pred in self.model.preds.items():
            try:
                if _series_sig_bool(self.cache.mask(pred)) == sig:
                    return True
            except Exception:
                continue
        return False

    def _register_pred(self, key: str, pred: Predicate) -> str:
        """Register predicate in model (with collision-avoidance) and return final key."""
        k = key
        if k in self.model.preds:
            i = 2
            k = f"{key}__{i}"
            while k in self.model.preds:
                i += 1
                k = f"{key}__{i}"
        self.model.preds[k] = pred
        if k not in self.model.boolean_cols:
            self.model.boolean_cols.append(k)
            self.model.boolean_cols.sort()
        return k

    # --------------- public API ----------------
    def discover_order_hypotheses(
        self,
        condition: Optional[Predicate] = None,
        *,
        use_base_if_none: bool = True,
        require_finite: bool = True,
        # selection thresholds
        min_support: float = 0.10,     # lower bound on support wrt domain
        max_support: float = 0.95,     # upper bound to avoid tautologies
        min_side_rate: float = 0.70,   # choose side only if its rate ≥ this
        min_balance: float = 0.10,     # ensure incomparability is real (both sides present)
        # equality / near-universal ordering handling
        tol_eq: float = 1e-9,
        promote_eq_if_near_universal: bool = True,
        near_universal_rate: float = 0.98,  # if max(rate(x≤y), rate(y≤x)) ≥ this
        # naming
        name_style: str = "pretty",         # "pretty": "(x < y)" ; "slug": "lt_x_y"
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of candidates with diagnostics and any registered keys.
        """
        df = self.model.df
        domain = self._domain_mask(condition, use_base_if_none)
        n_dom = int(domain.sum())
        if n_dom == 0:
            return pd.DataFrame(columns=[
                "inv1","inv2","n_rows","support",
                "rate_lt","rate_gt","rate_eq","balance",
                "selected","kind","key",
            ])

        # Pre-eval numeric columns
        num_cols = list(self.model.numeric_cols)
        eval_cache: Dict[str, pd.Series] = {c: self.model.exprs[c].eval(df) for c in num_cols}

        arrays: Dict[str, np.ndarray] = {}
        finite_masks: Dict[str, np.ndarray] = {}
        for c in num_cols:
            s = eval_cache[c]
            if require_finite:
                s = s.replace([np.inf, -np.inf], np.nan)
            a = s.to_numpy(dtype=float, copy=False)
            arrays[c] = a
            finite_masks[c] = np.isfinite(a) if require_finite else np.ones_like(a, dtype=bool)

        def _nm(x: str, y: str, kind: str) -> str:
            if name_style == "slug":
                if kind == "lt": return f"lt_{x}_{y}"
                if kind == "gt": return f"gt_{x}_{y}"
                if kind == "eq": return f"eq_{x}_{y}"
            # pretty
            if kind == "lt": return f"({x} < {y})"
            if kind == "gt": return f"({x} < {y})"  # printed as (y < x) below if swapped
            if kind == "eq": return f"({x} == {y})"
            return f"({x} ? {y})"

        rows = []
        domain_series = pd.Series(domain, index=df.index, dtype=bool)

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
                eq = np.isclose(axm, aym, atol=tol_eq)

                n_lt = int(lt.sum())
                n_gt = int(gt.sum())
                n_eq = int(eq.sum())

                rate_lt = n_lt / n
                rate_gt = n_gt / n
                rate_eq = n_eq / n
                balance = min(rate_lt, rate_gt)
                support = n / n_dom

                selected = False
                key = None
                kind = None

                # 1) Incomparability with a dominant side → register that strict side
                both_sides = (n_lt > 0) and (n_gt > 0)
                if both_sides and (balance >= float(min_balance)) and (support >= float(min_support)):
                    if rate_lt >= float(min_side_rate) and support <= float(max_support):
                        # candidate: (x < y)
                        pred = LT(self.model.exprs[x], self.model.exprs[y])
                        mask = self.cache.mask(pred) & domain_series
                        # avoid trivial: identical to domain or empty
                        if 0 < int(mask.sum()) < n_dom and not self._already_have_mask(mask):
                            key = self._register_pred(_nm(x, y, "lt"), pred)
                            selected, kind = True, "lt"

                    elif rate_gt >= float(min_side_rate) and support <= float(max_support):
                        # candidate: (y < x)
                        pred = LT(self.model.exprs[y], self.model.exprs[x])
                        mask = self.cache.mask(pred) & domain_series
                        if 0 < int(mask.sum()) < n_dom and not self._already_have_mask(mask):
                            # display as (y < x)
                            kname = f"({y} < {x})" if name_style == "pretty" else f"lt_{y}_{x}"
                            key = self._register_pred(kname, pred)
                            selected, kind = True, "gt"

                # 2) Near-universal ordering → optionally promote equality core
                if (not selected) and promote_eq_if_near_universal:
                    if max(rate_lt + rate_eq, rate_gt + rate_eq) >= float(near_universal_rate):
                        # Try equality core, but nontrivial (not ~0% or ~100%)
                        eq_mask_full = self.cache.mask(EQ(self.model.exprs[x], self.model.exprs[y])) & domain_series
                        k = int(eq_mask_full.sum())
                        if (support >= float(min_support)) and (0 < k < n_dom) and (k / n_dom <= float(max_support)):
                            if not self._already_have_mask(eq_mask_full):
                                key = self._register_pred(_nm(x, y, "eq"), EQ(self.model.exprs[x], self.model.exprs[y]))
                                selected, kind = True, "eq"

                rows.append({
                    "inv1": x, "inv2": y,
                    "n_rows": n, "support": support,
                    "rate_lt": rate_lt, "rate_gt": rate_gt, "rate_eq": rate_eq,
                    "balance": balance,
                    "selected": selected, "kind": kind, "key": key,
                })

        out = pd.DataFrame(rows, columns=[
            "inv1","inv2","n_rows","support",
            "rate_lt","rate_gt","rate_eq","balance",
            "selected","kind","key",
        ])
        return out.sort_values(
            ["selected","support","balance","inv1","inv2"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
