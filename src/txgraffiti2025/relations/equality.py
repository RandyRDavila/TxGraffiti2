# src/txgraffiti2025/relations/equality.py

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import DataModel, MaskCache
from ..forms.predicates import Predicate
from ..forms.generic_conjecture import TRUE, Conjecture, Eq
from ..forms.utils import Expr, Const


__all__ = ["EqualityMiner"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _domain_mask(
        model: DataModel,
        cache: MaskCache,
        condition: Optional[Predicate],
        use_base_if_none: bool,
    ) -> np.ndarray:
    if condition is not None:
        return cache.mask(condition)
    base = getattr(model, "base_pred", TRUE)
    if use_base_if_none and (base is not TRUE):
        return cache.mask(base)
    return np.ones(len(model.df), dtype=bool)


def _rationalize(x: float, max_denom: int) -> Fraction:
    try:
        return Fraction(x).limit_denominator(max_denom)
    except Exception:
        # fall back: treat as real-only
        return Fraction(0, 1)  # caller should ignore if needed


# ──────────────────────────────────────────────────────────────────────────────
# Equality miner
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class EqualityMiner:
    """
    Mine constant-value columns and near-equalities x ≈ y within a chosen domain.

    Returns tidy DataFrames with diagnostics, and (optionally) produces Eq(...)
    Conjecture objects you can pass to your reporting layer.

    Typical usage
    -------------
    em = EqualityMiner(model, cache)

    const_df = em.analyze_constants(
        condition=None, use_base_if_none=True,
        tol=1e-9, require_finite=True, rationalize=True, max_denom=64
    )

    pair_df = em.analyze_pair_equalities(
        condition=None, use_base_if_none=True,
        tol=1e-9, require_finite=True,
        min_support=0.10, min_eq_rate=0.95
    )

    # Optionally create conjecture objects for rows flagged as selected=True:
    conjs = em.make_eq_conjectures(constants=const_df, pairs=pair_df, condition=None)
    """

    model: DataModel
    cache: MaskCache

    # ──────────────────────────────────────────────────────────────────────
    # 1) Column-wise constants on a domain
    # ──────────────────────────────────────────────────────────────────────
    def analyze_constants(
        self,
        condition: Optional[Predicate] = None,
        *,
        use_base_if_none: bool = True,
        require_finite: bool = True,
        tol: float = 1e-9,
        # presentation
        rationalize: bool = True,
        max_denom: int = 64,
        # selection rules
        min_support: float = 0.10,   # fraction of domain rows that are finite for this column
    ) -> pd.DataFrame:
        """
        Detect numeric columns that are (near) constant on the domain.

        Returns
        -------
        DataFrame columns:
            inv, n_domain, n_finite, support, value, value_frac, max_abs_dev, selected
        """
        mask = _domain_mask(self.model, self.cache, condition, use_base_if_none)
        n_domain = int(mask.sum())
        if n_domain == 0:
            return pd.DataFrame(columns=[
                "inv", "n_domain", "n_finite", "support", "value",
                "value_frac", "max_abs_dev", "selected"
            ])

        eval_cache: Dict[str, pd.Series] = {c: self.model.exprs[c].eval(self.model.df) for c in self.model.numeric_cols}

        rows: List[dict] = []
        for c in self.model.numeric_cols:
            s = eval_cache[c]
            if require_finite:
                s = s.replace([np.inf, -np.inf], np.nan)

            m = mask & np.isfinite(s.to_numpy(dtype=float, copy=False))
            n_finite = int(m.sum())
            if n_finite == 0:
                # No usable rows for this column on the domain
                rows.append({
                    "inv": c, "n_domain": n_domain, "n_finite": 0,
                    "support": 0.0, "value": np.nan, "value_frac": None,
                    "max_abs_dev": np.nan, "selected": False
                })
                continue

            vals = s[m].to_numpy(dtype=float, copy=False)
            # Use median as the robust center
            v = float(np.nanmedian(vals))
            max_abs_dev = float(np.nanmax(np.abs(vals - v)))
            support = n_finite / n_domain

            if rationalize and np.isfinite(v):
                frac = _rationalize(v, max_denom)
                value_frac = str(frac) if frac.denominator <= max_denom else None
            else:
                value_frac = None

            selected = (support >= float(min_support)) and (max_abs_dev <= float(tol))

            rows.append({
                "inv": c,
                "n_domain": n_domain,
                "n_finite": n_finite,
                "support": support,
                "value": v,
                "value_frac": value_frac,
                "max_abs_dev": max_abs_dev,
                "selected": bool(selected),
            })

        out = pd.DataFrame(rows, columns=[
            "inv", "n_domain", "n_finite", "support",
            "value", "value_frac", "max_abs_dev", "selected",
        ])
        if out.empty:
            return out
        return out.sort_values(["selected", "support", "inv"], ascending=[False, False, True]).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # 2) Pairwise near-equalities x ≈ y on a domain
    # ──────────────────────────────────────────────────────────────────────
    def analyze_pair_equalities(
        self,
        condition: Optional[Predicate] = None,
        *,
        use_base_if_none: bool = True,
        require_finite: bool = True,
        tol: float = 1e-9,
        # selection rules
        min_support: float = 0.10,   # coverage fraction within domain
        min_eq_rate: float = 0.95,   # fraction of equal rows among usable pair rows
    ) -> pd.DataFrame:
        """
        Detect unordered pairs (x, y) that agree on most rows within the domain.

        Equality test: |x - y| ≤ tol.

        Returns
        -------
        DataFrame columns:
            inv1, inv2, n_domain, n_rows, n_eq, n_neq,
            rate_eq, support, max_abs_gap, selected
        """
        mask = _domain_mask(self.model, self.cache, condition, use_base_if_none)
        n_domain = int(mask.sum())
        if n_domain == 0:
            return pd.DataFrame(columns=[
                "inv1", "inv2", "n_domain", "n_rows", "n_eq", "n_neq",
                "rate_eq", "support", "max_abs_gap", "selected",
            ])

        # Pre-eval numeric columns once
        eval_cache: Dict[str, pd.Series] = {c: self.model.exprs[c].eval(self.model.df) for c in self.model.numeric_cols}

        arrays: Dict[str, np.ndarray] = {}
        finite_masks: Dict[str, np.ndarray] = {}
        for c in self.model.numeric_cols:
            s = eval_cache[c]
            if require_finite:
                s = s.replace([np.inf, -np.inf], np.nan)
            a = s.to_numpy(dtype=float, copy=False)
            arrays[c] = a
            finite_masks[c] = np.isfinite(a) if require_finite else np.ones_like(a, dtype=bool)

        rows: List[dict] = []
        cols = list(self.model.numeric_cols)
        for i in range(len(cols)):
            x = cols[i]; ax = arrays[x]; fx = finite_masks[x]
            for j in range(i + 1, len(cols)):
                y = cols[j]; ay = arrays[y]; fy = finite_masks[y]

                dm = mask & fx & fy
                n = int(dm.sum())
                if n == 0:
                    continue

                axm = ax[dm]; aym = ay[dm]
                gaps = np.abs(axm - aym)
                eq = gaps <= float(tol)

                n_eq = int(eq.sum())
                n_neq = n - n_eq
                rate_eq = n_eq / n
                support = n / n_domain
                max_abs_gap = float(np.nanmax(gaps)) if n > 0 else np.nan
                selected = (support >= float(min_support)) and (rate_eq >= float(min_eq_rate))

                rows.append({
                    "inv1": x, "inv2": y,
                    "n_domain": n_domain,
                    "n_rows": n,
                    "n_eq": n_eq, "n_neq": n_neq,
                    "rate_eq": rate_eq,
                    "support": support,
                    "max_abs_gap": max_abs_gap,
                    "selected": bool(selected),
                })

        out = pd.DataFrame(rows, columns=[
            "inv1", "inv2", "n_domain", "n_rows", "n_eq", "n_neq",
            "rate_eq", "support", "max_abs_gap", "selected",
        ])
        if out.empty:
            return out
        return out.sort_values(
            ["selected", "rate_eq", "support", "inv1", "inv2"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────
    # 3) Optional: turn selections into Eq(...) Conjectures
    # ──────────────────────────────────────────────────────────────────────
    def make_eq_conjectures(
        self,
        *,
        constants: Optional[pd.DataFrame] = None,
        pairs: Optional[pd.DataFrame] = None,
        condition: Optional[Predicate] = None,  # None means TRUE
        rationalize_constants: bool = True,
        max_denom: int = 64,
        only_selected: bool = True,
    ) -> List[Conjecture]:
        """
        Convert selected rows from analyze_* into Eq(...) conjectures.

        - For constants: produces Eq(inv, Const(value)) under `condition`.
          If `rationalize_constants`, try to use a small Fraction.

        - For pairs: produces Eq(inv1, inv2) under `condition`.

        Returns
        -------
        List[Conjecture]
        """
        H: Predicate = condition if condition is not None else TRUE
        out: List[Conjecture] = []

        # Constants
        if constants is not None and len(constants) > 0:
            iterable = constants.itertuples(index=False)
            for row in iterable:
                if only_selected and not getattr(row, "selected", False):
                    continue
                inv = getattr(row, "inv")
                val = float(getattr(row, "value"))
                if rationalize_constants and np.isfinite(val):
                    frac = _rationalize(val, max_denom)
                    # Use fractional form if it is exact (or very close)
                    if frac.denominator <= max_denom and abs(float(frac) - val) <= 1e-12:
                        rhs = Const(frac)
                    else:
                        rhs = Const(val)
                else:
                    rhs = Const(val)
                out.append(Eq(self.model.exprs[inv], rhs, H))

        # Pair equalities
        if pairs is not None and len(pairs) > 0:
            iterable = pairs.itertuples(index=False)
            for row in iterable:
                if only_selected and not getattr(row, "selected", False):
                    continue
                x = getattr(row, "inv1"); y = getattr(row, "inv2")
                out.append(Eq(self.model.exprs[x], self.model.exprs[y], H))

        return out
