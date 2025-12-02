from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from .core import DataModel, MaskCache
from ..forms.predicates import Predicate, EQ, LE, GE
from ..forms.generic_conjecture import TRUE


__all__ = ["BooleanDiscoveryMiner"]


@dataclass(slots=True)
class BooleanDiscoveryMiner:
    """
    Discover high-confidence boolean predicates from numeric expressions
    and REGISTER them into `model.preds` (and `model.boolean_cols`) so
    downstream ClassLogic and ClassRelationsMiner can reason with them.

    What it can register:
      • Near equalities    : x ≈ y  → predicate  (x == y)  (within tol)
      • Dominant orderings : x ≤ y  or  y ≤ x    (few violations)
      • Constant ties      : x ≈ c  → predicate  (x == c)
    """
    model: DataModel
    cache: MaskCache

    # ---------- internals ----------
    def _domain_mask(self, condition: Optional[Predicate]) -> np.ndarray:
        if condition is not None:
            return np.asarray(self.cache.mask(condition), dtype=bool)
        base = getattr(self.model, "base_pred", TRUE)
        if base is not TRUE:
            return np.asarray(self.cache.mask(base), dtype=bool)
        return np.ones(len(self.model.df), dtype=bool)

    def _unique_pred_key(self, name: str) -> str:
        # Avoid collisions in model.preds
        if name not in self.model.preds:
            return name
        i = 2
        key = f"{name}__{i}"
        while key in self.model.preds:
            i += 1
            key = f"{name}__{i}"
        return key

    def _register_pred(self, key: str, pred: Predicate) -> str:
        """Registers pred in model.preds and boolean_cols; returns key used."""
        k = self._unique_pred_key(key)
        self.model.preds[k] = pred
        # keep boolean_cols list in sync
        if k not in self.model.boolean_cols:
            self.model.boolean_cols.append(k)
            self.model.boolean_cols.sort()
        return k

    # ---------- public API ----------
    def discover_pairwise_rules(
        self,
        condition: Optional[Predicate] = None,
        *,
        require_finite: bool = True,
        # equality rule
        tol: float = 1e-9,
        min_support: float = 0.10,
        min_eq_rate: float = 0.95,
        # ordering rule
        min_rule_rate: float = 0.95,     # e.g., ≥95% of rows satisfy x≤y (few violations)
        max_violations: Optional[int] = None,
        name_style: str = "pretty",      # "pretty": "(x == y)", "(x ≤ y)"; "slug": "eq_x_y", "le_x_y"
    ) -> pd.DataFrame:
        """
        For each numeric pair (x, y) under `condition`:
          • If eq_rate ≥ min_eq_rate → register EQ(x, y)
          • Else if max(rate(x≤y), rate(y≤x)) ≥ min_rule_rate (and support ok) → register LE/GE accordingly

        Returns table with diagnostics and the registered keys (if any).
        """
        df = self.model.df
        domain = self._domain_mask(condition)
        n_dom = int(domain.sum())
        if n_dom == 0:
            return pd.DataFrame(columns=[
                "inv1","inv2","n_rows","support",
                "eq_rate","le_rate","ge_rate",
                "registered_eq","registered_le","registered_ge",
                "key_eq","key_le","key_ge"
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
            # naming policy
            a, b = (x, y)
            if name_style == "slug":
                if kind == "eq": return f"eq_{a}_{b}"
                if kind == "le": return f"le_{a}_{b}"
                if kind == "ge": return f"ge_{a}_{b}"
            # pretty
            if kind == "eq": return f"({a} == {b})"
            if kind == "le": return f"({a} ≤ {b})"
            if kind == "ge": return f"({a} ≥ {b})"
            return f"({a} ? {b})"

        rows = []
        for i in range(len(num_cols)):
            x = num_cols[i]; ax = arrays[x]; fx = finite_masks[x]
            for j in range(i + 1, len(num_cols)):
                y = num_cols[j]; ay = arrays[y]; fy = finite_masks[y]

                m = domain & fx & fy
                n = int(m.sum())
                if n == 0:
                    continue

                axm = ax[m]; aym = ay[m]
                # equality within tol
                eq_mask = np.isclose(axm, aym, atol=tol)
                # orderings
                le_mask = axm <= aym
                ge_mask = axm >= aym

                eq_rate = float(eq_mask.mean())
                le_rate = float(le_mask.mean())
                ge_rate = float(ge_mask.mean())
                support = n / n_dom

                # thresholds
                support_ok = (support >= float(min_support))

                # count violations (for ordering)
                n_le_viol = int((~le_mask).sum())
                n_ge_viol = int((~ge_mask).sum())
                viol_ok_le = True if max_violations is None else (n_le_viol <= int(max_violations))
                viol_ok_ge = True if max_violations is None else (n_ge_viol <= int(max_violations))

                # register decisions (disjoint priority: prefer equality first)
                reg_eq = reg_le = reg_ge = False
                key_eq = key_le = key_ge = None

                if support_ok and (eq_rate >= float(min_eq_rate)):
                    # equality predicate
                    key = self._register_pred(_nm(x, y, "eq"), EQ(self.model.exprs[x], self.model.exprs[y]))
                    reg_eq, key_eq = True, key
                else:
                    # try orderings (pick the stronger one, if above min_rule_rate)
                    if support_ok and (le_rate >= float(min_rule_rate)) and viol_ok_le:
                        key = self._register_pred(_nm(x, y, "le"), LE(self.model.exprs[x], self.model.exprs[y]))
                        reg_le, key_le = True, key
                    if support_ok and (ge_rate >= float(min_rule_rate)) and viol_ok_ge:
                        key = self._register_pred(_nm(x, y, "ge"), GE(self.model.exprs[x], self.model.exprs[y]))
                        reg_ge, key_ge = True, key

                rows.append({
                    "inv1": x, "inv2": y,
                    "n_rows": n, "support": support,
                    "eq_rate": eq_rate, "le_rate": le_rate, "ge_rate": ge_rate,
                    "registered_eq": reg_eq, "registered_le": reg_le, "registered_ge": reg_ge,
                    "key_eq": key_eq, "key_le": key_le, "key_ge": key_ge,
                })

        out = pd.DataFrame(rows, columns=[
            "inv1","inv2","n_rows","support",
            "eq_rate","le_rate","ge_rate",
            "registered_eq","registered_le","registered_ge",
            "key_eq","key_le","key_ge"
        ])
        return out.sort_values(
            ["registered_eq","registered_le","registered_ge","support","inv1","inv2"],
            ascending=[False, False, False, False, True, True],
        ).reset_index(drop=True)

    def discover_constant_ties(
        self,
        condition: Optional[Predicate] = None,
        *,
        require_finite: bool = True,
        tol: float = 1e-9,
        min_support: float = 0.10,
        min_const_rate: float = 0.95,
        rationalize: bool = True,
        max_denom: int = 64,
        name_style: str = "pretty",  # "(x == 3/2)" vs "eqc_x_3/2"
    ) -> pd.DataFrame:
        """
        For each invariant x, if ~all values equal (within tol), register a constant-equality predicate.
        """
        df = self.model.df
        domain = self._domain_mask(condition)
        n_dom = int(domain.sum())
        if n_dom == 0:
            return pd.DataFrame(columns=[
                "inv","n_rows","support","value","const_rate","registered","key"
            ])

        num_cols = list(self.model.numeric_cols)
        eval_cache: Dict[str, pd.Series] = {c: self.model.exprs[c].eval(df) for c in num_cols}

        rows = []
        for c in num_cols:
            s = eval_cache[c]
            if require_finite:
                s = s.replace([np.inf, -np.inf], np.nan)
            a = s.to_numpy(dtype=float, copy=False)

            m = domain & np.isfinite(a)
            n = int(m.sum())
            if n == 0:
                continue

            am = a[m]
            # pick a representative value (median) then compute within-tol agreement
            median = float(np.nanmedian(am))
            const_mask = np.isclose(am, median, atol=tol)
            const_rate = float(const_mask.mean())
            support = n / n_dom

            registered = False
            key = None
            if (support >= float(min_support)) and (const_rate >= float(min_const_rate)):
                val = median
                if rationalize:
                    try:
                        from fractions import Fraction
                        val = Fraction(median).limit_denominator(int(max_denom))
                        disp = f"{val.numerator}/{val.denominator}" if val.denominator != 1 else f"{val.numerator}"
                        v_for_eq = float(val)  # stable numeric compare
                    except Exception:
                        disp = f"{median:.12g}"
                        v_for_eq = median
                else:
                    disp = f"{median:.12g}"
                    v_for_eq = median

                if name_style == "slug":
                    key_name = f"eqc_{c}_{disp}"
                else:
                    key_name = f"({c} == {disp})"

                k = self._register_pred(key_name, EQ(self.model.exprs[c], v_for_eq))
                registered, key = True, k

            rows.append({
                "inv": c,
                "n_rows": n,
                "support": support,
                "value": median,
                "const_rate": const_rate,
                "registered": registered,
                "key": key,
            })

        out = pd.DataFrame(rows, columns=[
            "inv","n_rows","support","value","const_rate","registered","key"
        ])
        return out.sort_values(
            ["registered","const_rate","support","inv"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
