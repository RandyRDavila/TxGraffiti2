# src/txgraffiti2025/forms/asymptotic_miner.py
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd

from txgraffiti2025.forms.utils import to_expr, Expr
from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.graffiti_asymptotic import (
    AsymptoticConjecture,
    lim_to_infinity, lim_to_zero, lim_to_const, lim_ratio_const,
)
from txgraffiti2025.graffiti_relations import GraffitiClassRelations


# ───────────────────────────── Config ───────────────────────────── #

@dataclass
class AsymptoticSearchConfig:
    # correlation gate to call a monotone trend (full support)
    min_abs_rho: float = 0.40
    # tail window by quantile of parameter t
    tail_quantile: float = 0.70
    # minimal rows on hypothesis support
    min_support_n: int = 20
    # “zero” vs “const” separation
    const_cv_max: float = 0.08          # coefficient of variation on tail
    const_drift_max: float = 0.05       # |mean_tail - mean_mid| / max(1, |mean_mid|)
    ratio_cv_max: float = 0.06
    ratio_drift_max: float = 0.04
    # rationalization for constants
    max_denom: int = 50


# ────────────────────────────── Miner ───────────────────────────── #

class AsymptoticMiner:
    """
    Search asymptotic limit patterns for a *target* against candidate parameters,
    under each (nonredundant) hypothesis from GraffitiClassRelations.

    Emits `AsymptoticConjecture` objects with clean pretty/LaTeX (∞ rendered as '∞').
    """

    def __init__(self, gcr: GraffitiClassRelations, *, cfg: Optional[AsymptoticSearchConfig] = None):
        self.gcr = gcr
        self.df = gcr.df
        self.cfg = cfg or AsymptoticSearchConfig()

    # ------------------------- public API ------------------------- #

    def generate_asymptotics_for_target(
        self,
        target: str | Expr,
        *,
        t_candidates: Optional[Iterable[str]] = None,
        ratio_denominators: Optional[Iterable[str]] = None,
        hyps: Optional[Iterable[Tuple[str, Predicate]]] = None,
        drop_constant_cols: bool = True,
    ) -> List[AsymptoticConjecture]:
        """
        For each hypothesis H and each parameter t in `t_candidates`,
        propose one of: lim f→∞, lim f→0, lim f→const,
        and (optionally) ratio-constant limits lim f/g → const for each g.

        Results are deduped by signature and globally sorted by (support, |rho| surrogate).
        """
        y_expr = to_expr(target) if isinstance(target, str) else target
        y_name = getattr(y_expr, "name", None) or str(target)

        # candidate parameters and ratio denominators default to numeric columns minus y
        num_cols = [c for c in self.gcr.expr_cols if c != y_name]
        t_cols = list(t_candidates or num_cols)
        g_cols = list(ratio_denominators or num_cols)

        hyps_iter = list(hyps or getattr(self.gcr, "nonredundant_conjunctions_", []))
        if not hyps_iter:
            # fall back to base hypothesis explicitly
            hyps_iter = [(self.gcr.base_hypothesis_name, self.gcr.base_hypothesis)]

        out: List[AsymptoticConjecture] = []
        seen = set()

        for hyp_name, H in hyps_iter:
            mask = self._mask(H)
            n = int(mask.sum())
            if n < self.cfg.min_support_n:
                continue

            dfH = self.df.loc[mask]
            num = dfH.apply(pd.to_numeric, errors="coerce")

            y = num.get(y_name)
            if y is None or y.notna().sum() < self.cfg.min_support_n:
                continue
            if drop_constant_cols and y.nunique(dropna=True) <= 1:
                continue

            for t_name in t_cols:
                if t_name == y_name:
                    continue
                t = num.get(t_name)
                if t is None or t.notna().sum() < self.cfg.min_support_n:
                    continue
                if drop_constant_cols and t.nunique(dropna=True) <= 1:
                    continue

                # full-support monotonicity gate (Spearman)
                rho = _spearman(t, y)
                if not np.isfinite(rho):
                    continue

                # Tail window on t (top quantile)
                tail_idx = self._tail_index_by_t(t, q=self.cfg.tail_quantile)
                if tail_idx.size < max(8, int(0.2 * self.cfg.min_support_n)):
                    continue

                y_tail = y.to_numpy()[tail_idx]
                y_mid = y.to_numpy()[self._mid_index_by_t(t, q=self.cfg.tail_quantile)]

                # Decide limit type for f(t)
                cj = self._decide_limit_f_of_t(
                    hyp=H, y_name=y_name, t_name=t_name,
                    rho=rho, y_tail=y_tail, y_mid=y_mid
                )
                if cj is not None:
                    sig = cj.signature()
                    if sig not in seen:
                        seen.add(sig)
                        out.append(cj)

                # Ratio-constant: for each g (≠ y), test const on tail
                for g_name in g_cols:
                    if g_name in (y_name, t_name):
                        continue
                    g = num.get(g_name)
                    if g is None or g.notna().sum() < self.cfg.min_support_n:
                        continue
                    if drop_constant_cols and g.nunique(dropna=True) <= 1:
                        continue

                    g_arr = g.to_numpy()
                    tail = tail_idx
                    # require positive/finite denominator on the tail
                    valid = np.isfinite(y.to_numpy()[tail]) & np.isfinite(g_arr[tail]) & (g_arr[tail] != 0)
                    if valid.sum() < max(6, int(0.3 * self.cfg.min_support_n)):
                        continue
                    r = (y.to_numpy()[tail][valid]) / (g_arr[tail][valid])

                    if r.size == 0:
                        continue

                    if _is_tail_constant(r, cv_max=self.cfg.ratio_cv_max, drift_max=self.cfg.ratio_drift_max):
                        c = _nice_fraction(np.nanmean(r), self.cfg.max_denom)
                        ac = lim_ratio_const(f=y_name, g=g_name, t=t_name, c=c, condition=H, max_denom=self.cfg.max_denom)
                        sig = ac.signature()
                        if sig not in seen:
                            seen.add(sig)
                            out.append(ac)

        # sort: put base/large support first by approximating support with frequency of hyp mask,
        # then rough “strength” proxy by preferring ratio-constants and strong rho decisions first
        out.sort(key=lambda c: (self._support_size(c.condition), _relation_strength_key(c)), reverse=True)
        return out

    # ------------------------- decisions ------------------------- #

    def _decide_limit_f_of_t(self, *, hyp: Predicate, y_name: str, t_name: str,
                             rho: float, y_tail: np.ndarray, y_mid: np.ndarray) -> Optional[AsymptoticConjecture]:
        """
        Decide among: lim→∞, lim→0, lim→const, or None.
        Uses full-support rho and tail shape/statistics.
        """
        abs_rho = abs(float(rho))
        if abs_rho >= self.cfg.min_abs_rho:
            # strong monotone trend: check growth vs decay
            m_tail = np.nanmedian(y_tail)
            m_mid  = np.nanmedian(y_mid)
            # robust slope sign via medians:
            if m_mid == 0:
                ratio = np.inf if m_tail > 0 else (-np.inf if m_tail < 0 else 1.0)
            else:
                ratio = m_tail / m_mid

            if rho > 0 and ratio > (1.0 + 4 * self.cfg.const_drift_max):
                # increasing & tail clearly larger → ∞
                return lim_to_infinity(f=y_name, t=t_name, condition=hyp)

            if rho < 0 and ratio < (1.0 - 4 * self.cfg.const_drift_max):
                # decreasing & tail clearly smaller → 0 (if tail concentrates near 0)
                if _looks_zero(y_tail):
                    return lim_to_zero(f=y_name, t=t_name, condition=hyp)

        # If not clearly monotone to ∞/0, try “const” on tail
        if _is_tail_constant(y_tail, cv_max=self.cfg.const_cv_max, drift_max=self.cfg.const_drift_max, mid=y_mid):
            c = _nice_fraction(np.nanmean(y_tail), self.cfg.max_denom)
            return lim_to_const(f=y_name, t=t_name, c=c, condition=hyp, max_denom=self.cfg.max_denom)

        return None

    # ------------------------- helpers ------------------------- #

    def _mask(self, H: Predicate) -> np.ndarray:
        m = H.mask(self.df).reindex(self.df.index, fill_value=False)
        if m.dtype is not bool:
            m = m.fillna(False).astype(bool, copy=False)
        return m.to_numpy(dtype=bool, copy=False)

    def _support_size(self, H: Predicate) -> int:
        try:
            return int(self._mask(H).sum())
        except Exception:
            return 0

    def _tail_index_by_t(self, t: pd.Series, *, q: float) -> np.ndarray:
        """Indices of the top-quantile of t (tie-inclusive, stable)."""
        a = t.to_numpy(dtype=float, copy=False)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.array([], dtype=int)
        thr = np.nanquantile(t, q)
        idx = np.flatnonzero(t.to_numpy(dtype=float, copy=False) >= thr)
        return idx

    def _mid_index_by_t(self, t: pd.Series, *, q: float) -> np.ndarray:
        """Middle window to compare with tail (between 0.4 and 0.6 unless q is smaller)."""
        lo = min(0.40, q - 0.1) if q > 0.5 else 0.40
        hi = min(0.60, q - 0.05) if q > 0.6 else 0.60
        a = t.to_numpy(dtype=float, copy=False)
        if not np.isfinite(a).any():
            return np.array([], dtype=int)
        lo_thr = np.nanquantile(t, lo)
        hi_thr = np.nanquantile(t, hi)
        idx = np.flatnonzero((a >= lo_thr) & (a <= hi_thr))
        return idx


# ─────────────────────────── stats helpers ─────────────────────────── #

def _spearman(x: pd.Series, y: pd.Series) -> float:
    try:
        return float(x.corr(y, method="spearman"))
    except Exception:
        # simple fallback via ranks
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        xc = xr - np.nanmean(xr); yc = yr - np.nanmean(yr)
        num = np.nansum(xc * yc)
        den = np.sqrt(np.nansum(xc * xc) * np.nansum(yc * yc))
        return float(num / den) if den > 0 else np.nan


def _is_tail_constant(values: np.ndarray, *, cv_max: float, drift_max: float, mid: Optional[np.ndarray] = None) -> bool:
    v = values[np.isfinite(values)]
    if v.size < 6:
        return False
    mu = np.nanmean(v)
    sig = np.nanstd(v)
    cv = 0.0 if mu == 0 else (sig / max(1.0, abs(mu)))
    if cv > cv_max:
        return False
    if mid is not None and np.isfinite(mid).any():
        mu_mid = np.nanmean(mid[np.isfinite(mid)])
        drift = abs(mu - mu_mid) / max(1.0, abs(mu_mid))
        if drift > drift_max:
            return False
    return True


def _looks_zero(values: np.ndarray) -> bool:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return False
    # “near zero” if 95% quantile is tiny relative to mid-scale of the series
    q95 = np.nanquantile(v, 0.95)
    scale = max(1.0, abs(np.nanmedian(v)))
    return (q95 / scale) < 0.02


def _nice_fraction(x: float, max_denom: int) -> Fraction:
    try:
        fr = Fraction(float(x)).limit_denominator(max_denom)
        return fr
    except Exception:
        return Fraction(0, 1)


def _relation_strength_key(cj: AsymptoticConjecture) -> float:
    """Tiny heuristic for sorting: ratio-const slightly preferred over plain const, then ∞/0."""
    kind = getattr(cj.relation, "kind", None)
    if not kind:
        return 0.0
    name = kind.value
    if "ratio" in name:
        return 3.0
    if "to_const" in name:
        return 2.0
    if "to_infty" in name or "to_zero" in name:
        return 1.5
    return 0.0
